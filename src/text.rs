use std::collections::HashMap;

use crate::mapdata::Label;

/// Vertex for text rendering — anchored in world space, sized in screen pixels.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TextVertex {
    pub world_pos: [f32; 2],
    pub pixel_offset: [f32; 2],
    pub uv: [f32; 2],
    pub color: [f32; 4],
    /// 0.0 for "large" labels (State/City/District/Marker), 1.0 for "small"
    /// labels (Street/Building/POI/Park). The shader uses this to multiply
    /// only small labels by `camera.small_label_alpha`, so street names fade
    /// out alongside the z14 tiles when zoom drops below threshold while
    /// state/city names stay put.
    pub kind_flag: f32,
}

/// Metrics for a single glyph in the atlas.
#[derive(Clone, Copy)]
struct GlyphInfo {
    uv_min: [f32; 2],
    uv_max: [f32; 2],
    size: [f32; 2],
    bearing: [f32; 2],
    advance: f32,
}

/// Fast ASCII glyph lookup table (chars 32..127 → index 0..94).
const GLYPH_TABLE_SIZE: usize = 95; // 127 - 32

/// The text rendering system: glyph atlas + GPU pipeline.
pub struct TextSystem {
    pipeline: wgpu::RenderPipeline,
    atlas_texture: wgpu::Texture,
    atlas_bind_group: wgpu::BindGroup,
    vertex_buffer: Option<wgpu::Buffer>,
    index_buffer: Option<wgpu::Buffer>,
    num_indices: u32,
    glyph_table: [Option<GlyphInfo>; GLYPH_TABLE_SIZE],
    font_size: f32,
}

impl TextSystem {
    #[inline(always)]
    fn get_glyph(&self, ch: char) -> Option<&GlyphInfo> {
        let idx = ch as u32;
        if idx >= 32 && idx < 127 {
            self.glyph_table[(idx - 32) as usize].as_ref()
        } else {
            None
        }
    }
}

const ATLAS_SIZE: u32 = 2048;
const FONT_SIZE: f32 = 64.0;
/// Tighten letter spacing (1.0 = normal, <1.0 = tighter)
/// Global baseline kerning (applied at atlas level)
const KERNING: f32 = 0.82;

impl TextSystem {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, config: &wgpu::SurfaceConfiguration, camera_bind_group_layout: &wgpu::BindGroupLayout, msaa_samples: u32) -> Self {
        // Load a system font
        let font = load_system_font();
        let font_size = FONT_SIZE;

        // Build glyph atlas and convert HashMap to fast array lookup
        let (atlas_data, glyph_map) = build_glyph_atlas(&font, font_size);
        let mut glyph_table: [Option<GlyphInfo>; GLYPH_TABLE_SIZE] = std::array::from_fn(|_| None);
        for (ch, info) in glyph_map {
            let idx = ch as u32;
            if idx >= 32 && idx < 127 {
                glyph_table[(idx - 32) as usize] = Some(info);
            }
        }

        // Create atlas texture
        let atlas_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Glyph Atlas"),
            size: wgpu::Extent3d {
                width: ATLAS_SIZE,
                height: ATLAS_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &atlas_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &atlas_data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(ATLAS_SIZE),
                rows_per_image: Some(ATLAS_SIZE),
            },
            wgpu::Extent3d {
                width: ATLAS_SIZE,
                height: ATLAS_SIZE,
                depth_or_array_layers: 1,
            },
        );

        let atlas_view = atlas_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Glyph Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Atlas bind group
        let atlas_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Atlas Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let atlas_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Atlas Bind Group"),
            layout: &atlas_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&atlas_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        // Shader + pipeline
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Text Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("text.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Text Pipeline Layout"),
            bind_group_layouts: &[camera_bind_group_layout, &atlas_bind_group_layout],
            push_constant_ranges: &[],
        });

        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<TextVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute { offset: 0, shader_location: 0, format: wgpu::VertexFormat::Float32x2 },   // world_pos
                wgpu::VertexAttribute { offset: 8, shader_location: 1, format: wgpu::VertexFormat::Float32x2 },   // pixel_offset
                wgpu::VertexAttribute { offset: 16, shader_location: 2, format: wgpu::VertexFormat::Float32x2 },  // uv
                wgpu::VertexAttribute { offset: 24, shader_location: 3, format: wgpu::VertexFormat::Float32x4 },  // color
                wgpu::VertexAttribute { offset: 40, shader_location: 4, format: wgpu::VertexFormat::Float32 },    // kind_flag
            ],
        };

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Text Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[vertex_layout],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: crate::gpu::DEPTH_FORMAT,
                depth_write_enabled: false, // Text always on top, don't write depth
                depth_compare: wgpu::CompareFunction::Always,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: msaa_samples,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        Self {
            pipeline,
            atlas_texture,
            atlas_bind_group,
            vertex_buffer: None,
            index_buffer: None,
            num_indices: 0,
            glyph_table,
            font_size,
        }
    }

    /// Rebuild the glyph atlas with a new font. `font_bytes` is a raw TTF/OTF
    /// file. On success, rewrites the atlas texture and glyph table in place;
    /// the caller should mark labels_dirty so the vertex buffer rebuilds with
    /// the new UVs. Returns false if the font bytes were unparseable.
    pub fn reload_font(&mut self, queue: &wgpu::Queue, font_bytes: &[u8]) -> bool {
        let settings = fontdue::FontSettings {
            collection_index: 0,
            ..Default::default()
        };
        let font = match fontdue::Font::from_bytes(font_bytes.to_vec(), settings) {
            Ok(f) => f,
            Err(_) => return false,
        };
        let (atlas_data, glyph_map) = build_glyph_atlas(&font, self.font_size);
        let mut new_table: [Option<GlyphInfo>; GLYPH_TABLE_SIZE] = std::array::from_fn(|_| None);
        for (ch, info) in glyph_map {
            let idx = ch as u32;
            if idx >= 32 && idx < 127 {
                new_table[(idx - 32) as usize] = Some(info);
            }
        }
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.atlas_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &atlas_data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(ATLAS_SIZE),
                rows_per_image: Some(ATLAS_SIZE),
            },
            wgpu::Extent3d {
                width: ATLAS_SIZE,
                height: ATLAS_SIZE,
                depth_or_array_layers: 1,
            },
        );
        self.glyph_table = new_table;
        true
    }

    /// Reset to the embedded default font.
    pub fn reset_font(&mut self, queue: &wgpu::Queue) {
        self.reload_font(queue, EMBEDDED_FONT);
    }

    /// Estimate the world-space collision radius of a label, accounting for zoom.
    /// When zoomed out, labels occupy more world-space per pixel, so the radius
    /// must increase to prevent overlapping labels on screen.
    fn label_world_radius(&self, label: &Label, zoom: f32) -> f32 {
        let scale = label.font_scale();
        let total_width: f32 = label.text.chars()
            .filter_map(|c| self.get_glyph(c))
            .map(|g| g.advance * scale)
            .sum();
        // Base conversion at zoom=1: ~120 pixels ≈ 1 world unit.
        // The text shader scales by pow(2, zoom*0.5), so at lower zoom the text
        // covers more world-space. Invert that scaling for collision radius.
        let zoom_factor = 2.0f32.powf(-zoom * 0.5).max(0.1);
        let px_to_world = zoom_factor / 120.0;
        total_width * 1.2 * px_to_world
    }

    /// Generate text quads for all labels and upload to GPU.
    pub fn upload_labels(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, labels: &[Label]) {
        self.upload_labels_at_zoom(device, queue, labels, 1.0, 1000.0);
    }

    pub fn upload_labels_at_zoom(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, labels: &[Label], zoom: f32, viewport_min: f32) {
        self.upload_labels_themed(device, queue, labels, zoom, [0.0; 4], [0.0; 4], viewport_min);
    }

    pub fn upload_labels_themed(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        labels: &[Label],
        zoom: f32,
        road_tint: [f32; 4],
        land_tint: [f32; 4],
        viewport_min: f32,
    ) {
        let mut vertices: Vec<TextVertex> = Vec::with_capacity(256 * 20);
        let mut indices: Vec<u32> = Vec::with_capacity(256 * 30);

        // Adapt label colors to the theme.
        // If road tint is active, compute luminance — dark roads get light text, light roads get dark text.
        let road_active = road_tint[3] > 0.5;
        let road_lum = road_tint[0] * 0.299 + road_tint[1] * 0.587 + road_tint[2] * 0.114;
        let land_active = land_tint[3] > 0.5;
        let land_lum = land_tint[0] * 0.299 + land_tint[1] * 0.587 + land_tint[2] * 0.114;

        let (text_color, halo_color) = if land_active && land_lum < 0.3 {
            ([1.0_f32, 1.0, 1.0, 1.0], [0.0_f32, 0.0, 0.0, 0.9])
        } else {
            ([0.0_f32, 0.0, 0.0, 1.0], [1.0_f32, 1.0, 1.0, 0.9])
        };

        let (street_text, street_halo) = if road_active {
            if road_lum < 0.3 {
                ([1.0_f32, 1.0, 1.0, 1.0], [0.0_f32, 0.0, 0.0, 0.8])
            } else {
                ([0.0_f32, 0.0, 0.0, 1.0], [1.0_f32, 1.0, 1.0, 0.8])
            }
        } else {
            ([0.0_f32, 0.0, 0.0, 0.9], [1.0_f32, 1.0, 1.0, 0.8])
        };

        // Real world-space AABB collision. Replaces the old spatial-grid
        // heuristic — we now measure each label's actual visual extent and
        // do rectangle-rectangle intersection tests, so labels can't visually
        // overlap regardless of their size differences.
        // world_per_px matches the curved-label placement derivation so our
        // collision bounds correspond to actual rendered pixel extents.
        let vp_scale = 1000.0 / viewport_min.max(1.0);
        let world_per_px = 0.1 / 2.0_f32.powf(zoom * 0.5) * vp_scale;

        struct LabelAabb {
            min_x: f32, max_x: f32, min_y: f32, max_y: f32,
        }
        let mut accepted: Vec<LabelAabb> = Vec::with_capacity(512);
        let max_labels = 1500;
        let mut label_count = 0u32;

        // Priority order (lowest number = highest priority):
        // State > City > Subdivision > Listing > Park > Street (Road) > Poi (Business) > Building
        let mut sorted_labels: Vec<&Label> = labels.iter().collect();
        sorted_labels.sort_by_key(|l| match l.kind {
            crate::mapdata::LabelKind::State => 0,
            crate::mapdata::LabelKind::City => 1,
            crate::mapdata::LabelKind::District => 2,
            crate::mapdata::LabelKind::Marker => 3,
            crate::mapdata::LabelKind::Park => 4,
            crate::mapdata::LabelKind::Street => 5,
            crate::mapdata::LabelKind::Poi => 6,
            crate::mapdata::LabelKind::Building => 7,
        });

        for label in &sorted_labels {
            if label_count >= max_labels {
                break;
            }

            // Filter labels by zoom level
            let min_zoom = match label.kind {
                crate::mapdata::LabelKind::State => -10.0,
                crate::mapdata::LabelKind::City => -10.0,
                crate::mapdata::LabelKind::District => -4.0,
                crate::mapdata::LabelKind::Marker => -10.0,
                crate::mapdata::LabelKind::Park => -2.0,
                crate::mapdata::LabelKind::Street => -10.0,
                crate::mapdata::LabelKind::Building => 0.5,
                crate::mapdata::LabelKind::Poi => 0.0,
            };
            if zoom < min_zoom {
                continue;
            }

            let is_listing = matches!(label.kind, crate::mapdata::LabelKind::Marker);
            let scale = label.font_scale();
            let spacing = label.letter_spacing();

            let (label_text_color, label_halo_color) = match label.kind {
                crate::mapdata::LabelKind::Marker => {
                    if land_active && land_lum < 0.3 {
                        ([1.0_f32, 1.0, 1.0, 1.0], [0.0_f32, 0.0, 0.0, 0.9])
                    } else {
                        ([0.0_f32, 0.0, 0.0, 1.0], [1.0_f32, 1.0, 1.0, 0.9])
                    }
                }
                crate::mapdata::LabelKind::Street => {
                    (text_color, halo_color)
                }
                _ => (text_color, halo_color),
            };

            // Per-vertex kind flag — 1.0 for "small" labels that fade alongside
            // z14 tiles when zoom drops below threshold, 0.0 for large place
            // names (State / City / District / user Marker) which stay visible.
            let kind_flag: f32 = match label.kind {
                crate::mapdata::LabelKind::State
                | crate::mapdata::LabelKind::City
                | crate::mapdata::LabelKind::District
                | crate::mapdata::LabelKind::Marker => 0.0,
                _ => 1.0,
            };

            // Measure total width for centering and collision bounds
            let total_width: f32 = label.text.chars()
                .filter_map(|c| self.get_glyph(c))
                .map(|g| g.advance * scale * spacing)
                .sum();

            // World-space AABB based on actual rendered label extent.
            // Small inflation hides hairline edge-touching at the pixel level.
            let half_w_world = total_width * 0.5 * world_per_px * 1.05;
            let half_h_world = self.font_size * scale * 0.5 * world_per_px * 1.5;
            let my_aabb = LabelAabb {
                min_x: label.position[0] - half_w_world,
                max_x: label.position[0] + half_w_world,
                min_y: label.position[1] - half_h_world,
                max_y: label.position[1] + half_h_world,
            };

            // Listings (user's own homes) always render — they're the primary overlay.
            if !is_listing {
                let overlaps = accepted.iter().any(|a| {
                    !(my_aabb.max_x < a.min_x
                        || my_aabb.min_x > a.max_x
                        || my_aabb.max_y < a.min_y
                        || my_aabb.min_y > a.max_y)
                });
                if overlaps {
                    continue;
                }
            }
            accepted.push(my_aabb);
            label_count += 1;

            // Curved road labels: place each glyph at its own world position along the path
            if let Some(ref path) = label.path {
                if path.len() >= 2 {
                    self.emit_curved_label(
                        path, &label.text, &label.position, scale, spacing, total_width, zoom,
                        label_text_color, label_halo_color,
                        viewport_min,
                        &mut vertices, &mut indices,
                    );
                    continue;
                }
            }

            // POI labels: wrap into multiple centered lines
            if matches!(label.kind, crate::mapdata::LabelKind::Poi) {
                let lines = wrap_text(&label.text, 14);
                let line_height = self.font_size * scale * 1.2;
                let total_h = line_height * lines.len() as f32;
                let rotate = |x: f32, y: f32| -> (f32, f32) { (x, y) };

                for (li, line) in lines.iter().enumerate() {
                    let line_w: f32 = line.chars()
                        .filter_map(|c| self.get_glyph(c))
                        .map(|g| g.advance * scale * spacing)
                        .sum();
                    let line_y = -total_h * 0.5 + li as f32 * line_height;

                    // Halo
                    for &(ox, oy) in &[
                        (-1.5_f32, 0.0_f32), (1.5, 0.0), (0.0, -1.5), (0.0, 1.5),
                        (-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0),
                    ] {
                        let mut cx = -line_w * 0.5 + ox;
                        let cy = line_y + oy;
                        for ch in line.chars() {
                            if let Some(glyph) = self.get_glyph(ch) {
                                self.emit_glyph_quad_rotated(
                                    label.position,
                                    cx + glyph.bearing[0] * scale,
                                    cy + glyph.bearing[1] * scale,
                                    glyph, scale, label_halo_color, kind_flag, &rotate,
                                    &mut vertices, &mut indices,
                                );
                                cx += glyph.advance * scale * spacing;
                            }
                        }
                    }
                    // Text
                    let mut cx = -line_w * 0.5;
                    let cy = line_y;
                    for ch in line.chars() {
                        if let Some(glyph) = self.get_glyph(ch) {
                            self.emit_glyph_quad_rotated(
                                label.position,
                                cx + glyph.bearing[0] * scale,
                                cy + glyph.bearing[1] * scale,
                                glyph, scale, label_text_color, kind_flag, &rotate,
                                &mut vertices, &mut indices,
                            );
                            cx += glyph.advance * scale * spacing;
                        }
                    }
                }
                continue;
            }

            // Straight labels (non-road or fallback)
            let cos_a = label.angle.cos();
            let sin_a = label.angle.sin();
            let rotate = |x: f32, y: f32| -> (f32, f32) {
                (x * cos_a - y * sin_a, x * sin_a + y * cos_a)
            };

            // Render halo (outline) — 8 offset copies for bolder appearance
            for &(ox, oy) in &[
                (-1.5_f32, 0.0_f32), (1.5, 0.0), (0.0, -1.5), (0.0, 1.5),
                (-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0),
            ] {
                let mut cursor_x = -total_width * 0.5 + ox;
                let cursor_y = -self.font_size * scale * 0.5 + oy;

                for ch in label.text.chars() {
                    if let Some(glyph) = self.get_glyph(ch) {
                        self.emit_glyph_quad_rotated(
                            label.position,
                            cursor_x + glyph.bearing[0] * scale,
                            cursor_y + glyph.bearing[1] * scale,
                            glyph,
                            scale,
                            label_halo_color,
                            kind_flag,
                            &rotate,
                            &mut vertices,
                            &mut indices,
                        );
                        cursor_x += glyph.advance * scale * spacing;
                    }
                }
            }

            // Render text (on top of halo)
            let mut cursor_x = -total_width * 0.5;
            let cursor_y = -self.font_size * scale * 0.5;

            for ch in label.text.chars() {
                if let Some(glyph) = self.get_glyph(ch) {
                    self.emit_glyph_quad_rotated(
                        label.position,
                        cursor_x + glyph.bearing[0] * scale,
                        cursor_y + glyph.bearing[1] * scale,
                        glyph,
                        scale,
                        label_text_color,
                        kind_flag,
                        &rotate,
                        &mut vertices,
                        &mut indices,
                    );
                    cursor_x += glyph.advance * scale * spacing;
                }
            }
        }

        if vertices.is_empty() {
            return;
        }


        self.vertex_buffer = Some(crate::gpu::safe_buffer(
            device, queue, "Text Vertex Buffer",
            bytemuck::cast_slice(&vertices), wgpu::BufferUsages::VERTEX,
        ));

        self.index_buffer = Some(crate::gpu::safe_buffer(
            device, queue, "Text Index Buffer",
            bytemuck::cast_slice(&indices), wgpu::BufferUsages::INDEX,
        ));

        self.num_indices = indices.len() as u32;
    }

    fn emit_glyph_quad_rotated(
        &self,
        world_pos: [f32; 2],
        px: f32, py: f32,
        glyph: &GlyphInfo,
        scale: f32,
        color: [f32; 4],
        kind_flag: f32,
        rotate: &dyn Fn(f32, f32) -> (f32, f32),
        vertices: &mut Vec<TextVertex>,
        indices: &mut Vec<u32>,
    ) {
        let w = glyph.size[0] * scale;
        let h = glyph.size[1] * scale;

        if w < 0.1 || h < 0.1 {
            return;
        }

        let base = vertices.len() as u32;

        // Rotate each corner's pixel offset around the anchor point
        let (r0x, r0y) = rotate(px, py);
        let (r1x, r1y) = rotate(px + w, py);
        let (r2x, r2y) = rotate(px, py + h);
        let (r3x, r3y) = rotate(px + w, py + h);

        vertices.push(TextVertex { world_pos, pixel_offset: [r0x, r0y], uv: [glyph.uv_min[0], glyph.uv_min[1]], color, kind_flag });
        vertices.push(TextVertex { world_pos, pixel_offset: [r1x, r1y], uv: [glyph.uv_max[0], glyph.uv_min[1]], color, kind_flag });
        vertices.push(TextVertex { world_pos, pixel_offset: [r2x, r2y], uv: [glyph.uv_min[0], glyph.uv_max[1]], color, kind_flag });
        vertices.push(TextVertex { world_pos, pixel_offset: [r3x, r3y], uv: [glyph.uv_max[0], glyph.uv_max[1]], color, kind_flag });

        indices.extend_from_slice(&[base, base + 1, base + 2, base + 1, base + 3, base + 2]);
    }

    /// Place glyphs along a world-space polyline path, curving with the road.
    fn emit_curved_label(
        &self,
        path: &[[f32; 2]],
        text: &str,
        anchor: &[f32; 2],
        scale: f32,
        spacing: f32,
        total_width_px: f32,
        zoom: f32,
        text_color: [f32; 4],
        halo_color: [f32; 4],
        viewport_min: f32,
        vertices: &mut Vec<TextVertex>,
        indices: &mut Vec<u32>,
    ) {
        // Compute cumulative arc-length along path
        let mut cum_len = vec![0.0f32; path.len()];
        for i in 1..path.len() {
            let dx = path[i][0] - path[i - 1][0];
            let dy = path[i][1] - path[i - 1][1];
            cum_len[i] = cum_len[i - 1] + (dx * dx + dy * dy).sqrt();
        }
        let path_len = *cum_len.last().unwrap_or(&0.0);
        if path_len < 1e-6 { return; }

        // Find the arc-length distance of the anchor point on the path
        let mut anchor_dist = 0.0f32;
        let mut best_sq = f32::MAX;
        for i in 0..path.len().saturating_sub(1) {
            let ax = path[i][0]; let ay = path[i][1];
            let bx = path[i + 1][0]; let by = path[i + 1][1];
            let dx = bx - ax; let dy = by - ay;
            let seg_sq = dx * dx + dy * dy;
            if seg_sq < 1e-12 { continue; }
            let t = ((anchor[0] - ax) * dx + (anchor[1] - ay) * dy) / seg_sq;
            let t = t.clamp(0.0, 1.0);
            let px = ax + dx * t;
            let py = ay + dy * t;
            let dist_sq = (anchor[0] - px) * (anchor[0] - px) + (anchor[1] - py) * (anchor[1] - py);
            if dist_sq < best_sq {
                best_sq = dist_sq;
                anchor_dist = cum_len[i] + t * (cum_len[i + 1] - cum_len[i]);
            }
        }

        let glyphs: Vec<(char, f32)> = text.chars()
            .filter_map(|c| self.get_glyph(c).map(|g| (c, g.advance * scale * spacing)))
            .collect();
        if glyphs.is_empty() { return; }

        // Convert pixel advance to world units, matching the text shader's zoom scaling.
        // Shader zoom_scale = 2^(z*0.5), ortho projection scale = 100 / 2^z.
        // Derivation: world_per_pixel_screen = 100 / (viewport * pow(2, z*0.5)).
        // Constant 0.1 comes from 100 / 1000 (the viewport_min normalizer).
        // Previously 0.05 was tuned for the old shader's extra (1000/ref_size) factor;
        // with that removed, the correct base is 0.1.
        let vp_scale = 1000.0 / viewport_min.max(1.0);
        let world_per_px = 0.1 / 2.0_f32.powf(zoom * 0.5) * vp_scale;

        // Starting distance along the path (center the text on the anchor)
        let start_dist = anchor_dist - total_width_px * world_per_px * 0.5;
        let end_dist = start_dist + total_width_px * world_per_px;

        // Skip label entirely if it doesn't fit within the path
        if start_dist < 0.0 || end_dist > path_len { return; }

        // Decide text direction ONCE for the whole label.
        // Sample the overall tangent at the label center — if it points leftward,
        // flip the whole label so text reads left-to-right with the font baseline
        // toward the bottom of the screen.
        let (mid_tx, mid_ty) = sample_tangent(path, &cum_len, anchor_dist);
        let mid_angle = (-mid_ty).atan2(mid_tx);
        let flip = mid_angle.abs() > std::f32::consts::FRAC_PI_2;

        // Emit each glyph along the path — skip glyphs that fall outside the path
        let emit_glyphs = |color: [f32; 4], ox: f32, oy: f32, verts: &mut Vec<TextVertex>, idxs: &mut Vec<u32>| {
            let mut cursor_px = 0.0f32;
            for &(ch, advance_px) in &glyphs {
                let glyph = match self.get_glyph(ch) {
                    Some(g) => g,
                    None => { cursor_px += advance_px; continue; }
                };

                // Center of this glyph in world-distance
                let char_center_px = cursor_px + advance_px * 0.5;
                let d = if flip {
                    start_dist + (total_width_px - char_center_px) * world_per_px
                } else {
                    start_dist + char_center_px * world_per_px
                };

                // Skip glyphs that fall outside the path bounds
                if d < 0.0 || d > path_len {
                    cursor_px += advance_px;
                    continue;
                }

                let (wx, wy) = sample_path(path, &cum_len, d);
                let (tx, ty) = sample_tangent(path, &cum_len, d);

                // Per-character angle from tangent, with consistent flip for all chars
                let mut angle = (-ty).atan2(tx);
                if flip { angle += std::f32::consts::PI; }

                let cos_a = angle.cos();
                let sin_a = angle.sin();
                let rotate = |x: f32, y: f32| -> (f32, f32) {
                    (x * cos_a - y * sin_a, x * sin_a + y * cos_a)
                };

                let px = -advance_px * 0.5 + glyph.bearing[0] * scale + ox;
                let py = -self.font_size * scale * 0.5 + glyph.bearing[1] * scale + oy;
                // Curved labels are always streets — kind_flag = 1.0 (small).
                self.emit_glyph_quad_rotated(
                    [wx, wy], px, py, glyph, scale, color, 1.0, &rotate,
                    verts, idxs,
                );
                cursor_px += advance_px;
            }
        };

        // Halo
        for &(ox, oy) in &[(-1.0_f32, 0.0_f32), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0)] {
            emit_glyphs(halo_color, ox, oy, vertices, indices);
        }
        // Text
        emit_glyphs(text_color, 0.0, 0.0, vertices, indices);
    }

    pub fn render<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>, camera_bind_group: &'a wgpu::BindGroup) {
        if let (Some(vb), Some(ib)) = (&self.vertex_buffer, &self.index_buffer) {
            render_pass.set_pipeline(&self.pipeline);
            render_pass.set_bind_group(0, camera_bind_group, &[]);
            render_pass.set_bind_group(1, &self.atlas_bind_group, &[]);
            render_pass.set_vertex_buffer(0, vb.slice(..));
            render_pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..self.num_indices, 0, 0..1);
        }
    }
}

// --- Font loading and atlas building ---

/// Embedded font for WASM and as fallback.
const EMBEDDED_FONT: &[u8] = include_bytes!("../assets/font.ttf");

fn load_system_font() -> fontdue::Font {
    let settings = fontdue::FontSettings {
        collection_index: 0,
        ..Default::default()
    };

    // On native, try system fonts first
    #[cfg(not(target_arch = "wasm32"))]
    {
        let font_paths = [
            "/System/Library/Fonts/SFNS.ttf",
            "/System/Library/Fonts/SFNSText.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/Geneva.ttf",
            "/System/Library/Fonts/Supplemental/Verdana.ttf",
        ];

        for path in &font_paths {
            if let Ok(data) = std::fs::read(path) {
                if let Ok(font) = fontdue::Font::from_bytes(data, settings) {
                    return font;
                }
            }
        }
    }

    // Fall back to embedded font (always used on WASM)
    fontdue::Font::from_bytes(EMBEDDED_FONT.to_vec(), settings)
        .expect("Failed to load embedded font")
}

/// Convert a coverage bitmap into a signed distance field.
/// Positive values = inside the glyph, negative = outside.
/// Normalized to 0-255 where 128 = edge.
fn bitmap_to_sdf(bitmap: &[u8], w: u32, h: u32, spread: f32) -> Vec<u8> {
    let w = w as i32;
    let h = h as i32;
    let search = spread.ceil() as i32;
    let mut sdf = vec![0u8; (w * h) as usize];

    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) as usize;
            let inside = bitmap.get(idx).copied().unwrap_or(0) > 127;
            let mut min_dist_sq = (search * search + 1) as f32;

            // Search neighborhood for nearest edge pixel
            for sy in (y - search).max(0)..=(y + search).min(h - 1) {
                for sx in (x - search).max(0)..=(x + search).min(w - 1) {
                    let si = (sy * w + sx) as usize;
                    let s_inside = bitmap.get(si).copied().unwrap_or(0) > 127;
                    if s_inside != inside {
                        let dx = (sx - x) as f32;
                        let dy = (sy - y) as f32;
                        min_dist_sq = min_dist_sq.min(dx * dx + dy * dy);
                    }
                }
            }

            let dist = min_dist_sq.sqrt();
            let signed = if inside { dist } else { -dist };
            // Map to 0-255: 128 = edge, 255 = deep inside, 0 = far outside
            let normalized = (signed / spread * 127.0 + 128.0).clamp(0.0, 255.0);
            sdf[idx] = normalized as u8;
        }
    }
    sdf
}

fn build_glyph_atlas(font: &fontdue::Font, font_size: f32) -> (Vec<u8>, HashMap<char, GlyphInfo>) {
    let mut atlas = vec![0u8; (ATLAS_SIZE * ATLAS_SIZE) as usize];
    let mut glyphs = HashMap::new();

    let mut cursor_x: u32 = 1;
    let mut cursor_y: u32 = 1;
    let mut row_height: u32 = 0;
    let sdf_spread = 8.0_f32; // pixel radius for distance field

    for code in 32u8..127 {
        let ch = code as char;
        let (metrics, bitmap) = font.rasterize(ch, font_size);

        let w = metrics.width as u32;
        let h = metrics.height as u32;

        if w == 0 || h == 0 {
            glyphs.insert(ch, GlyphInfo {
                uv_min: [0.0, 0.0],
                uv_max: [0.0, 0.0],
                size: [0.0, 0.0],
                bearing: [0.0, 0.0],
                advance: metrics.advance_width,
            });
            continue;
        }

        // Wrap to next row if needed
        if cursor_x + w + 1 >= ATLAS_SIZE {
            cursor_x = 1;
            cursor_y += row_height + 1;
            row_height = 0;
        }

        if cursor_y + h + 1 >= ATLAS_SIZE {
            log::warn!("Glyph atlas full, skipping remaining characters");
            break;
        }

        // Convert rasterized bitmap to SDF
        let sdf = bitmap_to_sdf(&bitmap, w, h, sdf_spread);

        // Copy SDF into atlas
        for y in 0..h {
            for x in 0..w {
                let src_idx = (y * w + x) as usize;
                let dst_idx = ((cursor_y + y) * ATLAS_SIZE + cursor_x + x) as usize;
                if src_idx < sdf.len() && dst_idx < atlas.len() {
                    atlas[dst_idx] = sdf[src_idx];
                }
            }
        }

        let uv_min = [
            cursor_x as f32 / ATLAS_SIZE as f32,
            cursor_y as f32 / ATLAS_SIZE as f32,
        ];
        let uv_max = [
            (cursor_x + w) as f32 / ATLAS_SIZE as f32,
            (cursor_y + h) as f32 / ATLAS_SIZE as f32,
        ];

        glyphs.insert(ch, GlyphInfo {
            uv_min,
            uv_max,
            size: [w as f32, h as f32],
            bearing: [metrics.xmin as f32, -(metrics.ymin as f32 + h as f32 - font_size)],
            advance: metrics.advance_width * KERNING,
        });

        cursor_x += w + 1;
        row_height = row_height.max(h);
    }


    (atlas, glyphs)
}

/// Sample a point on a polyline at the given arc-length distance.
/// Clamps to path endpoints if distance is out of range.
fn sample_path(path: &[[f32; 2]], cum_len: &[f32], dist: f32) -> (f32, f32) {
    if path.is_empty() { return (0.0, 0.0); }
    if dist <= 0.0 { return (path[0][0], path[0][1]); }
    let total = *cum_len.last().unwrap_or(&0.0);
    if dist >= total {
        let p = path.last().unwrap();
        return (p[0], p[1]);
    }
    for i in 1..path.len() {
        if cum_len[i] >= dist {
            let seg_len = cum_len[i] - cum_len[i - 1];
            if seg_len < 1e-8 { continue; }
            let t = (dist - cum_len[i - 1]) / seg_len;
            let x = path[i - 1][0] + (path[i][0] - path[i - 1][0]) * t;
            let y = path[i - 1][1] + (path[i][1] - path[i - 1][1]) * t;
            return (x, y);
        }
    }
    let p = path.last().unwrap();
    (p[0], p[1])
}

/// Sample the unit tangent direction at a given arc-length distance.
fn sample_tangent(path: &[[f32; 2]], cum_len: &[f32], dist: f32) -> (f32, f32) {
    if path.len() < 2 { return (1.0, 0.0); }
    let total = *cum_len.last().unwrap_or(&0.0);
    let d = dist.clamp(0.0, total);
    for i in 1..path.len() {
        if cum_len[i] >= d {
            let dx = path[i][0] - path[i - 1][0];
            let dy = path[i][1] - path[i - 1][1];
            let len = (dx * dx + dy * dy).sqrt();
            if len < 1e-8 { continue; }
            return (dx / len, dy / len);
        }
    }
    let dx = path[path.len() - 1][0] - path[path.len() - 2][0];
    let dy = path[path.len() - 1][1] - path[path.len() - 2][1];
    let len = (dx * dx + dy * dy).sqrt();
    if len < 1e-8 { return (1.0, 0.0); }
    (dx / len, dy / len)
}

/// Word-wrap text into lines of approximately `max_chars` characters.
/// Breaks at spaces; words longer than max_chars are kept whole on their own line.
fn wrap_text(text: &str, max_chars: usize) -> Vec<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() { return vec![text.to_string()]; }

    let mut lines = Vec::new();
    let mut current = String::new();

    for word in words {
        if current.is_empty() {
            current = word.to_string();
        } else if current.len() + 1 + word.len() <= max_chars {
            current.push(' ');
            current.push_str(word);
        } else {
            lines.push(current);
            current = word.to_string();
        }
    }
    if !current.is_empty() {
        lines.push(current);
    }
    lines
}
