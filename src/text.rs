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

const ATLAS_SIZE: u32 = 512;
const FONT_SIZE: f32 = 22.0;

impl TextSystem {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, config: &wgpu::SurfaceConfiguration, camera_bind_group_layout: &wgpu::BindGroupLayout) -> Self {
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
                count: crate::gpu::MSAA_SAMPLES,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        Self {
            pipeline,
            atlas_bind_group,
            vertex_buffer: None,
            index_buffer: None,
            num_indices: 0,
            glyph_table,
            font_size,
        }
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
        self.upload_labels_at_zoom(device, queue, labels, 1.0);
    }

    pub fn upload_labels_at_zoom(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, labels: &[Label], zoom: f32) {
        let mut vertices: Vec<TextVertex> = Vec::with_capacity(256 * 20);
        let mut indices: Vec<u32> = Vec::with_capacity(256 * 30);

        let text_color = [0.2, 0.2, 0.2, 1.0];
        let halo_color = [1.0, 1.0, 1.0, 0.85];

        // Spatial grids for O(1) collision detection — separate grids so
        // streets don't compete with city/park labels for placement
        let grid_cell = 2.0f32 * 2.0f32.powf(-zoom * 0.5);
        let mut grid_main: std::collections::HashSet<(i32, i32)> = std::collections::HashSet::new();
        let mut grid_streets: std::collections::HashSet<(i32, i32)> = std::collections::HashSet::new();
        let max_labels = 500;
        let mut label_count = 0u32;

        // Sort labels by priority: Cities first, then Streets, then Parks, then rest
        let mut sorted_labels: Vec<&Label> = labels.iter().collect();
        sorted_labels.sort_by_key(|l| match l.kind {
            crate::mapdata::LabelKind::City => 0,
            crate::mapdata::LabelKind::Listing => 1,
            crate::mapdata::LabelKind::Street => 2,
            crate::mapdata::LabelKind::Park => 3,
            crate::mapdata::LabelKind::Building => 4,
        });

        for label in &sorted_labels {
            if label_count >= max_labels {
                break;
            }

            // Filter labels by zoom level
            let min_zoom = match label.kind {
                crate::mapdata::LabelKind::City => -10.0,
                crate::mapdata::LabelKind::Listing => -10.0,
                crate::mapdata::LabelKind::Park => -2.0,
                crate::mapdata::LabelKind::Street => -10.0,
                crate::mapdata::LabelKind::Building => 0.5,
            };
            if zoom < min_zoom {
                continue;
            }

            let is_listing = matches!(label.kind, crate::mapdata::LabelKind::Listing);
            let is_street = matches!(label.kind, crate::mapdata::LabelKind::Street);

            // Streets use a smaller grid cell (half size) so more labels fit
            let cell = if is_street { grid_cell * 0.5 } else { grid_cell };
            let gx = (label.position[0] / cell).floor() as i32;
            let gy = (label.position[1] / cell).floor() as i32;

            // Streets use their own collision grid so they don't compete
            // with city/park labels
            let grid = if is_street { &mut grid_streets } else { &mut grid_main };

            if !is_listing {
                let occupied = (-1..=1).any(|dx| {
                    (-1..=1).any(|dy| grid.contains(&(gx + dx, gy + dy)))
                });
                if occupied {
                    continue;
                }
            }
            grid.insert((gx, gy));
            label_count += 1;

            let scale = label.font_scale();

            let (label_text_color, label_halo_color) = match label.kind {
                crate::mapdata::LabelKind::Listing => {
                    ([0.0_f32, 0.0, 0.0, 1.0], [1.0_f32, 1.0, 1.0, 0.9])
                }
                crate::mapdata::LabelKind::Street => {
                    ([0.35_f32, 0.35, 0.35, 0.85], [1.0_f32, 1.0, 1.0, 0.7])
                }
                _ => (text_color, halo_color),
            };
            let cos_a = label.angle.cos();
            let sin_a = label.angle.sin();

            // Rotate a pixel offset by the label angle
            let rotate = |x: f32, y: f32| -> (f32, f32) {
                (x * cos_a - y * sin_a, x * sin_a + y * cos_a)
            };

            // Measure total width for centering
            let total_width: f32 = label.text.chars()
                .filter_map(|c| self.get_glyph(c))
                .map(|g| g.advance * scale)
                .sum();

            // Render halo (white outline) — 4 offset copies
            for &(ox, oy) in &[(-1.0_f32, 0.0_f32), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0)] {
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
                            &rotate,
                            &mut vertices,
                            &mut indices,
                        );
                        cursor_x += glyph.advance * scale;
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
                        &rotate,
                        &mut vertices,
                        &mut indices,
                    );
                    cursor_x += glyph.advance * scale;
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

        vertices.push(TextVertex { world_pos, pixel_offset: [r0x, r0y], uv: [glyph.uv_min[0], glyph.uv_min[1]], color });
        vertices.push(TextVertex { world_pos, pixel_offset: [r1x, r1y], uv: [glyph.uv_max[0], glyph.uv_min[1]], color });
        vertices.push(TextVertex { world_pos, pixel_offset: [r2x, r2y], uv: [glyph.uv_min[0], glyph.uv_max[1]], color });
        vertices.push(TextVertex { world_pos, pixel_offset: [r3x, r3y], uv: [glyph.uv_max[0], glyph.uv_max[1]], color });

        indices.extend_from_slice(&[base, base + 1, base + 2, base + 1, base + 3, base + 2]);
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

fn build_glyph_atlas(font: &fontdue::Font, font_size: f32) -> (Vec<u8>, HashMap<char, GlyphInfo>) {
    let mut atlas = vec![0u8; (ATLAS_SIZE * ATLAS_SIZE) as usize];
    let mut glyphs = HashMap::new();

    let mut cursor_x: u32 = 1;
    let mut cursor_y: u32 = 1;
    let mut row_height: u32 = 0;

    for code in 32u8..127 {
        let ch = code as char;
        let (metrics, bitmap) = font.rasterize(ch, font_size);

        let w = metrics.width as u32;
        let h = metrics.height as u32;

        if w == 0 || h == 0 {
            // Space or empty glyph
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

        // Copy glyph bitmap into atlas
        for y in 0..h {
            for x in 0..w {
                let src_idx = (y * w + x) as usize;
                let dst_idx = ((cursor_y + y) * ATLAS_SIZE + cursor_x + x) as usize;
                if src_idx < bitmap.len() && dst_idx < atlas.len() {
                    atlas[dst_idx] = bitmap[src_idx];
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
            advance: metrics.advance_width,
        });

        cursor_x += w + 1;
        row_height = row_height.max(h);
    }


    (atlas, glyphs)
}
