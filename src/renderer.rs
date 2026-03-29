use crate::gpu::GpuState;
use crate::mapdata::{MapData, MapVertex};
use crate::text::TextSystem;
use crate::texture::TextureSystem;

/// A single cloud billboard vertex: position (vec3), uv (vec2), alpha (f32) = 24 bytes.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CloudVertex {
    position: [f32; 3],
    uv: [f32; 2],
    alpha: f32,
}

/// Cloud billboard sprite system.
pub struct CloudSystem {
    vertex_buffer: Option<wgpu::Buffer>,
    index_buffer: Option<wgpu::Buffer>,
    num_indices: u32,
    texture_bind_group: Option<wgpu::BindGroup>,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    /// Last camera position used for cloud generation (triggers rebuild on significant move).
    last_gen_pos: [f32; 2],
}

impl CloudSystem {
    fn new(device: &wgpu::Device) -> Self {
        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Cloud Texture Bind Group Layout"),
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

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Cloud Atlas Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        Self {
            vertex_buffer: None,
            index_buffer: None,
            num_indices: 0,
            texture_bind_group: None,
            bind_group_layout,
            sampler,
            last_gen_pos: [f32::MAX, f32::MAX],
        }
    }

    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }

    /// Upload the cloud atlas texture (called from JS via command).
    pub fn upload_texture(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        rgba_data: &[u8],
    ) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Cloud Atlas Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            rgba_data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * width),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        self.texture_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Cloud Atlas Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        }));

        // Force cloud geometry regeneration now that we have a texture
        self.last_gen_pos = [f32::MAX, f32::MAX];

    }

    /// Simple hash function: maps vec2(a,b) -> f32 in [0,1).
    fn hash21(a: f32, b: f32) -> f32 {
        let mut x = a * 127.1 + b * 311.7;
        x = (x.sin()) * 43758.5453;
        x - x.floor()
    }

    /// Regenerate cloud billboard geometry around the given camera position.
    /// Uses a deterministic grid with hash-based offsets so clouds don't pop in/out.
    pub fn maybe_regenerate(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, cam_x: f32, cam_y: f32) {
        // Only regenerate if camera moved significantly (>2 world units)
        let dx = cam_x - self.last_gen_pos[0];
        let dy = cam_y - self.last_gen_pos[1];
        if dx * dx + dy * dy < 2500.0 { // regenerate when camera moves >50 world units
            return;
        }
        self.last_gen_pos = [cam_x, cam_y];

        let mut vertices: Vec<CloudVertex> = Vec::with_capacity(500 * 8); // shadow + cloud quads * cluster size
        let mut indices: Vec<u32> = Vec::with_capacity(500 * 12);

        let grid_spacing = 150.0_f32; // tighter spacing = more clouds

        for grid_x in -8..8_i32 {
            for grid_y in -8..8_i32 {
                let gx = grid_x as f32;
                let gy = grid_y as f32;

                // Grid cell center in world space (snapped to grid around camera)
                let cam_grid_x = (cam_x / grid_spacing).round() * grid_spacing;
                let cam_grid_y = (cam_y / grid_spacing).round() * grid_spacing;
                let base_x = cam_grid_x + gx * grid_spacing;
                let base_y = cam_grid_y + gy * grid_spacing;

                // Deterministic hash from the snapped grid cell (not camera position)
                let cell_x = (base_x / grid_spacing).round();
                let cell_y = (base_y / grid_spacing).round();

                let h1 = Self::hash21(cell_x, cell_y);
                let h2 = Self::hash21(cell_y, cell_x);
                let h3 = Self::hash21(cell_x + 7.3, cell_y + 13.1);
                let h4 = Self::hash21(cell_x + 31.7, cell_y + 5.9);

                // Skip some cells for gaps between clusters
                if h1 > 0.6 {
                    continue;
                }

                // Place a cluster of 3-5 clouds per cell
                let cluster_count = 3 + (h2 * 3.0).floor() as i32; // 3-5 clouds per cluster
                let cluster_cx = base_x + (h1 - 0.5) * 80.0;
                let cluster_cy = base_y + (h2 - 0.5) * 80.0;

                for ci in 0..cluster_count {
                    let ci_f = ci as f32;
                    let ch1 = Self::hash21(cell_x + ci_f * 17.3, cell_y + ci_f * 11.7);
                    let ch2 = Self::hash21(cell_y + ci_f * 23.1, cell_x + ci_f * 7.3);
                    let ch3 = Self::hash21(cell_x + ci_f * 31.9, cell_y + ci_f * 19.3);
                    let ch4 = Self::hash21(cell_x + ci_f * 5.7, cell_y + ci_f * 41.1);

                // Scatter within cluster — tighter grouping
                let world_x = cluster_cx + (ch1 - 0.5) * 80.0;
                let world_y = cluster_cy + (ch2 - 0.5) * 60.0;

                // Half the size — clusters create the illusion of variety
                let size = 160.0 + ch1 * 240.0; // 160-400 world units
                let height = 5.0 + ch3 * 3.0;
                let atlas_index = (ch4 * 4.0).floor().min(3.0) as u32;
                let alpha = 0.25 + ch1 * 0.2; // 25-45% opacity

                // Atlas UV for this cloud (2x2 grid)
                let atlas_col = atlas_index % 2;
                let atlas_row = atlas_index / 2;
                let u0 = atlas_col as f32 * 0.5;
                let v0 = atlas_row as f32 * 0.5;
                let u1 = u0 + 0.5;
                let v1 = v0 + 0.5;

                // Parallax: clouds at higher altitude drift relative to camera
                // This creates depth — clouds move slightly faster than the ground
                let parallax_factor = height * 0.08; // higher clouds = more parallax
                let parallax_x = (world_x - cam_x) * parallax_factor + world_x;
                let parallax_y = (world_y - cam_y) * parallax_factor + world_y;

                let half = size * 0.5;
                let half_h = half * 0.6; // clouds wider than tall

                // --- Shadow quad (on ground, offset by sun) ---
                let shadow_offset_x = height * 0.25;
                let shadow_offset_y = height * -0.15;
                let sx = parallax_x + shadow_offset_x;
                let sy = parallax_y + shadow_offset_y;
                let neg_alpha = -(0.3 + ch1 * 0.2); // negative = shadow flag

                let base_idx = vertices.len() as u32;
                vertices.push(CloudVertex { position: [sx - half, sy - half_h, 0.002], uv: [u0, v1], alpha: neg_alpha });
                vertices.push(CloudVertex { position: [sx + half, sy - half_h, 0.002], uv: [u1, v1], alpha: neg_alpha });
                vertices.push(CloudVertex { position: [sx + half, sy + half_h, 0.002], uv: [u1, v0], alpha: neg_alpha });
                vertices.push(CloudVertex { position: [sx - half, sy + half_h, 0.002], uv: [u0, v0], alpha: neg_alpha });
                indices.extend_from_slice(&[base_idx, base_idx+1, base_idx+2, base_idx, base_idx+2, base_idx+3]);

                // --- Cloud quad (floating above) ---
                let base_idx = vertices.len() as u32;
                vertices.push(CloudVertex {
                    position: [parallax_x - half, parallax_y - half_h, height],
                    uv: [u0, v1],
                    alpha,
                });
                vertices.push(CloudVertex {
                    position: [parallax_x + half, parallax_y - half_h, height],
                    uv: [u1, v1],
                    alpha,
                });
                vertices.push(CloudVertex {
                    position: [parallax_x + half, parallax_y + half_h, height],
                    uv: [u1, v0],
                    alpha,
                });
                vertices.push(CloudVertex {
                    position: [parallax_x - half, parallax_y + half_h, height],
                    uv: [u0, v0],
                    alpha,
                });

                // Two triangles
                indices.push(base_idx);
                indices.push(base_idx + 1);
                indices.push(base_idx + 2);
                indices.push(base_idx);
                indices.push(base_idx + 2);
                indices.push(base_idx + 3);

                } // end cluster loop (ci)
            }
        }

        if vertices.is_empty() {
            self.num_indices = 0;
            return;
        }

        self.vertex_buffer = Some(crate::gpu::safe_buffer(
            device, queue, "Cloud Vertex Buffer",
            bytemuck::cast_slice(&vertices), wgpu::BufferUsages::VERTEX,
        ));

        self.index_buffer = Some(crate::gpu::safe_buffer(
            device, queue, "Cloud Index Buffer",
            bytemuck::cast_slice(&indices), wgpu::BufferUsages::INDEX,
        ));

        self.num_indices = indices.len() as u32;
    }

    /// Render cloud billboards. Only renders if texture and geometry are available.
    fn render<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        cloud_pipeline: &'a wgpu::RenderPipeline,
        camera_bind_group: &'a wgpu::BindGroup,
    ) {
        if self.num_indices == 0 {
            return;
        }
        if let (Some(tex_bg), Some(vb), Some(ib)) =
            (&self.texture_bind_group, &self.vertex_buffer, &self.index_buffer)
        {
            render_pass.set_pipeline(cloud_pipeline);
            render_pass.set_bind_group(0, camera_bind_group, &[]);
            render_pass.set_bind_group(1, tex_bg, &[]);
            render_pass.set_vertex_buffer(0, vb.slice(..));
            render_pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..self.num_indices, 0, 0..1);
        }
    }
}

pub struct Renderer {
    map_pipeline: wgpu::RenderPipeline,
    shadow_pipeline: wgpu::RenderPipeline,
    cloud_pipeline: wgpu::RenderPipeline,
    vertex_buffer: Option<wgpu::Buffer>,
    index_buffer: Option<wgpu::Buffer>,
    num_indices: u32,
    shadow_vertex_buffer: Option<wgpu::Buffer>,
    shadow_index_buffer: Option<wgpu::Buffer>,
    num_shadow_indices: u32,
    pub text: TextSystem,
    pub textures: TextureSystem,
    pub clouds: CloudSystem,
}

impl Renderer {
    pub fn new(gpu: &GpuState) -> Self {
        // Create TextureSystem first so we can reference its material layout in pipelines
        let textures = TextureSystem::new(
            &gpu.device,
            &gpu.queue,
            &gpu.config,
            &gpu.camera_bind_group_layout,
        );

        let clouds = CloudSystem::new(&gpu.device);

        let map_shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Map Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("map.wgsl").into()),
            });

        let pipeline_layout =
            gpu.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Map Pipeline Layout"),
                    bind_group_layouts: &[
                        &gpu.camera_bind_group_layout,
                        textures.material_bind_group_layout(),
                    ],
                    push_constant_ranges: &[],
                });

        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<MapVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3, // position xyz
                },
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x4, // color
                },
                wgpu::VertexAttribute {
                    offset: 28,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32, // material
                },
            ],
        };

        let map_pipeline =
            gpu.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("Map Pipeline"),
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &map_shader,
                        entry_point: Some("vs_main"),
                        buffers: &[vertex_layout.clone()],
                        compilation_options: Default::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &map_shader,
                        entry_point: Some("fs_main"),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: gpu.config.format,
                            blend: Some(wgpu::BlendState::REPLACE),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                        compilation_options: Default::default(),
                    }),
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleList,
                        strip_index_format: None,
                        front_face: wgpu::FrontFace::Ccw,
                        cull_mode: None,
                        polygon_mode: wgpu::PolygonMode::Fill,
                        unclipped_depth: false,
                        conservative: false,
                    },
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: crate::gpu::DEPTH_FORMAT,
                        depth_write_enabled: true,
                        depth_compare: wgpu::CompareFunction::LessEqual,
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

        // Shadow pipeline: same vertex shader, shadow fragment shader, alpha blending
        let shadow_pipeline =
            gpu.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("Shadow Pipeline"),
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &map_shader,
                        entry_point: Some("vs_main"),
                        buffers: &[vertex_layout],
                        compilation_options: Default::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &map_shader,
                        entry_point: Some("fs_shadow"),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: gpu.config.format,
                            blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                        compilation_options: Default::default(),
                    }),
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleList,
                        strip_index_format: None,
                        front_face: wgpu::FrontFace::Ccw,
                        cull_mode: None,
                        polygon_mode: wgpu::PolygonMode::Fill,
                        unclipped_depth: false,
                        conservative: false,
                    },
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: crate::gpu::DEPTH_FORMAT,
                        depth_write_enabled: false, // Don't write depth — shadows are overlays
                        depth_compare: wgpu::CompareFunction::LessEqual,
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

        // Cloud pipeline: textured billboard quads, alpha blending, vertex buffer
        let cloud_vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<CloudVertex>() as wgpu::BufferAddress, // 24 bytes
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3, // position
                },
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2, // uv
                },
                wgpu::VertexAttribute {
                    offset: 20,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32, // alpha
                },
            ],
        };

        let cloud_pipeline_layout =
            gpu.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Cloud Pipeline Layout"),
                    bind_group_layouts: &[
                        &gpu.camera_bind_group_layout,
                        clouds.bind_group_layout(),
                    ],
                    push_constant_ranges: &[],
                });

        let cloud_pipeline =
            gpu.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("Cloud Pipeline"),
                    layout: Some(&cloud_pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &map_shader,
                        entry_point: Some("vs_cloud"),
                        buffers: &[cloud_vertex_layout],
                        compilation_options: Default::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &map_shader,
                        entry_point: Some("fs_cloud"),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: gpu.config.format,
                            blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                        compilation_options: Default::default(),
                    }),
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleList,
                        strip_index_format: None,
                        front_face: wgpu::FrontFace::Ccw,
                        cull_mode: None,
                        polygon_mode: wgpu::PolygonMode::Fill,
                        unclipped_depth: false,
                        conservative: false,
                    },
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: crate::gpu::DEPTH_FORMAT,
                        depth_write_enabled: false,
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

        let text = TextSystem::new(
            &gpu.device,
            &gpu.queue,
            &gpu.config,
            &gpu.camera_bind_group_layout,
        );

        Self {
            map_pipeline,
            shadow_pipeline,
            cloud_pipeline,
            vertex_buffer: None,
            index_buffer: None,
            num_indices: 0,
            shadow_vertex_buffer: None,
            shadow_index_buffer: None,
            num_shadow_indices: 0,
            text,
            textures,
            clouds,
        }
    }

    /// Upload map geometry to GPU buffers.
    pub fn upload_map_data(&mut self, gpu: &GpuState, data: &MapData) {
        if data.vertices.is_empty() || data.indices.is_empty() {
            return;
        }

        self.vertex_buffer = Some(crate::gpu::safe_buffer(
            &gpu.device, &gpu.queue, "Map Vertex Buffer",
            bytemuck::cast_slice(&data.vertices), wgpu::BufferUsages::VERTEX,
        ));

        self.index_buffer = Some(crate::gpu::safe_buffer(
            &gpu.device, &gpu.queue, "Map Index Buffer",
            bytemuck::cast_slice(&data.indices), wgpu::BufferUsages::INDEX,
        ));

        self.num_indices = data.indices.len() as u32;

        // Upload shadow geometry
        if !data.shadow_vertices.is_empty() {
            self.shadow_vertex_buffer = Some(crate::gpu::safe_buffer(
                &gpu.device, &gpu.queue, "Shadow Vertex Buffer",
                bytemuck::cast_slice(&data.shadow_vertices), wgpu::BufferUsages::VERTEX,
            ));

            self.shadow_index_buffer = Some(crate::gpu::safe_buffer(
                &gpu.device, &gpu.queue, "Shadow Index Buffer",
                bytemuck::cast_slice(&data.shadow_indices), wgpu::BufferUsages::INDEX,
            ));

            self.num_shadow_indices = data.shadow_indices.len() as u32;
        }

        // Upload labels for text rendering
        self.text.upload_labels(&gpu.device, &gpu.queue, &data.labels);

    }

    pub fn render(&self, gpu: &GpuState) -> Result<(), wgpu::SurfaceError> {
        if gpu.config.width == 0 || gpu.config.height == 0 {
            return Ok(());
        }
        let output = gpu.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Map Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &gpu.msaa_view,
                    resolve_target: Some(&view),
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.95,
                            g: 0.90,
                            b: 0.85,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &gpu.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            // Pass 0: Background texture (tiled, behind everything)
            self.textures.render_background(&mut render_pass, &gpu.camera_bind_group);

            // Pass 1: Map geometry (opaque)
            if self.num_indices > 0 {
                if let (Some(vb), Some(ib)) = (&self.vertex_buffer, &self.index_buffer) {
                    render_pass.set_pipeline(&self.map_pipeline);
                    render_pass.set_bind_group(0, &gpu.camera_bind_group, &[]);
                    render_pass.set_bind_group(1, self.textures.material_bind_group().unwrap(), &[]);
                    render_pass.set_vertex_buffer(0, vb.slice(..));
                    render_pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                    render_pass.draw_indexed(0..self.num_indices, 0, 0..1);
                }
            }

            // Pass 2: Shadows (alpha blended, darkening overlay)
            if self.num_shadow_indices > 0 {
                if let (Some(vb), Some(ib)) = (&self.shadow_vertex_buffer, &self.shadow_index_buffer) {
                    render_pass.set_pipeline(&self.shadow_pipeline);
                    render_pass.set_bind_group(0, &gpu.camera_bind_group, &[]);
                    render_pass.set_bind_group(1, self.textures.material_bind_group().unwrap(), &[]);
                    render_pass.set_vertex_buffer(0, vb.slice(..));
                    render_pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                    render_pass.draw_indexed(0..self.num_shadow_indices, 0, 0..1);
                }
            }

            // Pass 3: Text labels (alpha blended, on top)
            self.text.render(&mut render_pass, &gpu.camera_bind_group);

            // Pass 4: Billboard cloud sprites (disabled)
            // self.clouds.render(&mut render_pass, &self.cloud_pipeline, &gpu.camera_bind_group);
        }

        gpu.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

    /// Render multiple tiles (used by tile-based loading system).
    pub fn render_tiles<'a>(
        &'a self,
        gpu: &'a GpuState,
        tiles: impl Iterator<Item = &'a crate::tiles::LoadedTile>,
    ) -> Result<(), wgpu::SurfaceError> {
        if gpu.config.width == 0 || gpu.config.height == 0 {
            return Ok(());
        }
        let output = gpu.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Tile Render Encoder"),
        });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Tile Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &gpu.msaa_view,
                    resolve_target: Some(&view),
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.95, g: 0.90, b: 0.85, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &gpu.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            // Pass 0: Background texture (tiled, behind everything)
            self.textures.render_background(&mut render_pass, &gpu.camera_bind_group);

            // Collect tiles, deduplicating by z14 tile to prevent duplicate geometry
            let tile_list: Vec<_> = {
                let mut seen = std::collections::HashSet::with_capacity(48);
                let mut v = Vec::with_capacity(48);
                for tile in tiles {
                    if tile.num_indices > 0 && seen.insert(tile.z14_tile) {
                        v.push(tile);
                    }
                }
                v
            };

            // Pass 1: Opaque geometry for all tiles
            render_pass.set_pipeline(&self.map_pipeline);
            render_pass.set_bind_group(0, &gpu.camera_bind_group, &[]);
            render_pass.set_bind_group(1, self.textures.material_bind_group().unwrap(), &[]);
            for tile in &tile_list {
                render_pass.set_vertex_buffer(0, tile.vertex_buffer.slice(..));
                render_pass.set_index_buffer(tile.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.draw_indexed(0..tile.num_indices, 0, 0..1);
            }

            // Pass 2: Shadows for all tiles
            render_pass.set_pipeline(&self.shadow_pipeline);
            render_pass.set_bind_group(0, &gpu.camera_bind_group, &[]);
            render_pass.set_bind_group(1, self.textures.material_bind_group().unwrap(), &[]);
            for tile in &tile_list {
                if tile.num_shadow_indices > 0 {
                    if let (Some(svb), Some(sib)) = (&tile.shadow_vertex_buffer, &tile.shadow_index_buffer) {
                        render_pass.set_vertex_buffer(0, svb.slice(..));
                        render_pass.set_index_buffer(sib.slice(..), wgpu::IndexFormat::Uint32);
                        render_pass.draw_indexed(0..tile.num_shadow_indices, 0, 0..1);
                    }
                }
            }

            // Pass 3: Text labels
            self.text.render(&mut render_pass, &gpu.camera_bind_group);

            // Pass 4: Billboard cloud sprites (disabled)
            // self.clouds.render(&mut render_pass, &self.cloud_pipeline, &gpu.camera_bind_group);
        }

        gpu.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}
