use crate::cars::CarsSystem;
use crate::config::LayerVisibility;
use crate::gpu::GpuState;
use crate::mapdata::{MapData, MapVertex};
use crate::text::TextSystem;
use crate::texture::TextureSystem;

pub struct Renderer {
    map_pipeline: wgpu::RenderPipeline,
    shadow_pipeline: wgpu::RenderPipeline,
    cloud_pipeline: wgpu::RenderPipeline,
    vertex_buffer: Option<wgpu::Buffer>,
    /// Parallel birth-time VB for the non-tile render() path. Pipeline requires
    /// a second vertex buffer slot even for the single-shot MapData case; fill
    /// with zeros (= fully faded-in).
    birth_buffer: Option<wgpu::Buffer>,
    index_buffer: Option<wgpu::Buffer>,
    num_indices: u32,
    shadow_vertex_buffer: Option<wgpu::Buffer>,
    shadow_birth_buffer: Option<wgpu::Buffer>,
    shadow_index_buffer: Option<wgpu::Buffer>,
    num_shadow_indices: u32,
    pub text: TextSystem,
    pub cars: CarsSystem,
    pub textures: TextureSystem,
}

impl Renderer {
    pub fn new(gpu: &GpuState) -> Self {
        let textures = TextureSystem::new(
            &gpu.device,
            &gpu.queue,
            &gpu.config,
            &gpu.camera_bind_group_layout,
            gpu.msaa_samples,
        );

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
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: 28,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32,
                },
            ],
        };
        // Parallel vertex buffer carrying each vertex's tile birth-time (in
        // seconds since app start). One f32 per vertex. Used by the map and
        // shadow pipelines to fade the tile in on load.
        let birth_layout = wgpu::VertexBufferLayout {
            array_stride: 4 as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32,
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
                        buffers: &[vertex_layout.clone(), birth_layout.clone()],
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
                        count: gpu.msaa_samples,
                        mask: !0,
                        alpha_to_coverage_enabled: false,
                    },
                    multiview: None,
                    cache: None,
                });

        // Shadow pipeline
        let shadow_pipeline =
            gpu.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("Shadow Pipeline"),
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &map_shader,
                        entry_point: Some("vs_main"),
                        buffers: &[vertex_layout, birth_layout],
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
                        depth_write_enabled: false,
                        depth_compare: wgpu::CompareFunction::LessEqual,
                        stencil: wgpu::StencilState::default(),
                        bias: wgpu::DepthBiasState::default(),
                    }),
                    multisample: wgpu::MultisampleState {
                        count: gpu.msaa_samples,
                        mask: !0,
                        alpha_to_coverage_enabled: false,
                    },
                    multiview: None,
                    cache: None,
                });

        // Procedural cloud overlay — fullscreen quad, camera-only bind group
        let cloud_pipeline_layout =
            gpu.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Cloud Pipeline Layout"),
                    bind_group_layouts: &[&gpu.camera_bind_group_layout],
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
                        buffers: &[], // fullscreen quad from vertex_index
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
                        count: gpu.msaa_samples,
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
            gpu.msaa_samples,
        );

        let cars = CarsSystem::new(gpu);

        Self {
            map_pipeline,
            shadow_pipeline,
            cloud_pipeline,
            vertex_buffer: None,
            birth_buffer: None,
            index_buffer: None,
            num_indices: 0,
            shadow_vertex_buffer: None,
            shadow_birth_buffer: None,
            shadow_index_buffer: None,
            num_shadow_indices: 0,
            text,
            cars,
            textures,
        }
    }

    pub fn upload_map_data(&mut self, gpu: &GpuState, data: &MapData) {
        if data.vertices.is_empty() || data.indices.is_empty() {
            return;
        }

        self.vertex_buffer = Some(crate::gpu::safe_buffer(
            &gpu.device, &gpu.queue, "Map Vertex Buffer",
            bytemuck::cast_slice(&data.vertices), wgpu::BufferUsages::VERTEX,
        ));

        // Stub birth-time VB — 0.0 means "already fully faded in" since this
        // path is the single-shot native/pre-loaded case.
        let birth_data: Vec<f32> = vec![0.0; data.vertices.len().max(1)];
        self.birth_buffer = Some(crate::gpu::safe_buffer(
            &gpu.device, &gpu.queue, "Map Birth Buffer",
            bytemuck::cast_slice(&birth_data), wgpu::BufferUsages::VERTEX,
        ));

        self.index_buffer = Some(crate::gpu::safe_buffer(
            &gpu.device, &gpu.queue, "Map Index Buffer",
            bytemuck::cast_slice(&data.indices), wgpu::BufferUsages::INDEX,
        ));

        self.num_indices = data.indices.len() as u32;

        if !data.shadow_vertices.is_empty() {
            self.shadow_vertex_buffer = Some(crate::gpu::safe_buffer(
                &gpu.device, &gpu.queue, "Shadow Vertex Buffer",
                bytemuck::cast_slice(&data.shadow_vertices), wgpu::BufferUsages::VERTEX,
            ));

            let shadow_birth: Vec<f32> = vec![0.0; data.shadow_vertices.len()];
            self.shadow_birth_buffer = Some(crate::gpu::safe_buffer(
                &gpu.device, &gpu.queue, "Shadow Birth Buffer",
                bytemuck::cast_slice(&shadow_birth), wgpu::BufferUsages::VERTEX,
            ));

            self.shadow_index_buffer = Some(crate::gpu::safe_buffer(
                &gpu.device, &gpu.queue, "Shadow Index Buffer",
                bytemuck::cast_slice(&data.shadow_indices), wgpu::BufferUsages::INDEX,
            ));

            self.num_shadow_indices = data.shadow_indices.len() as u32;
        }

        self.text.upload_labels(&gpu.device, &gpu.queue, &data.labels);
    }

    fn render_clouds<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        camera_bind_group: &'a wgpu::BindGroup,
    ) {
        render_pass.set_pipeline(&self.cloud_pipeline);
        render_pass.set_bind_group(0, camera_bind_group, &[]);
        render_pass.draw(0..3, 0..1); // fullscreen triangle
    }

    pub fn render(&self, gpu: &GpuState, layers: &LayerVisibility, clear_color: [f64; 3]) -> Result<(), wgpu::SurfaceError> {
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
            let (color_view, resolve_target) = if gpu.msaa_samples > 1 {
                (&gpu.msaa_view, Some(&view))
            } else {
                (&view, None)
            };
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Map Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: color_view,
                    resolve_target,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: clear_color[0],
                            g: clear_color[1],
                            b: clear_color[2],
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

            // Pass 0: Background texture
            self.textures.render_background(&mut render_pass, &gpu.camera_bind_group);

            // Pass 1: Map geometry (opaque)
            if self.num_indices > 0 {
                if let (Some(vb), Some(bb), Some(ib)) = (&self.vertex_buffer, &self.birth_buffer, &self.index_buffer) {
                    render_pass.set_pipeline(&self.map_pipeline);
                    render_pass.set_bind_group(0, &gpu.camera_bind_group, &[]);
                    render_pass.set_bind_group(1, self.textures.material_bind_group().unwrap(), &[]);
                    render_pass.set_vertex_buffer(0, vb.slice(..));
                    render_pass.set_vertex_buffer(1, bb.slice(..));
                    render_pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                    render_pass.draw_indexed(0..self.num_indices, 0, 0..1);
                }
            }

            // Pass 2: Shadows
            if layers.shadows && self.num_shadow_indices > 0 {
                if let (Some(vb), Some(bb), Some(ib)) = (&self.shadow_vertex_buffer, &self.shadow_birth_buffer, &self.shadow_index_buffer) {
                    render_pass.set_pipeline(&self.shadow_pipeline);
                    render_pass.set_bind_group(0, &gpu.camera_bind_group, &[]);
                    render_pass.set_bind_group(1, self.textures.material_bind_group().unwrap(), &[]);
                    render_pass.set_vertex_buffer(0, vb.slice(..));
                    render_pass.set_vertex_buffer(1, bb.slice(..));
                    render_pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                    render_pass.draw_indexed(0..self.num_shadow_indices, 0, 0..1);
                }
            }

            // Pass 3: Cars
            if layers.cars {
                self.cars.render(&mut render_pass, &gpu.camera_bind_group);
            }

            // Pass 4: Text labels
            if layers.labels {
                self.text.render(&mut render_pass, &gpu.camera_bind_group);
            }

            // Pass 5: Procedural cloud overlay (fullscreen)
            if layers.clouds {
                self.render_clouds(&mut render_pass, &gpu.camera_bind_group);
            }
        }

        gpu.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

    pub fn render_tiles<'a>(
        &'a self,
        gpu: &'a GpuState,
        tiles: impl Iterator<Item = &'a crate::tiles::LoadedTile>,
        layers: &LayerVisibility,
        clear_color: [f64; 3],
        camera_zoom: f32,
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
            let (color_view, resolve_target) = if gpu.msaa_samples > 1 {
                (&gpu.msaa_view, Some(&view))
            } else {
                (&view, None)
            };
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Tile Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: color_view,
                    resolve_target,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: clear_color[0], g: clear_color[1], b: clear_color[2], a: 1.0 }),
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

            // Pass 0: Background texture
            self.textures.render_background(&mut render_pass, &gpu.camera_bind_group);

            // Collect tiles, dedup by z14
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

            // Pass 1: Opaque geometry (slot 0 = geometry VB, slot 1 = birth-time VB)
            render_pass.set_pipeline(&self.map_pipeline);
            render_pass.set_bind_group(0, &gpu.camera_bind_group, &[]);
            render_pass.set_bind_group(1, self.textures.material_bind_group().unwrap(), &[]);
            for tile in &tile_list {
                render_pass.set_vertex_buffer(0, tile.vertex_buffer.slice(..));
                render_pass.set_vertex_buffer(1, tile.birth_buffer.slice(..));
                render_pass.set_index_buffer(tile.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.draw_indexed(0..tile.num_indices, 0, 0..1);
            }

            // Pass 2: Shadows — skip at low zoom where shadows are pixel-dust anyway.
            // Saves ~40-50% GPU time when zoomed out since shadow pass is near-parity
            // vertex count with the main pass.
            if layers.shadows && camera_zoom >= 1.0 {
                render_pass.set_pipeline(&self.shadow_pipeline);
                render_pass.set_bind_group(0, &gpu.camera_bind_group, &[]);
                render_pass.set_bind_group(1, self.textures.material_bind_group().unwrap(), &[]);
                for tile in &tile_list {
                    if tile.num_shadow_indices > 0 {
                        if let (Some(svb), Some(sbb), Some(sib)) = (
                            &tile.shadow_vertex_buffer,
                            &tile.shadow_birth_buffer,
                            &tile.shadow_index_buffer,
                        ) {
                            render_pass.set_vertex_buffer(0, svb.slice(..));
                            render_pass.set_vertex_buffer(1, sbb.slice(..));
                            render_pass.set_index_buffer(sib.slice(..), wgpu::IndexFormat::Uint32);
                            render_pass.draw_indexed(0..tile.num_shadow_indices, 0, 0..1);
                        }
                    }
                }
            }

            // Pass 3: Cars
            if layers.cars {
                self.cars.render(&mut render_pass, &gpu.camera_bind_group);
            }

            // Pass 4: Text labels
            if layers.labels {
                self.text.render(&mut render_pass, &gpu.camera_bind_group);
            }

            // Pass 5: Procedural cloud overlay (fullscreen)
            if layers.clouds {
                self.render_clouds(&mut render_pass, &gpu.camera_bind_group);
            }
        }

        gpu.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}
