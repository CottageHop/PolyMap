use crate::gpu::GpuState;
use wgpu::util::DeviceExt;

/// Maximum noise sources uploaded per frame. Keeps the uniform buffer size
/// under WebGPU's 64 KB uniform limit (128 × 16 = 2 KB). For more sources
/// we'd spatial-cull on the CPU and only upload the 128 most relevant.
pub const MAX_NOISE_SOURCES: usize = 128;

/// A point noise source. Line sources (roads, rails) should be sampled by the
/// host into multiple points along the polyline at spawn time.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct NoiseSource {
    pub pos: [f32; 2],
    /// Decibels at 1 world-unit distance. The fragment shader applies
    /// inverse-square falloff from there.
    pub db: f32,
    pub _pad: f32,
}

/// Matches `NoiseUniform` in noise.wgsl: u32 count + 3 × u32 pad + N sources.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct NoiseUniform {
    count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    sources: [NoiseSource; MAX_NOISE_SOURCES],
}

pub struct NoiseSystem {
    pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    /// Canonical CPU-side copy. Written by the host via update(); uploaded to
    /// GPU on change via queue.write_buffer.
    state: NoiseUniform,
    dirty: bool,
}

impl NoiseSystem {
    pub fn new(gpu: &GpuState) -> Self {
        let shader = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Noise Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("noise.wgsl").into()),
        });

        let uniform_size = std::mem::size_of::<NoiseUniform>() as u64;
        let uniform_buffer = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Noise Uniform Buffer"),
            contents: bytemuck::cast_slice(&[NoiseUniform {
                count: 0,
                _pad0: 0, _pad1: 0, _pad2: 0,
                sources: [NoiseSource { pos: [0.0, 0.0], db: 0.0, _pad: 0.0 }; MAX_NOISE_SOURCES],
            }]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bgl = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Noise BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(uniform_size),
                },
                count: None,
            }],
        });
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Noise Bind Group"),
            layout: &bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let pipeline_layout = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Noise Pipeline Layout"),
            bind_group_layouts: &[&gpu.camera_bind_group_layout, &bgl],
            push_constant_ranges: &[],
        });

        let pipeline = gpu.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Noise Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
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
            // Overlay: render on top of everything, no depth testing.
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

        Self {
            pipeline,
            uniform_buffer,
            bind_group,
            state: NoiseUniform {
                count: 0,
                _pad0: 0, _pad1: 0, _pad2: 0,
                sources: [NoiseSource { pos: [0.0, 0.0], db: 0.0, _pad: 0.0 }; MAX_NOISE_SOURCES],
            },
            dirty: false,
        }
    }

    /// Replace the source list. Excess sources past MAX_NOISE_SOURCES are dropped.
    pub fn set_sources(&mut self, sources: &[NoiseSource]) {
        let n = sources.len().min(MAX_NOISE_SOURCES);
        self.state.count = n as u32;
        for i in 0..n {
            self.state.sources[i] = sources[i];
        }
        self.dirty = true;
    }

    /// Upload any pending state changes to the GPU. Call once per frame
    /// before the render pass, outside of any active pass.
    pub fn flush(&mut self, gpu: &GpuState) {
        if !self.dirty {
            return;
        }
        gpu.queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[self.state]));
        self.dirty = false;
    }

    /// Render the heat-map overlay. Expects the caller to have already bound
    /// the camera bind group at slot 0; we set our own at slot 1.
    pub fn render<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        camera_bind_group: &'a wgpu::BindGroup,
    ) {
        if self.state.count == 0 {
            return;
        }
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, camera_bind_group, &[]);
        render_pass.set_bind_group(1, &self.bind_group, &[]);
        render_pass.draw(0..3, 0..1);
    }
}
