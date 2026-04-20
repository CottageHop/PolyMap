use crate::gpu::GpuState;
use crate::mapdata::Car;

/// Vertex for a car quad — two triangles per car, fully expanded (no index buffer).
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CarVertex {
    pub world_pos: [f32; 3],
    pub color: [f32; 3],
}

/// Car rendering system: owns its pipeline and a dynamic vertex buffer that the
/// CPU rebuilds each frame with current animated positions.
pub struct CarsSystem {
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: Option<wgpu::Buffer>,
    vertex_capacity: usize,
    num_vertices: u32,
    /// Preallocated working buffer to avoid per-frame allocation.
    scratch: Vec<CarVertex>,
}

const CAR_LENGTH: f32 = 3.0;
const CAR_WIDTH: f32 = 1.25;
/// Slight Z lift so cars sit on top of roads without z-fighting.
const CAR_Z: f32 = 0.25;

impl CarsSystem {
    pub fn new(gpu: &GpuState) -> Self {
        let shader = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Cars Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("cars.wgsl").into()),
        });

        let pipeline_layout = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Cars Pipeline Layout"),
            bind_group_layouts: &[&gpu.camera_bind_group_layout],
            push_constant_ranges: &[],
        });

        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<CarVertex>() as wgpu::BufferAddress,
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
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        };

        let pipeline = gpu.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Cars Pipeline"),
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

        Self {
            pipeline,
            vertex_buffer: None,
            vertex_capacity: 0,
            num_vertices: 0,
            scratch: Vec::with_capacity(4096),
        }
    }

    /// Rebuild the vertex buffer with each car's position at the given time.
    pub fn update<'a, I>(&mut self, gpu: &GpuState, cars: I, time: f32)
    where
        I: Iterator<Item = &'a Car>,
    {
        self.scratch.clear();
        let half_l = CAR_LENGTH * 0.5;
        let half_w = CAR_WIDTH * 0.5;

        for car in cars {
            if car.path_length <= 0.0 || car.path.len() < 2 {
                continue;
            }
            let t = ((car.offset + car.speed * time).rem_euclid(car.path_length)).max(0.0);
            let (pos, tangent) = sample_path(&car.path, t);
            if tangent[0].is_nan() || tangent[1].is_nan() {
                continue;
            }

            let tx = tangent[0];
            let ty = tangent[1];
            // 4 corners in car-local space → rotate by tangent → translate to pos.
            // Corner order gives two triangles via indices [0,1,2,1,3,2].
            let corners: [[f32; 2]; 4] = [
                [-half_l, -half_w],
                [ half_l, -half_w],
                [-half_l,  half_w],
                [ half_l,  half_w],
            ];
            let mut world = [[0.0_f32, 0.0_f32]; 4];
            for i in 0..4 {
                let lx = corners[i][0];
                let ly = corners[i][1];
                world[i] = [
                    pos[0] + tx * lx - ty * ly,
                    pos[1] + ty * lx + tx * ly,
                ];
            }

            let tri_order = [0usize, 1, 2, 1, 3, 2];
            for &idx in &tri_order {
                self.scratch.push(CarVertex {
                    world_pos: [world[idx][0], world[idx][1], CAR_Z],
                    color: car.color,
                });
            }
        }

        self.num_vertices = self.scratch.len() as u32;

        if self.scratch.is_empty() {
            return;
        }

        // Grow the buffer geometrically if needed; otherwise write in place.
        if self.vertex_buffer.is_none() || self.scratch.len() > self.vertex_capacity {
            let new_cap = (self.scratch.len() * 2).max(256);
            let byte_size = (new_cap * std::mem::size_of::<CarVertex>()) as u64;
            self.vertex_buffer = Some(gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Cars VB"),
                size: byte_size,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.vertex_capacity = new_cap;
        }

        if let Some(vb) = &self.vertex_buffer {
            gpu.queue.write_buffer(vb, 0, bytemuck::cast_slice(&self.scratch));
        }
    }

    pub fn render<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        camera_bind_group: &'a wgpu::BindGroup,
    ) {
        if self.num_vertices == 0 {
            return;
        }
        let Some(vb) = &self.vertex_buffer else { return };
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, camera_bind_group, &[]);
        render_pass.set_vertex_buffer(0, vb.slice(..));
        render_pass.draw(0..self.num_vertices, 0..1);
    }
}

/// Walk `path` to distance `t` and return (position, unit_tangent).
fn sample_path(path: &[[f32; 2]], t: f32) -> ([f32; 2], [f32; 2]) {
    let mut walked = 0.0_f32;
    for i in 0..path.len() - 1 {
        let dx = path[i + 1][0] - path[i][0];
        let dy = path[i + 1][1] - path[i][1];
        let seg_len = (dx * dx + dy * dy).sqrt();
        if seg_len < 1e-6 {
            continue;
        }
        if walked + seg_len >= t {
            let s = (t - walked) / seg_len;
            return (
                [path[i][0] + dx * s, path[i][1] + dy * s],
                [dx / seg_len, dy / seg_len],
            );
        }
        walked += seg_len;
    }
    // Shouldn't reach here if t <= path_length, but return a sensible fallback.
    let last = path[path.len() - 1];
    let prev = path[path.len() - 2];
    let dx = last[0] - prev[0];
    let dy = last[1] - prev[1];
    let seg_len = (dx * dx + dy * dy).sqrt().max(1e-6);
    (last, [dx / seg_len, dy / seg_len])
}
