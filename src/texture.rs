/// Background texture tiling system.
/// Renders a fullscreen quad with a repeating texture in world space,
/// positioned behind all map geometry.

const BG_SHADER: &str = r#"
struct CameraUniform {
    view_proj: mat4x4<f32>,
    position: vec2<f32>,
    viewport: vec2<f32>,
    zoom: f32,
    time: f32,
    tilt: f32,
    cloud_opacity: f32,
    cloud_speed: f32,
    label_alpha: f32,
    _pad2b: f32,
    _pad2c: f32,
    water_tint: vec4<f32>,
    park_tint: vec4<f32>,
    building_tint: vec4<f32>,
    road_tint: vec4<f32>,
    land_tint: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

@group(1) @binding(0)
var bg_texture: texture_2d<f32>;
@group(1) @binding(1)
var bg_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    // Fullscreen triangle: 3 vertices cover the entire screen
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );

    var uv = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(2.0, 1.0),
        vec2<f32>(0.0, -1.0),
    );

    var out: VertexOutput;
    out.clip_position = vec4<f32>(pos[vi], 0.999, 1.0);
    out.uv = uv[vi];
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let tile_scale = 6.0;
    let zoom_scale = 100.0 / pow(2.0, camera.zoom);
    let aspect = camera.viewport.x / camera.viewport.y;
    let screen_offset = (in.uv - vec2(0.5)) * vec2(aspect, 1.0) * zoom_scale;
    let world_pos = screen_offset + camera.position;
    let tile_uv = world_pos / tile_scale;
    return textureSample(bg_texture, bg_sampler, tile_uv);
}
"#;

/// Number of material texture layers in the array.
const MATERIAL_LAYER_COUNT: u32 = 9;

pub struct TextureSystem {
    bg_pipeline: wgpu::RenderPipeline,
    bg_bind_group_layout: wgpu::BindGroupLayout,
    bg_texture_bind_group: Option<wgpu::BindGroup>,
    sampler_repeat: wgpu::Sampler,
    material_bind_group_layout: wgpu::BindGroupLayout,
    material_bind_group: Option<wgpu::BindGroup>,
    sampler_repeat_linear: wgpu::Sampler,
}

impl TextureSystem {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: &wgpu::SurfaceConfiguration,
        camera_bind_group_layout: &wgpu::BindGroupLayout,
        msaa_samples: u32,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Background Shader"),
            source: wgpu::ShaderSource::Wgsl(BG_SHADER.into()),
        });

        let bg_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Background Texture Bind Group Layout"),
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

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Background Pipeline Layout"),
            bind_group_layouts: &[camera_bind_group_layout, &bg_bind_group_layout],
            push_constant_ranges: &[],
        });

        let bg_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Background Pipeline"),
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
                    format: config.format,
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
                depth_write_enabled: false,
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

        let sampler_repeat = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Background Repeat Sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Material texture array: bind group layout
        let material_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Material Texture Array Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2Array,
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

        // Linear repeat sampler for material textures
        let sampler_repeat_linear = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Material Repeat Linear Sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Create 1x1x9 neutral-gray placeholder so the bind group is always valid
        let placeholder_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Material Placeholder Texture"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: MATERIAL_LAYER_COUNT,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // Fill all 9 layers with neutral gray (128, 128, 128, 255)
        let gray_pixel: [u8; 4] = [128, 128, 128, 255];
        for layer in 0..MATERIAL_LAYER_COUNT {
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &placeholder_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x: 0, y: 0, z: layer },
                    aspect: wgpu::TextureAspect::All,
                },
                &gray_pixel,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(4),
                    rows_per_image: Some(1),
                },
                wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
            );
        }

        let placeholder_view = placeholder_texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });

        let material_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Material Texture Array Bind Group (placeholder)"),
            layout: &material_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&placeholder_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler_repeat_linear),
                },
            ],
        }));

        Self {
            bg_pipeline,
            bg_bind_group_layout,
            bg_texture_bind_group: None,
            sampler_repeat,
            material_bind_group_layout,
            material_bind_group,
            sampler_repeat_linear,
        }
    }

    /// Upload RGBA image data as the background texture.
    pub fn upload_background(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        rgba_data: &[u8],
    ) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Background Texture"),
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

        self.bg_texture_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Background Texture Bind Group"),
            layout: &self.bg_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler_repeat),
                },
            ],
        }));

    }

    /// Render the tiled background. Call this first in the render pass,
    /// before any map geometry.
    pub fn render_background<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        camera_bind_group: &'a wgpu::BindGroup,
    ) {
        if let Some(ref bg_bind_group) = self.bg_texture_bind_group {
            render_pass.set_pipeline(&self.bg_pipeline);
            render_pass.set_bind_group(0, camera_bind_group, &[]);
            render_pass.set_bind_group(1, bg_bind_group, &[]);
            render_pass.draw(0..3, 0..1);
        }
    }

    /// Upload material textures as a 2D texture array.
    /// `layers` is a slice of RGBA byte vectors, one per layer. All layers must
    /// have the same width * height * 4 size.
    pub fn upload_material_textures(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        layers: &[Vec<u8>],
    ) {
        let num_layers = layers.len() as u32;
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Material Texture Array"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: num_layers,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        for (i, layer_data) in layers.iter().enumerate() {
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x: 0, y: 0, z: i as u32 },
                    aspect: wgpu::TextureAspect::All,
                },
                layer_data,
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
        }

        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });

        self.material_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Material Texture Array Bind Group"),
            layout: &self.material_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler_repeat_linear),
                },
            ],
        }));

    }

    /// Bind group layout for material texture array (used by map/shadow pipelines).
    pub fn material_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.material_bind_group_layout
    }

    /// Material texture array bind group. Always `Some` after construction.
    pub fn material_bind_group(&self) -> Option<&wgpu::BindGroup> {
        self.material_bind_group.as_ref()
    }
}
