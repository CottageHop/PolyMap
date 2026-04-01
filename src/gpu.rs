use std::sync::Arc;
use wgpu::{self, util::DeviceExt};
use winit::window::Window;

use crate::camera::CameraUniform;

pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth24Plus;

/// Default MSAA sample count. Safari/iOS may not support 4x — GpuState
/// detects supported counts at init and stores the actual value.
pub const DEFAULT_MSAA_SAMPLES: u32 = 4;

/// WebGPU `mappedAtCreation` size limit — buffers above this use the staging path.
/// Some browsers/devices fail well under 1MB; use 256KB as a safe threshold.
const MAPPED_CREATION_MAX: u64 = 256 * 1024;

/// Create a GPU buffer, choosing the fastest safe path:
/// - Small buffers: `create_buffer_init` (direct map, GPU-optimal memory)
/// - Large buffers: `create_buffer` + `queue.write_buffer` (staging copy, avoids WebGPU size limit)
pub fn safe_buffer(device: &wgpu::Device, queue: &wgpu::Queue, label: &str, data: &[u8], usage: wgpu::BufferUsages) -> wgpu::Buffer {
    let size = (data.len() as u64).max(4);

    if size <= MAPPED_CREATION_MAX {
        // Fast path: direct mapped write, no COPY_DST needed
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: if data.is_empty() { &[0u8; 4] } else { data },
            usage,
        })
    } else {
        // Large buffer: staging copy to avoid mappedAtCreation size limit
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: usage | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if !data.is_empty() {
            queue.write_buffer(&buf, 0, data);
        }
        buf
    }
}

pub struct GpuState {
    pub surface: wgpu::Surface<'static>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,
    pub size: winit::dpi::PhysicalSize<u32>,
    pub camera_buffer: wgpu::Buffer,
    pub camera_bind_group: wgpu::BindGroup,
    pub camera_bind_group_layout: wgpu::BindGroupLayout,
    pub depth_view: wgpu::TextureView,
    pub msaa_view: wgpu::TextureView,
    pub msaa_samples: u32,
}

impl GpuState {
    /// Create a GpuState from a canvas element directly (WASM).
    /// This avoids re-entrant RefCell borrows in winit on Safari
    /// by creating the surface outside of the event loop handler.
    #[cfg(target_arch = "wasm32")]
    pub async fn new_from_canvas(canvas: web_sys::HtmlCanvasElement) -> Self {
        let w = canvas.width();
        let h = canvas.height();
        let size = winit::dpi::PhysicalSize::new(w.max(1), h.max(1));

        // Try WebGPU first, fall back to WebGL2 (for Safari < 17)
        let (instance, surface) = {
            let inst = wgpu::Instance::new(&wgpu::InstanceDescriptor {
                backends: wgpu::Backends::BROWSER_WEBGPU | wgpu::Backends::GL,
                ..Default::default()
            });
            match inst.create_surface(wgpu::SurfaceTarget::Canvas(canvas.clone())) {
                Ok(s) => (inst, s),
                Err(e) => {
                    log::warn!("WebGPU surface failed ({e}), trying WebGL2 only");
                    let inst = wgpu::Instance::new(&wgpu::InstanceDescriptor {
                        backends: wgpu::Backends::GL,
                        ..Default::default()
                    });
                    let s = inst
                        .create_surface(wgpu::SurfaceTarget::Canvas(canvas))
                        .expect("Neither WebGPU nor WebGL2 available");
                    (inst, s)
                }
            }
        };

        Self::init(instance, surface, size).await
    }

    pub async fn new(window: Arc<Window>) -> Self {
        let mut size = window.inner_size();

        // Fallback if canvas not laid out yet
        if size.width == 0 || size.height == 0 {
            size = winit::dpi::PhysicalSize::new(1280, 800);
        }

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let surface = instance.create_surface(window).unwrap();

        Self::init(instance, surface, size).await
    }

    async fn init(
        instance: wgpu::Instance,
        surface: wgpu::Surface<'static>,
        size: winit::dpi::PhysicalSize<u32>,
    ) -> Self {

        // Try high performance first, fall back to low power (integrated GPU)
        let adapter = match instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
        {
            Some(a) => a,
            None => instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::LowPower,
                    compatible_surface: Some(&surface),
                    force_fallback_adapter: false,
                })
                .await
                .expect("Failed to find a suitable GPU adapter"),
        };


        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("PolyMap Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_webgl2_defaults()
                        .using_resolution(adapter.limits()),
                    ..Default::default()
                },
                None,
            )
            .await
            .expect("Failed to create device");

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        // Detect supported MSAA sample count (Safari/iOS may only support 1)
        let msaa_samples = {
            let features = adapter.get_texture_format_features(surface_format);
            if features.flags.contains(wgpu::TextureFormatFeatureFlags::MULTISAMPLE_X4) {
                DEFAULT_MSAA_SAMPLES
            } else {
                1
            }
        };

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let camera_uniform = CameraUniform::default();
        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Camera Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Camera Bind Group"),
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        let depth_view = create_depth_texture(&device, size.width, size.height, msaa_samples);
        let msaa_view = create_msaa_texture(&device, &config, size.width, size.height, msaa_samples);

        Self {
            surface,
            device,
            queue,
            config,
            size,
            camera_buffer,
            camera_bind_group,
            camera_bind_group_layout,
            depth_view,
            msaa_view,
            msaa_samples,
        }
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            self.depth_view = create_depth_texture(&self.device, new_size.width, new_size.height, self.msaa_samples);
            self.msaa_view = create_msaa_texture(&self.device, &self.config, new_size.width, new_size.height, self.msaa_samples);
        }
    }
}

fn create_depth_texture(device: &wgpu::Device, width: u32, height: u32, sample_count: u32) -> wgpu::TextureView {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Depth Texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

fn create_msaa_texture(
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
    width: u32,
    height: u32,
    sample_count: u32,
) -> wgpu::TextureView {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("MSAA Color Texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count,
        dimension: wgpu::TextureDimension::D2,
        format: config.format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}
