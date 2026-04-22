pub mod camera;
pub mod cars;
pub mod config;
pub mod gpu;
pub mod mapdata;
pub mod mvt;
pub mod mvt_convert;
pub mod noise;
pub mod renderer;
pub mod text;
pub mod texture;
pub mod tiles;

#[cfg(target_arch = "wasm32")]
pub mod api;
#[cfg(target_arch = "wasm32")]
pub mod pmtiles;

use std::sync::Arc;
use web_time::Instant;

use camera::Camera;
use config::LayerVisibility;
use glam::Vec2;
use gpu::GpuState;
use mapdata::MapData;
use renderer::Renderer;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

pub const DEFAULT_SOUTH: f64 = 40.758;
pub const DEFAULT_WEST: f64 = -73.990;
pub const DEFAULT_NORTH: f64 = 40.772;
pub const DEFAULT_EAST: f64 = -73.970;
pub const DEFAULT_ZOOM: f32 = -0.5;

/// A map marker placed by the host application.
#[derive(Clone, Debug)]
pub struct Marker {
    pub id: String,
    pub lat: f64,
    pub lon: f64,
    pub world_pos: [f32; 2],
}

pub struct App {
    window: Option<Arc<Window>>,
    gpu: Option<GpuState>,
    renderer: Option<Renderer>,
    camera: Camera,
    map_data: Option<MapData>,
    tile_manager: Option<tiles::TileManager>,
    use_tiles: bool,
    start_time: Instant,
    last_frame: Instant,
    mouse_pressed: bool,
    last_mouse_pos: Option<Vec2>,
    cursor_pos: Vec2,
    markers: Vec<Marker>,
    layer_visibility: LayerVisibility,
    cloud_opacity: f32,
    cloud_speed: f32,
    water_tint: [f32; 4],
    park_tint: [f32; 4],
    building_tint: [f32; 4],
    road_tint: [f32; 4],
    land_tint: [f32; 4],
    rail_tint: [f32; 4],
    ready_emitted: bool,
    markers_dirty: bool,
    labels_dirty: bool,
    last_label_zoom: f32,
    frames_since_camera_moved: u32,
    first_render_done: bool,
    /// Current label opacity, eased toward a target each frame so labels
    /// fade out while the camera is moving (hiding the lag between the
    /// gesture and the rebuild) and fade back in once the camera settles.
    label_fade: f32,
    /// Frames since GPU was initialized — used to re-sync canvas size on WASM
    frames_since_gpu_init: u32,
    #[cfg(target_arch = "wasm32")]
    gpu_pending: Option<std::pin::Pin<Box<dyn std::future::Future<Output = GpuState>>>>,
}

impl App {
    pub fn new(map_data: Option<MapData>) -> Self {
        let mut layer_vis = LayerVisibility::default();

        // Apply initial config if set (WASM)
        #[cfg(target_arch = "wasm32")]
        {
            api::INIT_CONFIG.with(|c| {
                if let Some(ref cfg) = *c.borrow() {
                    if let Some(ref layers) = cfg.layers {
                        layer_vis.apply_config(layers);
                    }
                }
            });
        }

        // If map_data is provided, use single-shot mode. Otherwise, tile mode.
        let use_tiles = map_data.is_none();

        Self {
            window: None,
            gpu: None,
            renderer: None,
            camera: Camera::new(800.0, 600.0),
            map_data,
            tile_manager: None,
            use_tiles,
            start_time: Instant::now(),
            last_frame: Instant::now(),
            mouse_pressed: false,
            last_mouse_pos: None,
            cursor_pos: Vec2::ZERO,
            markers: Vec::new(),
            layer_visibility: layer_vis,
            cloud_opacity: 0.5,
            cloud_speed: 1.0,
            water_tint: [0.0; 4],
            park_tint: [0.0; 4],
            building_tint: [0.0; 4],
            road_tint: [0.0; 4],
            land_tint: [0.0; 4],
            rail_tint: [0.0; 4],
            ready_emitted: false,
            markers_dirty: false,
            labels_dirty: false,
            last_label_zoom: -999.0,
            frames_since_camera_moved: 0,
            first_render_done: false,
            label_fade: 1.0,
            frames_since_gpu_init: 0,
            #[cfg(target_arch = "wasm32")]
            gpu_pending: None,
        }
    }

    /// Get the starting center lat/lon from config, map data, or defaults.
    fn get_start_center(&self) -> (f64, f64) {
        // 1. Check WASM config
        #[cfg(target_arch = "wasm32")]
        {
            let from_config = api::INIT_CONFIG.with(|c| {
                let c = c.borrow();
                if let Some(cfg) = c.as_ref() {
                    if let Some(ref center) = cfg.center {
                        return Some((center.lat, center.lon));
                    }
                    if let Some(ref bbox) = cfg.bbox {
                        return Some(((bbox.south + bbox.north) / 2.0, (bbox.west + bbox.east) / 2.0));
                    }
                }
                None
            });
            if let Some(c) = from_config {
                return c;
            }
        }

        // 2. From loaded map data
        if let Some(data) = &self.map_data {
            return (data.center_lat, data.center_lon);
        }

        // 3. Default: Manhattan
        ((DEFAULT_SOUTH + DEFAULT_NORTH) / 2.0, (DEFAULT_WEST + DEFAULT_EAST) / 2.0)
    }

    pub fn add_marker(&mut self, id: String, lat: f64, lon: f64) {
        let center = if let Some(data) = &self.map_data {
            Some((data.center_lat, data.center_lon))
        } else if let Some(tm) = &self.tile_manager {
            Some((tm.center_lat, tm.center_lon))
        } else {
            None
        };
        if let Some((clat, clon)) = center {
            let world_pos = mapdata::project_pub(lat, lon, clat, clon);
            self.markers.push(Marker { id, lat, lon, world_pos });
        }
    }

    pub fn remove_marker(&mut self, id: &str) {
        self.markers.retain(|m| m.id != id);
    }

    /// Process pending commands from the JS API.
    /// Uses mem::take to swap the queue out, avoiding borrow conflicts
    /// when command handlers trigger JS callbacks that push new commands.
    #[cfg(target_arch = "wasm32")]
    fn drain_commands(&mut self) {
        let commands: std::collections::VecDeque<api::Command> = api::COMMAND_QUEUE.with(|q| {
            std::mem::take(&mut *q.borrow_mut())
        });
        for cmd in commands {
                match cmd {
                    api::Command::SetView { lat, lon, zoom } => {
                        // Get center from map_data or tile_manager
                        let center = if let Some(data) = &self.map_data {
                            Some((data.center_lat, data.center_lon))
                        } else if let Some(tm) = &self.tile_manager {
                            Some((tm.center_lat, tm.center_lon))
                        } else {
                            None
                        };
                        if let Some((clat, clon)) = center {
                            let pos = mapdata::project_pub(lat, lon, clat, clon);
                            self.camera.set_target_position(pos[0], pos[1]);
                            if let Some(z) = zoom {
                                self.camera.set_target_zoom(z);
                            }
                        }
                    }
                    api::Command::SetZoom(z) => self.camera.set_target_zoom(z),
                    api::Command::SetTilt(t) => self.camera.set_target_tilt(t),
                    api::Command::PanBy { dx, dy } => self.camera.pan_by(Vec2::new(dx, dy)),
                    api::Command::AddMarker { id, lat, lon } => {
                        self.add_marker(id, lat, lon);
                        self.markers_dirty = true;
                    },
                    api::Command::RemoveMarker(id) => {
                        self.remove_marker(&id);
                        self.markers_dirty = true;
                    },
                    api::Command::ClearMarkers => {
                        self.markers.clear();
                        self.markers_dirty = true;
                    },
                    api::Command::SetLayerVisible { layer, visible } => {
                        match layer.as_str() {
                            "buildings" => self.layer_visibility.buildings = visible,
                            "roads" => self.layer_visibility.roads = visible,
                            "water" => self.layer_visibility.water = visible,
                            "parks" => self.layer_visibility.parks = visible,
                            "trees" => self.layer_visibility.trees = visible,
                            "shadows" => self.layer_visibility.shadows = visible,
                            "labels" => self.layer_visibility.labels = visible,
                            "clouds" => self.layer_visibility.clouds = visible,
                            "cars" => self.layer_visibility.cars = visible,
                            "noise" => self.layer_visibility.noise = visible,
                            _ => log::warn!("Unknown layer: {}", layer),
                        }
                    }
                    api::Command::SetCloudOpacity(opacity) => {
                        self.cloud_opacity = opacity.clamp(0.0, 1.0);
                    }
                    api::Command::SetCloudSpeed(speed) => {
                        self.cloud_speed = speed.clamp(0.0, 5.0);
                    }
                    api::Command::SetColors(colors) => {
                        if let Some(c) = colors.water { self.water_tint = c; }
                        if let Some(c) = colors.park { self.park_tint = c; }
                        if let Some(c) = colors.building { self.building_tint = c; }
                        if let Some(c) = colors.road { self.road_tint = c; }
                        if let Some(c) = colors.land { self.land_tint = c; }
                        if let Some(c) = colors.rail { self.rail_tint = c; }
                        self.labels_dirty = true; // rebuild labels with new theme colors
                    }
                    api::Command::SetNoiseSources(sources) => {
                        // Project lat/lon → world via the active projection origin.
                        let center = if let Some(data) = &self.map_data {
                            Some((data.center_lat, data.center_lon))
                        } else if let Some(tm) = &self.tile_manager {
                            Some((tm.center_lat, tm.center_lon))
                        } else {
                            None
                        };
                        if let Some((clat, clon)) = center {
                            let packed: Vec<crate::noise::NoiseSource> = sources.iter().map(|(lat, lon, db)| {
                                let pos = mapdata::project_pub(*lat, *lon, clat, clon);
                                crate::noise::NoiseSource { pos, db: *db, _pad: 0.0 }
                            }).collect();
                            if let Some(renderer) = &mut self.renderer {
                                renderer.noise.set_sources(&packed);
                            }
                        }
                    }
                    api::Command::UploadBackgroundTexture { width, height, rgba_data } => {
                        if let (Some(gpu), Some(renderer)) = (&self.gpu, &mut self.renderer) {
                            renderer.textures.upload_background(
                                &gpu.device,
                                &gpu.queue,
                                width,
                                height,
                                &rgba_data,
                            );
                        }
                    }
                    api::Command::UploadMaterialTextures { width, height, num_layers, rgba_data } => {
                        if let (Some(gpu), Some(renderer)) = (&self.gpu, &mut self.renderer) {
                            let layer_size = (width * height * 4) as usize;
                            let layers: Vec<Vec<u8>> = (0..num_layers as usize)
                                .map(|i| {
                                    let start = i * layer_size;
                                    let end = start + layer_size;
                                    rgba_data[start..end].to_vec()
                                })
                                .collect();
                            renderer.textures.upload_material_textures(
                                &gpu.device,
                                &gpu.queue,
                                width,
                                height,
                                &layers,
                            );
                        }
                    }
                    api::Command::UploadCloudTexture { .. } => {
                        // Clouds are now procedural — texture upload is a no-op
                    }
                    api::Command::LoadBbox { south, west, north, east } => {
                        let s = south; let w = west; let n = north; let e = east;
                        wasm_bindgen_futures::spawn_local(async move {
                            // Fetch new data and push it back
                            if let Some(data) = fetch_map_data_wasm_bbox(s, w, n, e).await {
                                // TODO: need a way to push new MapData back to App
                            }
                        });
                    }
                    api::Command::UploadTile {
                        col, row, vertices, indices,
                        shadow_vertices, shadow_indices,
                        labels_json, z14_tile,
                    } => {
                        if let Some(gpu) = &self.gpu {
                            let coord = tiles::TileCoord { col, row };

                            // Parse z14 tile coords from "tx,ty"
                            let z14 = {
                                let parts: Vec<&str> = z14_tile.split(',').collect();
                                if parts.len() == 2 {
                                    let tx = parts[0].parse::<u32>().unwrap_or(0);
                                    let ty = parts[1].parse::<u32>().unwrap_or(0);
                                    (tx, ty)
                                } else {
                                    (0, 0)
                                }
                            };

                            // MapVertex stride in f32s: [f32;3] + [f32;4] + f32 = 8
                            const VERT_STRIDE_F32S: usize = 8;
                            let num_verts = vertices.len() / VERT_STRIDE_F32S;
                            let num_shadow_verts = shadow_vertices.len() / VERT_STRIDE_F32S;

                            // Create vertex/index buffers from raw f32/u32 data
                            let vertex_buffer = crate::gpu::safe_buffer(
                                &gpu.device, &gpu.queue, "Worker Tile Vertex Buffer",
                                bytemuck::cast_slice(&vertices), wgpu::BufferUsages::VERTEX,
                            );
                            let index_buffer = crate::gpu::safe_buffer(
                                &gpu.device, &gpu.queue, "Worker Tile Index Buffer",
                                bytemuck::cast_slice(&indices), wgpu::BufferUsages::INDEX,
                            );
                            let num_indices = indices.len() as u32;

                            // Parallel birth-time buffer for tile fade-in.
                            let now_secs = (Instant::now() - self.start_time).as_secs_f32();
                            let birth_data: Vec<f32> = vec![now_secs; num_verts.max(1)];
                            let birth_buffer = crate::gpu::safe_buffer(
                                &gpu.device, &gpu.queue, "Worker Tile Birth VB",
                                bytemuck::cast_slice(&birth_data), wgpu::BufferUsages::VERTEX,
                            );

                            // Shadow buffers (optional)
                            let (shadow_vb, shadow_bb, shadow_ib, num_shadow) = if !shadow_vertices.is_empty() {
                                let svb = crate::gpu::safe_buffer(
                                    &gpu.device, &gpu.queue, "Worker Tile Shadow Vertex Buffer",
                                    bytemuck::cast_slice(&shadow_vertices), wgpu::BufferUsages::VERTEX,
                                );
                                let shadow_birth: Vec<f32> = vec![now_secs; num_shadow_verts.max(1)];
                                let sbb = crate::gpu::safe_buffer(
                                    &gpu.device, &gpu.queue, "Worker Tile Shadow Birth VB",
                                    bytemuck::cast_slice(&shadow_birth), wgpu::BufferUsages::VERTEX,
                                );
                                let sib = crate::gpu::safe_buffer(
                                    &gpu.device, &gpu.queue, "Worker Tile Shadow Index Buffer",
                                    bytemuck::cast_slice(&shadow_indices), wgpu::BufferUsages::INDEX,
                                );
                                (Some(svb), Some(sbb), Some(sib), shadow_indices.len() as u32)
                            } else {
                                (None, None, None, 0)
                            };

                            // Parse labels from JSON
                            let labels: Vec<mapdata::Label> = if labels_json.is_empty() {
                                Vec::new()
                            } else {
                                serde_json::from_str(&labels_json).unwrap_or_else(|e| {
                                    log::warn!("Failed to parse labels JSON: {}", e);
                                    Vec::new()
                                })
                            };

                            let loaded = tiles::LoadedTile {
                                vertex_buffer,
                                birth_buffer,
                                index_buffer,
                                num_indices,
                                shadow_vertex_buffer: shadow_vb,
                                shadow_birth_buffer: shadow_bb,
                                shadow_index_buffer: shadow_ib,
                                num_shadow_indices: num_shadow,
                                labels,
                                cars: Vec::new(),
                                last_visible_frame: 0,
                                z14_tile: z14,
                            };

                            // Ensure tile_manager exists
                            if self.tile_manager.is_none() {
                                let (clat, clon) = self.get_start_center();
                                self.tile_manager = Some(tiles::TileManager::new(clat, clon));
                            }
                            if let Some(tm) = &mut self.tile_manager {
                                tm.insert_loaded_tile(coord, loaded);
                                tm.decrement_in_flight();
                            }
                        }
                    }
                    api::Command::TileFailed { col, row } => {
                        if let Some(tm) = &mut self.tile_manager {
                            let coord = tiles::TileCoord { col, row };
                            tm.decrement_in_flight();
                            tm.record_failure_pub(coord);
                        }
                    }
                }
        }
    }

    /// Update camera state readable by JS.
    #[cfg(target_arch = "wasm32")]
    fn update_camera_state(&self) {
        let has_tm = self.tile_manager.is_some();
        let has_md = self.map_data.is_some();

        let (center_lat, center_lon) = if let Some(tm) = &self.tile_manager {
            (tm.center_lat, tm.center_lon)
        } else if let Some(data) = &self.map_data {
            (data.center_lat, data.center_lon)
        } else {
            self.get_start_center()
        };

        let (lat, lon) = mapdata::unproject_pub(
            self.camera.position.x,
            self.camera.position.y,
            center_lat,
            center_lon,
        );

        api::CAMERA_STATE.with(|cs| {
            let mut state = cs.borrow_mut();
            state.lat = lat;
            state.lon = lon;
            state.zoom = self.camera.zoom;
            state.tilt = self.camera.tilt;
        });
    }

    /// Emit marker screen positions to JS callback.
    #[cfg(target_arch = "wasm32")]
    fn emit_marker_positions(&self) {
        use wasm_bindgen::JsValue;

        api::MARKER_CALLBACK.with(|cb| {
            let cb = cb.borrow();
            let callback = match cb.as_ref() {
                Some(f) => f,
                None => return,
            };

            let dpr = web_sys::window().map(|w| w.device_pixel_ratio()).unwrap_or(1.0) as f32;
            let array = js_sys::Array::new();

            for m in &self.markers {
                let screen = self.camera.project_to_screen(m.world_pos[0], m.world_pos[1], 0.0);
                if let Some(sp) = screen {
                    let css_x = sp.x / dpr;
                    let css_y = sp.y / dpr;
                    let css_w = self.camera.viewport.x / dpr;
                    let css_h = self.camera.viewport.y / dpr;

                    let obj = js_sys::Object::new();
                    let _ = js_sys::Reflect::set(&obj, &"id".into(), &JsValue::from_str(&m.id));
                    let _ = js_sys::Reflect::set(&obj, &"x".into(), &JsValue::from_f64(css_x as f64));
                    let _ = js_sys::Reflect::set(&obj, &"y".into(), &JsValue::from_f64(css_y as f64));
                    let visible = css_x >= -50.0 && css_x <= css_w + 50.0
                               && css_y >= -50.0 && css_y <= css_h + 50.0;
                    let _ = js_sys::Reflect::set(&obj, &"visible".into(), &JsValue::from_bool(visible));
                    array.push(&obj);
                }
            }

            let _ = callback.call1(&JsValue::NULL, &array);
        });
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.gpu.is_some() {
            return;
        }

        let mut attrs = Window::default_attributes()
            .with_title("PolyMap");

        #[cfg(not(target_arch = "wasm32"))]
        {
            attrs = attrs.with_inner_size(winit::dpi::LogicalSize::new(1280, 800));
        }

        #[cfg(target_arch = "wasm32")]
        {
            use wasm_bindgen::JsCast;
            use winit::platform::web::WindowAttributesExtWebSys;

            let canvas_selector = api::INIT_CONFIG.with(|c| {
                c.borrow().as_ref().and_then(|cfg| cfg.canvas.clone())
            }).unwrap_or_else(|| "#polymap-canvas".to_string());

            let canvas = web_sys::window()
                .and_then(|win| win.document())
                .and_then(|doc| doc.query_selector(&canvas_selector).ok().flatten())
                .and_then(|el| el.dyn_into::<web_sys::HtmlCanvasElement>().ok());
            if let Some(canvas) = canvas {
                let dpr = web_sys::window()
                    .map(|w| w.device_pixel_ratio())
                    .unwrap_or(1.0);
                let css_w = canvas.client_width().max(1) as f64;
                let css_h = canvas.client_height().max(1) as f64;

                // Set canvas backing store to physical pixels
                canvas.set_width((css_w * dpr) as u32);
                canvas.set_height((css_h * dpr) as u32);

                // Tell winit to use this canvas and request the physical size
                attrs = attrs
                    .with_canvas(Some(canvas))
                    .with_inner_size(winit::dpi::PhysicalSize::new(
                        (css_w * dpr) as u32,
                        (css_h * dpr) as u32,
                    ));
            }
        }

        let window = Arc::new(
            event_loop
                .create_window(attrs)
                .expect("Failed to create window"),
        );

        #[cfg(not(target_arch = "wasm32"))]
        {
            let gpu = pollster::block_on(GpuState::new(window.clone()));
            let mut renderer = Renderer::new(&gpu);

            self.camera.resize(gpu.size.width as f32, gpu.size.height as f32);
            self.camera.set_zoom(DEFAULT_ZOOM);

            if let Some(data) = &self.map_data {
                renderer.upload_map_data(&gpu, data);
            }

            self.gpu = Some(gpu);
            self.renderer = Some(renderer);
            self.window = Some(window);
        }

        #[cfg(target_arch = "wasm32")]
        {
            use wasm_bindgen::JsCast;
            self.window = Some(window.clone());

            // Create surface from canvas OUTSIDE winit's event handler to avoid
            // Safari RefCell re-entrancy panic. We look up the canvas directly
            // and pass it to GpuState::new_from_canvas instead of going through
            // winit's window surface creation.
            let canvas_selector = api::INIT_CONFIG.with(|c| {
                c.borrow().as_ref().and_then(|cfg| cfg.canvas.clone())
            }).unwrap_or_else(|| "#polymap-canvas".to_string());

            let canvas = web_sys::window()
                .and_then(|win| win.document())
                .and_then(|doc| doc.query_selector(&canvas_selector).ok().flatten())
                .and_then(|el| el.dyn_into::<web_sys::HtmlCanvasElement>().ok());

            if let Some(canvas) = canvas {
                self.gpu_pending = Some(Box::pin(GpuState::new_from_canvas(canvas)));
            } else {
                // Fallback: use winit's window (may fail on Safari)
                self.gpu_pending = Some(Box::pin(GpuState::new(window)));
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::Resized(physical_size) => {
                if let Some(gpu) = &mut self.gpu {
                    gpu.resize(physical_size);
                    self.camera.resize(physical_size.width as f32, physical_size.height as f32);

                    #[cfg(target_arch = "wasm32")]
                    {
                        let obj = js_sys::Object::new();
                        let _ = js_sys::Reflect::set(&obj, &"width".into(), &wasm_bindgen::JsValue::from_f64(physical_size.width as f64));
                        let _ = js_sys::Reflect::set(&obj, &"height".into(), &wasm_bindgen::JsValue::from_f64(physical_size.height as f64));
                        api::emit_event("resize", &obj.into());
                    }
                }
            }

            WindowEvent::MouseInput { state, button, .. } => {
                if button == MouseButton::Left {
                    self.mouse_pressed = state == ElementState::Pressed;
                    if !self.mouse_pressed {
                        self.last_mouse_pos = None;
                    }
                }
            }

            WindowEvent::CursorMoved { position, .. } => {
                let current = Vec2::new(position.x as f32, position.y as f32);
                self.cursor_pos = current;
                if self.mouse_pressed {
                    if let Some(last) = self.last_mouse_pos {
                        let delta = current - last;
                        self.camera.pan_by(delta);
                    }
                }
                self.last_mouse_pos = Some(current);
            }

            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y * 0.5,
                    MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.01,
                };
                self.camera.zoom_at(scroll, self.cursor_pos);
            }

            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == ElementState::Pressed {
                    use winit::keyboard::{Key, NamedKey};
                    match event.logical_key {
                        Key::Named(NamedKey::Escape) => event_loop.exit(),
                        Key::Named(NamedKey::ArrowUp) => self.camera.tilt_by(-0.1),
                        Key::Named(NamedKey::ArrowDown) => self.camera.tilt_by(0.1),
                        _ => {}
                    }
                }
            }

            WindowEvent::RedrawRequested => {
                // On WASM, drain the command queue and poll GPU init
                #[cfg(target_arch = "wasm32")]
                {
                    self.drain_commands();

                    if self.gpu.is_none() {
                        if let Some(fut) = &mut self.gpu_pending {
                            use std::task::{Context, Poll, Wake};
                            struct NoopWaker;
                            impl Wake for NoopWaker {
                                fn wake(self: Arc<Self>) {}
                            }
                            let waker = Arc::new(NoopWaker).into();
                            let mut cx = Context::from_waker(&waker);
                            if let Poll::Ready(mut gpu) = fut.as_mut().poll(&mut cx) {
                                // Re-sync GPU surface with actual canvas physical pixel size
                                {
                                    use wasm_bindgen::JsCast;
                                    let canvas_selector = api::INIT_CONFIG.with(|c| {
                                        c.borrow().as_ref().and_then(|cfg| cfg.canvas.clone())
                                    }).unwrap_or_else(|| "#polymap-canvas".to_string());

                                    if let Some(canvas) = web_sys::window()
                                        .and_then(|w| w.document())
                                        .and_then(|d| d.query_selector(&canvas_selector).ok().flatten())
                                        .and_then(|e| e.dyn_into::<web_sys::HtmlCanvasElement>().ok())
                                    {
                                        let w = canvas.width();
                                        let h = canvas.height();
                                        if w > 0 && h > 0 && (w != gpu.size.width || h != gpu.size.height) {
                                            gpu.resize(winit::dpi::PhysicalSize::new(w, h));
                                        }
                                    }
                                }

                                let mut renderer = Renderer::new(&gpu);
                                self.camera.resize(gpu.size.width as f32, gpu.size.height as f32);

                                // Apply config zoom/tilt
                                let (zoom, tilt) = api::INIT_CONFIG.with(|c| {
                                    let c = c.borrow();
                                    let z = c.as_ref().and_then(|cfg| cfg.zoom).unwrap_or(DEFAULT_ZOOM);
                                    let t = c.as_ref().and_then(|cfg| cfg.tilt);
                                    (z, t)
                                });
                                self.camera.set_zoom(zoom);
                                if let Some(t) = tilt {
                                    self.camera.tilt = t;
                                    self.camera.set_target_tilt(t);
                                }

                                if let Some(data) = &self.map_data {
                                    renderer.upload_map_data(&gpu, data);
                                }
                                self.gpu = Some(gpu);
                                self.renderer = Some(renderer);
                                self.gpu_pending = None;
                                self.frames_since_gpu_init = 0; // Reset so canvas sync kicks in
                                log::info!("GPU initialized: {}x{}",
                                    self.gpu.as_ref().unwrap().size.width,
                                    self.gpu.as_ref().unwrap().size.height);
                            }
                        }
                        return;
                    }
                }

                let now = Instant::now();
                let dt = (now - self.last_frame).as_secs_f32();
                self.last_frame = now;
                let elapsed = (now - self.start_time).as_secs_f32();

                let camera_moved = self.camera.update(dt);

                // WASM DPR fix: sync canvas physical pixels with GPU surface.
                // Throttled to every 30th frame to avoid per-frame DOM reflow cost.
                #[cfg(target_arch = "wasm32")]
                {
                    self.frames_since_gpu_init += 1;
                    use wasm_bindgen::JsCast;

                    if self.frames_since_gpu_init % 30 == 0 {
                    if let Some(canvas) = web_sys::window()
                        .and_then(|w| w.document())
                        .and_then(|d| d.query_selector("canvas").ok().flatten())
                        .and_then(|e| e.dyn_into::<web_sys::HtmlCanvasElement>().ok())
                    {
                        let dpr = web_sys::window()
                            .map(|w| w.device_pixel_ratio()).unwrap_or(1.0);
                        let physical_w = (canvas.client_width() as f64 * dpr).round() as u32;
                        let physical_h = (canvas.client_height() as f64 * dpr).round() as u32;

                        if physical_w > 0 && physical_h > 0 {
                            if canvas.width() != physical_w || canvas.height() != physical_h {
                                canvas.set_width(physical_w);
                                canvas.set_height(physical_h);
                            }
                            if let Some(gpu) = &mut self.gpu {
                                if physical_w != gpu.size.width || physical_h != gpu.size.height {
                                    gpu.resize(winit::dpi::PhysicalSize::new(physical_w, physical_h));
                                    self.camera.resize(physical_w as f32, physical_h as f32);
                                    self.labels_dirty = true;
                                }
                            }
                        }
                    }
                    } // end throttled DOM check
                }

                // Tile-based loading: throttle heavy work to keep panning smooth
                if self.use_tiles {
                    if self.tile_manager.is_none() {
                        let (clat, clon) = self.get_start_center();
                        self.tile_manager = Some(tiles::TileManager::new(clat, clon));
                    }

                    if let (Some(gpu), Some(tm)) = (&self.gpu, &mut self.tile_manager) {
                        // Always poll for completed tiles (cheap — just checks a channel)
                        tm.poll_completed(gpu, elapsed);

                        // Expensive tile work (visibility calc, requests, eviction):
                        // Run every 10th frame to keep panning smooth, but always on first 3 frames
                        let frame = tm.frame_counter;
                        tm.frame_counter += 1;
                        if frame % 10 == 0 || frame < 3 {
                            tm.update_detail(tiles::detail_for_zoom(self.camera.zoom));
                            let visible = tm.visible_tiles(&self.camera);
                            tm.request_visible_tiles(&visible);
                            tm.update_visibility(&visible);
                        }

                        // New tiles or zoom changes dirty labels but we defer the
                        // actual rebuild until the camera settles — cloning all
                        // labels across 100+ tiles is ~5ms per call, too slow to
                        // do every frame during pan or zoom.
                        let zoom_changed = (self.camera.zoom - self.last_label_zoom).abs() > 0.1;
                        if tm.tiles_changed || zoom_changed {
                            self.labels_dirty = true;
                        }
                        let camera_settled = self.frames_since_camera_moved > 10;
                        let should_rebuild = self.labels_dirty && camera_settled;

                        if should_rebuild {
                            let label_refs: Vec<_> = tm.all_labels().into_iter().cloned().collect();
                            let viewport_min = gpu.size.width.min(gpu.size.height) as f32;
                            if let Some(renderer) = &mut self.renderer {
                                renderer.text.upload_labels_themed(&gpu.device, &gpu.queue, &label_refs, self.camera.zoom, self.road_tint, self.land_tint, viewport_min);
                            }
                            self.last_label_zoom = self.camera.zoom;
                            self.labels_dirty = false;
                        }
                    }
                }

                // Track how long since camera last moved
                if camera_moved {
                    if self.frames_since_camera_moved > 5 {
                        // Camera started moving again after being settled
                        self.labels_dirty = true;
                    }
                    self.frames_since_camera_moved = 0;
                } else {
                    // Emit camera:move exactly once when camera settles
                    #[cfg(target_arch = "wasm32")]
                    if self.frames_since_camera_moved == 5 && self.ready_emitted {
                        api::emit_event("camera:move", &wasm_bindgen::JsValue::UNDEFINED);
                    }
                    self.frames_since_camera_moved = self.frames_since_camera_moved.saturating_add(1);
                }

                // Update car positions (needs &mut renderer) before the render block's immutable borrow.
                if self.layer_visibility.cars {
                    if let (Some(gpu), Some(renderer), Some(tm)) =
                        (&self.gpu, self.renderer.as_mut(), self.tile_manager.as_ref())
                    {
                        let cars_iter = tm.loaded_tiles().flat_map(|t| t.cars.iter());
                        renderer.cars.update(gpu, cars_iter, elapsed);
                    }
                }

                // Flush any pending noise-source uploads (cheap no-op when clean).
                if let (Some(gpu), Some(renderer)) = (&self.gpu, self.renderer.as_mut()) {
                    renderer.noise.flush(gpu);
                }

                if let (Some(gpu), Some(renderer)) = (&self.gpu, &self.renderer) {
                    // Always update camera state FIRST so getCamera() returns valid
                    // data when JS callbacks (like ready) call it synchronously.
                    #[cfg(target_arch = "wasm32")]
                    {
                        self.update_camera_state();
                    }

                    // Emit "ready" event before idle-skip so it's not blocked
                    #[cfg(target_arch = "wasm32")]
                    if !self.ready_emitted {
                        self.ready_emitted = true;
                        let ready_data = js_sys::Object::new();
                        let _ = js_sys::Reflect::set(&ready_data, &"width".into(),
                            &wasm_bindgen::JsValue::from_f64(gpu.size.width as f64));
                        let _ = js_sys::Reflect::set(&ready_data, &"height".into(),
                            &wasm_bindgen::JsValue::from_f64(gpu.size.height as f64));
                        let dpr = web_sys::window()
                            .map(|w| w.device_pixel_ratio()).unwrap_or(1.0);
                        let _ = js_sys::Reflect::set(&ready_data, &"dpr".into(),
                            &wasm_bindgen::JsValue::from_f64(dpr));
                        api::emit_event("ready", &ready_data.into());
                    }

                    // Emit marker positions only when camera moved or markers changed
                    #[cfg(target_arch = "wasm32")]
                    {
                        if !self.markers.is_empty() || self.markers_dirty {
                            self.emit_marker_positions();
                            self.markers_dirty = false;
                        }
                    }

                    // Only render when something changed — saves CPU when idle
                    let has_markers = !self.markers.is_empty();
                    let has_loading_tiles = self.use_tiles && self.tile_manager.as_ref()
                        .map_or(false, |tm| tm.tiles.values().any(|s| matches!(s, crate::tiles::TileState::Loading)));
                    let clouds_animating = self.layer_visibility.clouds && self.cloud_opacity > 0.01 && self.cloud_speed > 0.01;
                    let cars_animating = self.layer_visibility.cars && self.use_tiles && self.tile_manager.as_ref()
                        .map_or(false, |tm| tm.loaded_tiles().any(|t| !t.cars.is_empty()));
                    // Noise rings animate based on camera.time; if overlay is on, keep rendering.
                    let noise_animating = self.layer_visibility.noise
                        && self.renderer.as_ref().map_or(false, |_| true);
                    let needs_render = camera_moved
                        || self.frames_since_camera_moved < 3
                        || (self.use_tiles && self.tile_manager.as_ref().map_or(false, |tm| tm.tiles_changed))
                        || self.labels_dirty
                        || has_markers
                        || has_loading_tiles
                        || !self.ready_emitted
                        || !self.first_render_done
                        || self.frames_since_gpu_init < 300
                        || clouds_animating
                        || cars_animating
                        || noise_animating;

                    if !needs_render {
                        return;
                    }
                    self.first_render_done = true;

                    // Ease label_fade toward 0.4 during camera motion, 1.0 when settled.
                    // Mask for the label rebuild lag so stale labels don't linger at full opacity.
                    let label_target = if self.frames_since_camera_moved <= 10 { 0.4 } else { 1.0 };
                    self.label_fade += (label_target - self.label_fade) * 0.3;

                    let mut uniform = self.camera.uniform(elapsed);
                    uniform.cloud_opacity = self.cloud_opacity;
                    uniform.cloud_speed = self.cloud_speed;
                    uniform.label_alpha = self.label_fade;
                    uniform.water_tint = self.water_tint;
                    uniform.park_tint = self.park_tint;
                    uniform.building_tint = self.building_tint;
                    uniform.road_tint = self.road_tint;
                    uniform.land_tint = self.land_tint;
                    uniform.rail_tint = self.rail_tint;
                    gpu.queue.write_buffer(&gpu.camera_buffer, 0, bytemuck::cast_slice(&[uniform]));

                    // Clear color must be in linear space for sRGB surfaces.
                    // land_tint values from JS are already linear (sRGB-to-linear in hexToRgba).
                    // Default cream [0.95, 0.90, 0.85] is sRGB — convert to linear.
                    let clear = if self.land_tint[3] > 0.5 {
                        [self.land_tint[0] as f64, self.land_tint[1] as f64, self.land_tint[2] as f64]
                    } else {
                        // sRGB cream → linear
                        [0.880, 0.787, 0.694]
                    };

                    let render_result = if self.use_tiles {
                        if let Some(tm) = &self.tile_manager {
                            let cam_aabb = self.camera.world_aabb();
                            renderer.render_tiles(gpu, tm.loaded_tiles_in_aabb(cam_aabb), &self.layer_visibility, clear, self.camera.zoom)
                        } else {
                            renderer.render(gpu, &self.layer_visibility, clear)
                        }
                    } else {
                        renderer.render(gpu, &self.layer_visibility, clear)
                    };

                    match render_result {
                        Ok(_) => {}
                        Err(wgpu::SurfaceError::Lost) => {
                            let size = gpu.size;
                            if let Some(gpu) = &mut self.gpu {
                                gpu.resize(size);
                            }
                        }
                        Err(wgpu::SurfaceError::OutOfMemory) => {
                            log::error!("Out of GPU memory");
                            event_loop.exit();
                        }
                        Err(e) => log::warn!("Surface error: {:?}", e),
                    }
                }

                // Marker/camera state emission moved above idle-skip check
            }

            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        // Only redraw when something needs updating — saves CPU/GPU when idle
        let needs_redraw = self.camera.dirty
            || self.frames_since_camera_moved < 30
            || self.gpu.is_none()
            || self.tile_manager.as_ref().map_or(false, |tm| tm.is_busy());

        #[cfg(target_arch = "wasm32")]
        let needs_redraw = needs_redraw
            || api::COMMAND_QUEUE.with(|q| !q.borrow().is_empty());

        if needs_redraw {
            if let Some(window) = &self.window {
                window.request_redraw();
            }
        }
    }
}

pub fn run_app(map_data: Option<MapData>) {
    let event_loop = EventLoop::new().expect("Failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new(map_data);
    event_loop.run_app(&mut app).expect("Event loop failed");
}

// ── WASM bootstrap ───────────────────────────────────────────────────

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

/// One frame of the main loop. Extracted so it can be called from either
/// winit's event loop (native) or a raw requestAnimationFrame loop (WASM).
#[cfg(target_arch = "wasm32")]
impl App {
    fn tick(&mut self) {
        self.drain_commands();

        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32();
        self.last_frame = now;
        let elapsed = (now - self.start_time).as_secs_f32();

        let camera_moved = self.camera.update(dt);

        // WASM DPR fix: sync canvas physical pixels with GPU surface.
        {
            use wasm_bindgen::JsCast;
            self.frames_since_gpu_init += 1;
            if self.frames_since_gpu_init % 30 == 0 {
                if let Some(canvas) = web_sys::window()
                    .and_then(|w| w.document())
                    .and_then(|d| d.query_selector("canvas").ok().flatten())
                    .and_then(|e| e.dyn_into::<web_sys::HtmlCanvasElement>().ok())
                {
                    let dpr = web_sys::window()
                        .map(|w| w.device_pixel_ratio()).unwrap_or(1.0);
                    let physical_w = (canvas.client_width() as f64 * dpr).round() as u32;
                    let physical_h = (canvas.client_height() as f64 * dpr).round() as u32;

                    if physical_w > 0 && physical_h > 0 {
                        if canvas.width() != physical_w || canvas.height() != physical_h {
                            canvas.set_width(physical_w);
                            canvas.set_height(physical_h);
                        }
                        if let Some(gpu) = &mut self.gpu {
                            if physical_w != gpu.size.width || physical_h != gpu.size.height {
                                gpu.resize(winit::dpi::PhysicalSize::new(physical_w, physical_h));
                                self.camera.resize(physical_w as f32, physical_h as f32);
                                self.labels_dirty = true;
                            }
                        }
                    }
                }
            }
        }

        // Tile-based loading
        if self.use_tiles {
            if self.tile_manager.is_none() {
                let (clat, clon) = self.get_start_center();
                self.tile_manager = Some(tiles::TileManager::new(clat, clon));
            }
            if let (Some(gpu), Some(tm)) = (&self.gpu, &mut self.tile_manager) {
                tm.poll_completed(gpu, elapsed);
                let frame = tm.frame_counter;
                tm.frame_counter += 1;
                if frame % 10 == 0 || frame < 3 {
                    tm.update_detail(tiles::detail_for_zoom(self.camera.zoom));
                    let visible = tm.visible_tiles(&self.camera);
                    tm.request_visible_tiles(&visible);
                    tm.update_visibility(&visible);
                }
                // New tiles and zoom changes both dirty the labels, but we wait
                // for camera settle before rebuilding — `all_labels().cloned().collect()`
                // over 100+ tiles is ~5ms and produces a visible stutter if fired
                // every frame during pan OR zoom gestures.
                let zoom_changed = (self.camera.zoom - self.last_label_zoom).abs() > 0.1;
                if tm.tiles_changed || zoom_changed {
                    self.labels_dirty = true;
                }
                let camera_settled = self.frames_since_camera_moved > 10;
                let should_rebuild = self.labels_dirty && camera_settled;
                if should_rebuild {
                    let label_refs: Vec<_> = tm.all_labels().into_iter().cloned().collect();
                    let viewport_min = gpu.size.width.min(gpu.size.height) as f32;
                    if let Some(renderer) = &mut self.renderer {
                        renderer.text.upload_labels_themed(&gpu.device, &gpu.queue, &label_refs, self.camera.zoom, self.road_tint, self.land_tint, viewport_min);
                    }
                    self.last_label_zoom = self.camera.zoom;
                    self.labels_dirty = false;
                }
            }
        }

        // Track camera settle
        if camera_moved {
            if self.frames_since_camera_moved > 5 {
                self.labels_dirty = true;
            }
            self.frames_since_camera_moved = 0;
        } else {
            if self.frames_since_camera_moved == 5 && self.ready_emitted {
                api::emit_event("camera:move", &wasm_bindgen::JsValue::UNDEFINED);
            }
            self.frames_since_camera_moved = self.frames_since_camera_moved.saturating_add(1);
        }

        // Update car positions (needs &mut renderer) before the render block's immutable borrow.
        if self.layer_visibility.cars {
            if let (Some(gpu), Some(renderer), Some(tm)) =
                (&self.gpu, self.renderer.as_mut(), self.tile_manager.as_ref())
            {
                let cars_iter = tm.loaded_tiles().flat_map(|t| t.cars.iter());
                renderer.cars.update(gpu, cars_iter, elapsed);
            }
        }

        // Flush pending noise-source uploads.
        if let (Some(gpu), Some(renderer)) = (&self.gpu, self.renderer.as_mut()) {
            renderer.noise.flush(gpu);
        }

        if let (Some(gpu), Some(renderer)) = (&self.gpu, &self.renderer) {
            self.update_camera_state();

            if !self.ready_emitted {
                self.ready_emitted = true;
                let ready_data = js_sys::Object::new();
                let _ = js_sys::Reflect::set(&ready_data, &"width".into(),
                    &wasm_bindgen::JsValue::from_f64(gpu.size.width as f64));
                let _ = js_sys::Reflect::set(&ready_data, &"height".into(),
                    &wasm_bindgen::JsValue::from_f64(gpu.size.height as f64));
                let dpr = web_sys::window()
                    .map(|w| w.device_pixel_ratio()).unwrap_or(1.0);
                let _ = js_sys::Reflect::set(&ready_data, &"dpr".into(),
                    &wasm_bindgen::JsValue::from_f64(dpr));
                api::emit_event("ready", &ready_data.into());
            }

            if !self.markers.is_empty() || self.markers_dirty {
                self.emit_marker_positions();
                self.markers_dirty = false;
            }

            let has_markers = !self.markers.is_empty();
            let has_loading_tiles = self.use_tiles && self.tile_manager.as_ref()
                .map_or(false, |tm| tm.tiles.values().any(|s| matches!(s, crate::tiles::TileState::Loading)));
            let clouds_animating = self.layer_visibility.clouds && self.cloud_opacity > 0.01 && self.cloud_speed > 0.01;
            let noise_animating = self.layer_visibility.noise;
            let needs_render = camera_moved
                || self.frames_since_camera_moved < 3
                || (self.use_tiles && self.tile_manager.as_ref().map_or(false, |tm| tm.tiles_changed))
                || self.labels_dirty
                || has_markers
                || has_loading_tiles
                || !self.ready_emitted
                || !self.first_render_done
                || self.frames_since_gpu_init < 300
                || clouds_animating
                || noise_animating;

            if !needs_render { return; }
            self.first_render_done = true;

            // Ease label_fade toward 0.4 during camera motion, 1.0 when settled.
            let label_target = if self.frames_since_camera_moved <= 10 { 0.4 } else { 1.0 };
            self.label_fade += (label_target - self.label_fade) * 0.3;

            let mut uniform = self.camera.uniform(elapsed);
            uniform.cloud_opacity = self.cloud_opacity;
            uniform.cloud_speed = self.cloud_speed;
            uniform.label_alpha = self.label_fade;
            uniform.water_tint = self.water_tint;
            uniform.park_tint = self.park_tint;
            uniform.building_tint = self.building_tint;
            uniform.road_tint = self.road_tint;
            uniform.land_tint = self.land_tint;
            uniform.rail_tint = self.rail_tint;
            gpu.queue.write_buffer(&gpu.camera_buffer, 0, bytemuck::cast_slice(&[uniform]));

            let clear = if self.land_tint[3] > 0.5 {
                [self.land_tint[0] as f64, self.land_tint[1] as f64, self.land_tint[2] as f64]
            } else {
                [0.880, 0.787, 0.694]
            };

            let render_result = if self.use_tiles {
                if let Some(tm) = &self.tile_manager {
                    let cam_aabb = self.camera.world_aabb();
                    renderer.render_tiles(gpu, tm.loaded_tiles_in_aabb(cam_aabb), &self.layer_visibility, clear, self.camera.zoom)
                } else {
                    renderer.render(gpu, &self.layer_visibility, clear)
                }
            } else {
                renderer.render(gpu, &self.layer_visibility, clear)
            };

            match render_result {
                Ok(_) => {}
                Err(wgpu::SurfaceError::Lost) => {
                    let size = gpu.size;
                    if let Some(gpu) = &mut self.gpu {
                        gpu.resize(size);
                    }
                }
                Err(e) => log::warn!("Surface error: {:?}", e),
            }
        }
    }
}

/// WASM bootstrap: bypasses winit entirely to avoid Safari RefCell panics.
/// Uses requestAnimationFrame + raw DOM event listeners.
#[cfg(target_arch = "wasm32")]
pub async fn wasm_init() {
    use std::cell::RefCell;
    use std::rc::Rc;
    use wasm_bindgen::JsCast;

    // Look up canvas
    let canvas_selector = api::INIT_CONFIG.with(|c| {
        c.borrow().as_ref().and_then(|cfg| cfg.canvas.clone())
    }).unwrap_or_else(|| "#polymap-canvas".to_string());

    let canvas: web_sys::HtmlCanvasElement = web_sys::window()
        .and_then(|w| w.document())
        .and_then(|d| d.query_selector(&canvas_selector).ok().flatten())
        .and_then(|e| e.dyn_into().ok())
        .expect("PolyMap: canvas not found");

    // Create GPU surface directly from canvas (no winit)
    let gpu = GpuState::new_from_canvas(canvas.clone()).await;
    let renderer = Renderer::new(&gpu);

    let mut app = App::new(None);
    app.camera.resize(gpu.size.width as f32, gpu.size.height as f32);

    // Apply config
    let (zoom, tilt) = api::INIT_CONFIG.with(|c| {
        let c = c.borrow();
        let z = c.as_ref().and_then(|cfg| cfg.zoom).unwrap_or(DEFAULT_ZOOM);
        let t = c.as_ref().and_then(|cfg| cfg.tilt);
        (z, t)
    });
    app.camera.set_zoom(zoom);
    if let Some(t) = tilt {
        app.camera.tilt = t;
        app.camera.set_target_tilt(t);
    }

    app.gpu = Some(gpu);
    app.renderer = Some(renderer);

    let app = Rc::new(RefCell::new(app));

    // ── DOM event listeners ──────────────────────────────────────────
    // All handlers use try_borrow_mut — if the app is borrowed (e.g. during
    // tick()), the event is silently dropped. This prevents RefCell panics
    // when the browser dispatches DOM events synchronously during rendering.
    {
        let app = app.clone();
        let cb = Closure::<dyn FnMut(_)>::new(move |_: web_sys::MouseEvent| {
            if let Ok(mut a) = app.try_borrow_mut() { a.mouse_pressed = true; }
        });
        canvas.add_event_listener_with_callback("mousedown", cb.as_ref().unchecked_ref()).ok();
        cb.forget();
    }
    // Register mouseup on document so releasing over a marker still ends the pan
    {
        let app = app.clone();
        let cb = Closure::<dyn FnMut(_)>::new(move |_: web_sys::MouseEvent| {
            if let Ok(mut a) = app.try_borrow_mut() {
                a.mouse_pressed = false;
                a.last_mouse_pos = None;
            }
        });
        let doc = web_sys::window().unwrap().document().unwrap();
        doc.add_event_listener_with_callback("mouseup", cb.as_ref().unchecked_ref()).ok();
        cb.forget();
    }
    // Register mousemove on document so panning continues even when cursor is over markers
    {
        let app = app.clone();
        let canvas_ref = canvas.clone();
        let dpr = web_sys::window().map(|w| w.device_pixel_ratio()).unwrap_or(1.0) as f32;
        let cb = Closure::<dyn FnMut(_)>::new(move |e: web_sys::MouseEvent| {
            if let Ok(mut a) = app.try_borrow_mut() {
                let rect = canvas_ref.get_bounding_client_rect();
                let x = (e.client_x() as f64 - rect.left()) as f32 * dpr;
                let y = (e.client_y() as f64 - rect.top()) as f32 * dpr;
                let current = Vec2::new(x, y);
                a.cursor_pos = current;
                if a.mouse_pressed {
                    if let Some(last) = a.last_mouse_pos {
                        let delta = current - last;
                        a.camera.pan_by(delta);
                    }
                }
                a.last_mouse_pos = Some(current);
            }
        });
        let doc = web_sys::window().unwrap().document().unwrap();
        doc.add_event_listener_with_callback("mousemove", cb.as_ref().unchecked_ref()).ok();
        cb.forget();
    }
    {
        let app = app.clone();
        let dpr = web_sys::window().map(|w| w.device_pixel_ratio()).unwrap_or(1.0) as f32;
        let canvas_ref = canvas.clone();
        let cb = Closure::<dyn FnMut(_)>::new(move |e: web_sys::WheelEvent| {
            e.prevent_default();
            if let Ok(mut a) = app.try_borrow_mut() {
                let scroll = -e.delta_y() as f32 * 0.01;
                // Use clientX/Y minus canvas rect (works for both native and synthetic events)
                let rect = canvas_ref.get_bounding_client_rect();
                let x = (e.client_x() as f64 - rect.left()) as f32 * dpr;
                let y = (e.client_y() as f64 - rect.top()) as f32 * dpr;
                a.camera.zoom_at(scroll, Vec2::new(x, y));
            }
        });
        let opts = web_sys::AddEventListenerOptions::new();
        opts.set_passive(false);
        canvas.add_event_listener_with_callback_and_add_event_listener_options(
            "wheel", cb.as_ref().unchecked_ref(), &opts,
        ).ok();
        cb.forget();
    }
    // Touch support for mobile Safari
    {
        let app = app.clone();
        let dpr = web_sys::window().map(|w| w.device_pixel_ratio()).unwrap_or(1.0) as f32;
        let cb = Closure::<dyn FnMut(_)>::new(move |e: web_sys::TouchEvent| {
            e.prevent_default();
            if let Ok(mut a) = app.try_borrow_mut() {
                if let Some(touch) = e.touches().get(0) {
                    let rect = e.target().and_then(|t| t.dyn_into::<web_sys::Element>().ok())
                        .map(|el| el.get_bounding_client_rect());
                    if let Some(rect) = rect {
                        let x = (touch.client_x() as f64 - rect.left()) as f32 * dpr;
                        let y = (touch.client_y() as f64 - rect.top()) as f32 * dpr;
                        a.cursor_pos = Vec2::new(x, y);
                        a.mouse_pressed = true;
                        a.last_mouse_pos = Some(Vec2::new(x, y));
                    }
                }
            }
        });
        let opts = web_sys::AddEventListenerOptions::new();
        opts.set_passive(false);
        canvas.add_event_listener_with_callback_and_add_event_listener_options(
            "touchstart", cb.as_ref().unchecked_ref(), &opts,
        ).ok();
        cb.forget();
    }
    {
        let app = app.clone();
        let dpr = web_sys::window().map(|w| w.device_pixel_ratio()).unwrap_or(1.0) as f32;
        let cb = Closure::<dyn FnMut(_)>::new(move |e: web_sys::TouchEvent| {
            e.prevent_default();
            if let Ok(mut a) = app.try_borrow_mut() {
                if let Some(touch) = e.touches().get(0) {
                    let rect = e.target().and_then(|t| t.dyn_into::<web_sys::Element>().ok())
                        .map(|el| el.get_bounding_client_rect());
                    if let Some(rect) = rect {
                        let x = (touch.client_x() as f64 - rect.left()) as f32 * dpr;
                        let y = (touch.client_y() as f64 - rect.top()) as f32 * dpr;
                        let current = Vec2::new(x, y);
                        if let Some(last) = a.last_mouse_pos {
                            let delta = current - last;
                            a.camera.pan_by(delta);
                        }
                        a.last_mouse_pos = Some(current);
                    }
                }
            }
        });
        let opts = web_sys::AddEventListenerOptions::new();
        opts.set_passive(false);
        canvas.add_event_listener_with_callback_and_add_event_listener_options(
            "touchmove", cb.as_ref().unchecked_ref(), &opts,
        ).ok();
        cb.forget();
    }
    {
        let app = app.clone();
        let cb = Closure::<dyn FnMut(_)>::new(move |_: web_sys::TouchEvent| {
            if let Ok(mut a) = app.try_borrow_mut() {
                a.mouse_pressed = false;
                a.last_mouse_pos = None;
            }
        });
        canvas.add_event_listener_with_callback("touchend", cb.as_ref().unchecked_ref()).ok();
        cb.forget();
    }

    // ── requestAnimationFrame loop ───────────────────────────────────
    let f: Rc<RefCell<Option<Closure<dyn FnMut()>>>> = Rc::new(RefCell::new(None));
    let g = f.clone();
    let app_raf = app.clone();

    *g.borrow_mut() = Some(Closure::new(move || {
        if let Ok(mut a) = app_raf.try_borrow_mut() { a.tick(); }
        // Schedule next frame
        if let Some(win) = web_sys::window() {
            let _ = win.request_animation_frame(
                f.borrow().as_ref().unwrap().as_ref().unchecked_ref()
            );
        }
    }));

    // Kick off the first frame
    if let Some(win) = web_sys::window() {
        let _ = win.request_animation_frame(
            g.borrow().as_ref().unwrap().as_ref().unchecked_ref()
        );
    }
}

/// Also keep the old auto-start for backward compatibility.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub fn wasm_main() {
    // Only init logger/panic hook. Actual start happens via PolyMap constructor.
    console_error_panic_hook::set_once();
    let _ = console_log::init_with_level(log::Level::Warn);
}

// ── Backward-compatible free functions ───────────────────────────────

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn set_marker_callback(callback: js_sys::Function) {
    api::MARKER_CALLBACK.with(|cb| *cb.borrow_mut() = Some(callback));
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn add_marker(id: &str, lat: f64, lon: f64) {
    api::push_command(api::Command::AddMarker { id: id.to_string(), lat, lon });
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn remove_marker(id: &str) {
    api::push_command(api::Command::RemoveMarker(id.to_string()));
}

// ── WASM data fetching ───────────────────────────────────────────────

#[cfg(target_arch = "wasm32")]
async fn wasm_fetch_text(url: &str, method: &str, content_type: Option<&str>, body: Option<&str>) -> Option<String> {
    use wasm_bindgen::JsCast;
    use web_sys::{Request, RequestInit, RequestMode, Response};

    let opts = RequestInit::new();
    opts.set_method(method);
    opts.set_mode(RequestMode::Cors);
    if let Some(b) = body {
        opts.set_body(&JsValue::from_str(b));
    }

    let request = Request::new_with_str_and_init(url, &opts).ok()?;
    if let Some(ct) = content_type {
        request.headers().set("Content-Type", ct).ok()?;
    }

    // Attach API key if configured
    let api_key = api::INIT_CONFIG.with(|c| {
        c.borrow().as_ref().and_then(|cfg| cfg.api_key.clone())
    });
    if let Some(key) = api_key {
        request.headers().set("X-Map-Key", &key).ok()?;
    }

    let window = web_sys::window()?;
    let resp_value = wasm_bindgen_futures::JsFuture::from(window.fetch_with_request(&request))
        .await
        .ok()?;
    let resp: Response = resp_value.dyn_into().ok()?;

    if !resp.ok() {
        return None;
    }

    let text = wasm_bindgen_futures::JsFuture::from(resp.text().ok()?)
        .await
        .ok()?;
    text.as_string()
}

/// Fetch binary data from a URL using an HTTP Range request.
/// Used by the PMTiles reader to fetch header, directories, and tile data.
/// `end` is exclusive: fetches bytes [start, end).
#[cfg(target_arch = "wasm32")]
pub(crate) async fn wasm_fetch_bytes(url: &str, start: u64, end: u64) -> Option<Vec<u8>> {
    use wasm_bindgen::JsCast;
    use web_sys::{Request, RequestInit, RequestMode, Response};

    let opts = RequestInit::new();
    opts.set_method("GET");
    opts.set_mode(RequestMode::Cors);

    let request = Request::new_with_str_and_init(url, &opts).ok()?;
    request.headers().set("Range", &format!("bytes={}-{}", start, end - 1)).ok()?;

    let window = web_sys::window()?;
    let resp_value = wasm_bindgen_futures::JsFuture::from(window.fetch_with_request(&request))
        .await
        .ok()?;
    let resp: Response = resp_value.dyn_into().ok()?;

    let status = resp.status();
    if status != 200 && status != 206 {
        return None;
    }

    let buf = wasm_bindgen_futures::JsFuture::from(resp.array_buffer().ok()?)
        .await
        .ok()?;
    let array = js_sys::Uint8Array::new(&buf);
    Some(array.to_vec())
}

#[cfg(target_arch = "wasm32")]
async fn fetch_map_data_wasm_bbox(south: f64, west: f64, north: f64, east: f64) -> Option<MapData> {
    // Try custom data URL from config
    let (data_url, api_base) = api::INIT_CONFIG.with(|c| {
        let cfg = c.borrow();
        let cfg = cfg.as_ref();
        (
            cfg.and_then(|c| c.data_url.clone()),
            cfg.and_then(|c| c.api_base.clone()),
        )
    });

    // Try configured API
    let base = api_base.as_deref().unwrap_or(mapdata::API_BASE);
    let api_url = data_url.unwrap_or_else(|| {
        format!("{}/map/osm?south={}&west={}&north={}&east={}",
            base, south, west, north, east)
    });


    let body_str = wasm_fetch_text(&api_url, "GET", None, None).await?;
    match mapdata::parse_osm_json(&body_str, south, west, north, east) {
        Ok(data) => {
            Some(data)
        }
        Err(e) => {
            log::error!("Failed to parse: {}", e);
            None
        }
    }
}
