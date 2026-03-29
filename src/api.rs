#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};

use crate::config::{ColorConfig, LayerConfig, PolyMapConfig};

/// Commands sent from JS to the running App via the command queue.
pub enum Command {
    SetView { lat: f64, lon: f64, zoom: Option<f32> },
    SetZoom(f32),
    SetTilt(f32),
    PanBy { dx: f32, dy: f32 },
    AddMarker { id: String, lat: f64, lon: f64 },
    RemoveMarker(String),
    ClearMarkers,
    SetLayerVisible { layer: String, visible: bool },
    SetColors(ColorConfig),
    LoadBbox { south: f64, west: f64, north: f64, east: f64 },
    UploadBackgroundTexture {
        width: u32,
        height: u32,
        rgba_data: Vec<u8>,
    },
    UploadMaterialTextures {
        width: u32,
        height: u32,
        num_layers: u32,
        rgba_data: Vec<u8>,
    },
    UploadCloudTexture {
        width: u32,
        height: u32,
        rgba_data: Vec<u8>,
    },
    UploadTile {
        col: i32,
        row: i32,
        vertices: Vec<f32>,
        indices: Vec<u32>,
        shadow_vertices: Vec<f32>,
        shadow_indices: Vec<u32>,
        labels_json: String,
        z14_tile: String,
    },
    TileFailed {
        col: i32,
        row: i32,
    },
}

/// Camera state readable from JS without a round-trip.
#[derive(Clone, Debug, Default)]
pub struct CameraState {
    pub lat: f64,
    pub lon: f64,
    pub zoom: f32,
    pub tilt: f32,
}

thread_local! {
    pub static COMMAND_QUEUE: RefCell<VecDeque<Command>> = RefCell::new(VecDeque::new());
    pub static EVENT_CALLBACKS: RefCell<HashMap<String, js_sys::Function>> = RefCell::new(HashMap::new());
    pub static CAMERA_STATE: RefCell<CameraState> = RefCell::new(CameraState::default());
    pub static INIT_CONFIG: RefCell<Option<PolyMapConfig>> = RefCell::new(None);
    pub static MARKER_DATA: RefCell<Vec<(String, f64, f64)>> = RefCell::new(Vec::new());
    pub static MARKER_CALLBACK: RefCell<Option<js_sys::Function>> = RefCell::new(None);
    pub static TILE_CALLBACK: RefCell<Option<js_sys::Function>> = RefCell::new(None);
}

/// Push a command to the queue (called from PolyMap methods).
/// Uses try_borrow_mut to avoid panicking if called re-entrantly during drain.
pub fn push_command(cmd: Command) {
    COMMAND_QUEUE.with(|q| {
        match q.try_borrow_mut() {
            Ok(mut queue) => queue.push_back(cmd),
            Err(_) => {
                // Queue is borrowed (render loop is draining) — defer to next frame
                // by spawning a microtask
                #[cfg(target_arch = "wasm32")]
                {
                    let cmd_cell = std::cell::RefCell::new(Some(cmd));
                    wasm_bindgen_futures::spawn_local(async move {
                        if let Some(c) = cmd_cell.borrow_mut().take() {
                            COMMAND_QUEUE.with(|q| q.borrow_mut().push_back(c));
                        }
                    });
                }
                #[cfg(not(target_arch = "wasm32"))]
                {
                    log::warn!("Command queue busy, dropping command");
                }
            }
        }
    });
}

/// Emit a named event to the registered JS callback.
#[cfg(target_arch = "wasm32")]
pub fn emit_event(name: &str, data: &JsValue) {
    EVENT_CALLBACKS.with(|cbs| {
        if let Some(cb) = cbs.borrow().get(name) {
            let _ = cb.call1(&JsValue::NULL, data);
        }
    });
}

// ── JS-facing PolyMap handle ──────────────────────────────────────────

/// The main PolyMap API exposed to JavaScript.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct PolyMap {
    _private: (),
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl PolyMap {
    /// Create a new PolyMap instance. Accepts a configuration object.
    #[wasm_bindgen(constructor)]
    pub fn new(config: JsValue) -> PolyMap {
        console_error_panic_hook::set_once();
        let _ = console_log::init_with_level(log::Level::Warn);

        let parsed: PolyMapConfig = if config.is_undefined() || config.is_null() {
            PolyMapConfig::default()
        } else {
            serde_wasm_bindgen::from_value(config).unwrap_or_default()
        };

        // Reset all stale state from previous instance (SPA navigation)
        INIT_CONFIG.with(|c| *c.borrow_mut() = Some(parsed));
        EVENT_CALLBACKS.with(|c| c.borrow_mut().clear());
        COMMAND_QUEUE.with(|c| c.borrow_mut().clear());
        MARKER_CALLBACK.with(|c| *c.borrow_mut() = None);
        CAMERA_STATE.with(|c| *c.borrow_mut() = CameraState::default());

        // Kick off async init
        wasm_bindgen_futures::spawn_local(crate::wasm_init());

        PolyMap { _private: () }
    }

    // ── Camera ────────────────────────────────────────────────────────

    /// Smoothly pan and zoom to a lat/lon position.
    #[wasm_bindgen(js_name = setView)]
    pub fn set_view(&self, lat: f64, lon: f64, zoom: Option<f32>) {
        push_command(Command::SetView { lat, lon, zoom });
    }

    /// Set the zoom level.
    #[wasm_bindgen(js_name = setZoom)]
    pub fn set_zoom(&self, zoom: f32) {
        push_command(Command::SetZoom(zoom));
    }

    /// Set the camera tilt (radians, 0.1 = horizon, 1.4 = top-down).
    #[wasm_bindgen(js_name = setTilt)]
    pub fn set_tilt(&self, tilt: f32) {
        push_command(Command::SetTilt(tilt));
    }

    /// Pan the camera by pixel offset.
    #[wasm_bindgen(js_name = panBy)]
    pub fn pan_by(&self, dx: f32, dy: f32) {
        push_command(Command::PanBy { dx, dy });
    }

    /// Get the current camera state (lat, lon, zoom, tilt).
    #[wasm_bindgen(js_name = getCamera)]
    pub fn get_camera(&self) -> JsValue {
        CAMERA_STATE.with(|cs| {
            let state = cs.borrow();
            let obj = js_sys::Object::new();
            let _ = js_sys::Reflect::set(&obj, &"lat".into(), &JsValue::from_f64(state.lat));
            let _ = js_sys::Reflect::set(&obj, &"lon".into(), &JsValue::from_f64(state.lon));
            let _ = js_sys::Reflect::set(&obj, &"zoom".into(), &JsValue::from_f64(state.zoom as f64));
            let _ = js_sys::Reflect::set(&obj, &"tilt".into(), &JsValue::from_f64(state.tilt as f64));
            obj.into()
        })
    }

    // ── Markers ───────────────────────────────────────────────────────

    /// Add a marker at a lat/lon. Its screen position is sent to the marker callback each frame.
    #[wasm_bindgen(js_name = addMarker)]
    pub fn add_marker(&self, id: &str, lat: f64, lon: f64) {
        push_command(Command::AddMarker { id: id.to_string(), lat, lon });
    }

    /// Remove a marker by ID.
    #[wasm_bindgen(js_name = removeMarker)]
    pub fn remove_marker(&self, id: &str) {
        push_command(Command::RemoveMarker(id.to_string()));
    }

    /// Remove all markers.
    #[wasm_bindgen(js_name = clearMarkers)]
    pub fn clear_markers(&self) {
        push_command(Command::ClearMarkers);
    }

    /// Set a callback that receives marker screen positions each frame.
    /// Callback receives: `[{ id: string, x: number, y: number, visible: boolean }]`
    #[wasm_bindgen(js_name = setMarkerCallback)]
    pub fn set_marker_callback(&self, callback: js_sys::Function) {
        MARKER_CALLBACK.with(|cb| *cb.borrow_mut() = Some(callback));
    }

    /// Set a callback for tile processing. When set, WASM calls this instead of
    /// fetching tiles internally. Callback receives: (col, row, south, west, north, east, centerLat, centerLon, detail)
    #[wasm_bindgen(js_name = setTileCallback)]
    pub fn set_tile_callback(&self, callback: js_sys::Function) {
        TILE_CALLBACK.with(|cb| *cb.borrow_mut() = Some(callback));
    }

    // ── Textures ──────────────────────────────────────────────────────

    /// Set the background texture from RGBA pixel data.
    #[wasm_bindgen(js_name = setBackgroundTexture)]
    pub fn set_background_texture(&self, width: u32, height: u32, data: &js_sys::Uint8Array) {
        push_command(Command::UploadBackgroundTexture {
            width,
            height,
            rgba_data: data.to_vec(),
        });
    }

    /// Upload material textures as a 2D array. Data is a packed RGBA Uint8Array
    /// containing `num_layers` images of size `width * height * 4` each, laid out
    /// sequentially (layer 0 first, then layer 1, etc.).
    #[wasm_bindgen(js_name = setMaterialTextures)]
    pub fn set_material_textures(
        &self,
        width: u32,
        height: u32,
        num_layers: u32,
        data: &js_sys::Uint8Array,
    ) {
        push_command(Command::UploadMaterialTextures {
            width,
            height,
            num_layers,
            rgba_data: data.to_vec(),
        });
    }

    /// Set the cloud atlas texture from RGBA pixel data (2048x1024, 4x3 grid of cloud sprites).
    #[wasm_bindgen(js_name = setCloudTexture)]
    pub fn set_cloud_texture(&self, width: u32, height: u32, data: &js_sys::Uint8Array) {
        push_command(Command::UploadCloudTexture {
            width,
            height,
            rgba_data: data.to_vec(),
        });
    }

    // ── Layers ────────────────────────────────────────────────────────

    /// Toggle layer visibility. Layers: "buildings", "roads", "water", "parks", "trees", "shadows", "labels".
    #[wasm_bindgen(js_name = setLayerVisible)]
    pub fn set_layer_visible(&self, layer: &str, visible: bool) {
        push_command(Command::SetLayerVisible { layer: layer.to_string(), visible });
    }

    // ── Events ────────────────────────────────────────────────────────

    /// Register an event callback. Events: "ready", "camera:move", "click", "resize".
    pub fn on(&self, event: &str, callback: js_sys::Function) {
        EVENT_CALLBACKS.with(|cbs| {
            cbs.borrow_mut().insert(event.to_string(), callback);
        });
    }

    /// Remove an event callback.
    pub fn off(&self, event: &str) {
        EVENT_CALLBACKS.with(|cbs| {
            cbs.borrow_mut().remove(event);
        });
    }

    // ── Data ──────────────────────────────────────────────────────────

    /// Load map data for a new bounding box.
    #[wasm_bindgen(js_name = loadBbox)]
    pub fn load_bbox(&self, south: f64, west: f64, north: f64, east: f64) {
        push_command(Command::LoadBbox { south, west, north, east });
    }

    /// Upload pre-processed tile data from a Web Worker.
    /// Vertices is a Float32Array (8 floats per vertex), indices is a Uint32Array.
    /// Shadow buffers follow the same layout. labels_json is a JSON array of label objects.
    /// z14_tile is "tx,ty" for dedup.
    #[wasm_bindgen(js_name = pushTileData)]
    pub fn push_tile_data(
        &self,
        col: i32,
        row: i32,
        vertices: &js_sys::Float32Array,
        indices: &js_sys::Uint32Array,
        shadow_vertices: &js_sys::Float32Array,
        shadow_indices: &js_sys::Uint32Array,
        labels_json: &str,
        z14_tile: &str,
    ) {
        push_command(Command::UploadTile {
            col,
            row,
            vertices: vertices.to_vec(),
            indices: indices.to_vec(),
            shadow_vertices: shadow_vertices.to_vec(),
            shadow_indices: shadow_indices.to_vec(),
            labels_json: labels_json.to_string(),
            z14_tile: z14_tile.to_string(),
        });
    }

    /// Notify WASM that a worker tile fetch failed or returned empty.
    /// Decrements the in-flight counter so new tiles can be requested.
    #[wasm_bindgen(js_name = notifyTileFailed)]
    pub fn notify_tile_failed(&self, col: i32, row: i32) {
        push_command(Command::TileFailed { col, row });
    }
}
