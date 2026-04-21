use std::collections::{HashMap, VecDeque};
use std::rc::Rc;

use crate::gpu::GpuState;
use crate::mapdata::{self, MapData, MapVertex, Label};
use crate::mvt_convert::DetailLevel;

/// LRU-tracked cache. The companion VecDeque stores keys in insertion/access
/// order so eviction on capacity overflow drops the single oldest entry
/// instead of wiping the whole map (which caused frame stutter on pan).
#[cfg(target_arch = "wasm32")]
struct LruCache<K: Eq + std::hash::Hash + Clone, V> {
    map: HashMap<K, V>,
    order: VecDeque<K>,
    cap: usize,
}

#[cfg(target_arch = "wasm32")]
impl<K: Eq + std::hash::Hash + Clone, V> LruCache<K, V> {
    fn new(cap: usize) -> Self {
        Self { map: HashMap::new(), order: VecDeque::new(), cap }
    }
    fn get(&self, key: &K) -> Option<&V> { self.map.get(key) }
    fn insert(&mut self, key: K, value: V) {
        if !self.map.contains_key(&key) {
            if self.map.len() >= self.cap {
                if let Some(oldest) = self.order.pop_front() {
                    self.map.remove(&oldest);
                }
            }
            self.order.push_back(key.clone());
        }
        self.map.insert(key, value);
    }
    fn clear(&mut self) {
        self.map.clear();
        self.order.clear();
    }
}

#[cfg(target_arch = "wasm32")]
thread_local! {
    static CURRENT_DETAIL: std::cell::Cell<DetailLevel> = std::cell::Cell::new(DetailLevel::High);
    /// Cache of generated MapData by (z14 tile x, y, detail level).
    /// Uses Rc to avoid cloning multi-MB vertex arrays on cache hit.
    /// LRU eviction at cap to avoid the frame-stutter sawtooth a full-clear would cause.
    static MAPDATA_CACHE: std::cell::RefCell<LruCache<(u32, u32, u8), Rc<MapData>>> =
        std::cell::RefCell::new(LruCache::new(64));
    /// Tracks which z14 tiles already have geometry rendered.
    static RENDERED_Z14: std::cell::RefCell<std::collections::HashSet<(u32, u32)>> =
        std::cell::RefCell::new(std::collections::HashSet::new());
}

/// Determine detail level from camera zoom.
/// Currently always High — the detail-level gating in mvt_convert.rs only
/// touches POIs and road outlines, so switching tiers triggers a full
/// MAPDATA_CACHE wipe + tile re-fetch for essentially zero geometry savings.
/// Re-introduce zoom-based LOD once more mvt_convert paths gate walls,
/// shadows, and trees by detail level.
pub fn detail_for_zoom(_zoom: f32) -> DetailLevel {
    DetailLevel::High
}

/// Set the current detail level for subsequent tile fetches.
#[cfg(target_arch = "wasm32")]
pub fn set_detail_level(detail: DetailLevel) {
    CURRENT_DETAIL.with(|d| d.set(detail));
}

/// Clear the z14 dedup set so new tiles can be rendered after eviction/pan.
#[cfg(target_arch = "wasm32")]
pub fn clear_rendered_z14() {
    RENDERED_Z14.with(|s| s.borrow_mut().clear());
}

/// Tile size in degrees (~1.1km per tile).
pub const TILE_SIZE: f64 = 0.01;

/// Maximum number of tiles kept in memory.
const MAX_TILES: usize = 128;

/// Maximum concurrent tile fetches.
const MAX_IN_FLIGHT: usize = 8;

/// Minimum seconds between tile fetch requests.
const FETCH_INTERVAL: f64 = 0.01;

/// Cooldown seconds after a 429 rate limit response.
const RATE_LIMIT_COOLDOWN_SECS: f64 = 2.0;

/// Maximum number of retries before giving up on a tile.
const MAX_TILE_RETRIES: u32 = 3;

/// Base backoff seconds for failed tiles (doubles each retry).
const FAIL_BACKOFF_BASE_SECS: f64 = 2.0;

#[derive(Clone, Copy, Hash, Eq, PartialEq, Debug)]
pub struct TileCoord {
    pub col: i32,
    pub row: i32,
}

impl TileCoord {
    pub fn from_latlon(lat: f64, lon: f64) -> Self {
        Self {
            col: (lon / TILE_SIZE).floor() as i32,
            row: (lat / TILE_SIZE).floor() as i32,
        }
    }

    /// Bounding box for this tile: (south, west, north, east).
    pub fn bbox(&self) -> (f64, f64, f64, f64) {
        let south = self.row as f64 * TILE_SIZE;
        let west = self.col as f64 * TILE_SIZE;
        (south, west, south + TILE_SIZE, west + TILE_SIZE)
    }
}

pub enum TileState {
    Loading,
    Loaded(LoadedTile),
    /// Old detail level — still rendered but will be re-fetched and replaced.
    Stale(LoadedTile),
    /// Failed — retry_after time for backoff.
    Failed(web_time::Instant),
}

pub struct LoadedTile {
    pub vertex_buffer: wgpu::Buffer,
    /// Parallel vertex buffer holding `birth_time: f32` per vertex — the tile's
    /// upload time (seconds since app start). Shader compares to camera.time
    /// to fade the tile in smoothly on load. Matches the vertex count of
    /// `vertex_buffer`.
    pub birth_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub num_indices: u32,
    pub shadow_vertex_buffer: Option<wgpu::Buffer>,
    pub shadow_birth_buffer: Option<wgpu::Buffer>,
    pub shadow_index_buffer: Option<wgpu::Buffer>,
    pub num_shadow_indices: u32,
    pub labels: Vec<Label>,
    pub cars: Vec<crate::mapdata::Car>,
    pub last_visible_frame: u64,
    /// z14 MVT tile coordinates — used for dedup in rendering
    pub z14_tile: (u32, u32),
}

impl LoadedTile {
    pub fn from_map_data(gpu: &GpuState, data: &MapData, z14_tile: (u32, u32), birth_time: f32) -> Self {
        let vertex_buffer = crate::gpu::safe_buffer(&gpu.device, &gpu.queue, "Tile VB",
            bytemuck::cast_slice(&data.vertices), wgpu::BufferUsages::VERTEX);
        let index_buffer = crate::gpu::safe_buffer(&gpu.device, &gpu.queue, "Tile IB",
            bytemuck::cast_slice(&data.indices), wgpu::BufferUsages::INDEX);

        // Parallel birth-time buffer: one f32 per vertex, all the same value.
        // Keeps the main vertex layout binary-compatible with polymap-worker
        // while letting the shader fade the tile in on load.
        let birth_data: Vec<f32> = vec![birth_time; data.vertices.len().max(1)];
        let birth_buffer = crate::gpu::safe_buffer(&gpu.device, &gpu.queue, "Tile Birth VB",
            bytemuck::cast_slice(&birth_data), wgpu::BufferUsages::VERTEX);

        let (shadow_vb, shadow_bb, shadow_ib, num_shadow) = if !data.shadow_vertices.is_empty() {
            let svb = crate::gpu::safe_buffer(&gpu.device, &gpu.queue, "Tile SVB",
                bytemuck::cast_slice(&data.shadow_vertices), wgpu::BufferUsages::VERTEX);
            let shadow_birth: Vec<f32> = vec![birth_time; data.shadow_vertices.len()];
            let sbb = crate::gpu::safe_buffer(&gpu.device, &gpu.queue, "Tile Shadow Birth VB",
                bytemuck::cast_slice(&shadow_birth), wgpu::BufferUsages::VERTEX);
            let sib = crate::gpu::safe_buffer(&gpu.device, &gpu.queue, "Tile SIB",
                bytemuck::cast_slice(&data.shadow_indices), wgpu::BufferUsages::INDEX);
            (Some(svb), Some(sbb), Some(sib), data.shadow_indices.len() as u32)
        } else {
            (None, None, None, 0)
        };

        Self {
            vertex_buffer,
            birth_buffer,
            index_buffer,
            num_indices: data.indices.len() as u32,
            shadow_vertex_buffer: shadow_vb,
            shadow_birth_buffer: shadow_bb,
            shadow_index_buffer: shadow_ib,
            num_shadow_indices: num_shadow,
            labels: data.labels.clone(),
            cars: data.cars.clone(),
            last_visible_frame: 0,
            z14_tile,
        }
    }
}

pub struct TileManager {
    pub tiles: HashMap<TileCoord, TileState>,
    pub center_lat: f64,
    pub center_lon: f64,
    pub frame_counter: u64,
    pub tiles_changed: bool,
    in_flight: usize,
    last_request_time: web_time::Instant,
    rate_limited_until: Option<web_time::Instant>,
    current_detail: DetailLevel,
    /// Tracks how many times each tile has failed (for backoff).
    fail_counts: HashMap<TileCoord, u32>,

    #[cfg(not(target_arch = "wasm32"))]
    receiver: std::sync::mpsc::Receiver<(TileCoord, Result<(MapData, bool), String>)>,
    #[cfg(not(target_arch = "wasm32"))]
    sender: std::sync::mpsc::Sender<(TileCoord, Result<(MapData, bool), String>)>,

    #[cfg(target_arch = "wasm32")]
    completed: std::rc::Rc<std::cell::RefCell<Vec<(TileCoord, Result<(MapData, bool), String>)>>>,
}

impl TileManager {
    pub fn new(center_lat: f64, center_lon: f64) -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        let (sender, receiver) = std::sync::mpsc::channel();

        Self {
            tiles: HashMap::new(),
            center_lat,
            center_lon,
            frame_counter: 0,
            tiles_changed: false,
            last_request_time: web_time::Instant::now(),
            rate_limited_until: None,
            in_flight: 0,
            current_detail: DetailLevel::High,
            fail_counts: HashMap::new(),

            #[cfg(not(target_arch = "wasm32"))]
            receiver,
            #[cfg(not(target_arch = "wasm32"))]
            sender,

            #[cfg(target_arch = "wasm32")]
            completed: std::rc::Rc::new(std::cell::RefCell::new(Vec::new())),
        }
    }

    /// Update the detail level. If it changed, clear caches and mark loaded
    /// tiles as stale so they get re-fetched — but keep rendering old tiles
    /// until replacements arrive (no flash of empty).
    pub fn update_detail(&mut self, detail: DetailLevel) {
        if detail != self.current_detail {
            self.current_detail = detail;
            #[cfg(target_arch = "wasm32")]
            {
                set_detail_level(detail);
                MAPDATA_CACHE.with(|c| c.borrow_mut().clear());
            }
            // Convert Loaded tiles to Stale — still rendered but eligible for re-fetch
            let stale_coords: Vec<_> = self.tiles.iter()
                .filter(|(_, s)| matches!(s, TileState::Loaded(_)))
                .map(|(c, _)| *c)
                .collect();
            for coord in stale_coords {
                if let Some(TileState::Loaded(tile)) = self.tiles.remove(&coord) {
                    self.tiles.insert(coord, TileState::Stale(tile));
                }
            }
            self.tiles_changed = true;
        }
    }

    /// Determine which tiles are visible from the current camera state.
    pub fn visible_tiles(&self, camera: &crate::camera::Camera) -> Vec<TileCoord> {
        let hw = camera.viewport.x * 0.5;
        let hh = camera.viewport.y * 0.5;

        // Unproject screen corners + center to world space
        let points = [
            camera.unproject_to_world(0.0, 0.0),
            camera.unproject_to_world(camera.viewport.x, 0.0),
            camera.unproject_to_world(0.0, camera.viewport.y),
            camera.unproject_to_world(camera.viewport.x, camera.viewport.y),
            camera.unproject_to_world(hw, hh), // screen center
            // Also include camera's world position directly
            camera.position,
        ];

        // Convert world coords to lat/lon
        let mut min_lat = f64::MAX;
        let mut max_lat = f64::MIN;
        let mut min_lon = f64::MAX;
        let mut max_lon = f64::MIN;

        for pt in &points {
            let (lat, lon) = mapdata::unproject_pub(pt.x, pt.y, self.center_lat, self.center_lon);
            min_lat = min_lat.min(lat);
            max_lat = max_lat.max(lat);
            min_lon = min_lon.min(lon);
            max_lon = max_lon.max(lon);
        }

        // Pad by 2 tiles to pre-load surrounding area for smooth panning
        let pad = TILE_SIZE * 2.0;
        min_lat -= pad;
        max_lat += pad;
        min_lon -= pad;
        max_lon += pad;

        // Enumerate tile coordinates
        let min_row = (min_lat / TILE_SIZE).floor() as i32;
        let max_row = (max_lat / TILE_SIZE).ceil() as i32;
        let min_col = (min_lon / TILE_SIZE).floor() as i32;
        let max_col = (max_lon / TILE_SIZE).ceil() as i32;

        // Get camera center in tile coordinates for prioritization
        let (cam_lat, cam_lon) = mapdata::unproject_pub(
            camera.position.x, camera.position.y,
            self.center_lat, self.center_lon,
        );
        let cam_col = (cam_lon / TILE_SIZE).floor() as i32;
        let cam_row = (cam_lat / TILE_SIZE).floor() as i32;

        let mut tiles = Vec::new();
        for row in min_row..=max_row {
            for col in min_col..=max_col {
                tiles.push(TileCoord { col, row });
            }
        }

        // Sort by distance from camera center — load nearest tiles first
        tiles.sort_by_key(|t| {
            let dc = (t.col - cam_col).abs();
            let dr = (t.row - cam_row).abs();
            dc * dc + dr * dr
        });

        tiles
    }

    /// Request tiles from the sorted visible list. Processes as many as allowed.
    pub fn request_visible_tiles(&mut self, visible: &[TileCoord]) {
        let now = web_time::Instant::now();

        // Respect rate limit cooldown
        if let Some(until) = self.rate_limited_until {
            if now < until {
                return;
            }
            self.rate_limited_until = None;
        }

        let max_in_flight = MAX_IN_FLIGHT;
        let fetch_interval = FETCH_INTERVAL;

        for coord in visible {
            if self.in_flight >= max_in_flight {
                break;
            }
            // Skip already loaded or currently loading tiles
            match self.tiles.get(coord) {
                Some(TileState::Loaded(_)) | Some(TileState::Loading) => continue,
                Some(TileState::Stale(_)) => {
                    // Keep stale tile for rendering, but re-fetch at new detail
                    // Don't remove it — poll_completed will replace it
                }
                Some(TileState::Failed(retry_after)) => {
                    let retries = self.fail_counts.get(coord).copied().unwrap_or(0);
                    if retries >= MAX_TILE_RETRIES {
                        continue; // Give up on this tile
                    }
                    if now < *retry_after {
                        continue; // Not yet time to retry
                    }
                    self.tiles.remove(coord);
                }
                None => {}
            }

            // Throttle: check time since last request
            let elapsed = now.duration_since(self.last_request_time).as_secs_f64();
            if elapsed < fetch_interval && self.in_flight > 0 {
                // Allow the very first request without throttle, throttle subsequent
                break;
            }

            self.dispatch_tile(*coord);
            self.last_request_time = now;
        }
    }

    /// Record a tile failure with exponential backoff (public for worker error path).
    pub fn record_failure_pub(&mut self, coord: TileCoord) {
        self.record_failure(coord);
    }

    /// Record a tile failure with exponential backoff.
    fn record_failure(&mut self, coord: TileCoord) {
        let retries = self.fail_counts.entry(coord).or_insert(0);
        *retries += 1;
        let backoff = FAIL_BACKOFF_BASE_SECS * 2.0_f64.powi((*retries - 1) as i32);
        self.tiles.insert(coord, TileState::Failed(
            web_time::Instant::now() + std::time::Duration::from_secs_f64(backoff),
        ));
    }

    /// Dispatch a single tile fetch.
    fn dispatch_tile(&mut self, coord: TileCoord) {

        self.tiles.insert(coord, TileState::Loading);
        self.in_flight += 1;

        let (south, west, north, east) = coord.bbox();
        let center_lat = self.center_lat;
        let center_lon = self.center_lon;

        #[cfg(not(target_arch = "wasm32"))]
        {
            let sender = self.sender.clone();
            std::thread::spawn(move || {
                let result = fetch_tile_native(south, west, north, east, center_lat, center_lon);
                let _ = sender.send((coord, result));
            });
        }

        #[cfg(target_arch = "wasm32")]
        {
            // If a JS tile callback is set (Web Worker pool), use it instead of
            // fetching on the main thread.
            let has_callback = crate::api::TILE_CALLBACK.with(|cb| cb.borrow().is_some());
            if has_callback {
                use wasm_bindgen::JsValue;
                let detail = self.current_detail as u8;
                crate::api::TILE_CALLBACK.with(|cb| {
                    if let Some(ref func) = *cb.borrow() {
                        let _ = func.call9(
                            &JsValue::NULL,
                            &JsValue::from(coord.col),
                            &JsValue::from(coord.row),
                            &JsValue::from(south),
                            &JsValue::from(west),
                            &JsValue::from(north),
                            &JsValue::from(east),
                            &JsValue::from(center_lat),
                            &JsValue::from(center_lon),
                            &JsValue::from(detail),
                        );
                    }
                });
            } else {
                // Fallback: fetch on main thread
                let completed = self.completed.clone();
                wasm_bindgen_futures::spawn_local(async move {
                    let result = fetch_tile_wasm(south, west, north, east, center_lat, center_lon).await;
                    completed.borrow_mut().push((coord, result));
                });
            }
        }
    }

    /// Poll for completed tile fetches and upload to GPU.
    /// `now_secs` is seconds since app start; stamped into each tile's
    /// vertex data so the map shader can fade the tile in on load.
    pub fn poll_completed(&mut self, gpu: &GpuState, now_secs: f32) {
        self.tiles_changed = false;

        #[cfg(not(target_arch = "wasm32"))]
        {
            while let Ok((coord, result)) = self.receiver.try_recv() {
                self.in_flight = self.in_flight.saturating_sub(1);
                self.tiles_changed = true;
                match result {
                    Ok((data, _)) => {
                        if !data.vertices.is_empty() {
                            let (s, w, n, e) = coord.bbox();
                            let mid_lat = (s + n) / 2.0;
                            let mid_lon = (w + e) / 2.0;
                            let (tx, ty) = latlon_to_zxy(mid_lat, mid_lon, 14);
                            let loaded = LoadedTile::from_map_data(gpu, &data, (tx, ty), now_secs);
                            self.fail_counts.remove(&coord);
                            self.tiles.insert(coord, TileState::Loaded(loaded));
                        } else {
                            self.record_failure(coord);
                        }
                    }
                    Err(e) => {
                        if e == "RATE_LIMITED" {
                            log::warn!("Rate limited! Cooling down for {}s", RATE_LIMIT_COOLDOWN_SECS);
                            self.rate_limited_until = Some(web_time::Instant::now()
                                + std::time::Duration::from_secs_f64(RATE_LIMIT_COOLDOWN_SECS));
                            // Re-queue this tile for retry
                            self.tiles.remove(&coord);
                        } else {
                            log::warn!("Tile {:?} failed: {}", coord, e);
                            self.record_failure(coord);
                        }
                    }
                }
            }
        }

        #[cfg(target_arch = "wasm32")]
        {
            let results: Vec<_> = self.completed.borrow_mut().drain(..).collect();
            for (coord, result) in results {
                self.in_flight = self.in_flight.saturating_sub(1);
                self.tiles_changed = true;
                match result {
                    Ok((data, _)) => {
                        if !data.vertices.is_empty() {
                            let (s, w, n, e) = coord.bbox();
                            let mid_lat = (s + n) / 2.0;
                            let mid_lon = (w + e) / 2.0;
                            let (tx, ty) = latlon_to_zxy(mid_lat, mid_lon, 14);
                            let loaded = LoadedTile::from_map_data(gpu, &data, (tx, ty), now_secs);
                            self.fail_counts.remove(&coord);
                            self.tiles.insert(coord, TileState::Loaded(loaded));
                        } else {
                            self.record_failure(coord);
                        }
                    }
                    Err(e) => {
                        if e == "RATE_LIMITED" {
                            log::warn!("Rate limited! Cooling down for {}s", RATE_LIMIT_COOLDOWN_SECS);
                            self.rate_limited_until = Some(web_time::Instant::now()
                                + std::time::Duration::from_secs_f64(RATE_LIMIT_COOLDOWN_SECS));
                            self.tiles.remove(&coord);
                        } else {
                            log::warn!("Tile {:?} failed: {}", coord, e);
                            self.record_failure(coord);
                        }
                    }
                }
            }
        }
    }

    /// Mark visible tiles and evict far-away ones if over the limit.
    pub fn update_visibility(&mut self, visible: &[TileCoord]) {
        for coord in visible {
            match self.tiles.get_mut(coord) {
                Some(TileState::Loaded(tile)) | Some(TileState::Stale(tile)) => {
                    tile.last_visible_frame = self.frame_counter;
                }
                _ => {}
            }
        }

        // Evict if over limit — only count loaded/stale tiles, never evict visible ones
        let current_frame = self.frame_counter;
        let loaded_count = self.tiles.values().filter(|s| matches!(s, TileState::Loaded(_) | TileState::Stale(_))).count();
        let mut evicted = 0;
        while loaded_count - evicted > MAX_TILES {
            let oldest = self.tiles.iter()
                .filter_map(|(coord, state)| {
                    match state {
                        TileState::Loaded(tile) | TileState::Stale(tile) => {
                            // Don't evict tiles seen this frame or last frame
                            if tile.last_visible_frame >= current_frame.saturating_sub(1) {
                                None
                            } else {
                                Some((*coord, tile.last_visible_frame))
                            }
                        }
                        TileState::Failed(_) => Some((*coord, 0)),
                        _ => None,
                    }
                })
                .min_by_key(|(_, frame)| *frame);

            if let Some((coord, _)) = oldest {
                self.tiles.remove(&coord);
                evicted += 1;
            } else {
                break; // all remaining tiles are currently visible — can't evict
            }
        }
    }

    /// Insert a pre-built LoadedTile (e.g. from Web Worker upload).
    pub fn insert_loaded_tile(&mut self, coord: TileCoord, tile: LoadedTile) {
        self.tiles.insert(coord, TileState::Loaded(tile));
        self.tiles_changed = true;
    }

    /// Decrement the in-flight counter (called when a worker tile completes).
    pub fn decrement_in_flight(&mut self) {
        self.in_flight = self.in_flight.saturating_sub(1);
    }

    /// True if tiles are being fetched or new tiles just arrived.
    pub fn is_busy(&self) -> bool {
        self.in_flight > 0 || self.tiles_changed
    }

    /// Get all loaded tiles for rendering.
    /// Loaded tiles whose projected world AABB intersects the given camera AABB.
    /// Culls cache entries that are loaded but off-screen, so the renderer
    /// never submits geometry for them.
    pub fn loaded_tiles_in_aabb(
        &self,
        cam_aabb: (f32, f32, f32, f32),
    ) -> impl Iterator<Item = &LoadedTile> {
        let (cmin_x, cmin_y, cmax_x, cmax_y) = cam_aabb;
        let center_lat = self.center_lat;
        let center_lon = self.center_lon;
        self.tiles.iter().filter_map(move |(coord, state)| {
            let tile = match state {
                TileState::Loaded(t) | TileState::Stale(t) => t,
                _ => return None,
            };
            let (s, w, n, e) = coord.bbox();
            // Project the four corners of the tile's lat/lon bbox to world.
            let sw = crate::mapdata::project_pub(s, w, center_lat, center_lon);
            let ne = crate::mapdata::project_pub(n, e, center_lat, center_lon);
            let tile_min_x = sw[0].min(ne[0]);
            let tile_max_x = sw[0].max(ne[0]);
            let tile_min_y = sw[1].min(ne[1]);
            let tile_max_y = sw[1].max(ne[1]);
            // AABB intersection test
            if tile_max_x < cmin_x || tile_min_x > cmax_x { return None; }
            if tile_max_y < cmin_y || tile_min_y > cmax_y { return None; }
            Some(tile)
        })
    }

    pub fn loaded_tiles(&self) -> impl Iterator<Item = &LoadedTile> {
        self.tiles.values().filter_map(|state| {
            match state {
                TileState::Loaded(tile) | TileState::Stale(tile) => Some(tile),
                _ => None,
            }
        })
    }

    /// Collect all labels from loaded tiles.
    pub fn all_labels(&self) -> Vec<&Label> {
        self.tiles.values().filter_map(|state| {
            match state {
                TileState::Loaded(tile) | TileState::Stale(tile) => Some(tile.labels.iter()),
                _ => None,
            }
        }).flatten().collect()
    }
}

// ── Tile fetching ────────────────────────────────────────────────────

#[cfg(not(target_arch = "wasm32"))]
fn fetch_tile_native(
    south: f64, west: f64, north: f64, east: f64,
    center_lat: f64, center_lon: f64,
) -> Result<(MapData, bool), String> {
    let total_start = std::time::Instant::now();

    let api_url = format!(
        "{}/map/osm?south={}&west={}&north={}&east={}",
        mapdata::API_BASE, south, west, north, east
    );

    let fetch_start = std::time::Instant::now();
    match ureq::get(&api_url)
        .timeout(std::time::Duration::from_secs(10))
        .call()
    {
        Ok(response) => {
            let fetch_ms = fetch_start.elapsed().as_millis();
            let body = response.into_string().map_err(|e| format!("Read failed: {}", e))?;
            let data = mapdata::parse_osm_json_centered(&body, south, west, north, east, center_lat, center_lon)?;
            Ok((data, true))
        }
        Err(ureq::Error::Status(429, _)) => {
            Err("RATE_LIMITED".to_string())
        }
        Err(e) => Err(format!("API fetch failed: {}", e)),
    }
}

#[cfg(target_arch = "wasm32")]
fn latlon_to_zxy(lat: f64, lon: f64, z: u8) -> (u32, u32) {
    let n = (1u32 << z) as f64;
    let x = ((lon + 180.0) / 360.0 * n).floor().max(0.0) as u32;
    let lat_rad = lat.to_radians();
    let y = ((1.0 - lat_rad.tan().asinh() / std::f64::consts::PI) / 2.0 * n)
        .floor()
        .max(0.0) as u32;
    (x, y)
}

// ── z14 MVT raw bytes cache ──────────────────────────────────────
// Multiple 0.01° PolyMap tiles map to the same z14 MVT tile (~0.022°).
// Cache raw decompressed bytes to avoid redundant HTTP fetches.
// LRU eviction so overflow drops one entry rather than stalling the
// frame with a catastrophic full-cache deallocation.
#[cfg(target_arch = "wasm32")]
thread_local! {
    static MVT_BYTES_CACHE: std::cell::RefCell<LruCache<(u32, u32), Rc<Vec<u8>>>> =
        std::cell::RefCell::new(LruCache::new(32));
    static PARCEL_BYTES_CACHE: std::cell::RefCell<LruCache<(u32, u32), Rc<Vec<u8>>>> =
        std::cell::RefCell::new(LruCache::new(64));
}

#[cfg(target_arch = "wasm32")]
async fn get_or_fetch_mvt_bytes(url: &str, z: u8, tx: u32, ty: u32) -> Result<Rc<Vec<u8>>, String> {
    let cached = MVT_BYTES_CACHE.with(|c| c.borrow().get(&(tx, ty)).cloned());
    if let Some(bytes) = cached {
        return Ok(bytes); // Rc clone — 16 bytes
    }

    let tile_data = crate::pmtiles::get_tile(url, z, tx, ty)
        .await
        .ok_or("PMTiles fetch failed")?;

    let rc_data = Rc::new(tile_data);
    MVT_BYTES_CACHE.with(|c| {
        c.borrow_mut().insert((tx, ty), Rc::clone(&rc_data));
    });

    Ok(rc_data)
}

#[cfg(target_arch = "wasm32")]
async fn get_or_fetch_parcel_bytes(url: &str, z: u8, tx: u32, ty: u32) -> Result<Rc<Vec<u8>>, String> {
    // Check cache first
    let cached = PARCEL_BYTES_CACHE.with(|c| c.borrow().get(&(tx, ty)).cloned());
    if let Some(bytes) = cached {
        return Ok(bytes); // Rc clone
    }

    let tile_data = crate::pmtiles::get_tile(url, z, tx, ty)
        .await
        .ok_or("Parcel PMTiles fetch failed")?;

    let rc_data = Rc::new(tile_data);
    PARCEL_BYTES_CACHE.with(|c| {
        c.borrow_mut().insert((tx, ty), Rc::clone(&rc_data));
    });

    Ok(rc_data)
}

#[cfg(target_arch = "wasm32")]
async fn fetch_tile_wasm(
    south: f64, west: f64, north: f64, east: f64,
    center_lat: f64, center_lon: f64,
) -> Result<(MapData, bool), String> {
    // Check if PMTiles URL is configured
    let pmtiles_url = crate::api::INIT_CONFIG.with(|c| {
        c.borrow().as_ref().and_then(|cfg| cfg.pmtiles_url.clone())
    });

    if let Some(url) = pmtiles_url {
        // ── PMTiles path: read MVT tile directly from GCS ──
        let mid_lat = (south + north) / 2.0;
        let mid_lon = (west + east) / 2.0;
        let z: u8 = 14;
        let (tx, ty) = latlon_to_zxy(mid_lat, mid_lon, z);

        let detail = CURRENT_DETAIL.with(|d| d.get());
        let detail_key = detail as u8;

        // Check MapData cache — avoids re-decoding + re-triangulating
        // for PolyMap tiles that share the same z14 tile.
        let cached = MAPDATA_CACHE.with(|c| {
            c.borrow().get(&(tx, ty, detail_key)).cloned()
        });

        let data: Rc<MapData> = if let Some(cached_data) = cached {
            cached_data // Rc clone — 16 bytes, not multi-MB
        } else {
            // Build geometry from scratch
            let mvt_bytes = get_or_fetch_mvt_bytes(&url, z, tx, ty).await?;
            let mvt_tile = crate::mvt::decode_tile(&mvt_bytes);
            let mut data = crate::mvt_convert::mvt_to_mapdata(
                &mvt_tile, z, tx, ty, center_lat, center_lon, detail,
            );

            // Parcel overlay disabled

            let rc_data = Rc::new(data);
            // Cache for other PolyMap tiles sharing this z14 tile.
            // LRU eviction in insert() drops only the oldest entry on overflow.
            MAPDATA_CACHE.with(|c| {
                c.borrow_mut().insert((tx, ty, detail_key), Rc::clone(&rc_data));
            });

            rc_data
        };

        // Deref Rc for GPU upload — only clones on first build, cache hits skip MVT decode
        Ok(((*data).clone(), true))
    } else {
        // ── Fallback: API path (original behavior) ──
        let api_base = crate::api::INIT_CONFIG.with(|c| {
            c.borrow().as_ref().and_then(|cfg| cfg.api_base.clone())
        });
        let base = api_base.as_deref().unwrap_or(mapdata::API_BASE);
        let api_url = format!(
            "{}/map/osm?south={}&west={}&north={}&east={}",
            base, south, west, north, east
        );

        let body = crate::wasm_fetch_text(&api_url, "GET", None, None)
            .await
            .ok_or("API fetch failed")?;

        let data = mapdata::parse_osm_json_centered(&body, south, west, north, east, center_lat, center_lon)?;
        Ok((data, true))
    }
}
