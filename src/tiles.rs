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

/// Maximum number of tiles whose GPU buffers (vertex/index/shadow) are
/// retained simultaneously. Each loaded tile holds several wgpu::Buffer
/// allocations totaling ~3-5 MB depending on density; 32 tiles ≈ 100-160 MB
/// of GPU memory at the high end. Beyond that, Safari starts hitting its
/// per-tab memory ceiling.
///
/// Eviction is LRU on `last_visible_frame`. Re-entering a previously-evicted
/// tile is cheap because the decoded `MapData` is still cached on the CPU
/// side (`MAPDATA_CACHE`, 64 entries) — we just rebuild the GPU buffers
/// without re-fetching or re-decoding.
const MAX_TILES: usize = 32;

/// Maximum concurrent tile fetches. Lower = less peak memory during pan
/// (fewer simultaneous decodes), slightly slower fill-in.
const MAX_IN_FLIGHT: usize = 6;

/// Minimum seconds between tile fetch requests.
const FETCH_INTERVAL: f64 = 0.01;

/// Cooldown seconds after a 429 rate limit response.
const RATE_LIMIT_COOLDOWN_SECS: f64 = 2.0;

/// Maximum number of retries before giving up on a tile.
const MAX_TILE_RETRIES: u32 = 3;

/// Base backoff seconds for failed tiles (doubles each retry).
const FAIL_BACKOFF_BASE_SECS: f64 = 2.0;

/// Camera zoom below which z14 detail isn't worth keeping around: the base
/// underlay carries the visual load and z14 GPU memory + fetch budget is
/// better spent on tiles the user can actually see. Tuned empirically — the
/// default initial zoom in `web/index.html` is 0.8.
pub const LOW_ZOOM_THRESHOLD: f32 = -1.75;

/// Seconds the fade-out animation runs before the tile is eligible for
/// eviction. Matches the shader's `smoothstep(0.0, 0.6, dying_age)` upper
/// bound in `tile_fade`.
pub const FADE_OUT_DURATION: f32 = 0.6;

/// Extra grace period after the fade completes before actually freeing GPU
/// buffers — gives the user a brief window to zoom back in and cancel the
/// fade without paying the rebuild cost.
pub const EVICT_DELAY_AFTER_FADE: f32 = 0.1;

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
    /// Parallel vertex buffer holding `birth_time: f32` per vertex.
    /// Encoding: `0` = legacy single-shot (always visible); `> 0` = born at
    /// that time (fade-in animation); `< 0` = dying, with `-birth_time` being
    /// the death time (fade-out animation). Same value across all vertices.
    pub birth_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub num_indices: u32,
    pub shadow_vertex_buffer: Option<wgpu::Buffer>,
    pub shadow_birth_buffer: Option<wgpu::Buffer>,
    pub shadow_index_buffer: Option<wgpu::Buffer>,
    pub num_shadow_indices: u32,
    pub labels: Vec<Label>,
    pub cars: Vec<crate::mapdata::Car>,
    pub noise_sources: Vec<crate::mapdata::NoiseSource>,
    pub last_visible_frame: u64,
    /// z14 MVT tile coordinates — used for dedup in rendering
    pub z14_tile: (u32, u32),
    /// Vertex count for the main buffer — needed when rewriting birth_buffer
    /// for fade-out.
    pub vertex_count: u32,
    pub shadow_vertex_count: u32,
    /// `Some(t)` if the tile is fading out; `t` is the death time. Eviction
    /// happens after `t + FADE_OUT_DURATION + EVICT_DELAY_AFTER_FADE`.
    pub death_time: Option<f32>,
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
        // COPY_DST so start_death / cancel_death can rewrite it in-place.
        let birth_data: Vec<f32> = vec![birth_time; data.vertices.len().max(1)];
        let birth_buffer = crate::gpu::safe_buffer(&gpu.device, &gpu.queue, "Tile Birth VB",
            bytemuck::cast_slice(&birth_data), wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST);

        let (shadow_vb, shadow_bb, shadow_ib, num_shadow) = if !data.shadow_vertices.is_empty() {
            let svb = crate::gpu::safe_buffer(&gpu.device, &gpu.queue, "Tile SVB",
                bytemuck::cast_slice(&data.shadow_vertices), wgpu::BufferUsages::VERTEX);
            let shadow_birth: Vec<f32> = vec![birth_time; data.shadow_vertices.len()];
            let sbb = crate::gpu::safe_buffer(&gpu.device, &gpu.queue, "Tile Shadow Birth VB",
                bytemuck::cast_slice(&shadow_birth), wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST);
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
            noise_sources: data.noise_sources.clone(),
            last_visible_frame: 0,
            z14_tile,
            vertex_count: data.vertices.len() as u32,
            shadow_vertex_count: data.shadow_vertices.len() as u32,
            death_time: None,
        }
    }

    /// Begin the fade-out animation by overwriting the birth buffer with a
    /// negative death sentinel: `birth_time = -now_secs` for every vertex.
    /// The shader interprets negative birth_time as "dying" and ramps alpha
    /// from 1 → 0 over `FADE_OUT_DURATION` seconds.
    pub fn start_death(&mut self, gpu: &GpuState, now_secs: f32) {
        if self.death_time.is_some() { return; }
        self.death_time = Some(now_secs);
        // Encode death as negative birth_time. now_secs is always positive in
        // practice, so -now_secs is < 0 and disambiguated from 0 (legacy).
        let neg = -now_secs;
        if self.vertex_count > 0 {
            let buf: Vec<f32> = vec![neg; self.vertex_count as usize];
            gpu.queue.write_buffer(&self.birth_buffer, 0, bytemuck::cast_slice(&buf));
        }
        if self.shadow_vertex_count > 0 {
            if let Some(b) = &self.shadow_birth_buffer {
                let sbuf: Vec<f32> = vec![neg; self.shadow_vertex_count as usize];
                gpu.queue.write_buffer(b, 0, bytemuck::cast_slice(&sbuf));
            }
        }
    }

    /// Abort an in-progress fade-out and snap the tile back to fully visible.
    /// Used when the camera zooms back in before the fade completes — without
    /// this, the tile would finish dying despite being visible again.
    /// The snap-back skips the fade-in animation by writing a birth_time well
    /// in the past (now − 1.0 s); the shader's `smoothstep(0, 0.4, age)` is
    /// already saturated at age = 1.0.
    pub fn cancel_death(&mut self, gpu: &GpuState, now_secs: f32) {
        if self.death_time.is_none() { return; }
        self.death_time = None;
        let birth = (now_secs - 1.0).max(0.001);
        if self.vertex_count > 0 {
            let buf: Vec<f32> = vec![birth; self.vertex_count as usize];
            gpu.queue.write_buffer(&self.birth_buffer, 0, bytemuck::cast_slice(&buf));
        }
        if self.shadow_vertex_count > 0 {
            if let Some(b) = &self.shadow_birth_buffer {
                let sbuf: Vec<f32> = vec![birth; self.shadow_vertex_count as usize];
                gpu.queue.write_buffer(b, 0, bytemuck::cast_slice(&sbuf));
            }
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
    /// True while the camera zoom is below `LOW_ZOOM_THRESHOLD`. While true,
    /// new z14 fetches are paused and any Loaded tiles fade out and evict.
    pub low_zoom_active: bool,

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
            low_zoom_active: false,

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
        // While zoomed out below threshold, the base-layer underlay carries
        // the visuals and z14 detail isn't worth fetching. Returning early
        // also stops new tiles from arriving mid-fade-out and confusing the
        // user.
        if self.low_zoom_active { return; }

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
        // Cap the exponent so repeated failures after eviction-then-revisit
        // don't produce `inf` and panic `Duration::from_secs_f64`. Capping at
        // 20 gives a maximum backoff of ~24 days, plenty.
        let exp = ((*retries as i64 - 1).clamp(0, 20)) as i32;
        let backoff = FAIL_BACKOFF_BASE_SECS * 2.0_f64.powi(exp);
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
                // Don't leak the retry counter — if the tile comes back into
                // view later we want a fresh budget, not to resume from an old
                // accumulated count (which could eventually overflow the
                // backoff exponent).
                self.fail_counts.remove(&coord);
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

    /// Collect all labels from loaded tiles. Dying tiles' labels are kept in
    /// the set so small (Street/Building/POI/Park) labels can fade smoothly
    /// via the shader's `small_label_alpha` while the tile geometry fades.
    /// They drop out naturally when the tile evicts.
    pub fn all_labels(&self) -> Vec<&Label> {
        self.tiles.values().filter_map(|state| {
            match state {
                TileState::Loaded(tile) | TileState::Stale(tile) => Some(tile.labels.iter()),
                _ => None,
            }
        }).flatten().collect()
    }

    /// React to zoom changes: when zoom drops below LOW_ZOOM_THRESHOLD, start
    /// fading out all Loaded z14 tiles. When zoom rises above, cancel any
    /// in-progress fades so the tiles snap back to fully visible without a
    /// rebuild.
    pub fn update_zoom_state(&mut self, gpu: &GpuState, zoom: f32, now_secs: f32) {
        let want_low = zoom < LOW_ZOOM_THRESHOLD;
        if want_low == self.low_zoom_active { return; }
        self.low_zoom_active = want_low;
        self.tiles_changed = true;

        if want_low {
            // Mark every currently-loaded tile dying; eviction follows the fade.
            for state in self.tiles.values_mut() {
                if let TileState::Loaded(tile) = state {
                    tile.start_death(gpu, now_secs);
                }
            }
        } else {
            // Cancel in-progress fades. Stale tiles aren't relevant here — the
            // detail-level path handles those.
            for state in self.tiles.values_mut() {
                if let TileState::Loaded(tile) = state {
                    if tile.death_time.is_some() {
                        tile.cancel_death(gpu, now_secs);
                    }
                }
            }
        }
    }

    /// Drop any tile whose fade-out animation has finished plus a small grace
    /// period. Frees the GPU buffers so the budget is available for tiles the
    /// user is actually looking at.
    pub fn evict_dead_tiles(&mut self, now_secs: f32) {
        let cutoff = FADE_OUT_DURATION + EVICT_DELAY_AFTER_FADE;
        let to_drop: Vec<_> = self.tiles.iter()
            .filter_map(|(coord, state)| {
                if let TileState::Loaded(t) = state {
                    if let Some(dt) = t.death_time {
                        if now_secs - dt > cutoff {
                            return Some(*coord);
                        }
                    }
                }
                None
            })
            .collect();
        if !to_drop.is_empty() {
            for coord in to_drop {
                self.tiles.remove(&coord);
                self.fail_counts.remove(&coord);
            }
            self.tiles_changed = true;
        }
    }

    /// True if any loaded tile is mid-fade-out — used by the render loop's
    /// `needs_render` check to keep drawing while the animation runs.
    pub fn has_dying_tiles(&self) -> bool {
        self.tiles.values().any(|state| match state {
            TileState::Loaded(t) => t.death_time.is_some(),
            _ => false,
        })
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

        let body = crate::app::wasm_fetch_text(&api_url, "GET", None, None)
            .await
            .ok_or("API fetch failed")?;

        let data = mapdata::parse_osm_json_centered(&body, south, west, north, east, center_lat, center_lon)?;
        Ok((data, true))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Low-resolution base-layer tiles (themed underlay)
//
// Fetches lower-zoom MVT tiles (e.g. z11) from the same PMTiles archive,
// decodes them at DetailLevel::Low (water + landuse + road fills only), and
// uploads each as a LoadedTile. The renderer draws these in a pass before the
// z14 opaque pass; depth test occludes them where high-res has rendered.
// Z values are shifted down by BASE_Z_OFFSET so the underlay sits below the
// z14 plane and never z-fights with high-res tiles.
// ─────────────────────────────────────────────────────────────────────────────

/// Maximum number of base tiles GPU-resident at once. A single z11 tile covers
/// ~8x8 of PolyMap's 0.01° tiles, so 8 base tiles cover a much larger area
/// than the typical viewport with pan padding.
#[cfg(target_arch = "wasm32")]
const MAX_BASE_TILES: usize = 8;

/// Concurrent base-tile fetches. Kept low so the base layer doesn't compete
/// with the z14 fetch budget — the underlay is best-effort, not critical.
#[cfg(target_arch = "wasm32")]
const MAX_BASE_IN_FLIGHT: usize = 2;

/// Z offset baked into every base-tile vertex. Pushes the entire base layer
/// below the z14 plane (Z_LANDUSE = -0.005 in mapdata.rs), so the depth test
/// reliably occludes base geometry where z14 has rendered. Far enough below
/// the deepest z14 feature (-0.005) to avoid z-fighting under tilt, but not
/// so far that the base layer pokes through above-ground features at any
/// viewing angle.
#[cfg(target_arch = "wasm32")]
const BASE_Z_OFFSET: f32 = -0.05;

#[derive(Clone, Copy, Hash, Eq, PartialEq, Debug)]
pub struct BaseTileCoord {
    pub z: u8,
    pub x: u32,
    pub y: u32,
}

pub struct BaseLoadedTile {
    pub inner: LoadedTile,
    pub coord: BaseTileCoord,
}

#[cfg(target_arch = "wasm32")]
thread_local! {
    /// Decoded MapData cache for base tiles. Separate from MAPDATA_CACHE so
    /// the well-tuned z14 cache isn't perturbed by the (much smaller) base
    /// tile working set.
    static BASE_MAPDATA_CACHE: std::cell::RefCell<LruCache<(u8, u32, u32), Rc<MapData>>> =
        std::cell::RefCell::new(LruCache::new(16));
}

/// Apply a uniform Z shift to every vertex (and shadow vertex) in a MapData.
/// Used to push base-layer geometry below the z14 plane.
fn shift_mapdata_z(data: &mut MapData, dz: f32) {
    for v in &mut data.vertices {
        v.position[2] += dz;
    }
    for v in &mut data.shadow_vertices {
        v.position[2] += dz;
    }
}

#[cfg(target_arch = "wasm32")]
pub struct BaseTileManager {
    tiles: HashMap<BaseTileCoord, TileState>,
    pub center_lat: f64,
    pub center_lon: f64,
    pub frame_counter: u64,
    pub tiles_changed: bool,
    in_flight: usize,
    last_request_time: web_time::Instant,
    rate_limited_until: Option<web_time::Instant>,
    fail_counts: HashMap<BaseTileCoord, u32>,
    /// Zoom level for base tiles (from PolyMapConfig::low_res_underlay).
    pub zoom: u8,
    completed: std::rc::Rc<std::cell::RefCell<Vec<(BaseTileCoord, Result<MapData, String>)>>>,
}

#[cfg(target_arch = "wasm32")]
impl BaseTileManager {
    pub fn new(center_lat: f64, center_lon: f64, zoom: u8) -> Self {
        Self {
            tiles: HashMap::new(),
            center_lat,
            center_lon,
            frame_counter: 0,
            tiles_changed: false,
            in_flight: 0,
            last_request_time: web_time::Instant::now(),
            rate_limited_until: None,
            fail_counts: HashMap::new(),
            zoom,
            completed: std::rc::Rc::new(std::cell::RefCell::new(Vec::new())),
        }
    }

    /// Tile coords visible from the current camera, in z11-tile space.
    /// Pads by 1 tile on each side for smooth pan.
    pub fn visible_tiles(&self, camera: &crate::camera::Camera) -> Vec<BaseTileCoord> {
        let hw = camera.viewport.x * 0.5;
        let hh = camera.viewport.y * 0.5;
        let points = [
            camera.unproject_to_world(0.0, 0.0),
            camera.unproject_to_world(camera.viewport.x, 0.0),
            camera.unproject_to_world(0.0, camera.viewport.y),
            camera.unproject_to_world(camera.viewport.x, camera.viewport.y),
            camera.unproject_to_world(hw, hh),
            camera.position,
        ];

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

        let z = self.zoom;
        let (x_min, y_max) = latlon_to_zxy(min_lat, min_lon, z);
        let (x_max, y_min) = latlon_to_zxy(max_lat, max_lon, z);
        // Pad by 1 z11 tile on each side. (z11 tiles are large; 1 is enough.)
        let x0 = x_min.saturating_sub(1);
        let x1 = x_max.saturating_add(1);
        let y0 = y_min.saturating_sub(1);
        let y1 = y_max.saturating_add(1);

        let mut out = Vec::new();
        for x in x0..=x1 {
            for y in y0..=y1 {
                out.push(BaseTileCoord { z, x, y });
            }
        }
        out
    }

    /// Dispatch fetches for visible tiles (respecting in-flight + cooldown).
    pub fn request_visible_tiles(&mut self, visible: &[BaseTileCoord]) {
        let now = web_time::Instant::now();
        if let Some(until) = self.rate_limited_until {
            if now < until { return; }
            self.rate_limited_until = None;
        }

        for coord in visible {
            if self.in_flight >= MAX_BASE_IN_FLIGHT { break; }
            match self.tiles.get(coord) {
                Some(TileState::Loaded(_)) | Some(TileState::Loading) => continue,
                Some(TileState::Stale(_)) => {} // shouldn't happen for base; treat as loaded
                Some(TileState::Failed(retry_after)) => {
                    let retries = self.fail_counts.get(coord).copied().unwrap_or(0);
                    if retries >= MAX_TILE_RETRIES { continue; }
                    if now < *retry_after { continue; }
                    self.tiles.remove(coord);
                }
                None => {}
            }
            let elapsed = now.duration_since(self.last_request_time).as_secs_f64();
            if elapsed < FETCH_INTERVAL && self.in_flight > 0 { break; }

            self.dispatch_tile(*coord);
            self.last_request_time = now;
        }
    }

    fn dispatch_tile(&mut self, coord: BaseTileCoord) {
        self.tiles.insert(coord, TileState::Loading);
        self.in_flight += 1;

        let center_lat = self.center_lat;
        let center_lon = self.center_lon;
        let completed = self.completed.clone();
        wasm_bindgen_futures::spawn_local(async move {
            let result = fetch_base_tile_wasm(coord.z, coord.x, coord.y, center_lat, center_lon).await;
            completed.borrow_mut().push((coord, result));
        });
    }

    fn record_failure(&mut self, coord: BaseTileCoord) {
        let retries = self.fail_counts.entry(coord).or_insert(0);
        *retries += 1;
        let exp = ((*retries as i64 - 1).clamp(0, 20)) as i32;
        let backoff = FAIL_BACKOFF_BASE_SECS * 2.0_f64.powi(exp);
        self.tiles.insert(coord, TileState::Failed(
            web_time::Instant::now() + std::time::Duration::from_secs_f64(backoff),
        ));
    }

    pub fn poll_completed(&mut self, gpu: &GpuState, now_secs: f32) {
        self.tiles_changed = false;
        let results: Vec<_> = self.completed.borrow_mut().drain(..).collect();
        for (coord, result) in results {
            self.in_flight = self.in_flight.saturating_sub(1);
            self.tiles_changed = true;
            match result {
                Ok(mut data) => {
                    if !data.vertices.is_empty() {
                        shift_mapdata_z(&mut data, BASE_Z_OFFSET);
                        // z14_tile field unused for base tiles — pass (0, 0).
                        let inner = LoadedTile::from_map_data(gpu, &data, (0, 0), now_secs);
                        self.fail_counts.remove(&coord);
                        self.tiles.insert(coord, TileState::Loaded(inner));
                    } else {
                        self.record_failure(coord);
                    }
                }
                Err(e) => {
                    if e == "RATE_LIMITED" {
                        self.rate_limited_until = Some(web_time::Instant::now()
                            + std::time::Duration::from_secs_f64(RATE_LIMIT_COOLDOWN_SECS));
                        self.tiles.remove(&coord);
                    } else {
                        log::warn!("Base tile {:?} failed: {}", coord, e);
                        self.record_failure(coord);
                    }
                }
            }
        }
    }

    /// Mark visible tiles fresh and evict the oldest non-visible tile if over budget.
    pub fn update_visibility(&mut self, visible: &[BaseTileCoord]) {
        for coord in visible {
            if let Some(TileState::Loaded(tile)) = self.tiles.get_mut(coord) {
                tile.last_visible_frame = self.frame_counter;
            }
        }
        let current_frame = self.frame_counter;
        let loaded_count = self.tiles.values()
            .filter(|s| matches!(s, TileState::Loaded(_)))
            .count();
        let mut evicted = 0;
        while loaded_count - evicted > MAX_BASE_TILES {
            let oldest = self.tiles.iter()
                .filter_map(|(coord, state)| match state {
                    TileState::Loaded(tile) => {
                        if tile.last_visible_frame >= current_frame.saturating_sub(1) {
                            None
                        } else {
                            Some((*coord, tile.last_visible_frame))
                        }
                    }
                    TileState::Failed(_) => Some((*coord, 0)),
                    _ => None,
                })
                .min_by_key(|(_, frame)| *frame);
            if let Some((coord, _)) = oldest {
                self.tiles.remove(&coord);
                self.fail_counts.remove(&coord);
                evicted += 1;
            } else {
                break;
            }
        }
    }

    /// Labels from base tiles, restricted to large place kinds (State, City,
    /// District). Used while zoomed out — the z14 tiles' Street/Building/POI
    /// labels are fading away with their tiles, but place names should remain.
    pub fn large_labels(&self) -> Vec<&Label> {
        use crate::mapdata::LabelKind;
        self.tiles.values().filter_map(|state| {
            if let TileState::Loaded(tile) = state {
                Some(tile.labels.iter().filter(|l| matches!(
                    l.kind,
                    LabelKind::State | LabelKind::City | LabelKind::District
                )))
            } else {
                None
            }
        }).flatten().collect()
    }

    /// Loaded base tiles + their coords (renderer dedups on the coord).
    /// Filters by camera AABB intersection so off-screen tiles are skipped.
    pub fn loaded_tiles_with_coords_in_aabb<'a>(
        &'a self,
        cam_aabb: (f32, f32, f32, f32),
    ) -> impl Iterator<Item = (&'a LoadedTile, BaseTileCoord)> + 'a {
        let (cmin_x, cmin_y, cmax_x, cmax_y) = cam_aabb;
        let center_lat = self.center_lat;
        let center_lon = self.center_lon;
        self.tiles.iter().filter_map(move |(coord, state)| {
            let tile = match state {
                TileState::Loaded(t) => t,
                _ => return None,
            };
            // z11 tile bbox in lat/lon (Web Mercator).
            let n = (1u64 << coord.z) as f64;
            let west = coord.x as f64 / n * 360.0 - 180.0;
            let east = (coord.x as f64 + 1.0) / n * 360.0 - 180.0;
            // Tile y is inverted: y=0 is north pole.
            let north_rad = (std::f64::consts::PI * (1.0 - 2.0 * coord.y as f64 / n)).sinh().atan();
            let south_rad = (std::f64::consts::PI * (1.0 - 2.0 * (coord.y as f64 + 1.0) / n)).sinh().atan();
            let north = north_rad.to_degrees();
            let south = south_rad.to_degrees();
            let sw = crate::mapdata::project_pub(south, west, center_lat, center_lon);
            let ne = crate::mapdata::project_pub(north, east, center_lat, center_lon);
            let tile_min_x = sw[0].min(ne[0]);
            let tile_max_x = sw[0].max(ne[0]);
            let tile_min_y = sw[1].min(ne[1]);
            let tile_max_y = sw[1].max(ne[1]);
            if tile_max_x < cmin_x || tile_min_x > cmax_x { return None; }
            if tile_max_y < cmin_y || tile_min_y > cmax_y { return None; }
            Some((tile, *coord))
        })
    }
}

#[cfg(target_arch = "wasm32")]
async fn fetch_base_tile_wasm(
    z: u8, x: u32, y: u32,
    center_lat: f64, center_lon: f64,
) -> Result<MapData, String> {
    let pmtiles_url = crate::api::INIT_CONFIG.with(|c| {
        c.borrow().as_ref().and_then(|cfg| cfg.pmtiles_url.clone())
    }).ok_or("PMTiles URL not configured")?;

    // Cache lookup
    let cached = BASE_MAPDATA_CACHE.with(|c| c.borrow().get(&(z, x, y)).cloned());
    if let Some(cached) = cached {
        return Ok((*cached).clone());
    }

    let mvt_bytes = crate::pmtiles::get_tile(&pmtiles_url, z, x, y)
        .await
        .ok_or("Base PMTiles fetch failed")?;
    let mvt_tile = crate::mvt::decode_tile(&mvt_bytes);
    let data = crate::mvt_convert::mvt_to_mapdata(
        &mvt_tile, z, x, y, center_lat, center_lon,
        crate::mvt_convert::DetailLevel::Low,
    );
    let rc_data = Rc::new(data.clone());
    BASE_MAPDATA_CACHE.with(|c| {
        c.borrow_mut().insert((z, x, y), Rc::clone(&rc_data));
    });
    Ok(data)
}
