# Tile Pipeline Upgrades

Scoping doc for three perceived-quality improvements to PolyMap's tile loading
and rendering. Order: smallest blast radius first.

## Status of the three asks

| Item | Status | Why |
| --- | --- | --- |
| **Center-out priority queue** | **Already done** | `tiles.rs:326–331` sorts visible tiles by squared distance from the camera tile before dispatch. Nearest tile fetches first, then a ring outward. No code change needed. |
| **Overview / "low-res surrounding"** | Not done | New layer required — see below. |
| **Multi-draw batching** | Not done | Optimization, not perceptual; defer. |

---

## 1. Center-out priority queue — verification only

**Where:** `src/tiles.rs:326-331`

```rust
tiles.sort_by_key(|t| {
    let dc = (t.col - cam_col).abs();
    let dr = (t.row - cam_row).abs();
    dc * dc + dr * dr
});
```

`request_visible_tiles()` iterates this sorted list respecting `MAX_IN_FLIGHT`.
The first dispatched fetch is always the tile under the camera.

**Possible refinement (not required):** the sort uses tile-grid distance, not
sub-tile camera position. When the camera sits between two tiles' centers,
either could be picked first. Using squared distance from camera *world
position* to tile *center world position* would be slightly more accurate.
File this under "polish if needed."

---

## 2. Overview layer — the real win

### Goal

User opens the app or pans to a new area → instantly sees a coherent
low-detail rendering of the whole viewport, not a blank screen filling in
tile-by-tile. As detail tiles arrive, they fade in over the overview, masking
the pop-in completely.

### Architecture decision: separate overview pmtiles

PolyMap fetches MVT tiles via PMTiles range requests against a single
`tiles.pmtiles` backed by Protomaps' build (or a custom build). The current
`tiles.pmtiles` carries z14-only data tuned for high-detail rendering; lower
zooms (z6–z10) are present but optimized for navigation, not for use as a
visual fallback.

Rather than retrofit the existing pmtiles, build a **dedicated
`overview.pmtiles`** at low zoom with aggressive simplification, and load it as
a separate layer that's always resident.

### `overview.pmtiles` build

New step in `parcels/run.sh` (or the basemap build, wherever you build
`tiles.pmtiles`):

```bash
# Source: same regional OSM extract you use for the basemap.
# Output: ~5–20 MB pmtiles covering the same area at z6–z9 only.

osmium tags-filter $REGIONAL_PBF \
    a/natural=water,coastline \
    a/landuse=forest,wood,park,residential,commercial,industrial \
    w/highway=motorway,trunk,primary \
    a/place=country,state \
    -o overview.pbf

osmium export overview.pbf -f geojsonseq -o overview.geojsonl

tippecanoe \
    --output=overview.pmtiles \
    --layer=overview \
    --minimum-zoom=4 \
    --maximum-zoom=9 \
    --simplification=20 \
    --drop-densest-as-needed \
    --no-tile-compression \
    --force overview.geojsonl
```

Expected size: 5–20 MB for a US state, 50–100 MB for the whole US. Loaded
once at app start, stays in memory forever.

### PolyMap-side changes

**A. `tiles.rs` — add an overview cache:**
- `TileManager` gets `overview_tiles: HashMap<(u8, u32, u32), LoadedTile>`
  keyed by (zoom, x, y) of the MVT tile in `overview.pmtiles`.
- New method `request_overview_tiles(camera)` that, given a camera position
  and zoom, decides which overview tiles are needed (typically z6–z9 covering
  the visible area at the chosen overview zoom).
- New constructor parameter `overview_pmtiles_url: Option<String>` so the
  feature is opt-in until the overview pmtiles is built and published.
- Eviction policy: overview tiles **never evict**. Use a separate fetcher
  budget (1–2 concurrent fetches) so they don't compete with detail tiles.

**B. `mvt_convert.rs` — handle the `overview` layer:**
- Add `"overview"` to the layer dispatch.
- The handler reads `kind` (or actual OSM tag) and routes into existing
  buckets: water → `water_polys`, forest/park → `park_polys`, residential
  → `landuse_polys`, etc. This reuses all existing rendering paths.

**C. `lib.rs` — kick off overview load on init:**
- After `TileManager::new`, call `request_overview_tiles` once. The overview
  doesn't need to wait for any visible-tile calculation — it's tied to broad
  region, not viewport.
- Re-call `request_overview_tiles` on major camera moves (cross-region pan)
  but not every frame.

**D. `renderer.rs` — render overview before detail tiles:**

Add a new pass *before* the existing tile pass at line 495:

```rust
// Pass 0: Overview — always-resident low-detail geometry.
// Detail tiles render on top via fade-in; this prevents blank-screen pan.
render_pass.set_pipeline(&self.map_pipeline);
render_pass.set_bind_group(0, &gpu.camera_bind_group, &[]);
render_pass.set_bind_group(1, self.textures.material_bind_group().unwrap(), &[]);
for tile in tiles.overview_tiles.values() {
    if tile.num_indices == 0 { continue; }
    render_pass.set_vertex_buffer(0, tile.vertex_buffer.slice(..));
    render_pass.set_vertex_buffer(1, tile.birth_buffer.slice(..));
    render_pass.set_index_buffer(tile.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
    render_pass.draw_indexed(0..tile.num_indices, 0, 0..1);
}

// Pass 1: Detail tiles (existing code at line 495)
```

The detail-tile fade-in (`birth_time` shader logic) already cross-fades over
whatever's under it, so the transition from overview → detail is automatic
once both are in the depth buffer.

**E. Z-ordering / depth:**
- Overview vertices should render *behind* detail. Either:
  - Offset overview vertices by a small negative z so they sort behind detail
    (simplest, no shader change), or
  - Disable depth writes during overview pass and enable for detail pass
    (cleaner, two-pipeline change).
- I recommend the small-z-offset approach. Add `OVERVIEW_Z_OFFSET = -0.001`
  in `mapdata.rs` and apply it in the overview-layer handlers.

### Effort

- pmtiles build: 1–2 h (add the bash step, run once per region)
- overview cache + dispatch in `tiles.rs`: ~3 h
- handler in `mvt_convert.rs`: ~1 h (mostly mirroring landuse/water logic)
- renderer pass + z-offset: ~1 h
- end-to-end test: ~2 h

**Total: ~1 day of focused work, ~1.5 days with debugging.**

### Risks / open questions

1. **Color matching:** the overview's water/park/landuse colors must match the
   detail tiles' colors exactly, otherwise the cross-fade will be visible as
   a hue shift. Use the same `COLOR_*` constants in both handler paths.
2. **Memory footprint:** at ~20 MB per state-sized overview, 50 states is
   ~1 GB. Don't load all 50 — load only the user's current state on init,
   re-load on cross-state pan.
3. **PMTiles range fetches at low zoom can be slow** because they're sparse.
   Worth pre-warming with a single bulk fetch of the few z6 tiles needed.

---

## 3. Multi-draw batching — optimization, defer

### Current state

`renderer.rs:499-504` issues one `set_vertex_buffer` + `draw_indexed` per
tile (~30–40 tiles in view × 2 passes = ~60–80 draw calls per frame). Each
call has a fixed CPU overhead in the wgpu submission path. Modern GPUs swallow
this; it's not the bottleneck. **You should not touch this until you have a
profiling reason.**

### When it would matter

If you find yourself:
- Rendering 200+ tiles per frame (very high zoom-out + dense data)
- Hitting CPU-side wgpu submission cost in profiles
- Targeting mobile, where draw-call cost is higher

### Implementation sketch

WebGPU supports `multi_draw_indexed_indirect` via the `MULTI_DRAW_INDIRECT`
feature (Chrome desktop only at time of writing — Safari and mobile don't
support it). This rules it out as a portable optimization.

A more portable alternative: **megabuffer + indirect**. Concatenate all visible
tiles' vertex/index data into one global buffer per frame (rebuilt on any
change), record one `draw_indexed_indirect` per tile via an indirect buffer.
This is a CPU-side simplification more than a GPU win — same draw calls, but
the per-tile bind/buffer-set is replaced with offset math.

### Effort

- Megabuffer manager: 1–2 days
- Indirect command recording: 1 day
- Validation across browsers: 1–2 days
- **Total: ~5 days** for a marginal CPU-side win.

### Recommendation

Skip until profiling proves a need.

---

## Implementation order

1. **Verify center-out priority is acceptable** (already done in code; one
   visual pan test confirms).
2. **Build `overview.pmtiles`** for one state (smallest one — RI or DE) as
   a smoke test. If the file size and visual look are right, repeat for
   target deploy regions.
3. **Wire the overview cache + render pass** behind a feature flag so it can
   be toggled on/off. Validate fade-in timing visually against a stale-tile
   pan.
4. **Iterate on z-offset / color matching** until the overview→detail
   transition is invisible.
5. **Skip multi-draw** until you have a profiling reason.

If the overview pass works as expected, the perceived "clunkiness" of
tile-by-tile pop-in should disappear — and the change is contained to four
files: `tiles.rs`, `mvt_convert.rs`, `renderer.rs`, and one new bash step in
the basemap build.
