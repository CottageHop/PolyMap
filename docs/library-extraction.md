# Library Extraction & WASM Consolidation

Scoping doc for two related refactors:

1. **Workspace consolidation** — eliminate the duplicated source between
   `PolyMap/src/` and `PolyMap/polymap-worker/src/`. One crate, two WASM
   outputs via Cargo features. No more drift.
2. **Generalize PolyMap as an open-source library** — remove curb_appeal-
   specific concepts (real-estate listings, NYC penthouse demo data) so
   PolyMap is usable by any project — hiking, city planning, restaurants,
   etc. — without surprise.

These can be done independently, but it's cleaner to ship them together.

---

## Part 1 — Workspace consolidation

### Current state

```
PolyMap/
├── Cargo.toml                 # main crate (renderer + data extraction)
├── src/
│   ├── lib.rs                 # wasm-bindgen exports for the renderer
│   ├── api.rs                 # high-level API
│   ├── camera.rs, gpu.rs, renderer.rs, text.rs, cars.rs, tiles.rs
│   ├── mapdata.rs             # ← duplicated
│   ├── mvt.rs                 # ← duplicated
│   ├── mvt_convert.rs         # ← duplicated
│   └── pmtiles.rs             # ← duplicated
└── polymap-worker/
    ├── Cargo.toml             # separate crate (data only)
    └── src/
        ├── lib.rs             # wasm-bindgen export for `process_tile`
        ├── mapdata.rs         # ← duplicated, drifts
        ├── mvt.rs             # ← duplicated
        ├── mvt_convert.rs     # ← duplicated, drifts
        └── pmtiles.rs         # ← duplicated
```

Symptoms:
- `make sync-worker` exists but **silently clobbers intentional divergence**
  (worker has no `Car` struct, no `LabelKind::State`/`Subdivision`, etc.).
- Geometry/material additions have to be hand-mirrored in both crates and we
  ship guidance memory ("update both files") to manage it.
- Two `.wasm` artifacts to maintain in curb_appeal: `polymap_bg.wasm` and
  `polymap_worker_bg.wasm`.

### Target state

```
PolyMap/
├── Cargo.toml                 # workspace root
├── crates/
│   ├── polymap/               # the only crate; produces both WASMs
│   │   ├── Cargo.toml         # cargo features: "renderer" (default), "worker"
│   │   └── src/
│   │       ├── lib.rs         # gates renderer module on feature
│   │       ├── api.rs         # gated on "renderer"
│   │       ├── camera.rs, gpu.rs, renderer.rs, text.rs, cars.rs, tiles.rs
│   │       │                  # all gated on "renderer"
│   │       ├── mapdata.rs     # always compiled
│   │       ├── mvt.rs         # always compiled
│   │       ├── mvt_convert.rs # always compiled
│   │       ├── pmtiles.rs     # always compiled
│   │       └── worker.rs      # process_tile() wasm-bindgen export, gated on "worker"
│   └── (no second crate)
└── Makefile
    # build-wasm:    cargo build --features renderer  → polymap.wasm (~3 MB)
    # build-worker:  cargo build --no-default-features --features worker → polymap-worker.wasm (~200 KB)
```

### Cargo manifest sketch

```toml
# crates/polymap/Cargo.toml
[package]
name = "polymap"
edition = "2024"

[features]
default = ["renderer"]
renderer = ["dep:wgpu", "dep:winit", "dep:fontdue"]
worker = []  # marker; no extra deps, just enables wasm-bindgen `process_tile`

[dependencies]
serde_json = "1"
earcutr = "0.4"
flate2 = { version = "1", default-features = false, features = ["rust_backend"] }
bytemuck = { version = "1", features = ["derive"] }
glam = "0.29"
log = "0.4"
web-time = "1"

# Renderer-only deps — only pulled in when the feature is enabled
wgpu = { version = "24", features = ["webgl"], optional = true }
winit = { version = "0.30", optional = true }
fontdue = { version = "0.9", optional = true }

[target.'cfg(target_arch = "wasm32")'.dependencies]
console_error_panic_hook = "0.1"
console_log = "1"
wasm-bindgen = "0.2"
wasm-bindgen-futures = "0.4"
serde-wasm-bindgen = "0.6"
js-sys = "0.3"
web-sys = { version = "0.3", features = [...] }
```

### `lib.rs` shape

```rust
// crates/polymap/src/lib.rs

pub mod mapdata;
pub mod mvt;
pub mod mvt_convert;
pub mod pmtiles;

#[cfg(feature = "renderer")]
mod camera;
#[cfg(feature = "renderer")]
mod gpu;
#[cfg(feature = "renderer")]
mod renderer;
#[cfg(feature = "renderer")]
mod text;
#[cfg(feature = "renderer")]
mod cars;
#[cfg(feature = "renderer")]
pub mod api;
#[cfg(feature = "renderer")]
pub mod tiles;

#[cfg(feature = "worker")]
mod worker;

// Re-export the public API for either context.
#[cfg(feature = "renderer")]
pub use api::PolyMap;

#[cfg(feature = "worker")]
pub use worker::process_tile;
```

### Makefile build steps

```make
build-wasm:
	cd crates/polymap && wasm-pack build --release --target web \
	    --out-dir ../../web/pkg \
	    -- --features renderer

build-worker:
	cd crates/polymap && wasm-pack build --release --target web \
	    --out-dir ../../web/worker-pkg \
	    -- --no-default-features --features worker
	cp web/worker-pkg/polymap_bg.wasm web/worker-pkg/polymap_worker_bg.wasm
	cp web/worker-pkg/polymap.js     web/worker-pkg/polymap_worker.js
```

(The `cp` lines preserve the file names tile-worker.js currently expects;
or just update tile-worker.js to import the new names.)

### Migration order (no rendering breaks)

1. Add features to current main `Cargo.toml` and gate the renderer modules.
   Verify `make build-wasm` still produces an identical-functioning
   renderer WASM.
2. Add a `worker` module with a `process_tile()` export that mirrors
   polymap-worker's. Keep the old worker crate temporarily.
3. Update PolyMap's Makefile so `build-worker` builds from the main crate
   with `--no-default-features --features worker`. Swap the artifact names.
4. Diff the new worker WASM's behavior against the old one (decode the
   same tile, compare output). Should be byte-identical or trivially close.
5. Once verified, **delete `polymap-worker/` entirely**. Delete `make
   sync-worker`. Delete the worker-sync memory.
6. Update curb_appeal's `public/polymap/` import paths to the new artifacts.
   No code change in curb_appeal — same file names, just one source.

### Effort

- Feature gating: ~3 h (mostly cfg attributes on existing modules).
- `worker` module + `process_tile`: ~1 h (lift from polymap-worker/lib.rs).
- Makefile rework: ~1 h.
- Diff / smoke testing: ~2 h.
- Delete the old crate + cleanup: ~30 min.

**Total: ~1 day.** Zero behavior change for end users.

---

## Part 2 — Generalize curb_appeal-specific concepts out of PolyMap

### What needs to leave PolyMap

From the audit:

| Today | Status |
| --- | --- |
| `HomeListing { id, lat, lon, sqft, beds, baths, price, image_url }` in `mapdata.rs` | Explicitly real-estate-named. **Generalize.** |
| `PlacedListing { listing, position }` | Same. **Generalize.** |
| `MapData::add_listings(&mut self, ...)` | API surface is realty-flavored. **Rename.** |
| `LabelKind::Listing` | Generic concept ("named pin label") with a real-estate name. **Rename.** |
| `LabelKind::Subdivision` | Realty term; civic equivalent is "district" or "neighborhood". **Rename.** |
| `MAT_PIN = 13.0` (// "Material for home listing pin") | Pin geometry is generic; just the comment is realty-specific. **Update comment.** |
| `web/index.html` — NYC Park Ave penthouse demo data, listing-popup CSS | The demo is a sales pitch for curb_appeal. **Replace with generic POI demo.** |

### What can stay (with documentation)

| Today | Why it can stay |
| --- | --- |
| `parcels_url`, `parcels: bool` config | Parcels are a real civic/geographic concept (property boundaries from OSM and city assessor data), useful for hiking, planning, surveying apps, not just real estate. Keep, but document as "optional parcel-boundary overlay" not "for real estate apps". |
| Building height, roof shapes, gable rendering | Generic geographic rendering. |
| Tile pipeline, materials, shaders | All generic. |

### Proposed renames

```rust
// Before (mapdata.rs)
pub struct HomeListing {
    pub id: u64,
    pub lat: f64,
    pub lon: f64,
    pub sqft: u32,
    pub beds: u32,
    pub baths: f32,
    pub price: u32,
    pub image_url: Option<String>,
}

// After
pub struct MapMarker {
    pub id: u64,
    pub lat: f64,
    pub lon: f64,
    /// User-defined metadata as a JSON value. PolyMap doesn't interpret this;
    /// it's passed to the JS side on hover/click events for the host app to
    /// render a popup or detail view.
    #[serde(default)]
    pub metadata: serde_json::Value,
}

pub struct PlacedMarker {
    pub marker: MapMarker,
    pub position: [f32; 2],
}

impl MapData {
    pub fn add_markers(&mut self, markers: Vec<PlacedMarker>) { ... }
}
```

```rust
// LabelKind enum
pub enum LabelKind {
    State,
    City,
    District,           // was: Subdivision
    Street,
    Park,
    Building,
    Marker,             // was: Listing
    Poi,
}
```

### What curb_appeal has to change

curb_appeal converts each listing → `MapMarker` at the boundary with PolyMap:

```ts
// curb_appeal side
function listingToMarker(l: Listing): MapMarker {
  return {
    id: l.id,
    lat: l.lat,
    lon: l.lon,
    metadata: {
      kind: "real_estate",
      price: l.price,
      sqft: l.sqft,
      beds: l.beds,
      baths: l.baths,
      address: l.address,
      image_url: l.image_url,
    },
  };
}

map.addMarkers(listings.map(listingToMarker));
```

curb_appeal's existing listing-popup UI reads from `metadata` instead of the
direct fields. ~30 lines of TypeScript change.

### Demo replacement

Replace `web/index.html`'s NYC penthouse listings with a generic dataset.
Suggested: a small list of NYC restaurants or museums (well-known data, no
proprietary feel, demonstrates the marker API without screaming "buy this
condo"). Or better: just *no* hardcoded markers — let the map render with
basemap data only, and add a small "Try `map.addMarkers(...)` from the
console" hint.

### Effort

- Renames in `mapdata.rs` + propagate through `lib.rs`, `api.rs`, `text.rs`:
  ~2 h.
- Remove direct realty fields, add generic `metadata` JSON pass-through: ~1 h.
- Update `web/index.html` demo data: ~30 min.
- Update curb_appeal call sites: ~30 min on the curb_appeal side
  (probably 5–10 lines).
- Verify both PolyMap demo and curb_appeal still work: ~1 h.

**Total: ~5 h** spread across both repos.

---

## Combined effort

~1.5 days total for both refactors. Net result:

- One crate, two WASM outputs, no source duplication.
- PolyMap's public API has zero realty-specific concepts.
- curb_appeal continues to work; just imports `polymap.wasm` /
  `polymap_worker.wasm` from one PolyMap build.
- Outside developers can adopt PolyMap for any geographic-rendering use
  case without confusion.

## What's deferred

- The tile-pipeline upgrades from `tile-pipeline-upgrades.md` (overview
  layer, multi-draw) are independent and can land any time.
- The generative-buildings work is independent.
- This refactor doesn't touch shaders, GPU code, or rendering logic at all
  beyond renames.

---

## Implementation order

If you want to execute, the safe path is:

1. **Part 1 (workspace) first.** Behavior-preserving, internal change. Verify
   PolyMap demo and curb_appeal both still work against the new artifacts.
2. **Part 2 (rename) second.** API-breaking change for curb_appeal but
   contained to one boundary file in curb_appeal. Update both repos in one
   PR pair.

Each part is independently shippable.
