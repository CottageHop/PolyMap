# PolyMap

![PolyMap — 3D OpenStreetMap rendering](https://raw.githubusercontent.com/CottageHop/PolyMap/assets/polymap-hero.png)

A 3D map renderer powered by WebGPU and Rust/WASM. Renders OpenStreetMap data as textured 3D geometry with buildings, roads, parks, water, trees, and labels — directly in the browser.

## Features

- **WebGPU rendering** — hardware-accelerated 3D map with materials and shadows
- **PMTiles support** — load vector tiles from a single static file (no tile server needed)
- **MVT decoding** — reads Mapbox Vector Tiles and generates 3D geometry
- **Tiled loading** — streams map data as you pan and zoom
- **Text labels** — city names, street names, parks, and buildings with collision avoidance
- **Material textures** — configurable texture layers for roads, buildings, water, etc.
- **WASM** — compiles to WebAssembly for near-native performance
- **JavaScript API** — simple JS wrapper for easy integration
- **Vue component** — drop-in Vue 3 component included

## Quick Start

### Prerequisites

- [Rust](https://rustup.rs/) (stable)
- [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/)
- Node.js 18+ (for the JS wrapper)

### Build

```bash
# Build the WASM module
make build-wasm

# Or build and serve locally
make serve
```

### Usage (JavaScript)

```html
<div id="map" style="width: 100%; height: 500px;"></div>
<script type="module">
  import { createPolyMap } from './polymap.js';

  const map = await createPolyMap('#map', {
    center: { lat: 43.615, lon: -116.202 },
    zoom: -0.5,
    tilt: 1.45,
    pmtilesUrl: 'https://your-server.com/tiles.pmtiles',
  });
</script>
```

### Usage (Vue 3)

```vue
<template>
  <PolyMap
    :center="{ lat: 43.615, lon: -116.202 }"
    :zoom="-0.5"
    :tilt="1.45"
    pmtiles-url="https://your-server.com/tiles.pmtiles"
  />
</template>
```

## Architecture

```
src/           Core Rust library (WebGPU renderer, tile manager, MVT decoder)
  lib.rs       Application entry point and event loop
  gpu.rs       WebGPU device/surface initialization
  renderer.rs  Render pipeline (map geometry, shadows, text, clouds)
  tiles.rs     Tile manager (loading, caching, visibility)
  camera.rs    3D camera with smooth zoom/pan/tilt
  mapdata.rs   OSM data parsing and geometry generation
  mvt.rs       Mapbox Vector Tile protobuf decoder
  mvt_convert.rs  MVT to 3D geometry converter
  pmtiles.rs   PMTiles v3 reader (HTTP range requests)
  text.rs      Text rendering (glyph atlas, label placement)
  texture.rs   Material texture system
  api.rs       JavaScript API bindings (wasm-bindgen)
  config.rs    Configuration types
  map.wgsl     Main vertex/fragment shader
  text.wgsl    Text rendering shader
  terrain.wgsl Terrain shader

js/            JavaScript wrapper library
  src/
    polymap.js   High-level JS API (createPolyMap, events, markers)
    polymap.d.ts TypeScript definitions

polymap-worker/  Web Worker for off-main-thread tile processing
  src/
    lib.rs       Worker entry point (MVT decode + triangulation)

web/           Demo application
  index.html   Standalone demo page
```

## Data Sources

PolyMap renders vector tile data in the [MVT format](https://docs.mapbox.com/data/tilesets/guides/vector-tiles-standards/). You can use:

- **PMTiles** — a single-file tile archive served over HTTP with range requests. Generate with [tippecanoe](https://github.com/felt/tippecanoe) from GeoJSON or use pre-built OpenStreetMap extracts.
- **Custom API** — provide a URL that returns OSM JSON for a bounding box.

## Configuration

```javascript
const map = await createPolyMap('#map', {
  // Map position
  center: { lat: 43.615, lon: -116.202 },
  zoom: -0.5,        // logarithmic: 0 = default, +1 = 2x closer
  tilt: 1.45,        // radians: 0 = top-down, PI/2 = horizon

  // Data sources (provide at least one)
  pmtilesUrl: '...',  // URL to a PMTiles file
  apiBase: '...',     // Base URL for /map/osm?south=...&west=...
  apiKey: '...',      // API key sent as X-Map-Key header

  // Layers (all default to visible)
  layers: {
    buildings: true,
    roads: true,
    water: true,
    parks: true,
    trees: true,
    shadows: true,
    labels: true,
  },
});

// Events
map.on('ready', () => { ... });
map.on('camera:move', () => { ... });

// Methods
map.setView(lat, lon, zoom);
map.setZoom(zoom);
map.setTilt(tilt);
map.addMarker(id, lat, lon);
map.removeMarker(id);
map.clearMarkers();
map.getCamera();  // { lat, lon, zoom, tilt, bounds }
map.destroy();
```

## Building from Source

```bash
# Install Rust and wasm-pack
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
cargo install wasm-pack

# Clone and build
git clone https://github.com/CottageHop/PolyMap.git
cd PolyMap
make build-wasm

# Serve the demo
make serve
# Open http://localhost:8080
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License. See [LICENSE](LICENSE).
