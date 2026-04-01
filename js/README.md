# polymap

3D map renderer powered by WebGPU/WASM.

## Installation

```bash
npm install polymap
```

## Usage

```js
import { createPolyMap } from 'polymap';

const map = await createPolyMap('#map', {
  center: { lat: 40.765, lon: -73.980 },
  zoom: 1.0,
  tilt: 1.35,
});

// Add markers
map.addMarker('apt-1', 40.764, -73.978, {
  html: '<div class="pin">$4.2M</div>',
});

// Listen for events
map.on('ready', () => console.log('Map ready!'));
map.on('marker:click', ({ id, lat, lon }) => {
  console.log(`Clicked marker ${id}`);
});

// Camera control
map.setView(34.0522, -118.2437, 1.0); // Fly to LA
map.setZoom(2.0);
map.setTilt(1.0);

// Toggle layers
map.setLayerVisible('trees', false);
map.setLayerVisible('shadows', false);

// Theme colors
map.setColors({
  water: [0.06, 0.24, 0.29, 1.0],
  park: [0.17, 0.19, 0.11, 1.0],
  building: [0.29, 0.25, 0.15, 1.0],
  road: [0.21, 0.10, 0.03, 1.0],
  land: [0.82, 0.73, 0.55, 1.0],
});

// Cloud controls
map.setCloudOpacity(0.5);
map.setCloudSpeed(1.5);
```

## API

### `createPolyMap(container, options)`

Creates a map instance inside a container element.

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `center` | `{ lat, lon }` | `{ lat: 40.765, lon: -73.980 }` | Map center |
| `zoom` | `number` | `0.8` | Zoom level |
| `tilt` | `number` | `1.35` | Camera tilt in radians |
| `pmtilesUrl` | `string` | — | URL to PMTiles file |
| `wasmUrl` | `string` | — | Custom WASM module URL |
| `layers` | `Object` | `{}` | Layer visibility: `{ buildings, roads, water, parks, trees, shadows, labels, clouds }` |
| `showControls` | `boolean` | `true` | Show the customize map controls panel |

### Methods

| Method | Description |
|--------|-------------|
| `setView(lat, lon, zoom?)` | Pan and zoom to position |
| `setZoom(zoom)` | Set zoom level |
| `setTilt(tilt)` | Set camera tilt (radians) |
| `panBy(dx, dy)` | Pan by pixel offset |
| `getCamera()` | Get `{ lat, lon, zoom, tilt }` |
| `addMarker(id, lat, lon, options?)` | Add a marker |
| `removeMarker(id)` | Remove a marker |
| `clearMarkers()` | Remove all markers |
| `setLayerVisible(layer, visible)` | Toggle layer visibility |
| `setColors(config)` | Set color overrides `{ water, park, building, road, land }` |
| `setCloudOpacity(opacity)` | Set cloud opacity (0.0–1.0) |
| `setCloudSpeed(speed)` | Set cloud animation speed (0.0–3.0) |
| `setBackgroundTexture(url)` | Set tiled background texture |
| `setCloudTexture(url)` | Set cloud atlas texture |
| `showControls()` | Show the customize map panel |
| `hideControls()` | Hide the customize map panel |
| `on(event, callback)` | Register event listener |
| `off(event, callback?)` | Remove event listener |
| `destroy()` | Clean up resources |

### Events

| Event | Payload | Description |
|-------|---------|-------------|
| `ready` | `{ width, height, dpr }` | Map initialized |
| `camera:move` | — | Camera settled after movement |
| `click` | `{ screenX, screenY }` | Map clicked |
| `resize` | `{ width, height }` | Map resized |
| `marker:click` | `{ id, lat, lon, element }` | Marker clicked |
