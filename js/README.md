# polymap

3D map renderer powered by WebGPU/WASM. Available as a vanilla JS library and a Vue 3 component.

## Installation

```bash
npm install polymap
```

## Vanilla JS

```js
import { createPolyMap } from 'polymap/polymap';

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
```

## Vue 3

```vue
<template>
  <PolyMap
    :center="{ lat: 40.765, lon: -73.980 }"
    :zoom="1.0"
    :tilt="1.35"
    :markers="listings"
    :layers="{ trees: true, shadows: true }"
    width="100%"
    height="600px"
    @ready="onReady"
    @marker:click="onMarkerClick"
    @update:center="center = $event"
    @update:zoom="zoom = $event"
  >
    <!-- Custom overlay that has access to the map instance -->
    <template #overlay="{ map, camera }">
      <div class="camera-info">
        Zoom: {{ camera.zoom.toFixed(1) }}
      </div>
    </template>
  </PolyMap>
</template>

<script setup>
import { ref } from 'vue';
import { PolyMap } from 'polymap';

const center = ref({ lat: 40.765, lon: -73.980 });
const zoom = ref(1.0);

const listings = ref([
  { id: 'apt-1', lat: 40.764, lon: -73.978, label: '$4.2M', className: 'listing-pin' },
  { id: 'apt-2', lat: 40.766, lon: -73.982, label: '$1.8M' },
]);

function onReady(map) {
  console.log('Map ready!', map.getCamera());
}

function onMarkerClick({ id }) {
  console.log('Clicked:', id);
}
</script>
```

### Props

| Prop | Type | Default | Description |
|------|------|---------|-------------|
| `center` | `{ lat, lon }` | `{ lat: 40.765, lon: -73.980 }` | Map center (two-way with `v-model:center`) |
| `zoom` | `number` | `0.8` | Zoom level (two-way with `v-model:zoom`) |
| `tilt` | `number` | `1.35` | Camera tilt in radians |
| `markers` | `Array` | `[]` | Markers: `[{ id, lat, lon, html?, label?, className? }]` |
| `layers` | `Object` | `{}` | Layer visibility: `{ buildings, roads, water, parks, trees, shadows, labels }` |
| `width` | `string` | `'100%'` | Container width |
| `height` | `string` | `'400px'` | Container height |
| `dataUrl` | `string` | — | Custom data API URL |
| `wasmUrl` | `string` | — | Custom WASM module URL |

### Events

| Event | Payload | Description |
|-------|---------|-------------|
| `ready` | `map` | Map initialized and first tiles loaded |
| `camera:move` | `{ lat, lon, zoom, tilt }` | Camera position changed |
| `click` | `{ screenX, screenY }` | Map clicked |
| `resize` | `{ width, height }` | Map resized |
| `marker:click` | `{ id, lat, lon, element }` | Marker clicked |
| `update:center` | `{ lat, lon }` | Center changed (for v-model) |
| `update:zoom` | `number` | Zoom changed (for v-model) |

### Slots

| Slot | Props | Description |
|------|-------|-------------|
| `overlay` | `{ map, camera }` | Custom overlay with access to map instance and camera state |
| `markers` | `{ map, addMarker, removeMarker }` | Custom marker management |
