<template>
  <div ref="container" class="polymap-container" :style="containerStyle">
    <slot name="overlay" :map="mapInstance" :camera="cameraState" />
    <slot
      name="markers"
      :map="mapInstance"
      :addMarker="addMarker"
      :removeMarker="removeMarker"
    />
  </div>
</template>

<script>
import { defineComponent, ref, watch, onMounted, onBeforeUnmount, toRefs, computed } from 'vue';

export default defineComponent({
  name: 'PolyMap',

  props: {
    center: {
      type: Object,
      default: () => ({ lat: 40.765, lon: -73.980 }),
    },
    zoom: {
      type: Number,
      default: 0.8,
    },
    tilt: {
      type: Number,
      default: 1.35,
    },
    dataUrl: {
      type: String,
      default: undefined,
    },
    wasmUrl: {
      type: String,
      default: undefined,
    },
    layers: {
      type: Object,
      default: () => ({}),
    },
    markers: {
      type: Array,
      default: () => [],
    },
    width: {
      type: String,
      default: '100%',
    },
    height: {
      type: String,
      default: '400px',
    },
  },

  emits: [
    'ready',
    'camera:move',
    'click',
    'resize',
    'marker:click',
    'update:center',
    'update:zoom',
    'error',
  ],

  setup(props, { emit }) {
    const container = ref(null);
    const mapInstance = ref(null);
    const cameraState = ref({ lat: 0, lon: 0, zoom: 0, tilt: 0 });
    const isReady = ref(false);
    const { center, zoom, tilt, layers, markers } = toRefs(props);

    const containerStyle = computed(() => ({
      width: props.width,
      height: props.height,
      position: 'relative',
      overflow: 'hidden',
    }));

    const renderedMarkerIds = new Set();

    onMounted(async () => {
      // SSR guard — only run on client
      if (typeof window === 'undefined' || typeof document === 'undefined') return;

      try {
        // Dynamic import to avoid SSR issues
        const { createPolyMap } = await import('./polymap.js');

        // Wait for container to have layout dimensions
        await waitForLayout(container.value);

        const map = await createPolyMap(container.value, {
          center: center.value,
          zoom: zoom.value,
          tilt: tilt.value,
          dataUrl: props.dataUrl,
          wasmUrl: props.wasmUrl,
          layers: layers.value,
        });

        mapInstance.value = map;

        map.on('ready', () => {
          isReady.value = true;
          emit('ready', map);
          syncMarkers();
        });

        map.on('camera:move', (cam) => {
          cameraState.value = cam;
          emit('camera:move', cam);
          emit('update:center', { lat: cam.lat, lon: cam.lon });
          emit('update:zoom', cam.zoom);
        });

        map.on('click', (data) => emit('click', data));
        map.on('resize', (data) => emit('resize', data));
        map.on('marker:click', (data) => emit('marker:click', data));

        for (const [layer, visible] of Object.entries(layers.value)) {
          map.setLayerVisible(layer, visible);
        }
      } catch (e) {
        console.error('PolyMap init failed:', e);
        emit('error', { error: e, message: e.message || String(e) });
      }
    });

    onBeforeUnmount(() => {
      if (mapInstance.value) {
        mapInstance.value.destroy();
        mapInstance.value = null;
      }
    });

    /** Wait until the element has non-zero dimensions. */
    function waitForLayout(el, maxAttempts = 10) {
      return new Promise((resolve) => {
        let attempts = 0;
        const check = () => {
          if (el.clientWidth > 0 && el.clientHeight > 0) {
            resolve();
          } else if (attempts < maxAttempts) {
            attempts++;
            requestAnimationFrame(check);
          } else {
            resolve(); // Give up, proceed with whatever we have
          }
        };
        check();
      });
    }

    function syncMarkers() {
      const map = mapInstance.value;
      if (!map || !isReady.value) return;

      const newIds = new Set(markers.value.map((m) => m.id));

      for (const id of renderedMarkerIds) {
        if (!newIds.has(id)) {
          map.removeMarker(id);
          renderedMarkerIds.delete(id);
        }
      }

      for (const m of markers.value) {
        if (!renderedMarkerIds.has(m.id)) {
          map.addMarker(m.id, m.lat, m.lon, {
            html: m.html || `<div class="polymap-default-pin">${m.label || ''}</div>`,
            className: m.className,
          });
          renderedMarkerIds.add(m.id);
        }
      }
    }

    function addMarker(id, lat, lon, options) {
      if (mapInstance.value) {
        return mapInstance.value.addMarker(id, lat, lon, options);
      }
    }

    function removeMarker(id) {
      if (mapInstance.value) {
        mapInstance.value.removeMarker(id);
        renderedMarkerIds.delete(id);
      }
    }

    watch(center, (val) => {
      if (mapInstance.value && val) {
        mapInstance.value.setView(val.lat, val.lon);
      }
    }, { deep: true });

    watch(zoom, (val) => {
      if (mapInstance.value && val !== undefined) {
        mapInstance.value.setZoom(val);
      }
    });

    watch(tilt, (val) => {
      if (mapInstance.value && val !== undefined) {
        mapInstance.value.setTilt(val);
      }
    });

    watch(layers, (val) => {
      if (mapInstance.value && val) {
        for (const [layer, visible] of Object.entries(val)) {
          mapInstance.value.setLayerVisible(layer, visible);
        }
      }
    }, { deep: true });

    watch(markers, () => syncMarkers(), { deep: true });

    return {
      container,
      mapInstance,
      cameraState,
      containerStyle,
      addMarker,
      removeMarker,
    };
  },
});
</script>

<style>
.polymap-container {
  background: #eeedea;
}

.polymap-container canvas {
  width: 100%;
  height: 100%;
  display: block;
}

.polymap-marker {
  position: absolute;
  transform: translate(-50%, -100%);
  pointer-events: auto;
  cursor: pointer;
  transition: opacity 0.15s;
}

.polymap-default-pin {
  background: #c0392b;
  color: white;
  padding: 6px 12px;
  border-radius: 20px;
  font: bold 13px -apple-system, BlinkMacSystemFont, sans-serif;
  white-space: nowrap;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
  position: relative;
}

.polymap-default-pin::after {
  content: '';
  position: absolute;
  left: 50%;
  bottom: -6px;
  transform: translateX(-50%);
  border: 6px solid transparent;
  border-top-color: #c0392b;
}
</style>
