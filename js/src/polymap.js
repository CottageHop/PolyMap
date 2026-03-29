/**
 * PolyMap - 3D Map Renderer
 * Lightweight JS wrapper around the PolyMap WASM engine.
 */

let wasmModule = null;

/**
 * Initialize the WASM module. Called automatically by createPolyMap.
 */
export async function initWasm(wasmUrl) {
  if (wasmModule) return wasmModule;
  if (typeof window === 'undefined') return null; // SSR guard

  const mod = await import(wasmUrl || '/polymap/polymap.js');
  await mod.default();
  wasmModule = mod;
  return mod;
}

/**
 * Create a PolyMap instance inside a container element.
 * @param {string|HTMLElement} container
 * @param {object} options
 * @returns {Promise<PolyMapInstance>}
 */
export async function createPolyMap(container, options = {}) {
  if (typeof window === 'undefined') {
    throw new Error('PolyMap: cannot run in SSR environment');
  }

  const el = typeof container === 'string'
    ? document.querySelector(container)
    : container;

  if (!el) throw new Error(`PolyMap: container "${container}" not found`);

  // Create canvas if not present
  let canvas = el.querySelector('canvas');
  if (!canvas) {
    canvas = document.createElement('canvas');
    canvas.style.width = '100%';
    canvas.style.height = '100%';
    canvas.style.display = 'block';
    el.appendChild(canvas);
  }

  // Set canvas ID for WASM to find
  const canvasId = canvas.id || `polymap-canvas-${Date.now()}`;
  canvas.id = canvasId;

  // Set canvas backing store to native resolution
  // Retry until the element has layout dimensions
  const dpr = window.devicePixelRatio || 1;
  await ensureCanvasSize(canvas, dpr);

  // Create marker overlay container
  let markerContainer = el.querySelector('.polymap-markers');
  if (!markerContainer) {
    markerContainer = document.createElement('div');
    markerContainer.className = 'polymap-markers';
    markerContainer.style.cssText =
      'position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;overflow:hidden;';
    el.style.position = el.style.position || 'relative';
    el.appendChild(markerContainer);
  }

  // Init WASM
  await initWasm(options.wasmUrl);

  // Create the WASM map instance
  const wasmMap = new wasmModule.PolyMap({
    canvas: `#${canvasId}`,
    center: options.center,
    bbox: options.bbox,
    zoom: options.zoom,
    tilt: options.tilt,
    data_url: options.dataUrl,
    api_base: options.apiBase,
    api_key: options.apiKey,
    layers: options.layers,
  });

  // Set up Web Worker pool for off-main-thread tile processing
  const workerCount = Math.min(navigator.hardwareConcurrency || 4, 4);
  const workers = [];
  const pendingTiles = new Map(); // id -> { col, row }
  let tileIdCounter = 0;

  for (let i = 0; i < workerCount; i++) {
    const w = new Worker('/polymap/tile-worker.js', { type: 'module' });
    w.busy = false;
    w.onmessage = (e) => {
      const { type, id, error, empty, vertices, indices, shadowVertices, shadowIndices, labels, z14Tile } = e.data;
      if (type === 'ready') return;
      if (type === 'error') {
        console.warn('[PolyMap Worker] Error:', error);
      }
      w.busy = false;

      if (type === 'result' && id != null) {
        const tile = pendingTiles.get(id);
        pendingTiles.delete(id);
        if (tile && !empty && vertices && indices) {
          wasmMap.pushTileData(
            tile.col, tile.row,
            new Float32Array(vertices.buffer || vertices),
            new Uint32Array(indices.buffer || indices),
            new Float32Array(shadowVertices?.buffer || shadowVertices || []),
            new Uint32Array(shadowIndices?.buffer || shadowIndices || []),
            labels || '[]',
            z14Tile || '0,0',
          );
        }
      }

      // Process queued work
      drainQueue();
    };
    w.postMessage({ type: 'init' });
    workers.push(w);
  }

  const tileQueue = [];

  function drainQueue() {
    while (tileQueue.length > 0) {
      const idle = workers.find(w => !w.busy);
      if (!idle) break;
      const job = tileQueue.shift();
      idle.busy = true;
      idle.postMessage(job);
    }
  }

  // Wire up the tile callback so WASM dispatches to workers instead of main thread
  const pmtilesUrl = options.pmtilesUrl || '';
  if (pmtilesUrl) {
    wasmMap.setTileCallback((col, row, south, west, north, east, centerLat, centerLon, detail) => {
      const id = tileIdCounter++;
      pendingTiles.set(id, { col, row });

      // Convert lat/lon to z14 tile coordinates
      const midLat = (south + north) / 2;
      const midLon = (west + east) / 2;
      const z = 14;
      const n = Math.pow(2, z);
      const tx = Math.floor((midLon + 180) / 360 * n);
      const latRad = midLat * Math.PI / 180;
      const ty = Math.floor((1 - Math.log(Math.tan(latRad) + 1 / Math.cos(latRad)) / Math.PI) / 2 * n);

      const job = {
        type: 'process',
        id,
        url: pmtilesUrl,
        z,
        tx,
        ty,
        centerLat,
        centerLon,
        detail,
      };

      const idle = workers.find(w => !w.busy);
      if (idle) {
        idle.busy = true;
        idle.postMessage(job);
      } else {
        tileQueue.push(job);
      }
    });
  }
  // If no pmtilesUrl, WASM falls back to main-thread API fetch

  // Handle resize — keep canvas backing store at native resolution
  let destroyed = false;
  let resizeObserver = null;

  if (typeof ResizeObserver !== 'undefined') {
    resizeObserver = new ResizeObserver(() => {
      if (destroyed) return;
      const currentDpr = window.devicePixelRatio || 1;
      const w = Math.round(el.clientWidth * currentDpr);
      const h = Math.round(el.clientHeight * currentDpr);
      if (w > 0 && h > 0) {
        canvas.width = w;
        canvas.height = h;
      }
    });
    resizeObserver.observe(el);
  }

  const instance = new PolyMapInstance(
    wasmMap, canvas, markerContainer, el, resizeObserver, () => { destroyed = true; }
  );
  return instance;
}

/**
 * Ensure canvas has physical pixel dimensions. Retries until layout is ready.
 */
function ensureCanvasSize(canvas, dpr, maxAttempts = 10) {
  return new Promise((resolve) => {
    let attempts = 0;
    const check = () => {
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      if (w > 0 && h > 0) {
        canvas.width = Math.round(w * dpr);
        canvas.height = Math.round(h * dpr);
        resolve();
      } else if (attempts < maxAttempts) {
        attempts++;
        requestAnimationFrame(check);
      } else {
        // Fallback: use parent dimensions or defaults
        const parent = canvas.parentElement;
        const fw = parent?.clientWidth || 1280;
        const fh = parent?.clientHeight || 800;
        canvas.width = Math.round(fw * dpr);
        canvas.height = Math.round(fh * dpr);
        resolve();
      }
    };
    check();
  });
}

class PolyMapInstance {
  constructor(wasmMap, canvas, markerContainer, rootEl, resizeObserver, onDestroy) {
    this._wasm = wasmMap;
    this._canvas = canvas;
    this._markerContainer = markerContainer;
    this._rootEl = rootEl;
    this._resizeObserver = resizeObserver;
    this._onDestroy = onDestroy;
    this._markers = new Map();
    this._listeners = new Map();
    this._destroyed = false;
    this._wasmEventsBound = new Set();

    // Wire up marker position updates
    this._wasm.setMarkerCallback((positions) => {
      if (!this._destroyed) this._updateMarkerPositions(positions);
    });
  }

  // ── Camera ──────────────────────────────────────────────────────

  setView(lat, lon, zoom) {
    this._wasm.setView(lat, lon, zoom ?? undefined);
  }

  setZoom(zoom) {
    this._wasm.setZoom(zoom);
  }

  setTilt(tilt) {
    this._wasm.setTilt(tilt);
  }

  panBy(dx, dy) {
    this._wasm.panBy(dx, dy);
  }

  getCamera() {
    return this._wasm.getCamera();
  }

  // ── Markers ─────────────────────────────────────────────────────

  addMarker(id, lat, lon, options = {}) {
    if (this._markers.has(id)) {
      this.removeMarker(id);
    }

    const el = document.createElement('div');
    el.className = `polymap-marker ${options.className || ''}`.trim();
    el.style.cssText = `
      position: absolute;
      transform: translate(-50%, -100%);
      pointer-events: auto;
      cursor: pointer;
      opacity: 0;
      will-change: left, top;
      transition: opacity 0.15s;
    `;
    el.setAttribute('data-marker-id', id);

    if (options.html) {
      el.innerHTML = options.html;
    }

    const clickHandler = (e) => {
      e.stopPropagation();
      this._emit('marker:click', { id, lat, lon, element: el });
    };
    el.addEventListener('click', clickHandler);

    this._markerContainer.appendChild(el);
    this._markers.set(id, { id, lat, lon, element: el, clickHandler });
    this._wasm.addMarker(id, lat, lon);

    return el;
  }

  removeMarker(id) {
    const marker = this._markers.get(id);
    if (marker) {
      marker.element.removeEventListener('click', marker.clickHandler);
      marker.element.remove();
      this._markers.delete(id);
      this._wasm.removeMarker(id);
    }
  }

  clearMarkers() {
    for (const [, marker] of this._markers) {
      marker.element.removeEventListener('click', marker.clickHandler);
      marker.element.remove();
    }
    this._markers.clear();
    this._wasm.clearMarkers();
  }

  getMarkerElement(id) {
    return this._markers.get(id)?.element || null;
  }

  // ── Textures ────────────────────────────────────────────────────

  /**
   * Set a tiled background texture from an image URL.
   * Fetches the image, decodes to RGBA, and uploads to the GPU.
   * @param {string} url - URL of the image to use as background texture
   * @returns {Promise<void>}
   */
  async setBackgroundTexture(url) {
    const res = await fetch(url);
    const blob = await res.blob();
    const bitmap = await createImageBitmap(blob);

    const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(bitmap, 0, 0);
    const imageData = ctx.getImageData(0, 0, bitmap.width, bitmap.height);

    this._wasm.setBackgroundTexture(bitmap.width, bitmap.height, imageData.data);
    bitmap.close();
  }

  /**
   * Set the cloud atlas texture from an image URL.
   * Fetches the image, decodes to RGBA, and uploads to the GPU.
   * @param {string} url - URL of the cloud atlas image (2048x1024, 4x3 grid)
   * @returns {Promise<void>}
   */
  async setCloudTexture(url) {
    try {
      const res = await fetch(url);
      const blob = await res.blob();
      const bitmap = await createImageBitmap(blob);

      const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
      const ctx = canvas.getContext('2d');
      ctx.drawImage(bitmap, 0, 0);
      const imageData = ctx.getImageData(0, 0, bitmap.width, bitmap.height);

      this._wasm.setCloudTexture(bitmap.width, bitmap.height, imageData.data);
      bitmap.close();
    } catch (e) {
      console.warn('[PolyMap] Failed to load cloud texture:', e);
    }
  }

  /**
   * Upload material textures from an array of image URLs.
   * Loads all images, packs them into a single RGBA buffer, and uploads to GPU.
   * All images must be the same size.
   * @param {string[]} urls - Array of image URLs (one per material layer)
   * @returns {Promise<void>}
   */
  async setMaterialTextures(urls) {
    try {
      const blobs = await Promise.all(urls.map(u => fetch(u).then(r => r.blob())));
      const bitmaps = await Promise.all(blobs.map(b => createImageBitmap(b)));

      const width = bitmaps[0].width;
      const height = bitmaps[0].height;
      const numLayers = bitmaps.length;
      const packed = new Uint8Array(width * height * 4 * numLayers);

      const canvas = new OffscreenCanvas(width, height);
      const ctx = canvas.getContext('2d');

      for (let i = 0; i < numLayers; i++) {
        ctx.clearRect(0, 0, width, height);
        ctx.drawImage(bitmaps[i], 0, 0);
        const imageData = ctx.getImageData(0, 0, width, height);
        packed.set(imageData.data, i * width * height * 4);
        bitmaps[i].close();
      }

      this._wasm.setMaterialTextures(width, height, numLayers, packed);
    } catch (e) {
      console.warn('[PolyMap] Failed to load material textures:', e);
    }
  }

  // ── Layers ──────────────────────────────────────────────────────

  setLayerVisible(layer, visible) {
    this._wasm.setLayerVisible(layer, visible);
  }

  // ── Events ──────────────────────────────────────────────────────

  on(event, callback) {
    if (!this._listeners.has(event)) {
      this._listeners.set(event, new Set());
    }
    this._listeners.get(event).add(callback);

    // Bind WASM event once per event type
    if (!this._wasmEventsBound.has(event) &&
        ['ready', 'camera:move', 'click', 'resize'].includes(event)) {
      this._wasmEventsBound.add(event);
      this._wasm.on(event, (...args) => {
        if (!this._destroyed) this._emit(event, ...args);
      });
    }

    return this;
  }

  off(event, callback) {
    const listeners = this._listeners.get(event);
    if (listeners) {
      callback ? listeners.delete(callback) : listeners.clear();
    }
    return this;
  }

  destroy() {
    if (this._destroyed) return;
    this._destroyed = true;
    this._onDestroy?.();
    this._resizeObserver?.disconnect();
    this.clearMarkers();
    this._markerContainer?.remove();
    this._listeners.clear();
    this._wasmEventsBound.clear();
  }

  // ── Internal ────────────────────────────────────────────────────

  _emit(event, ...args) {
    const listeners = this._listeners.get(event);
    if (listeners) {
      for (const cb of listeners) {
        try { cb(...args); } catch (e) { console.error(`PolyMap [${event}]:`, e); }
      }
    }
  }

  _updateMarkerPositions(positions) {
    for (const { id, x, y, visible } of positions) {
      const marker = this._markers.get(id);
      if (!marker) continue;
      const el = marker.element;
      el.style.left = `${x}px`;
      el.style.top = `${y}px`;
      el.style.opacity = visible ? '1' : '0';
      el.style.pointerEvents = visible ? 'auto' : 'none';
    }
  }
}

export default createPolyMap;
