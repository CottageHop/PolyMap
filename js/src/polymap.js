/**
 * PolyMap - 3D Map Renderer
 * Lightweight JS wrapper around the PolyMap WASM engine.
 */

let wasmModule = null;

/** Create a canvas for pixel manipulation (Safari fallback for OffscreenCanvas). */
function createPixelCanvas(w, h) {
  if (typeof OffscreenCanvas !== 'undefined') {
    return createPixelCanvas(w, h);
  }
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  return c;
}

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

      const tile = id != null ? pendingTiles.get(id) : null;
      if (tile) pendingTiles.delete(id);

      if (type === 'result' && tile && !empty && vertices && indices) {
        wasmMap.pushTileData(
          tile.col, tile.row,
          new Float32Array(vertices.buffer || vertices),
          new Uint32Array(indices.buffer || indices),
          new Float32Array(shadowVertices?.buffer || shadowVertices || []),
          new Uint32Array(shadowIndices?.buffer || shadowIndices || []),
          labels || '[]',
          z14Tile || '0,0',
        );
      } else if (tile) {
        // Empty or failed — release the in-flight slot
        wasmMap.notifyTileFailed(tile.col, tile.row);
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

  // Show customize controls panel if requested
  if (options.showControls !== false) {
    instance._controls = createControlsPanel(el, instance);
  }

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

    const canvas = createPixelCanvas(bitmap.width, bitmap.height);
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

      const canvas = createPixelCanvas(bitmap.width, bitmap.height);
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

      const canvas = createPixelCanvas(width, height);
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

  setCloudOpacity(opacity) {
    this._wasm.setCloudOpacity(opacity);
  }

  setColors(config) {
    this._wasm.setColors(config);
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

  showControls() {
    if (this._controls) this._controls.style.display = '';
  }

  hideControls() {
    if (this._controls) this._controls.style.display = 'none';
  }

  destroy() {
    if (this._destroyed) return;
    this._destroyed = true;
    this._onDestroy?.();
    this._resizeObserver?.disconnect();
    this.clearMarkers();
    this._markerContainer?.remove();
    this._controls?.remove();
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

/**
 * Create a controls panel for the map instance.
 * @param {HTMLElement} container
 * @param {PolyMapInstance} map
 * @returns {HTMLElement}
 */
function createControlsPanel(container, map) {
  const srgbToLinear = (c) =>
    c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
  const hexToRgba = (hex) => {
    const r = srgbToLinear(parseInt(hex.slice(1, 3), 16) / 255);
    const g = srgbToLinear(parseInt(hex.slice(3, 5), 16) / 255);
    const b = srgbToLinear(parseInt(hex.slice(5, 7), 16) / 255);
    return [r, g, b, 1.0];
  };

  const THEMES = {
    cottagecore: { water:'#99b3a6', park:'#8c9959', building:'#d9b08f', road:'#8c7061', land:'#f2e6d9', marker:'#c0392b', cloudOpacity:50, clouds:true, useDefaults:true },
    cyberpunk:   { water:'#0a1628', park:'#1a0a12', building:'#2a1525', road:'#00d4e8', land:'#0c1020', marker:'#f0c800', cloudOpacity:0, clouds:false },
    modern:      { water:'#42a5f5', park:'#8bc34a', building:'#e0e0e0', road:'#bdbdbd', land:'#f5f5f5', marker:'#1976d2', cloudOpacity:30, clouds:true },
    greyscale:   { water:'#888888', park:'#aaaaaa', building:'#666666', road:'#777777', land:'#f0f0f0', marker:'#444444', cloudOpacity:20, clouds:true },
    dark:        { water:'#1a3a4a', park:'#1e3a1e', building:'#2a2a2a', road:'#5a5a5a', land:'#1a1a1a', marker:'#e0e0e0', cloudOpacity:15, clouds:true },
    eighties:    { water:'#0099dd', park:'#7bef2a', building:'#ff6347', road:'#ff1493', land:'#ffd732', marker:'#0099dd', cloudOpacity:0, clouds:false },
    seventies:   { water:'#4ca8a8', park:'#f7c868', building:'#e87848', road:'#e03030', land:'#fdd998', marker:'#e03030', cloudOpacity:0, clouds:false },
    oldworld:    { water:'#5b7e8a', park:'#6b7c47', building:'#c4a265', road:'#8b4513', land:'#e8d5a3', marker:'#8b0000', cloudOpacity:0, clouds:false },
  };

  function applyTheme(name) {
    const t = THEMES[name];
    if (!t) return;
    for (const [key, id] of [['water','ctrl-water'],['park','ctrl-park'],['building','ctrl-building'],['road','ctrl-road'],['land','ctrl-land'],['marker','ctrl-marker']]) {
      const el = panel.querySelector('#' + id);
      if (el) el.value = t[key];
    }
    panel.querySelector('#ctrl-marker')?.dispatchEvent(new Event('input'));
    if (t.useDefaults) {
      map.setColors({ water:[0,0,0,0], park:[0,0,0,0], building:[0,0,0,0], road:[0,0,0,0], land:[0,0,0,0] });
    } else {
      map.setColors({ water:hexToRgba(t.water), park:hexToRgba(t.park), building:hexToRgba(t.building), road:hexToRgba(t.road), land:hexToRgba(t.land) });
    }
    map.setCloudOpacity(t.cloudOpacity / 100);
    map.setLayerVisible('clouds', t.clouds);
    const opSlider = panel.querySelector('#ctrl-opacity');
    const opVal = panel.querySelector('#ctrl-opacity-val');
    if (opSlider) opSlider.value = t.cloudOpacity;
    if (opVal) opVal.textContent = t.cloudOpacity + '%';
    panel.querySelectorAll('.pm-theme-btn').forEach(b => b.classList.remove('active'));
    panel.querySelector(`.pm-theme-btn[data-theme="${name}"]`)?.classList.add('active');
  }

  const panel = document.createElement('div');
  panel.className = 'polymap-controls collapsed';
  panel.style.cssText = 'position:absolute;top:12px;left:12px;z-index:30;width:280px;background:rgba(255,255,255,0.92);backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);border-radius:12px;box-shadow:0 2px 16px rgba(0,0,0,0.12);overflow:hidden;font:13px -apple-system,BlinkMacSystemFont,sans-serif;color:#333;';

  const style = document.createElement('style');
  style.textContent = `
    .polymap-controls.collapsed .pm-body,.polymap-controls.collapsed .pm-title{display:none}
    .pm-header{display:flex;align-items:center;padding:10px 12px;gap:8px;border-bottom:1px solid rgba(0,0,0,0.06);cursor:pointer;user-select:none}
    .pm-header:hover{background:rgba(0,0,0,0.03)}
    .pm-section{padding:6px 14px 10px}
    .pm-section+.pm-section{border-top:1px solid rgba(0,0,0,0.06)}
    .pm-lbl{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#999;margin-bottom:8px}
    .pm-theme-grid{display:grid;grid-template-columns:1fr 1fr;gap:6px}
    .pm-theme-btn{position:relative;border:2px solid #e0e0e0;border-radius:8px;padding:0;cursor:pointer;overflow:hidden;background:none;aspect-ratio:1.6;transition:border-color .15s}
    .pm-theme-btn:hover{border-color:#aaa}
    .pm-theme-btn.active{border-color:#5a8f5a;box-shadow:0 0 0 2px rgba(90,143,90,0.3)}
    .pm-swatches{display:flex;height:100%}
    .pm-swatches span{flex:1}
    .pm-theme-lbl{position:absolute;bottom:0;left:0;right:0;background:rgba(0,0,0,0.45);color:#fff;font-size:10px;font-weight:600;padding:2px 0;text-align:center}
    .pm-row{display:flex;align-items:center;justify-content:space-between;padding:3px 0}
    .pm-row label{font-size:13px}
    .pm-row input[type=color]{-webkit-appearance:none;appearance:none;width:28px;height:28px;border:2px solid #e0e0e0;border-radius:6px;cursor:pointer;padding:0;background:none}
    .pm-slider-row{display:flex;align-items:center;gap:8px;padding:4px 0}
    .pm-slider-row label{font-size:13px;min-width:54px}
    .pm-slider-row input[type=range]{flex:1;height:4px;-webkit-appearance:none;appearance:none;background:#ddd;border-radius:2px;outline:none}
    .pm-slider-row input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:14px;height:14px;background:#5a8f5a;border-radius:50%;cursor:pointer}
    .pm-slider-val{font-size:12px;color:#888;min-width:30px;text-align:right}
    .pm-reset{display:block;width:100%;margin-top:6px;padding:6px;background:none;border:1px solid #ddd;border-radius:6px;color:#888;font-size:12px;cursor:pointer}
    .pm-reset:hover{background:#f5f5f5;color:#555}
  `;
  panel.appendChild(style);

  const themeButtons = Object.entries(THEMES).map(([name, t]) => {
    const label = { cottagecore:'Cottage Core', cyberpunk:'Cyberpunk', modern:'Modern', greyscale:'Greyscale', dark:'Dark', eighties:"80's", seventies:"70's", oldworld:'Old World' }[name] || name;
    const colors = [t.land, t.building, t.park, t.water, t.road];
    return `<button class="pm-theme-btn${name==='cottagecore'?' active':''}" data-theme="${name}"><div class="pm-swatches">${colors.map(c=>`<span style="background:${c}"></span>`).join('')}</div><div class="pm-theme-lbl">${label}</div></button>`;
  }).join('');

  panel.innerHTML += `
    <div class="pm-header"><div style="font-size:16px;color:#555">&#9881;</div><div class="pm-title" style="font-weight:600;font-size:13px">Customize Map</div></div>
    <div class="pm-body">
      <div class="pm-section"><div class="pm-lbl">Theme</div><div class="pm-theme-grid">${themeButtons}</div></div>
      <div class="pm-section"><div class="pm-lbl">Clouds</div>
        <div class="pm-slider-row"><label>Opacity</label><input type="range" id="ctrl-opacity" min="0" max="100" value="50"><span class="pm-slider-val" id="ctrl-opacity-val">50%</span></div>
        <div class="pm-slider-row"><label>Speed</label><input type="range" id="ctrl-speed" min="0" max="300" value="100"><span class="pm-slider-val" id="ctrl-speed-val">1.0x</span></div>
      </div>
      <div class="pm-section"><div class="pm-lbl">Colors</div>
        <div class="pm-row"><label>Water</label><input type="color" id="ctrl-water" value="#99b3a6"></div>
        <div class="pm-row"><label>Green Space</label><input type="color" id="ctrl-park" value="#8c9959"></div>
        <div class="pm-row"><label>Buildings</label><input type="color" id="ctrl-building" value="#d9b08f"></div>
        <div class="pm-row"><label>Roads</label><input type="color" id="ctrl-road" value="#8c7061"></div>
        <div class="pm-row"><label>Background</label><input type="color" id="ctrl-land" value="#f2e6d9"></div>
        <div class="pm-row"><label>Markers</label><input type="color" id="ctrl-marker" value="#c0392b"></div>
        <button class="pm-reset">Reset Colors</button>
      </div>
    </div>`;

  // Toggle collapse
  panel.querySelector('.pm-header').addEventListener('click', () => panel.classList.toggle('collapsed'));

  // Theme buttons
  panel.querySelectorAll('.pm-theme-btn').forEach(btn => {
    btn.addEventListener('click', () => applyTheme(btn.dataset.theme));
  });

  // Cloud sliders
  panel.querySelector('#ctrl-opacity')?.addEventListener('input', (e) => {
    map.setCloudOpacity(e.target.value / 100);
    panel.querySelector('#ctrl-opacity-val').textContent = e.target.value + '%';
  });
  panel.querySelector('#ctrl-speed')?.addEventListener('input', (e) => {
    const v = e.target.value / 100;
    map.setCloudSpeed(v);
    panel.querySelector('#ctrl-speed-val').textContent = v.toFixed(1) + 'x';
  });

  // Color pickers
  for (const [id, key] of [['ctrl-water','water'],['ctrl-park','park'],['ctrl-building','building'],['ctrl-road','road'],['ctrl-land','land']]) {
    panel.querySelector('#' + id)?.addEventListener('input', (e) => {
      map.setColors({ [key]: hexToRgba(e.target.value) });
    });
  }
  panel.querySelector('#ctrl-marker')?.addEventListener('input', (e) => {
    document.documentElement.style.setProperty('--marker-color', e.target.value);
  });

  // Reset
  panel.querySelector('.pm-reset')?.addEventListener('click', () => applyTheme('cottagecore'));

  container.style.position = container.style.position || 'relative';
  container.appendChild(panel);
  return panel;
}

export default createPolyMap;
