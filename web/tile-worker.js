/**
 * PolyMap Tile Worker
 *
 * Runs polymap-worker WASM in a Web Worker context to process
 * PMTiles tiles off the main thread. Receives tile requests from
 * the main thread, fetches MVT data via PMTiles HTTP Range requests,
 * decodes geometry, and returns Float32Array/Uint32Array buffers
 * for GPU upload.
 */

let wasmReady = false;
let pendingJobs = [];

// Load the worker WASM module
import('./worker-pkg/polymap_worker.js').then(async (mod) => {
  await mod.default();
  wasmReady = true;
  self.postMessage({ type: 'ready' });

  // Process any jobs that arrived before WASM was ready
  for (const job of pendingJobs) {
    processJob(mod, job);
  }
  pendingJobs = [];

  self.onmessage = (e) => {
    if (e.data.type === 'process') {
      processJob(mod, e.data);
    }
  };
}).catch((err) => {
  console.error('[TileWorker] Failed to load WASM:', err);
});

self.onmessage = (e) => {
  if (e.data.type === 'init') return;
  if (e.data.type === 'process' && !wasmReady) {
    pendingJobs.push(e.data);
  }
};

async function processJob(mod, job) {
  const { id, url, z, tx, ty, centerLat, centerLon, detail } = job;

  try {
    const result = await mod.process_tile(
      url, z, tx, ty, centerLat, centerLon, detail, undefined
    );

    if (!result) {
      self.postMessage({ type: 'result', id, empty: true });
      return;
    }

    // Extract transferable arrays for zero-copy transfer
    const vertices = result.vertices;
    const indices = result.indices;
    const shadowVertices = result.shadowVertices;
    const shadowIndices = result.shadowIndices;
    const noiseSources = result.noiseSources;
    const labels = result.labels || '[]';
    const z14Tile = result.z14Tile || '0,0';

    const transfer = [];
    if (vertices?.buffer) transfer.push(vertices.buffer);
    if (indices?.buffer) transfer.push(indices.buffer);
    if (shadowVertices?.buffer) transfer.push(shadowVertices.buffer);
    if (shadowIndices?.buffer) transfer.push(shadowIndices.buffer);
    if (noiseSources?.buffer) transfer.push(noiseSources.buffer);

    self.postMessage({
      type: 'result',
      id,
      vertices,
      indices,
      shadowVertices,
      shadowIndices,
      noiseSources,
      labels,
      z14Tile,
    }, transfer);
  } catch (err) {
    self.postMessage({ type: 'error', id, error: String(err) });
  }
}
