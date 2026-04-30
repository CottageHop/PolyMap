//! PolyMap — 3D map renderer + tile data pipeline, compiled to WebAssembly.
//!
//! Two cargo features select which build you get:
//!   * `renderer` (default) — full WebGPU-based 3D map renderer.
//!   * `worker` — slim build that exposes `process_tile()` for use in a
//!     Web Worker (no wgpu/winit/fontdue dependencies).
//!
//! See `docs/library-extraction.md` for the design.

// --- Always-compiled data modules ---------------------------------------
// These contain pure data structures (MapVertex, Label, etc.), MVT decoding,
// MapData generation, and PMTiles HTTP fetching. They have no graphics deps.

pub mod config;
pub mod mapdata;
pub mod mvt;
pub mod mvt_convert;

#[cfg(target_arch = "wasm32")]
pub mod pmtiles;

// --- Renderer-only modules ----------------------------------------------
// These pull in wgpu/winit/fontdue and depend on a GPU context. Gated so the
// worker build doesn't compile (or download) any of it.

#[cfg(feature = "renderer")]
pub mod camera;
#[cfg(feature = "renderer")]
pub mod cars;
#[cfg(feature = "renderer")]
pub mod gpu;
#[cfg(feature = "renderer")]
pub mod noise;
#[cfg(feature = "renderer")]
pub mod renderer;
#[cfg(feature = "renderer")]
pub mod text;
#[cfg(feature = "renderer")]
pub mod texture;
#[cfg(feature = "renderer")]
pub mod tiles;

#[cfg(all(feature = "renderer", target_arch = "wasm32"))]
pub mod api;

// `app` holds the App struct + ApplicationHandler implementation + the
// renderer wasm-bindgen exports. Renderer-only.
#[cfg(feature = "renderer")]
pub mod app;

#[cfg(feature = "renderer")]
pub use app::*;

// --- Worker-only module --------------------------------------------------
// Exposes `process_tile()` via wasm-bindgen for the Web Worker context.

#[cfg(all(feature = "worker", target_arch = "wasm32"))]
pub mod worker;

// --- Shared HTTP fetch helper -------------------------------------------
// Used by `pmtiles` for HTTP Range requests. Uses the global `fetch()` so it
// works in both Window and Worker scopes.

#[cfg(target_arch = "wasm32")]
pub(crate) async fn wasm_fetch_bytes(url: &str, start: u64, end: u64) -> Option<Vec<u8>> {
    use wasm_bindgen::JsCast;
    use web_sys::{Request, RequestInit, RequestMode, Response};

    let opts = RequestInit::new();
    opts.set_method("GET");
    opts.set_mode(RequestMode::Cors);

    let request = Request::new_with_str_and_init(url, &opts).ok()?;
    request
        .headers()
        .set("Range", &format!("bytes={}-{}", start, end - 1))
        .ok()?;

    // Global fetch — works in both Window and Worker scopes.
    let global = js_sys::global();
    let fetch_fn = js_sys::Reflect::get(&global, &wasm_bindgen::JsValue::from_str("fetch")).ok()?;
    let fetch_fn: js_sys::Function = fetch_fn.dyn_into().ok()?;
    let resp_promise = fetch_fn.call1(&global, &request).ok()?;
    let resp_value =
        wasm_bindgen_futures::JsFuture::from(js_sys::Promise::from(resp_promise))
            .await
            .ok()?;
    let resp: Response = resp_value.dyn_into().ok()?;

    let status = resp.status();
    if status != 200 && status != 206 {
        return None;
    }

    let buf = wasm_bindgen_futures::JsFuture::from(resp.array_buffer().ok()?)
        .await
        .ok()?;
    let array = js_sys::Uint8Array::new(&buf);
    let bytes = array.to_vec();

    let expected = (end - start) as usize;
    if status == 206 && bytes.len() != expected {
        return None; // truncated
    }

    Some(bytes)
}
