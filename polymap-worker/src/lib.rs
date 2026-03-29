//! PolyMap Worker — WASM module for off-main-thread tile processing.
//!
//! This crate contains the computation-only parts of PolyMap:
//! PMTiles reading, MVT decoding, and MapData geometry generation.
//! It runs in a Web Worker and returns raw vertex/index arrays
//! that the main thread uploads to GPU.

pub mod mapdata;
pub mod mvt;
pub mod mvt_convert;
pub mod pmtiles;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use js_sys::{Float32Array, Uint32Array};

/// Initialize the worker WASM module.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub fn worker_start() {
    console_error_panic_hook::set_once();
    let _ = console_log::init_with_level(log::Level::Warn);
}

/// Process a single tile: fetch from PMTiles, decode MVT, generate geometry.
///
/// Returns a JS object with Float32Array/Uint32Array fields for vertices,
/// indices, shadow geometry, and serialized labels. Returns null if the
/// tile doesn't exist or processing fails.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub async fn process_tile(
    url: &str,
    z: u8,
    tx: u32,
    ty: u32,
    center_lat: f64,
    center_lon: f64,
    detail: u8,
    parcels_url: JsValue,
) -> JsValue {
    match process_tile_inner(url, z, tx, ty, center_lat, center_lon, detail).await {
        Some(result) => result,
        None => JsValue::NULL,
    }
}

#[cfg(target_arch = "wasm32")]
async fn process_tile_inner(
    url: &str,
    z: u8,
    tx: u32,
    ty: u32,
    center_lat: f64,
    center_lon: f64,
    detail: u8,
) -> Option<JsValue> {
    use mvt_convert::DetailLevel;

    let detail_level = match detail {
        0 => DetailLevel::Low,
        1 => DetailLevel::Medium,
        _ => DetailLevel::High,
    };

    // Fetch MVT tile from PMTiles
    let mvt_bytes = pmtiles::get_tile(url, z, tx, ty).await?;

    // Decode MVT — catch panics from malformed data
    let tile = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        mvt::decode_tile(&mvt_bytes)
    })).ok()?;

    // Convert to MapData
    let map_data = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        mvt_convert::mvt_to_mapdata(
            &tile, z, tx, ty, center_lat, center_lon, detail_level,
        )
    })).ok()?;

    // Serialize MapVertex arrays as raw bytes for zero-copy transfer
    let result = js_sys::Object::new();

    // Main geometry
    set_typed_array(&result, "vertices", &map_data.vertices);
    set_uint32_array(&result, "indices", &map_data.indices);

    // Shadow geometry
    set_typed_array(&result, "shadowVertices", &map_data.shadow_vertices);
    set_uint32_array(&result, "shadowIndices", &map_data.shadow_indices);

    // Labels as JSON
    let labels_json = serde_json::to_string(&map_data.labels).unwrap_or_default();
    js_sys::Reflect::set(
        &result,
        &"labels".into(),
        &JsValue::from_str(&labels_json),
    ).ok();

    // z14 tile coords for dedup
    js_sys::Reflect::set(&result, &"z14Tile".into(), &JsValue::from_str(
        &format!("{},{}", tx, ty),
    )).ok();

    Some(result.into())
}

/// Convert MapVertex slice to Float32Array (reinterpret as f32 array).
#[cfg(target_arch = "wasm32")]
fn set_typed_array(obj: &js_sys::Object, key: &str, vertices: &[mapdata::MapVertex]) {
    if vertices.is_empty() {
        js_sys::Reflect::set(obj, &key.into(), &JsValue::NULL).ok();
        return;
    }
    let bytes: &[u8] = bytemuck::cast_slice(vertices);
    let floats: &[f32] = bytemuck::cast_slice(bytes);
    let arr = Float32Array::new_with_length(floats.len() as u32);
    arr.copy_from(floats);
    js_sys::Reflect::set(obj, &key.into(), &arr).ok();
}

/// Convert u32 slice to Uint32Array.
#[cfg(target_arch = "wasm32")]
fn set_uint32_array(obj: &js_sys::Object, key: &str, indices: &[u32]) {
    if indices.is_empty() {
        js_sys::Reflect::set(obj, &key.into(), &JsValue::NULL).ok();
        return;
    }
    let arr = Uint32Array::new_with_length(indices.len() as u32);
    arr.copy_from(indices);
    js_sys::Reflect::set(obj, &key.into(), &arr).ok();
}

/// Fetch a byte range from a URL (used by pmtiles for HTTP Range requests).
/// Uses the global `fetch()` function which works in both Window and Worker contexts.
#[cfg(target_arch = "wasm32")]
pub(crate) async fn wasm_fetch_bytes(url: &str, start: u64, end: u64) -> Option<Vec<u8>> {
    use wasm_bindgen::JsCast;
    use web_sys::{Request, RequestInit, RequestMode, Response};

    let opts = RequestInit::new();
    opts.set_method("GET");
    opts.set_mode(RequestMode::Cors);

    let request = Request::new_with_str_and_init(url, &opts).ok()?;
    request.headers().set("Range", &format!("bytes={}-{}", start, end - 1)).ok()?;

    // Use global fetch() — works in both Window and Worker scopes
    let global = js_sys::global();
    let fetch_fn = js_sys::Reflect::get(&global, &JsValue::from_str("fetch")).ok()?;
    let fetch_fn: js_sys::Function = fetch_fn.dyn_into().ok()?;
    let resp_promise = fetch_fn.call1(&global, &request).ok()?;
    let resp_value = wasm_bindgen_futures::JsFuture::from(js_sys::Promise::from(resp_promise))
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

    // Validate we got the expected amount of data
    let expected = (end - start) as usize;
    if status == 206 && bytes.len() != expected {
        return None; // truncated response
    }

    Some(bytes)
}
