//! Worker-mode entry points.
//!
//! Built when the `worker` cargo feature is enabled — the same crate also
//! produces the renderer WASM under the `renderer` feature. This module
//! exposes only the data-decoding API (`process_tile`) needed by a Web
//! Worker for off-main-thread tile processing.

#![cfg(all(feature = "worker", target_arch = "wasm32"))]

use js_sys::{Float32Array, Uint32Array};
use wasm_bindgen::prelude::*;

use crate::mapdata;
use crate::mvt;
use crate::mvt_convert;
use crate::pmtiles;

/// Initialize the worker WASM module.
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
#[wasm_bindgen]
pub async fn process_tile(
    url: &str,
    z: u8,
    tx: u32,
    ty: u32,
    center_lat: f64,
    center_lon: f64,
    detail: u8,
    _parcels_url: JsValue,
) -> JsValue {
    match process_tile_inner(url, z, tx, ty, center_lat, center_lon, detail).await {
        Some(result) => result,
        None => JsValue::NULL,
    }
}

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

    let mvt_bytes = pmtiles::get_tile(url, z, tx, ty).await?;

    let tile = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        mvt::decode_tile(&mvt_bytes)
    }))
    .ok()?;

    let map_data = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        mvt_convert::mvt_to_mapdata(&tile, z, tx, ty, center_lat, center_lon, detail_level)
    }))
    .ok()?;

    let result = js_sys::Object::new();

    set_vertex_array(&result, "vertices", &map_data.vertices);
    set_uint32_array(&result, "indices", &map_data.indices);
    set_vertex_array(&result, "shadowVertices", &map_data.shadow_vertices);
    set_uint32_array(&result, "shadowIndices", &map_data.shadow_indices);

    let labels_json = serde_json::to_string(&map_data.labels).unwrap_or_default();
    js_sys::Reflect::set(&result, &"labels".into(), &JsValue::from_str(&labels_json)).ok();

    set_noise_sources(&result, "noiseSources", &map_data.noise_sources);

    js_sys::Reflect::set(
        &result,
        &"z14Tile".into(),
        &JsValue::from_str(&format!("{},{}", tx, ty)),
    )
    .ok();

    Some(result.into())
}

fn set_vertex_array(obj: &js_sys::Object, key: &str, vertices: &[mapdata::MapVertex]) {
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

fn set_noise_sources(obj: &js_sys::Object, key: &str, sources: &[mapdata::NoiseSource]) {
    if sources.is_empty() {
        js_sys::Reflect::set(obj, &key.into(), &JsValue::NULL).ok();
        return;
    }
    let bytes: &[u8] = bytemuck::cast_slice(sources);
    let floats: &[f32] = bytemuck::cast_slice(bytes);
    let arr = Float32Array::new_with_length(floats.len() as u32);
    arr.copy_from(floats);
    js_sys::Reflect::set(obj, &key.into(), &arr).ok();
}

fn set_uint32_array(obj: &js_sys::Object, key: &str, indices: &[u32]) {
    if indices.is_empty() {
        js_sys::Reflect::set(obj, &key.into(), &JsValue::NULL).ok();
        return;
    }
    let arr = Uint32Array::new_with_length(indices.len() as u32);
    arr.copy_from(indices);
    js_sys::Reflect::set(obj, &key.into(), &arr).ok();
}
