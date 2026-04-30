// PolyMap — Noise Heat Map Overlay Shader
// Fullscreen pass. For each pixel we:
//   1. Unproject the pixel to world-space (approximate — uses the camera's
//      orthographic extent, same trick as the cloud shader).
//   2. Sum log-power from every noise source: dB_total = 10·log10(Σ 10^(dB_i/10)).
//      Point sources attenuate by inverse-square (−20·log10(d/d_ref)).
//      d_ref is 1 world unit; source.db is the dB value at 1 unit.
//   3. Map the total dB to a color ramp (green → yellow → orange → red).

struct CameraUniform {
    view_proj: mat4x4<f32>,
    position: vec2<f32>,
    viewport: vec2<f32>,
    zoom: f32,
    time: f32,
    tilt: f32,
    cloud_opacity: f32,
    cloud_speed: f32,
    label_alpha: f32,
    _pad2b: f32,
    _pad2c: f32,
    water_tint: vec4<f32>,
    park_tint: vec4<f32>,
    building_tint: vec4<f32>,
    road_tint: vec4<f32>,
    land_tint: vec4<f32>,
    rail_tint: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

// 16-byte aligned per-source record: vec2 pos + f32 db + f32 _pad.
struct NoiseSource {
    pos: vec2<f32>,
    db: f32,
    _pad: f32,
};

// Fixed-size uniform array (128 sources). count tells us how many are live.
struct NoiseUniform {
    count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    sources: array<NoiseSource, 128>,
};

@group(1) @binding(0)
var<uniform> noise_data: NoiseUniform;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    // Fullscreen triangle — oversize, rasterizer clips to the viewport.
    var out: VertexOutput;
    let x = f32(i32(vi & 1u)) * 4.0 - 1.0;
    let y = f32(i32(vi >> 1u)) * 4.0 - 1.0;
    out.clip_position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>(x * 0.5 + 0.5, 1.0 - (y * 0.5 + 0.5));
    return out;
}

// Color ramp: 40 dB (background) → 90 dB (painfully loud).
// Low = transparent green, high = red, mid = yellow-orange.
fn ramp(db: f32) -> vec4<f32> {
    let t = clamp((db - 40.0) / 50.0, 0.0, 1.0);
    // three-stop gradient
    let green = vec3<f32>(0.20, 0.80, 0.35);
    let yellow = vec3<f32>(0.95, 0.85, 0.20);
    let red = vec3<f32>(0.90, 0.15, 0.15);
    var col: vec3<f32>;
    if t < 0.5 {
        col = mix(green, yellow, t * 2.0);
    } else {
        col = mix(yellow, red, (t - 0.5) * 2.0);
    }
    // Alpha grows with noise level so quiet areas are clear.
    let a = clamp(t * 0.7 + 0.1, 0.0, 0.6);
    return vec4<f32>(col, a);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    if noise_data.count == 0u {
        discard;
    }

    // Approximate world position from UV (flat-ortho path, same as clouds).
    let aspect = camera.viewport.x / camera.viewport.y;
    let zoom_scale = 100.0 / pow(2.0, camera.zoom);
    let world_x = camera.position.x + (in.uv.x - 0.5) * zoom_scale * aspect;
    let world_y = camera.position.y + (0.5 - in.uv.y) * zoom_scale;
    let pos = vec2<f32>(world_x, world_y);

    // Sum power contributions + gather per-source ring phase in one loop.
    var total_power = 0.0;
    var ring_accum = 0.0;
    // Tunable ripple params. wavelength = world-units between ring peaks.
    // speed = world-units/sec the rings travel outward.
    // sharpness = higher → thinner bright bands. inv_range = how far rings travel.
    let wavelength = 14.0;
    let speed = 5.0;
    let sharpness = 10.0;

    for (var i = 0u; i < noise_data.count; i = i + 1u) {
        let src = noise_data.sources[i];
        let d = max(distance(pos, src.pos), 1.0);
        let db_here = src.db - 20.0 * log2(d) / log2(10.0);
        // Only include audible contributions (prevents underflow).
        if db_here > 30.0 {
            let weight = pow(10.0, db_here / 10.0);
            total_power = total_power + weight;

            // Per-source phase offset so multiple sources don't pulse in lockstep.
            let phase_offset = src.pos.x * 0.1 + src.pos.y * 0.07;
            // Rings travel OUTWARD over time: as time grows, the peak moves
            // to larger d. `sin(d/λ − time*speed/λ + phase)`.
            let phase = d / wavelength - camera.time * speed / wavelength + phase_offset;
            // Crisp bright band (rather than smooth sine).
            let band = pow(max(0.0, sin(phase * 6.2831853)), sharpness);
            ring_accum = ring_accum + band * weight;
        }
    }
    if total_power < 1.0 {
        discard;
    }
    let total_db = 10.0 * log2(total_power) / log2(10.0);

    // Base heat-map color from summed dB.
    var col = ramp(total_db);

    // Ring highlight — normalize ring_accum against the source power so
    // louder spots get proportionally bright bands, quiet spots don't flash.
    let ring_strength = clamp(ring_accum / max(total_power, 1.0), 0.0, 1.0);
    // Boost alpha + brightness wherever a ring band is passing.
    col = vec4<f32>(col.rgb + vec3<f32>(ring_strength * 0.5), clamp(col.a + ring_strength * 0.35, 0.0, 0.9));
    return col;
}
