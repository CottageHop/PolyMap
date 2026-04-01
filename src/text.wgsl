// PolyMap — Text Label Shader (3D-aware)
// Renders glyph quads anchored in world space but sized in screen pixels.

struct CameraUniform {
    view_proj: mat4x4<f32>,
    position: vec2<f32>,
    viewport: vec2<f32>,
    zoom: f32,
    time: f32,
    tilt: f32,
    cloud_opacity: f32,
    cloud_speed: f32,
    _pad2a: f32,
    _pad2b: f32,
    _pad2c: f32,
    water_tint: vec4<f32>,
    park_tint: vec4<f32>,
    building_tint: vec4<f32>,
    road_tint: vec4<f32>,
    land_tint: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

@group(1) @binding(0)
var glyph_atlas: texture_2d<f32>;
@group(1) @binding(1)
var atlas_sampler: sampler;

struct VertexInput {
    @location(0) world_pos: vec2<f32>,
    @location(1) pixel_offset: vec2<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    // Transform world anchor through the 3D view-projection
    let anchor_clip = camera.view_proj * vec4<f32>(in.world_pos.x, in.world_pos.y, 0.0, 1.0);
    let ndc = anchor_clip.xy / anchor_clip.w;

    // Scale labels by zoom level. The pixel_to_ndc conversion already accounts for
    // viewport size (larger viewport = smaller NDC per pixel), so we only need zoom scaling.
    // Use min dimension as reference to keep labels consistent regardless of aspect ratio.
    let ref_size = min(camera.viewport.x, camera.viewport.y);
    let zoom_scale = pow(2.0, camera.zoom * 0.5) * (1000.0 / ref_size);
    let pixel_to_ndc = vec2<f32>(2.0 / camera.viewport.x, -2.0 / camera.viewport.y) * zoom_scale;
    let final_pos = ndc + in.pixel_offset * pixel_to_ndc;

    // Place text on top (z near camera)
    out.clip_position = vec4<f32>(final_pos, 0.0, 1.0);
    out.uv = in.uv;
    out.color = in.color;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let coverage = textureSample(glyph_atlas, atlas_sampler, in.uv).r;
    let alpha = smoothstep(0.1, 0.6, coverage) * in.color.a;

    if alpha < 0.01 {
        discard;
    }

    return vec4<f32>(in.color.rgb, alpha);
}
