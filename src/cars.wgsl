// PolyMap — Moving Cars Shader
// Simple opaque quads in world space, coloured per-car.
// CPU rebuilds the vertex buffer each frame with current positions.

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

struct VertexInput {
    @location(0) world_pos: vec3<f32>,
    @location(1) color: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(in.world_pos, 1.0);
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}
