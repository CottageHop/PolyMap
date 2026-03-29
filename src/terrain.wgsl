// PolyMap — Ray Marched Terrain Shader
// Renders a procedural terrain from a top-down map perspective using
// ray marching on the GPU. No tiles, no LOD popping — infinite smooth zoom.

struct CameraUniform {
    position: vec2<f32>,
    viewport: vec2<f32>,
    zoom: f32,
    time: f32,
    _padding: vec2<f32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// Full-screen triangle (no vertex buffer needed)
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    // Generate a full-screen triangle from vertex index
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index & 2u) * 2 - 1);
    out.clip_position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

// --- Noise functions for procedural terrain ---

fn hash2(p: vec2<f32>) -> f32 {
    let h = dot(p, vec2<f32>(127.1, 311.7));
    return fract(sin(h) * 43758.5453123);
}

fn hash2v(p: vec2<f32>) -> vec2<f32> {
    let h = vec2<f32>(dot(p, vec2<f32>(127.1, 311.7)), dot(p, vec2<f32>(269.5, 183.3)));
    return fract(sin(h) * 43758.5453123);
}

// Smooth value noise
fn noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f); // smoothstep

    let a = hash2(i + vec2<f32>(0.0, 0.0));
    let b = hash2(i + vec2<f32>(1.0, 0.0));
    let c = hash2(i + vec2<f32>(0.0, 1.0));
    let d = hash2(i + vec2<f32>(1.0, 1.0));

    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// Fractal Brownian Motion — multi-octave noise for natural terrain
fn fbm(p_in: vec2<f32>) -> f32 {
    var p = p_in;
    var value = 0.0;
    var amplitude = 0.5;
    var frequency = 1.0;

    // 6 octaves for detailed terrain
    for (var i = 0; i < 6; i++) {
        value += amplitude * noise(p * frequency);
        frequency *= 2.0;
        amplitude *= 0.5;
        // Rotate each octave to break grid alignment
        p = vec2<f32>(
            p.x * 0.866 - p.y * 0.5,
            p.x * 0.5 + p.y * 0.866
        );
    }
    return value;
}

// Voronoi / cellular noise for coastline detail
fn voronoi(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    var min_dist = 1.0;

    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let neighbor = vec2<f32>(f32(x), f32(y));
            let point = hash2v(i + neighbor);
            let diff = neighbor + point - f;
            let dist = length(diff);
            min_dist = min(min_dist, dist);
        }
    }
    return min_dist;
}

// --- Terrain evaluation ---

// Returns the terrain height at a world position
fn terrain_height(world_pos: vec2<f32>) -> f32 {
    let scale = 0.003; // Base scale of terrain features
    let p = world_pos * scale;

    // Large-scale continental shapes
    var h = fbm(p * 1.0) * 1.0;

    // Medium features (mountains, valleys)
    h += fbm(p * 3.0 + vec2<f32>(100.0, 200.0)) * 0.3;

    // Fine detail (ridges, erosion)
    h += fbm(p * 8.0 + vec2<f32>(50.0, 80.0)) * 0.08;

    // Voronoi for interesting coastline shapes
    h += voronoi(p * 2.0) * 0.15;

    return h;
}

// Sea level threshold — below this is water
const SEA_LEVEL: f32 = 0.55;

// --- Coloring ---

fn terrain_color(height: f32, world_pos: vec2<f32>) -> vec3<f32> {
    let h = height;

    // Water
    if h < SEA_LEVEL {
        let depth = (SEA_LEVEL - h) / SEA_LEVEL;
        let shallow = vec3<f32>(0.15, 0.55, 0.72);
        let deep = vec3<f32>(0.05, 0.12, 0.28);
        let water = mix(shallow, deep, smoothstep(0.0, 0.5, depth));

        // Subtle wave pattern
        let wave = sin(world_pos.x * 0.1 + camera.time * 0.5) *
                   cos(world_pos.y * 0.08 + camera.time * 0.3) * 0.02;
        return water + vec3<f32>(wave);
    }

    // Beach / sand
    if h < SEA_LEVEL + 0.02 {
        let t = (h - SEA_LEVEL) / 0.02;
        let sand = vec3<f32>(0.82, 0.78, 0.62);
        let grass_low = vec3<f32>(0.28, 0.52, 0.22);
        return mix(sand, grass_low, smoothstep(0.0, 1.0, t));
    }

    // Lowlands (green)
    if h < SEA_LEVEL + 0.15 {
        let t = (h - SEA_LEVEL - 0.02) / 0.13;
        let grass_low = vec3<f32>(0.28, 0.52, 0.22);
        let grass_high = vec3<f32>(0.22, 0.42, 0.18);
        // Add noise variation to grass
        let variation = noise(world_pos * 0.05) * 0.08;
        return mix(grass_low, grass_high, t) + vec3<f32>(variation, variation * 0.5, 0.0);
    }

    // Highlands / rock
    if h < SEA_LEVEL + 0.35 {
        let t = (h - SEA_LEVEL - 0.15) / 0.2;
        let rock_low = vec3<f32>(0.42, 0.38, 0.32);
        let rock_high = vec3<f32>(0.55, 0.52, 0.48);
        return mix(rock_low, rock_high, t);
    }

    // Snow-capped peaks
    let t = smoothstep(SEA_LEVEL + 0.35, SEA_LEVEL + 0.45, h);
    let rock = vec3<f32>(0.55, 0.52, 0.48);
    let snow = vec3<f32>(0.92, 0.94, 0.96);
    return mix(rock, snow, t);
}

// --- Lighting ---

fn calculate_normal(pos: vec2<f32>) -> vec3<f32> {
    let eps = 0.5; // Sample distance for normal calculation
    let hL = terrain_height(pos - vec2<f32>(eps, 0.0));
    let hR = terrain_height(pos + vec2<f32>(eps, 0.0));
    let hD = terrain_height(pos - vec2<f32>(0.0, eps));
    let hU = terrain_height(pos + vec2<f32>(0.0, eps));
    return normalize(vec3<f32>(hL - hR, 2.0 * eps * 0.003, hD - hU));
}

// --- Main fragment shader ---

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Convert pixel coordinates to world coordinates
    let aspect = camera.viewport.x / camera.viewport.y;
    let zoom_scale = 100.0 / pow(2.0, camera.zoom);

    // UV centered on screen, scaled by zoom
    let uv = (in.uv - vec2<f32>(0.5)) * vec2<f32>(aspect, 1.0) * zoom_scale;
    let world_pos = uv + camera.position;

    // Evaluate terrain
    let height = terrain_height(world_pos);
    var color = terrain_color(height, world_pos);

    // Simple top-down lighting (sun from top-left)
    if height >= SEA_LEVEL {
        let normal = calculate_normal(world_pos);
        let sun_dir = normalize(vec3<f32>(0.4, 1.0, 0.3));
        let diffuse = max(dot(normal, sun_dir), 0.0);
        let ambient = 0.35;
        color *= ambient + diffuse * (1.0 - ambient);
    }

    // SDF-based grid lines (visible at medium zoom levels)
    if camera.zoom > 4.0 && camera.zoom < 12.0 {
        let grid_size = 10.0;
        let grid = abs(fract(world_pos / grid_size + 0.5) - 0.5) * grid_size;
        let grid_dist = min(grid.x, grid.y);
        let line_width = zoom_scale * 0.002;
        let grid_alpha = 1.0 - smoothstep(0.0, line_width, grid_dist);
        let grid_opacity = smoothstep(4.0, 6.0, camera.zoom) * (1.0 - smoothstep(10.0, 12.0, camera.zoom));
        color = mix(color, vec3<f32>(0.0, 0.0, 0.0), grid_alpha * 0.15 * grid_opacity);
    }

    // Subtle vignette
    let vignette_dist = length(in.uv - vec2<f32>(0.5)) * 1.4;
    let vignette = 1.0 - vignette_dist * vignette_dist * 0.3;
    color *= vignette;

    // Tone mapping (simple Reinhard)
    color = color / (color + vec3<f32>(1.0));

    // Gamma correction
    color = pow(color, vec3<f32>(1.0 / 2.2));

    return vec4<f32>(color, 1.0);
}
