// PolyMap — 3D Procedurally Textured Map Shader
// Materials: 0=default, 1=asphalt, 2=building roof, 3=grass, 4=water, 5=building wall, 6=tree leaves, 7=tree trunk, 8=cobblestone, 9=glass roof, 10=glass wall, 11=fountain

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
var material_textures: texture_2d_array<f32>;
@group(1) @binding(1)
var material_sampler: sampler;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
    @location(2) material: f32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) world_pos: vec3<f32>,
    @location(2) material: f32,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    out.clip_position = camera.view_proj * vec4<f32>(in.position, 1.0);
    out.world_pos = in.position;
    out.material = in.material;

    // Subtle vertex AO: only brighten roofs, don't darken bases
    // (darkening bases causes visible artifacts at tile boundaries)
    let mat = i32(round(in.material));
    var vc = in.color;
    if mat == 2 || mat == 9 {
        // Roof materials only: slight brightness boost at height
        let ao = smoothstep(0.0, 0.2, in.position.z);
        let ao_factor = mix(1.0, 1.08, ao); // no darkening, just brighten roofs 8%
        vc = vec4<f32>(vc.rgb * ao_factor, vc.a);
    }
    out.color = vc;

    return out;
}

// ============================================================
//  Noise functions
// ============================================================

fn hash21(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * 0.1031);
    p3 += dot(p3, vec3<f32>(p3.y + 33.33, p3.z + 33.33, p3.x + 33.33));
    return fract((p3.x + p3.y) * p3.z);
}

fn hash22(p: vec2<f32>) -> vec2<f32> {
    let p3 = fract(vec3<f32>(p.x, p.y, p.x) * vec3<f32>(0.1031, 0.1030, 0.0973));
    let p4 = p3 + dot(p3, vec3<f32>(p3.y + 33.33, p3.z + 33.33, p3.x + 33.33));
    return fract(vec2<f32>((p4.x + p4.y) * p4.z, (p4.x + p4.z) * p4.y));
}

fn noise2(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = hash21(i);
    let b = hash21(i + vec2<f32>(1.0, 0.0));
    let c = hash21(i + vec2<f32>(0.0, 1.0));
    let d = hash21(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn fbm2(p_in: vec2<f32>) -> f32 {
    var p = p_in;
    var v = 0.5 * noise2(p);
    p = p * 2.0 + vec2<f32>(1.7, 3.1);
    v += 0.25 * noise2(p);
    return v;
}

fn fbm3(p_in: vec2<f32>) -> f32 {
    var p = p_in;
    var v = 0.0;
    var a = 0.5;
    for (var i = 0; i < 3; i++) {
        v += a * noise2(p);
        p = p * 2.0 + vec2<f32>(1.7, 3.1);
        a *= 0.5;
    }
    return v;
}

fn voronoi(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    var min_d = 1.0;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let n = vec2<f32>(f32(x), f32(y));
            let pt = hash22(i + n);
            let diff = n + pt - f;
            min_d = min(min_d, dot(diff, diff));
        }
    }
    return sqrt(min_d);
}

// ============================================================
//  Procedural material textures
// ============================================================

fn tex_asphalt(p: vec2<f32>, base_color: vec3<f32>) -> vec3<f32> {
    let uv = p * 8.0;
    let speckle = noise2(uv * 12.0) * 0.12 - 0.06;
    let roughness = fbm3(uv * 3.0) * 0.08 - 0.04;
    let v = voronoi(uv * 1.5);
    let crack = smoothstep(0.02, 0.06, v);
    let crack_darken = mix(-0.06, 0.0, crack);
    let tar = smoothstep(0.55, 0.6, noise2(uv * 0.8)) * 0.04;
    return base_color + vec3<f32>(speckle + roughness + crack_darken - tar);
}

fn tex_building(p: vec2<f32>, base_color: vec3<f32>) -> vec3<f32> {
    let uv = p * 16.0;
    let row = floor(uv.y);
    var brick_uv = uv;
    if (i32(row) % 2) == 1 { brick_uv.x += 0.5; }
    let brick_x = fract(brick_uv.x);
    let brick_y = fract(uv.y);
    let mortar_x = smoothstep(0.0, 0.04, brick_x) * smoothstep(0.0, 0.04, 1.0 - brick_x);
    let mortar_y = smoothstep(0.0, 0.06, brick_y) * smoothstep(0.0, 0.06, 1.0 - brick_y);
    let mortar = mortar_x * mortar_y;
    let brick_id = floor(brick_uv);
    let brick_hue = hash21(brick_id) * 0.06 - 0.03;
    let surface = noise2(uv * 8.0) * 0.04;
    let mortar_color = base_color + vec3<f32>(0.04, 0.04, 0.03);
    let brick_color = base_color + vec3<f32>(brick_hue + surface);
    return mix(mortar_color, brick_color, mortar);
}

fn tex_building_wall(p: vec2<f32>, z: f32, base_color: vec3<f32>) -> vec3<f32> {
    // Walls use world x/y for horizontal and z for vertical tiling
    let uv = vec2<f32>(p.x + p.y, z) * 16.0;
    let row = floor(uv.y);
    var brick_uv = uv;
    if (i32(row) % 2) == 1 { brick_uv.x += 0.5; }
    let brick_x = fract(brick_uv.x);
    let brick_y = fract(uv.y);
    let mortar_x = smoothstep(0.0, 0.05, brick_x) * smoothstep(0.0, 0.05, 1.0 - brick_x);
    let mortar_y = smoothstep(0.0, 0.07, brick_y) * smoothstep(0.0, 0.07, 1.0 - brick_y);
    let mortar = mortar_x * mortar_y;
    let brick_id = floor(brick_uv);
    let brick_hue = hash21(brick_id) * 0.05 - 0.025;
    let mortar_color = base_color + vec3<f32>(0.03);
    let brick_color = base_color + vec3<f32>(brick_hue);
    return mix(mortar_color, brick_color, mortar);
}

fn tex_grass(p: vec2<f32>, base_color: vec3<f32>) -> vec3<f32> {
    let uv = p * 6.0;
    let patches = fbm2(uv * 0.5) * 0.12;
    let clumps = noise2(uv * 2.0) * 0.08;
    let blades = noise2(uv * 15.0) * 0.06;
    let flower = smoothstep(0.88, 0.9, noise2(uv * 10.0 + vec2<f32>(50.0, 80.0)));
    let flower_color = vec3<f32>(0.08, -0.04, 0.02) * flower;
    let hue_shift = fbm2(uv * 0.3 + vec2<f32>(100.0, 200.0));
    let hue = vec3<f32>((hue_shift - 0.5) * 0.06, 0.0, (0.5 - hue_shift) * 0.04);
    return base_color + vec3<f32>(patches + clumps + blades) + hue + flower_color;
}

fn tex_water(p: vec2<f32>, base_color: vec3<f32>, time: f32) -> vec3<f32> {
    let uv = p * 4.0;
    let wave1 = sin(uv.x * 2.0 + time * 0.4) * cos(uv.y * 1.5 + time * 0.3) * 0.03;
    let wave2 = sin(uv.x * 6.0 - time * 0.8 + uv.y * 3.0) * 0.015;
    // Replace expensive voronoi caustics with cheap noise
    let caustic_uv = uv * 2.0 + vec2<f32>(time * 0.15, time * 0.1);
    let caustic_bright = noise2(caustic_uv * 1.5) * 0.05;
    let depth = fbm2(uv * 0.8) * 0.06;
    let spec_uv = uv + vec2<f32>(time * 0.2, time * 0.15);
    let spec = smoothstep(0.75, 0.85, noise2(spec_uv * 5.0)) * 0.10;
    let variation = wave1 + wave2 + caustic_bright - depth;
    return base_color + vec3<f32>(variation + spec * 0.5, variation + spec, variation + spec);
}

fn tex_cobblestone(p: vec2<f32>, base_color: vec3<f32>) -> vec3<f32> {
    let uv = p * 18.0;

    // Voronoi cells form the individual stones
    let stone_dist = voronoi(uv);

    // Mortar lines in the gaps between stones
    let mortar = smoothstep(0.04, 0.12, stone_dist);

    // Each stone gets a slightly different shade
    let stone_id = floor(uv) + hash22(floor(uv));
    let stone_hue = hash21(stone_id) * 0.08 - 0.04;

    // Surface roughness on each stone
    let roughness = noise2(uv * 6.0) * 0.04;

    // Worn/polished highlights
    let wear = smoothstep(0.6, 0.8, noise2(uv * 1.5)) * 0.03;

    let mortar_color = base_color - vec3<f32>(0.08);
    let stone_color = base_color + vec3<f32>(stone_hue + roughness + wear);

    return mix(mortar_color, stone_color, mortar);
}

// ============================================================
//  Fragment shader
// ============================================================

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let mat = i32(round(in.material));
    var color = in.color.rgb;

    // Apply dashboard color overrides.
    // Check specific materials first, then fall through to land for everything else.
    let orig_lum = dot(color, vec3<f32>(0.299, 0.587, 0.114));
    let is_water = mat == 4 || mat == 11;
    let is_park = mat == 3 || mat == 6;
    let is_building = mat == 2 || mat == 5 || mat == 9 || mat == 10;
    let is_road = mat == 1 || mat == 8;
    let is_land = !is_water && !is_park && !is_building && !is_road
                  && mat != 12 && mat != 13; // exclude clouds and pins

    if camera.water_tint.a > 0.5 && is_water {
        color = camera.water_tint.rgb;
    } else if camera.park_tint.a > 0.5 && is_park {
        color = camera.park_tint.rgb;
    } else if camera.building_tint.a > 0.5 && is_building {
        let tint = camera.building_tint.rgb;
        if mat == 5 || mat == 10 {
            let tint_lum = max(dot(tint, vec3<f32>(0.299, 0.587, 0.114)), 0.01);
            color = tint * (orig_lum / tint_lum);
        } else {
            color = tint;
        }
    } else if camera.road_tint.a > 0.5 && is_road {
        color = camera.road_tint.rgb;
    } else if camera.land_tint.a > 0.5 && is_land {
        color = camera.land_tint.rgb;
    }

    // Procedural materials
    switch mat {
        case 5: {
            color = tex_building_wall(in.world_pos.xy, in.world_pos.z, color);
        }
        case 6: {
            // Tree leaves — organic variation using noise
            let uv = in.world_pos.xy * 12.0 + vec2<f32>(in.world_pos.z * 5.0);
            let leaf_noise = fbm3(uv) * 0.15;
            let clump = noise2(uv * 3.0) * 0.08;
            color = color + vec3<f32>(leaf_noise + clump - 0.06, leaf_noise + clump, leaf_noise + clump - 0.04);
        }
        case 7: {
            // Tree trunk — bark texture
            let uv = vec2<f32>(in.world_pos.x + in.world_pos.y, in.world_pos.z) * 30.0;
            let bark = noise2(uv * vec2<f32>(1.0, 4.0)) * 0.08;
            let grain = noise2(uv * vec2<f32>(0.5, 8.0)) * 0.05;
            color = color + vec3<f32>(bark + grain - 0.04);
        }
        case 9: {
            // Glass roof — subtle reflection variation, no grid lines
            let uv = in.world_pos.xy * 4.0;
            let reflection = noise2(uv * 0.5) * 0.04;
            color = color + vec3<f32>(reflection);
        }
        case 10: {
            // Glass wall — subtle variation, no grid lines
            let uv = vec2<f32>(in.world_pos.x + in.world_pos.y, in.world_pos.z) * 4.0;
            let tint = noise2(uv * 0.8) * 0.05;
            let sky_reflect = smoothstep(0.4, 0.8, noise2(uv * 0.3)) * 0.04;
            color = color + vec3<f32>(tint + sky_reflect);
        }
        case 11: {
            // Fountain spray — animated shimmer and sparkle
            let uv = in.world_pos.xy * 20.0 + vec2<f32>(camera.time * 2.0, camera.time * 1.5);
            let sparkle = smoothstep(0.7, 0.9, noise2(uv * 3.0)) * 0.15;
            let shimmer = sin(in.world_pos.z * 40.0 + camera.time * 6.0) * 0.06;
            let flow = noise2(uv * 1.5 + vec2<f32>(0.0, camera.time * 3.0)) * 0.08;
            color = color + vec3<f32>(sparkle + shimmer + flow);
        }
        case 4: {
            // Water — animated waves with fake reflection
            let t = camera.time;
            let uv = in.world_pos.xy;

            // Layered animated waves
            let wave1 = sin(uv.x * 8.0 + t * 1.2) * cos(uv.y * 6.0 + t * 0.8) * 0.03;
            let wave2 = sin(uv.x * 15.0 - t * 1.8 + uv.y * 3.0) * 0.02;
            let wave3 = noise2((uv + vec2<f32>(t * 0.3, t * 0.2)) * 4.0) * 0.04;

            // Ripple caustics
            let caustic_uv = uv * 12.0 + vec2<f32>(t * 0.5, t * 0.3);
            let caustic = smoothstep(0.55, 0.7, noise2(caustic_uv)) * 0.06;

            // Cloud reflections — large soft shapes that drift slowly
            let cloud_uv = uv * 0.8 + vec2<f32>(t * 0.08, t * 0.05);
            let cloud1 = smoothstep(0.35, 0.65, noise2(cloud_uv));
            let cloud2 = smoothstep(0.3, 0.7, noise2(cloud_uv * 2.3 + vec2<f32>(3.7, 1.2)));
            let clouds = cloud1 * 0.6 + cloud2 * 0.4;
            let cloud_reflect = clouds * 0.12; // subtle white cloud reflection

            // Specular sparkle — sun glints on wave peaks
            let sparkle_uv = uv * 20.0 + vec2<f32>(t * 2.0, t * 1.5);
            let sparkle = smoothstep(0.85, 0.95, noise2(sparkle_uv)) * 0.15;

            let wave_n = wave1 + wave2 + wave3 + caustic;
            let spark_n = sparkle + cloud_reflect;
            if camera.water_tint.a > 0.5 {
                // Scale effects by tint brightness so dark water stays dark
                let brightness = dot(camera.water_tint.rgb, vec3<f32>(0.299, 0.587, 0.114));
                let effect_scale = clamp(brightness * 2.0, 0.05, 1.0);
                color = color + vec3<f32>((wave_n + spark_n) * effect_scale);
            } else {
                color = color + vec3<f32>(wave_n);
                color = color + vec3<f32>(cloud_reflect);
                color = color + vec3<f32>(sparkle * 0.7, sparkle * 0.85, sparkle);
            }
        }
        case 3: {
            // Grass/park — organic variation (skip green bias when tinted)
            let uv = in.world_pos.xy;
            let grass1 = fbm3(uv * 8.0) * 0.06;
            let grass2 = noise2(uv * 25.0) * 0.03;
            let clump = smoothstep(0.4, 0.6, noise2(uv * 4.0)) * 0.04;
            if camera.park_tint.a > 0.5 {
                // Neutral noise — no green bias
                let n = grass1 + grass2 * 0.5 + clump * 0.5;
                color = color + vec3<f32>(n, n, n);
            } else {
                color = color + vec3<f32>(grass1 * 0.5, grass1 + grass2 + clump, grass1 * 0.3);
            }
        }
        case 1: {
            // Asphalt roads — subtle grain
            let uv = in.world_pos.xy * 30.0;
            let grain = noise2(uv) * 0.03;
            let crack = smoothstep(0.8, 0.85, noise2(uv * 0.3)) * 0.04;
            color = color + vec3<f32>(grain - crack);
        }
        case 8: {
            // Cobblestone paths — subtle variation
            let uv = in.world_pos.xy * 20.0;
            let stone = noise2(uv * 2.0) * 0.04;
            color = color + vec3<f32>(stone);
        }
        case 13: {
            // Pin marker — glossy with subtle highlight
            let highlight = smoothstep(0.3, 0.7, in.world_pos.z / 0.5) * 0.15;
            color = color + vec3<f32>(highlight);
        }
        default: {
            let subtle = noise2(in.world_pos.xy * 20.0) * 0.015;
            color = color + vec3<f32>(subtle);
        }
    }

    // Screen-space edge fog — based on distance from screen center (NDC space)
    // This makes fog independent of world position — it's a viewport vignette
    var fog_color = vec3<f32>(0.95, 0.90, 0.85); // warm cream fog matching cottagecore land
    if camera.land_tint.a > 0.5 {
        fog_color = camera.land_tint.rgb;
    }
    let ndc = in.clip_position.xy / camera.viewport * 2.0 - 1.0; // -1..1 screen space
    let screen_dist = length(ndc); // 0 at center, ~1.4 at corners
    let fog_factor = smoothstep(0.85, 1.3, screen_dist) * 0.7; // starts near edge, 70% at corners
    color = mix(color, fog_color, fog_factor);

    return vec4<f32>(color, in.color.a);
}

// Shadow/overlay fragment shader — handles both shadows (black+alpha) and clouds (white+alpha)
@fragment
fn fs_shadow(in: VertexOutput) -> @location(0) vec4<f32> {
    let mat = i32(round(in.material));
    if mat == 12 {
        // Cloud: white with vertex alpha, slight animated variation
        let drift = vec2<f32>(camera.time * 0.06, camera.time * 0.04);
        let uv = in.world_pos.xy * 0.3 + drift;
        let density = fbm3(uv) * 0.4 + 0.6;
        let alpha = in.color.a * density;
        return vec4<f32>(1.0, 1.0, 1.0, alpha);
    }
    // Shadow: black with vertex alpha
    return vec4<f32>(0.0, 0.0, 0.0, in.color.a);
}

// ============================================================
//  Procedural fullscreen cloud overlay
//  One fullscreen triangle — all cloud logic in the fragment shader.
// ============================================================

struct CloudOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_cloud(@builtin(vertex_index) vi: u32) -> CloudOutput {
    // Fullscreen triangle (oversized, clipped by rasterizer)
    var out: CloudOutput;
    let x = f32(i32(vi & 1u)) * 4.0 - 1.0;
    let y = f32(i32(vi >> 1u)) * 4.0 - 1.0;
    out.clip_position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>(x * 0.5 + 0.5, 1.0 - (y * 0.5 + 0.5));
    return out;
}

// Smooth Voronoi-like cell noise for puffy cloud shapes
fn cloud_cell(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    var min_d = 1.0;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let neighbor = vec2<f32>(f32(x), f32(y));
            let point = hash22(i + neighbor);
            let diff = neighbor + point - f;
            min_d = min(min_d, dot(diff, diff));
        }
    }
    return 1.0 - sqrt(min_d);
}

@fragment
fn fs_cloud(in: CloudOutput) -> @location(0) vec4<f32> {
    // Skip if opacity is zero
    if camera.cloud_opacity < 0.01 {
        discard;
    }

    // Convert screen UV to world-space coordinates
    let aspect = camera.viewport.x / camera.viewport.y;
    let zoom_scale = 100.0 / pow(2.0, camera.zoom);
    let world_x = camera.position.x + (in.uv.x - 0.5) * zoom_scale * aspect;
    let world_y = camera.position.y + (0.5 - in.uv.y) * zoom_scale;

    // Drift leftward (negative X), slight upward
    let drift = vec2<f32>(-camera.time * camera.cloud_speed * 0.15, camera.time * camera.cloud_speed * 0.02);
    let cloud_uv = vec2<f32>(world_x, world_y) * 0.015 + drift;

    // Layered cloud density using cell noise + value noise
    let large = cloud_cell(cloud_uv * 1.0) * 0.6;
    let medium = cloud_cell(cloud_uv * 2.3 + vec2<f32>(5.3, 1.7)) * 0.25;
    let small = noise2(cloud_uv * 5.0 + vec2<f32>(3.1, 7.9)) * 0.15;
    let density = large + medium + small;

    // Threshold to create distinct cloud patches with soft edges
    let cloud = smoothstep(0.45, 0.65, density);

    if cloud < 0.005 {
        discard;
    }

    let alpha = cloud * 0.35 * camera.cloud_opacity;
    return vec4<f32>(1.0, 1.0, 1.0, alpha);
}
