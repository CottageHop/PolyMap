/// Convert decoded MVT tiles into PolyMap's `MapData` struct for GPU rendering.
///
/// This module bridges between the MVT decoder (`crate::mvt`) and PolyMap's
/// rendering pipeline (`crate::mapdata`). It replaces the role of
/// `parse_osm_json_centered` by producing the same `MapData` output from
/// MVT features instead of OSM JSON.

use crate::mapdata::{
    self, classify_road, expand_polygon,
    generate_fountain, generate_line_geometry, generate_poi_icon,
    is_closed, polygon_centroid, road_width,
    triangulate_polygon_at_height, Label, LabelKind, MapData, MapVertex, RoadType,
    COLOR_BUILDING, COLOR_COMMERCIAL, COLOR_INDUSTRIAL, COLOR_LAND,
    COLOR_PARK, COLOR_RESIDENTIAL, COLOR_ROAD_MAJOR, COLOR_ROAD_MINOR,
    COLOR_RAIL, COLOR_RAIL_TIE, COLOR_ROAD_OUTLINE, COLOR_SHADOW_CORE, COLOR_SHADOW_EDGE, COLOR_SHADOW_MID, COLOR_SIDEWALK,
    COLOR_SIDEWALK_OUTLINE, COLOR_SKYSCRAPER, COLOR_WATER, MAT_BUILDING, MAT_BUILDING_WALL,
    MAT_COBBLESTONE, MAT_COMMERCIAL, MAT_DEFAULT, MAT_GLASS, MAT_GLASS_WALL, MAT_GRASS,
    MAT_INDUSTRIAL, MAT_RAIL, MAT_RAIL_TIE, MAT_RESIDENTIAL, MAT_ROAD, MAT_WATER,
    SHADOW_BLUR, SHADOW_DIR, Z_LANDUSE, Z_LANDUSE_DETAIL, Z_PARK, Z_PATH_FILL,
    Z_PATH_OUTLINE, Z_ROAD_FILL, Z_ROAD_OUTLINE, Z_WATER,
};
use crate::mvt::{GeomType, Layer, Tile};

// ---------------------------------------------------------------------------
// Detail levels — reduce geometry at zoom-out to save GPU resources
// ---------------------------------------------------------------------------

/// Controls how much geometry is generated per tile.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum DetailLevel {
    /// Water + landuse + road fills only. No buildings, shadows, trees, parcels.
    Low,
    /// Adds flat building footprints and road outlines. No walls, shadows, trees.
    Medium,
    /// Full 3D: extruded buildings, shadows, trees, fountains, parcels.
    High,
}

// ---------------------------------------------------------------------------
// Coordinate conversion
// ---------------------------------------------------------------------------

/// Convert MVT tile-local integer coordinates to lat/lon.
///
/// MVT geometry is in tile-local space where (0,0) is the top-left corner
/// and (extent, extent) is the bottom-right. This converts to geographic
/// coordinates using the standard Web Mercator tile math.
fn tile_to_latlon(px: i32, py: i32, extent: u32, z: u8, x: u32, y: u32) -> (f64, f64) {
    let n = (1u64 << z) as f64;
    let lon = (x as f64 + px as f64 / extent as f64) / n * 360.0 - 180.0;
    let lat_rad = (std::f64::consts::PI * (1.0 - 2.0 * (y as f64 + py as f64 / extent as f64) / n))
        .sinh()
        .atan();
    let lat = lat_rad.to_degrees();
    (lat, lon)
}

/// Check if a ring's centroid is outside the tile boundary (with margin).
/// Features in the buffer zone are skipped — the adjacent tile will render them.
fn centroid_outside_tile(ring: &[[i32; 2]], extent: u32) -> bool {
    if ring.is_empty() { return true; }
    let e = extent as i32;
    let (sx, sy) = ring.iter().fold((0i64, 0i64), |(sx, sy), pt| {
        (sx + pt[0] as i64, sy + pt[1] as i64)
    });
    let cx = (sx / ring.len() as i64) as i32;
    let cy = (sy / ring.len() as i64) as i32;
    cx < 0 || cx > e || cy < 0 || cy > e
}

/// Check if a building ring's centroid is near any tile edge.
/// Buildings near edges get rendered by both tiles and z-fight — skip them.
fn centroid_near_tile_edge(ring: &[[i32; 2]], extent: u32) -> bool {
    if ring.is_empty() { return true; }
    let e = extent as i32;
    let margin = 200; // ~5% of tile width
    let (sx, sy) = ring.iter().fold((0i64, 0i64), |(sx, sy), pt| {
        (sx + pt[0] as i64, sy + pt[1] as i64)
    });
    let cx = (sx / ring.len() as i64) as i32;
    let cy = (sy / ring.len() as i64) as i32;
    cx < margin || cx > e - margin || cy < margin || cy > e - margin
}

/// Check if a line's midpoint is outside the tile boundary.
fn line_midpoint_outside_tile(ring: &[[i32; 2]], extent: u32) -> bool {
    if ring.len() < 2 { return true; }
    let mid = ring.len() / 2;
    let pt = ring[mid];
    let e = extent as i32;
    pt[0] < 0 || pt[0] > e || pt[1] < 0 || pt[1] > e
}

/// Check if an edge lies on or very near the tile boundary.
/// An edge is a boundary edge if both endpoints share the same near-boundary coordinate.
fn is_tile_boundary_edge(p1: [i32; 2], p2: [i32; 2], extent: u32) -> bool {
    let e = extent as i32;
    let tol = 128; // generous tolerance — MVT clipping can place vertices slightly inside
    // Both endpoints near the left edge
    (p1[0] <= tol && p2[0] <= tol) ||
    // Both endpoints near the right edge
    (p1[0] >= e - tol && p2[0] >= e - tol) ||
    // Both endpoints near the top edge
    (p1[1] <= tol && p2[1] <= tol) ||
    // Both endpoints near the bottom edge
    (p1[1] >= e - tol && p2[1] >= e - tol)
}



/// Convert a ring of MVT tile-local coordinates to world-space `[f32; 2]` points.
fn convert_ring(
    ring: &[[i32; 2]],
    extent: u32,
    z: u8,
    x: u32,
    y: u32,
    center_lat: f64,
    center_lon: f64,
) -> Vec<[f32; 2]> {
    ring.iter()
        .map(|pt| {
            let (lat, lon) = tile_to_latlon(pt[0], pt[1], extent, z, x, y);
            mapdata::project(lat, lon, center_lat, center_lon)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Main conversion entry point
// ---------------------------------------------------------------------------

/// Convert a decoded MVT tile into PolyMap's `MapData` for GPU rendering.
///
/// Geometry is produced in the same z-order as `parse_osm_json_centered`:
/// water, landuse, parks, building shadows, buildings, roads (outline then
/// fill), and labels.
pub fn mvt_to_mapdata(
    tile: &Tile,
    z: u8,
    x: u32,
    y: u32,
    center_lat: f64,
    center_lon: f64,
    detail: DetailLevel,
) -> MapData {
    // Pre-allocate to avoid repeated heap reallocations during tessellation
    let mut vertices: Vec<MapVertex> = Vec::with_capacity(4096);
    let mut indices: Vec<u32> = Vec::with_capacity(12288);
    let mut shadow_vertices: Vec<MapVertex> = Vec::with_capacity(1024);
    let mut shadow_indices: Vec<u32> = Vec::with_capacity(3072);
    let mut labels: Vec<Label> = Vec::with_capacity(64);

    // Collect geometry into categorized buckets
    let mut water_polys: Vec<Vec<[f32; 2]>> = Vec::with_capacity(16);
    let mut water_lines: Vec<(Vec<[f32; 2]>, f32)> = Vec::with_capacity(16);
    let mut landuse_polys: Vec<(Vec<[f32; 2]>, [f32; 4], f32, f32)> = Vec::with_capacity(32);
    let mut park_polys: Vec<(Vec<[f32; 2]>, Option<String>)> = Vec::with_capacity(16);
    // (coords, name, height, boundary_edge_mask) — mask[i] = true if edge i→i+1 is on tile boundary
    let mut building_polys: Vec<(Vec<[f32; 2]>, Option<String>, f32, Vec<bool>)> = Vec::with_capacity(128);
    let mut road_lines: Vec<(Vec<[f32; 2]>, RoadType, Option<String>)> = Vec::with_capacity(64);
    let mut fountain_positions: Vec<[f32; 2]> = Vec::with_capacity(8);
    let mut tree_positions: Vec<[f32; 2]> = Vec::with_capacity(32);
    let mut poi_icons: Vec<([f32; 2], String)> = Vec::with_capacity(32);

    // Process each MVT layer
    for layer in &tile.layers {
        let extent = layer.extent;
        match layer.name.as_str() {
            "water" => process_water_layer(
                layer, extent, z, x, y, center_lat, center_lon,
                &mut water_polys, &mut water_lines,
            ),
            "buildings" => process_buildings_layer(
                layer, extent, z, x, y, center_lat, center_lon,
                &mut building_polys,
            ),
            "roads" => process_roads_layer(
                layer, extent, z, x, y, center_lat, center_lon,
                &mut road_lines,
            ),
            "landuse" => process_landuse_layer(
                layer, extent, z, x, y, center_lat, center_lon,
                &mut landuse_polys, &mut park_polys,
            ),
            "pois" if detail == DetailLevel::High => process_pois_layer(
                layer, extent, z, x, y, center_lat, center_lon,
                &mut fountain_positions, &mut tree_positions, &mut labels, &mut poi_icons,
            ),
            "places" => process_places_layer(
                layer, extent, z, x, y, center_lat, center_lon,
                &mut labels,
            ),
            _ => {}
        }
    }

    // --- Build GPU geometry in render order (back to front) ---

    // Layer 0: Background quad REMOVED — the full-screen clear color matches COLOR_LAND,
    // so no per-tile background is needed. Per-tile quads caused visible seams at z14
    // tile boundaries due to floating-point precision in the Mercator projection.

    // Layer 1: Water polygons
    for coords in &water_polys {
        if coords.len() >= 3 {
            triangulate_polygon_at_height(
                coords, Z_WATER, COLOR_WATER, MAT_WATER,
                &mut vertices, &mut indices,
            );
        }
    }

    // Layer 1b: Linear waterways
    for (coords, width) in &water_lines {
        let start_vert = vertices.len();
        generate_line_geometry(coords, *width, COLOR_WATER, MAT_WATER, &mut vertices, &mut indices);
        for v in &mut vertices[start_vert..] {
            v.position[2] = Z_WATER;
        }
    }

    // Layer 2: Landuse areas
    for (coords, color, material, z) in &landuse_polys {
        if coords.len() >= 3 {
            triangulate_polygon_at_height(
                coords, *z, *color, *material,
                &mut vertices, &mut indices,
            );
        }
    }

    // Layer 3: Parks
    for (coords, name) in &park_polys {
        if coords.len() < 3 {
            continue;
        }
        triangulate_polygon_at_height(
            coords, Z_PARK, COLOR_PARK, MAT_GRASS,
            &mut vertices, &mut indices,
        );
        if let Some(name) = name {
            let center = polygon_centroid(coords);
            labels.push(Label {
                text: name.clone(),
                position: center,
                angle: 0.0,
                kind: LabelKind::Park,
                path: None,
            });
        }
    }

    // Sort buildings by height so taller buildings render on top
    building_polys.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

    // Layer 3.5: Building shadows
    {
    for (coords, _, height, _mask) in &building_polys {
        let height_z = height / 11.1;
        if !is_closed(coords) || coords.len() < 4 {
            continue;
        }

        // Offset polygon for shadow projection
        let offset: Vec<[f32; 2]> = coords
            .iter()
            .map(|p| {
                [
                    p[0] + SHADOW_DIR[0] * height_z,
                    p[1] + SHADOW_DIR[1] * height_z,
                ]
            })
            .collect();
        let centroid = polygon_centroid(coords);

        // Outer penumbra
        let outer = expand_polygon(&offset, centroid, SHADOW_BLUR * 2.0);
        triangulate_polygon_at_height(
            &outer, 0.001, COLOR_SHADOW_EDGE, MAT_DEFAULT,
            &mut shadow_vertices, &mut shadow_indices,
        );

        // Mid penumbra
        let mid = expand_polygon(&offset, centroid, SHADOW_BLUR);
        triangulate_polygon_at_height(
            &mid, 0.0011, COLOR_SHADOW_MID, MAT_DEFAULT,
            &mut shadow_vertices, &mut shadow_indices,
        );

        // Core shadow: footprint
        triangulate_polygon_at_height(
            coords, 0.0012, COLOR_SHADOW_CORE, MAT_DEFAULT,
            &mut shadow_vertices, &mut shadow_indices,
        );

        // Core shadow: offset roof projection
        triangulate_polygon_at_height(
            &offset, 0.0012, COLOR_SHADOW_CORE, MAT_DEFAULT,
            &mut shadow_vertices, &mut shadow_indices,
        );

        // Core shadow: connect footprint to offset with quads (the "sweep")
        let pts = if coords.len() > 1 && coords[0] == coords[coords.len() - 1] {
            &coords[..coords.len() - 1]
        } else {
            coords.as_slice()
        };
        for i in 0..pts.len() {
            let j = (i + 1) % pts.len();
            let a = pts[i];
            let b = pts[j];
            let ao = [
                a[0] + SHADOW_DIR[0] * height_z,
                a[1] + SHADOW_DIR[1] * height_z,
            ];
            let bo = [
                b[0] + SHADOW_DIR[0] * height_z,
                b[1] + SHADOW_DIR[1] * height_z,
            ];
            let base = shadow_vertices.len() as u32;
            shadow_vertices.push(MapVertex::at_height(a[0], a[1], 0.0012, COLOR_SHADOW_CORE, MAT_DEFAULT));
            shadow_vertices.push(MapVertex::at_height(b[0], b[1], 0.0012, COLOR_SHADOW_CORE, MAT_DEFAULT));
            shadow_vertices.push(MapVertex::at_height(ao[0], ao[1], 0.0012, COLOR_SHADOW_CORE, MAT_DEFAULT));
            shadow_vertices.push(MapVertex::at_height(bo[0], bo[1], 0.0012, COLOR_SHADOW_CORE, MAT_DEFAULT));
            shadow_indices.extend_from_slice(&[
                base,
                base + 1,
                base + 2,
                base + 1,
                base + 3,
                base + 2,
            ]);
        }
    }
    } // end High-detail shadow block

    // Layer 4: Buildings (sorted by height above)
    for (coords, name, height, _boundary_mask) in &building_polys {
        let height_z = height / 11.1;
        let (base_color, roof_mat, wall_mat) = (COLOR_BUILDING, MAT_BUILDING, MAT_BUILDING_WALL);

        let c = polygon_centroid(coords);
        let color = [base_color[0], base_color[1], base_color[2], 1.0];

        // Shrink taller buildings slightly to prevent overlap z-fighting
        let shrink = if *height > 5.0 {
            (*height - 5.0) * 0.0001
        } else {
            0.0
        };
        let render_coords = if shrink > 0.0 {
            expand_polygon(coords, c, -shrink)
        } else {
            coords.clone()
        };

        let effective_height = height_z;

        // Extrude walls
        {
            let pts = if render_coords.len() > 1 && render_coords[0] == render_coords[render_coords.len() - 1] {
                &render_coords[..render_coords.len() - 1]
            } else {
                render_coords.as_slice()
            };
            for i in 0..pts.len() {
                let j = (i + 1) % pts.len();
                let p0 = pts[i];
                let p1 = pts[j];
                let dx = p1[0] - p0[0];
                let dy = p1[1] - p0[1];
                let len = (dx * dx + dy * dy).sqrt();
                if len < 1e-8 { continue; }
                let nx = -dy / len;
                let ny = dx / len;
                let sun_dot = (nx * 0.5 + ny * -0.7).max(0.0);
                let shade = 0.6 + sun_dot * 0.4;
                let wc = [color[0] * shade, color[1] * shade, color[2] * shade, 1.0];
                let base = vertices.len() as u32;
                vertices.push(MapVertex::at_height(p0[0], p0[1], 0.0, wc, wall_mat));
                vertices.push(MapVertex::at_height(p1[0], p1[1], 0.0, wc, wall_mat));
                vertices.push(MapVertex::at_height(p0[0], p0[1], effective_height, wc, wall_mat));
                vertices.push(MapVertex::at_height(p1[0], p1[1], effective_height, wc, wall_mat));
                indices.extend_from_slice(&[base, base+1, base+2, base+1, base+3, base+2]);
            }
        }
        triangulate_polygon_at_height(
            &render_coords, effective_height, color, roof_mat,
            &mut vertices, &mut indices,
        );

        // Building label
        if let Some(name) = name {
            labels.push(Label {
                text: name.clone(),
                position: c,
                angle: 0.0,
                kind: LabelKind::Building,
                path: None,
            });
        }
    }

    // Layer 5: Sidewalks / footpaths (outline pass) — all detail levels
    {
    for (coords, road_type, name) in &road_lines {
        if !matches!(road_type, RoadType::Path) {
            continue;
        }
        let base_width = if name.is_some() { 0.25 } else { road_width(road_type) };
        let width = base_width + 0.04;
        let start_vert = vertices.len();
        generate_line_geometry(
            coords, width, COLOR_SIDEWALK_OUTLINE, MAT_COBBLESTONE,
            &mut vertices, &mut indices,
        );
        for v in &mut vertices[start_vert..] {
            v.position[2] = Z_PATH_OUTLINE;
        }
    }

    // Layer 5b: Sidewalks / footpaths (fill pass)
    for (coords, road_type, name) in &road_lines {
        if !matches!(road_type, RoadType::Path) {
            continue;
        }
        let base_width = if name.is_some() { 0.25 } else { road_width(road_type) };
        let start_vert = vertices.len();
        generate_line_geometry(
            coords, base_width, COLOR_SIDEWALK, MAT_COBBLESTONE,
            &mut vertices, &mut indices,
        );
        for v in &mut vertices[start_vert..] {
            v.position[2] = Z_PATH_FILL;
        }
    }
    } // end Medium+ sidewalk block

    // Layer 6: Road outlines (slightly wider, dark) — Medium+ detail
    if detail >= DetailLevel::Medium {
    for (coords, road_type, _) in &road_lines {
        if matches!(road_type, RoadType::Path | RoadType::Rail) {
            continue;
        }
        let width = road_width(road_type) + 0.20;
        let start_vert = vertices.len();
        generate_line_geometry(
            coords, width, COLOR_ROAD_OUTLINE, MAT_ROAD,
            &mut vertices, &mut indices,
        );
        for v in &mut vertices[start_vert..] {
            v.position[2] = Z_ROAD_OUTLINE;
        }
    }
    } // end Medium+ road outline block

    // Layer 7: Road fills (on top of outlines)
    for (coords, road_type, _) in &road_lines {
        if matches!(road_type, RoadType::Path) {
            continue;
        }
        let width = road_width(road_type);
        let (color, material) = match road_type {
            RoadType::Major => (COLOR_ROAD_MAJOR, MAT_ROAD),
            RoadType::Minor | RoadType::Residential => (COLOR_ROAD_MINOR, MAT_ROAD),
            RoadType::Rail => (COLOR_RAIL, MAT_RAIL),
            RoadType::Path => continue,
        };
        let start_vert = vertices.len();
        generate_line_geometry(
            coords, width, color, material,
            &mut vertices, &mut indices,
        );
        for v in &mut vertices[start_vert..] {
            v.position[2] = Z_ROAD_FILL;
        }

        // Railroad crossties — perpendicular bars along the rail path.
        if matches!(road_type, RoadType::Rail) && coords.len() >= 2 {
            const TIE_SPACING: f32 = 1.6;
            const TIE_HALF_WIDTH: f32 = 0.30;
            const TIE_HALF_LEN: f32 = 0.10;
            let tie_z = Z_ROAD_FILL + 0.0005;

            let mut walked = 0.0f32;
            let mut next_tie = TIE_SPACING * 0.5;
            for i in 0..coords.len() - 1 {
                let dx = coords[i + 1][0] - coords[i][0];
                let dy = coords[i + 1][1] - coords[i][1];
                let seg_len = (dx * dx + dy * dy).sqrt();
                if seg_len < 1e-6 { continue; }
                let ux = dx / seg_len;
                let uy = dy / seg_len;
                let px = -uy;
                let py = ux;

                while walked + seg_len >= next_tie {
                    let t_local = next_tie - walked;
                    let cx = coords[i][0] + ux * t_local;
                    let cy = coords[i][1] + uy * t_local;
                    let hl = TIE_HALF_LEN;
                    let hw = TIE_HALF_WIDTH;
                    let v0 = [cx - ux * hl - px * hw, cy - uy * hl - py * hw];
                    let v1 = [cx + ux * hl - px * hw, cy + uy * hl - py * hw];
                    let v2 = [cx - ux * hl + px * hw, cy - uy * hl + py * hw];
                    let v3 = [cx + ux * hl + px * hw, cy + uy * hl + py * hw];
                    let base = vertices.len() as u32;
                    vertices.push(MapVertex::at_height(v0[0], v0[1], tie_z, COLOR_RAIL_TIE, MAT_RAIL_TIE));
                    vertices.push(MapVertex::at_height(v1[0], v1[1], tie_z, COLOR_RAIL_TIE, MAT_RAIL_TIE));
                    vertices.push(MapVertex::at_height(v2[0], v2[1], tie_z, COLOR_RAIL_TIE, MAT_RAIL_TIE));
                    vertices.push(MapVertex::at_height(v3[0], v3[1], tie_z, COLOR_RAIL_TIE, MAT_RAIL_TIE));
                    indices.extend_from_slice(&[base, base + 1, base + 2, base + 1, base + 3, base + 2]);
                    next_tie += TIE_SPACING;
                }
                walked += seg_len;
            }
        }
    }

    // Road labels — place on straight sections of each road segment
    let label_spacing = 200.0f32;
    // Maximum angle change (radians) between adjacent segments to be considered "straight"
    let max_curvature = 0.35; // ~20 degrees
    for (coords, road_type, name) in &road_lines {
        if let Some(name) = name {
            if matches!(road_type, RoadType::Path | RoadType::Rail) {
                continue;
            }
            if coords.len() < 2 {
                continue;
            }

            // Find straight runs: sequences of segments with low curvature
            // Only place labels on segments that are part of a straight run
            let mut seg_angles: Vec<f32> = Vec::with_capacity(coords.len());
            for i in 0..coords.len() - 1 {
                let dx = coords[i + 1][0] - coords[i][0];
                let dy = coords[i + 1][1] - coords[i][1];
                seg_angles.push(dy.atan2(dx));
            }

            // Mark each segment as straight (true) if the angle change from
            // the previous segment is below the curvature threshold
            let mut is_straight = vec![true; seg_angles.len()];
            for i in 1..seg_angles.len() {
                let mut delta = (seg_angles[i] - seg_angles[i - 1]).abs();
                if delta > std::f32::consts::PI {
                    delta = 2.0 * std::f32::consts::PI - delta;
                }
                if delta > max_curvature {
                    is_straight[i] = false;
                    is_straight[i - 1] = false; // mark both sides of the bend
                }
            }

            let labels_before = labels.len();
            let mut walked = 0.0f32;
            let mut next_label_at = label_spacing * 0.5;
            for i in 0..coords.len() - 1 {
                let dx = coords[i + 1][0] - coords[i][0];
                let dy = coords[i + 1][1] - coords[i][1];
                let seg_len = (dx * dx + dy * dy).sqrt();

                while walked + seg_len >= next_label_at && seg_len > 1e-8 {
                    if is_straight[i] {
                        let t = (next_label_at - walked) / seg_len;
                        let pos = [coords[i][0] + dx * t, coords[i][1] + dy * t];
                        let mut angle = (-dy).atan2(dx);
                        if angle > std::f32::consts::FRAC_PI_2 {
                            angle -= std::f32::consts::PI;
                        } else if angle < -std::f32::consts::FRAC_PI_2 {
                            angle += std::f32::consts::PI;
                        }
                        labels.push(Label {
                            text: name.clone(),
                            position: pos,
                            angle,
                            kind: LabelKind::Street,
                            path: Some(coords.clone()),
                        });
                    }
                    next_label_at += label_spacing;
                }

                walked += seg_len;
            }

            // Coverage fallback: every named road with non-zero length gets
            // at least one label, even if entirely curved or too short for
            // the straight-run loop to hit.
            if labels.len() == labels_before {
                let mut total_len = 0.0f32;
                for i in 0..coords.len() - 1 {
                    let dx = coords[i + 1][0] - coords[i][0];
                    let dy = coords[i + 1][1] - coords[i][1];
                    total_len += (dx * dx + dy * dy).sqrt();
                }
                if total_len > 1e-6 {
                    let target = total_len * 0.5;
                    let mut walked = 0.0f32;
                    for i in 0..coords.len() - 1 {
                        let dx = coords[i + 1][0] - coords[i][0];
                        let dy = coords[i + 1][1] - coords[i][1];
                        let seg_len = (dx * dx + dy * dy).sqrt();
                        if walked + seg_len >= target && seg_len > 1e-8 {
                            let t = (target - walked) / seg_len;
                            let pos = [coords[i][0] + dx * t, coords[i][1] + dy * t];
                            let mut angle = (-dy).atan2(dx);
                            if angle > std::f32::consts::FRAC_PI_2 {
                                angle -= std::f32::consts::PI;
                            } else if angle < -std::f32::consts::FRAC_PI_2 {
                                angle += std::f32::consts::PI;
                            }
                            labels.push(Label {
                                text: name.clone(),
                                position: pos,
                                angle,
                                kind: LabelKind::Street,
                                path: Some(coords.clone()),
                            });
                            break;
                        }
                        walked += seg_len;
                    }
                }
            }
        }
    }

    // Layer 10: Fountains
    for pos in &fountain_positions {
        generate_fountain(*pos, Z_WATER, &mut vertices, &mut indices);
    }


    // No hard vertex cap — using queue.write_buffer avoids mappedAtCreation limits.
    // Building budget (80K verts) prevents unbounded growth in dense tiles.
    // Shadow cap only — prevent extreme shadow memory in dense areas
    let shadow_cap: usize = 80_000;
    if shadow_vertices.len() > shadow_cap {
        shadow_vertices.truncate(shadow_cap);
        let max_idx = shadow_cap as u32;
        shadow_indices.retain(|&i| i < max_idx);
        let tri_count = shadow_indices.len() / 3;
        shadow_indices.truncate(tri_count * 3);
    }

    MapData {
        vertices,
        indices,
        shadow_vertices,
        shadow_indices,
        labels,
        listings: Vec::new(),
        center_lat,
        center_lon,
    }
}

// ---------------------------------------------------------------------------
// Per-layer processing functions
// ---------------------------------------------------------------------------

/// Process the `water` MVT layer. Polygon features become water bodies;
/// line features become linear waterways with width based on kind.
fn process_water_layer(
    layer: &Layer,
    extent: u32,
    z: u8,
    x: u32,
    y: u32,
    center_lat: f64,
    center_lon: f64,
    water_polys: &mut Vec<Vec<[f32; 2]>>,
    water_lines: &mut Vec<(Vec<[f32; 2]>, f32)>,
) {
    for feature in &layer.features {
        match feature.geom_type {
            GeomType::Polygon => {
                for ring in &feature.geometry {
                    let coords = convert_ring(ring, extent, z, x, y, center_lat, center_lon);
                    if coords.len() >= 3 {
                        water_polys.push(coords);
                    }
                }
            }
            GeomType::LineString => {
                let kind = feature.get_str(layer, "kind").unwrap_or("");
                let width = match kind {
                    "river" => 3.0,
                    "canal" => 1.8,
                    "stream" => 0.6,
                    _ => 1.2,
                };
                for line in &feature.geometry {
                    let coords = convert_ring(line, extent, z, x, y, center_lat, center_lon);
                    if coords.len() >= 2 {
                        water_lines.push((coords, width));
                    }
                }
            }
            _ => {}
        }
    }
}

/// Process the `buildings` MVT layer.
fn process_buildings_layer(
    layer: &Layer,
    extent: u32,
    z: u8,
    x: u32,
    y: u32,
    center_lat: f64,
    center_lon: f64,
    building_polys: &mut Vec<(Vec<[f32; 2]>, Option<String>, f32, Vec<bool>)>,
) {
    for feature in &layer.features {
        if feature.geom_type != GeomType::Polygon {
            continue;
        }

        let height = feature.get_f64(layer, "render_height")
            .or_else(|| feature.get_f64(layer, "height"))
            .unwrap_or(10.0) as f32;
        let name = feature.get_str(layer, "name").map(|s| s.to_string());

        // MVT MultiPolygons have multiple rings — each outer ring is a separate building
        for ring in &feature.geometry {
            if ring.len() < 3 { continue; }
            if centroid_outside_tile(ring, extent) { continue; }

            let coords = convert_ring(ring, extent, z, x, y, center_lat, center_lon);
            if coords.len() < 4 || !is_closed(&coords) {
                continue;
            }
            // Compute boundary edge mask for wall generation
            let mask: Vec<bool> = (0..ring.len().saturating_sub(1))
                .map(|i| is_tile_boundary_edge(ring[i], ring[i + 1], extent))
                .collect();
            building_polys.push((coords, name.clone(), height, mask));
        }
    }
}

/// Process the `roads` MVT layer.
fn process_roads_layer(
    layer: &Layer,
    extent: u32,
    z: u8,
    x: u32,
    y: u32,
    center_lat: f64,
    center_lon: f64,
    road_lines: &mut Vec<(Vec<[f32; 2]>, RoadType, Option<String>)>,
) {
    for feature in &layer.features {
        if feature.geom_type != GeomType::LineString {
            continue;
        }

        let kind = feature.get_str(layer, "kind").unwrap_or("");
        let kind_detail = feature.get_str(layer, "kind_detail").unwrap_or("");
        let name = feature.get_str(layer, "name").map(|s| s.to_string());

        // Try to classify by kind_detail first (e.g. "motorway", "primary", "residential")
        let road_type = if !kind_detail.is_empty() {
            let classified = classify_road(kind_detail);
            // classify_road returns Residential for unknown values, so check if
            // kind_detail actually matched a known type or if we should fall back
            // to the broader `kind` field.
            match kind_detail {
                "motorway" | "trunk" | "primary" | "motorway_link" | "trunk_link"
                | "primary_link" | "secondary" | "tertiary" | "secondary_link"
                | "tertiary_link" | "footway" | "cycleway" | "path" | "pedestrian"
                | "steps" | "service" | "track" | "corridor" | "bridleway"
                | "residential" | "unclassified" | "living_street" => classified,
                "rail" | "subway" | "light_rail" | "tram" | "narrow_gauge"
                | "monorail" | "funicular" => RoadType::Rail,
                // kind_detail didn't match a known OSM highway value; fall back to kind
                _ => classify_kind_fallback(kind),
            }
        } else {
            classify_kind_fallback(kind)
        };

        for line in &feature.geometry {
            let coords = convert_ring(line, extent, z, x, y, center_lat, center_lon);
            if coords.len() >= 2 {
                road_lines.push((coords, road_type, name.clone()));
            }
        }
    }
}

/// Fallback road classification from Protomaps `kind` field.
fn classify_kind_fallback(kind: &str) -> RoadType {
    match kind {
        "highway" | "major_road" => RoadType::Major,
        "minor_road" => RoadType::Residential,
        "path" => RoadType::Path,
        "rail" | "railway" => RoadType::Rail,
        _ => RoadType::Residential,
    }
}

/// Process the `landuse` MVT layer.
///
/// Park/garden/playground/forest/grass features are separated into the
/// park bucket so they render at `Z_PARK` with the grass material. All
/// other landuse types go into the generic landuse bucket.
fn process_landuse_layer(
    layer: &Layer,
    extent: u32,
    z: u8,
    x: u32,
    y: u32,
    center_lat: f64,
    center_lon: f64,
    landuse_polys: &mut Vec<(Vec<[f32; 2]>, [f32; 4], f32, f32)>,
    park_polys: &mut Vec<(Vec<[f32; 2]>, Option<String>)>,
) {
    for feature in &layer.features {
        if feature.geom_type != GeomType::Polygon {
            continue;
        }

        let kind = feature.get_str(layer, "kind").unwrap_or("");
        let name = feature.get_str(layer, "name").map(|s| s.to_string());

        // MVT MultiPolygons have multiple rings
        for ring in &feature.geometry {
            let coords = convert_ring(ring, extent, z, x, y, center_lat, center_lon);
            if coords.len() < 4 || !is_closed(&coords) {
                continue;
            }

            match kind {
                "park" | "garden" | "playground" | "forest" | "wood" | "grass" | "meadow" => {
                    park_polys.push((coords, name.clone()));
                }
                "residential" => {
                    landuse_polys.push((coords, COLOR_RESIDENTIAL, MAT_RESIDENTIAL, Z_LANDUSE_DETAIL));
                }
                "commercial" | "retail" => {
                    landuse_polys.push((coords, COLOR_COMMERCIAL, MAT_COMMERCIAL, Z_LANDUSE_DETAIL));
                }
                "industrial" => {
                    landuse_polys.push((coords, COLOR_INDUSTRIAL, MAT_INDUSTRIAL, Z_LANDUSE_DETAIL));
                }
                _ => {
                    landuse_polys.push((coords, COLOR_LAND, MAT_DEFAULT, Z_LANDUSE));
                }
            }
        }
    }
}

/// Process the `pois` MVT layer. Extracts trees, fountains, and POI labels.
fn process_pois_layer(
    layer: &Layer,
    extent: u32,
    z: u8,
    x: u32,
    y: u32,
    center_lat: f64,
    center_lon: f64,
    fountain_positions: &mut Vec<[f32; 2]>,
    tree_positions: &mut Vec<[f32; 2]>,
    labels: &mut Vec<Label>,
    poi_icons: &mut Vec<([f32; 2], String)>,
) {
    for feature in &layer.features {
        let kind = feature.get_str(layer, "kind").unwrap_or("");

        if let Some(ring) = feature.geometry.first() {
            if let Some(&pt) = ring.first() {
                if centroid_outside_tile(&[pt], extent) { continue; }
                let (lat, lon) = tile_to_latlon(pt[0], pt[1], extent, z, x, y);
                let pos = mapdata::project(lat, lon, center_lat, center_lon);

                match kind {
                    "fountain" => fountain_positions.push(pos),
                    "tree" => {}
                    _ => {
                        if let Some(name) = feature.get_str(layer, "name") {
                            if !name.is_empty() {
                                labels.push(Label {
                                    text: name.to_string(),
                                    position: pos,
                                    angle: 0.0,
                                    kind: LabelKind::Poi,
                                    path: None,
                                });
                                poi_icons.push((pos, kind.to_string()));
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Process the `places` MVT layer for city/town/village labels.
fn process_places_layer(
    layer: &Layer,
    extent: u32,
    z: u8,
    x: u32,
    y: u32,
    center_lat: f64,
    center_lon: f64,
    labels: &mut Vec<Label>,
) {
    for feature in &layer.features {
        let name = match feature.get_str(layer, "name") {
            Some(n) if !n.is_empty() => n.to_string(),
            _ => continue,
        };

        let kind = feature.get_str(layer, "kind").unwrap_or("");
        let label_kind = match kind {
            "city" | "town" | "county" | "state" | "country" => LabelKind::City,
            "village" | "hamlet" | "suburb" | "neighbourhood" | "locality" => LabelKind::Park,
            _ => continue,
        };

        // Use the first point of the geometry as position
        if let Some(ring) = feature.geometry.first() {
            if let Some(pt) = ring.first() {
                let pos = {
                    let (lat, lon) = tile_to_latlon(pt[0], pt[1], extent, z, x, y);
                    mapdata::project(lat, lon, center_lat, center_lon)
                };
                labels.push(Label {
                    text: name,
                    position: pos,
                    angle: 0.0,
                    kind: label_kind,
                    path: None,
                });
            }
        }
    }
}
