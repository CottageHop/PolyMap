use std::collections::HashMap;

/// Material IDs for procedural textures.
pub const MAT_DEFAULT: f32 = 0.0;
pub const MAT_ROAD: f32 = 1.0;
pub const MAT_BUILDING: f32 = 2.0;
pub const MAT_GRASS: f32 = 3.0;
pub const MAT_WATER: f32 = 4.0;

/// Material for building walls (slightly darker for shadow effect).
pub const MAT_BUILDING_WALL: f32 = 5.0;

/// Material for tree foliage (uses grass-like procedural texture).
pub const MAT_TREE_LEAVES: f32 = 6.0;
/// Material for tree trunk.
pub const MAT_TREE_TRUNK: f32 = 7.0;
/// Material for cobblestone walkways.
pub const MAT_COBBLESTONE: f32 = 8.0;
/// Material for glass skyscraper facade.
pub const MAT_GLASS: f32 = 9.0;
/// Material for glass skyscraper walls.
pub const MAT_GLASS_WALL: f32 = 10.0;

/// Material for animated fountain spray.
pub const MAT_FOUNTAIN: f32 = 11.0;
/// Material for volumetric clouds.
pub const MAT_CLOUD: f32 = 12.0;
/// Material for home listing pin.
pub const MAT_PIN: f32 = 13.0;
/// Material for commercial zones.
pub const MAT_RAIL: f32 = 17.0;
pub const MAT_RAIL_TIE: f32 = 18.0;
pub const MAT_COMMERCIAL: f32 = 14.0;
/// Material for residential zones.
pub const MAT_RESIDENTIAL: f32 = 15.0;
/// Material for industrial zones.
pub const MAT_INDUSTRIAL: f32 = 16.0;

/// Height threshold (meters) above which a building is considered a skyscraper.
const SKYSCRAPER_HEIGHT: f32 = 50.0;

/// Max area (world units²) for a water body to be considered a fountain candidate.
const FOUNTAIN_MAX_AREA: f32 = 0.15;
/// Minimum circularity (0..1, 1=perfect circle) for fountain detection.
const FOUNTAIN_MIN_CIRCULARITY: f32 = 0.65;

/// A single vertex for the map geometry, sent directly to the GPU.
/// IMPORTANT: this layout must match `polymap-worker`'s MapVertex exactly.
/// Per-tile data that varies per-frame (fade-in birth time, etc.) lives in
/// a parallel vertex buffer owned by the main thread, not here.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MapVertex {
    pub position: [f32; 3], // x, y, z (z = height for 3D buildings)
    pub color: [f32; 4],
    pub material: f32,
}

impl MapVertex {
    pub fn new(x: f32, y: f32, color: [f32; 4], material: f32) -> Self {
        Self {
            position: [x, y, 0.0],
            color,
            material,
        }
    }

    pub fn at_height(x: f32, y: f32, z: f32, color: [f32; 4], material: f32) -> Self {
        Self {
            position: [x, y, z],
            color,
            material,
        }
    }
}

/// A text label to render on the map.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct Label {
    pub text: String,
    pub position: [f32; 2],
    pub angle: f32, // rotation in radians (0 = horizontal, positive = counter-clockwise)
    pub kind: LabelKind,
    /// Road polyline for curved street labels. Each glyph is placed along this path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<Vec<[f32; 2]>>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub enum LabelKind {
    State,
    City,
    Subdivision,
    Street,
    Park,
    Building,
    Listing,
    Poi,
}

impl Label {
    /// Font scale factor based on label type.
    pub fn font_scale(&self) -> f32 {
        match self.kind {
            LabelKind::State => 2.0,
            LabelKind::City => 1.5,
            LabelKind::Subdivision => 0.9,
            LabelKind::Street => 0.38,
            LabelKind::Park => 0.6,
            LabelKind::Building => 0.4,
            LabelKind::Listing => 0.65,
            LabelKind::Poi => 0.18,
        }
    }

    /// Extra letter spacing multiplier (1.0 = normal, >1 = wider)
    pub fn letter_spacing(&self) -> f32 {
        match self.kind {
            LabelKind::State => 1.5,
            LabelKind::City => 1.35,
            LabelKind::Subdivision => 1.25,
            LabelKind::Street => 1.35,
            LabelKind::Park => 1.1,
            LabelKind::Building => 1.0,
            LabelKind::Listing => 1.0,
            LabelKind::Poi => 1.0,
        }
    }
}

/// A home listing to display on the map.
#[derive(Clone, Debug)]
pub struct HomeListing {
    pub lat: f64,
    pub lon: f64,
    pub price: String,
    pub label: String,
    pub color: [f32; 4],
    pub sqft: u32,
    pub beds: u32,
    pub baths: f32,
    pub image_url: String,
}

/// A placed listing with its world-space position for picking.
#[derive(Clone, Debug)]
pub struct PlacedListing {
    pub listing: HomeListing,
    pub world_pos: [f32; 2],
}

/// One car travelling along a road polyline.
/// `offset` and `speed` are measured in world units.
#[derive(Clone, Debug)]
pub struct Car {
    pub path: Vec<[f32; 2]>,
    pub path_length: f32,
    pub offset: f32,
    pub speed: f32,
    pub color: [f32; 3],
}

/// All GPU-ready map geometry.
#[derive(Clone)]
pub struct MapData {
    pub vertices: Vec<MapVertex>,
    pub indices: Vec<u32>,
    pub shadow_vertices: Vec<MapVertex>,
    pub shadow_indices: Vec<u32>,
    pub labels: Vec<Label>,
    pub listings: Vec<PlacedListing>,
    pub cars: Vec<Car>,
    pub center_lat: f64,
    pub center_lon: f64,
}

pub(crate) const COLOR_PIN_DEFAULT: [f32; 4] = [0.65, 0.08, 0.08, 1.0]; // dark red
pub(crate) const COLOR_PIN_NEEDLE: [f32; 4] = [0.15, 0.15, 0.15, 1.0]; // near black
const PIN_HEAD_RADIUS: f32 = 0.5;
const PIN_NEEDLE_RADIUS: f32 = 0.06;
pub const PIN_HEIGHT: f32 = 20.0;

impl MapData {
    /// Add home listings to the map. Call this after `fetch_osm_data` / `parse_osm_json`.
    pub fn add_listings(&mut self, listings: &[HomeListing]) {
        for listing in listings {
            let pos = project(listing.lat, listing.lon, self.center_lat, self.center_lon);
            let pin_color = if listing.color[3] > 0.0 { listing.color } else { COLOR_PIN_DEFAULT };

            // Pin needle (thin cylinder from ground to head)
            generate_cylinder(
                pos, PIN_NEEDLE_RADIUS, 0.0, PIN_HEIGHT,
                COLOR_PIN_NEEDLE, MAT_PIN,
                6, &mut self.vertices, &mut self.indices,
            );

            // Pin head (sphere at top)
            generate_sphere(
                pos, PIN_HEIGHT, PIN_HEAD_RADIUS,
                pin_color, MAT_PIN,
                8, 6, &mut self.vertices, &mut self.indices,
            );

            // Price label above pin
            let label_text = if listing.price.is_empty() {
                listing.label.clone()
            } else {
                listing.price.clone()
            };

            if !label_text.is_empty() {
                self.labels.push(Label {
                    text: label_text,
                    position: pos,
                    angle: 0.0,
                    kind: LabelKind::Listing,
                    path: None,
                });
            }

            // Store for picking
            self.listings.push(PlacedListing {
                listing: listing.clone(),
                world_pos: pos,
            });
        }

    }
}

// --- Cottagecore color palette ---
// Warm creams, earthy peaches, sage greens, muted browns
pub(crate) const COLOR_WATER: [f32; 4] = [0.60, 0.70, 0.65, 1.0];       // muted sage-teal (row 2 sage tones)
pub(crate) const COLOR_LAND: [f32; 4] = [0.95, 0.90, 0.85, 1.0];        // warm cream (row 1 far right)
pub(crate) const COLOR_SIDEWALK: [f32; 4] = [0.89, 0.82, 0.74, 1.0];    // warm beige (row 1 mid-right)
pub(crate) const COLOR_SIDEWALK_OUTLINE: [f32; 4] = [0.78, 0.70, 0.62, 1.0]; // deeper warm beige
pub(crate) const COLOR_PARK: [f32; 4] = [0.55, 0.60, 0.35, 1.0];        // olive green (row 2 mid)
pub(crate) const COLOR_BUILDING: [f32; 4] = [0.85, 0.69, 0.56, 1.0];    // warm peach (row 1 col 4)
pub(crate) const COLOR_SKYSCRAPER: [f32; 4] = [0.67, 0.72, 0.65, 1.0];  // muted sage (row 2 right)
pub(crate) const COLOR_ROAD_MAJOR: [f32; 4] = [0.55, 0.44, 0.38, 1.0];  // warm brown (row 3 mid)
pub(crate) const COLOR_ROAD_MINOR: [f32; 4] = [0.62, 0.52, 0.45, 1.0];  // lighter warm brown
pub(crate) const COLOR_ROAD_OUTLINE: [f32; 4] = [0.42, 0.30, 0.22, 1.0]; // dark brown (row 3 left)
pub(crate) const COLOR_RAIL: [f32; 4] = [0.48, 0.35, 0.28, 1.0];        // deep brown (row 3)
pub(crate) const COLOR_RAIL_TIE: [f32; 4] = [0.38, 0.28, 0.20, 1.0];    // darker brown
pub(crate) const COLOR_RESIDENTIAL: [f32; 4] = [0.94, 0.88, 0.82, 1.0]; // pale peach (row 1 right)
pub(crate) const COLOR_COMMERCIAL: [f32; 4] = [0.88, 0.77, 0.68, 1.0];  // soft peach (row 1 col 5-6)
pub(crate) const COLOR_INDUSTRIAL: [f32; 4] = [0.75, 0.78, 0.70, 1.0];  // pale sage (row 2 right)
pub(crate) const COLOR_BUILDING_OUTLINE: [f32; 4] = [0.30, 0.22, 0.18, 1.0]; // warm dark brown
// Shadow colors: black with varying alpha for transparent darkening
pub(crate) const COLOR_SHADOW_CORE: [f32; 4] = [0.0, 0.0, 0.0, 0.22];
pub(crate) const COLOR_SHADOW_MID: [f32; 4] = [0.0, 0.0, 0.0, 0.12];
pub(crate) const COLOR_SHADOW_EDGE: [f32; 4] = [0.0, 0.0, 0.0, 0.0];

/// Shadow offset per unit of z-height. Near-noon sun — shadows stay tight to buildings.
pub(crate) const SHADOW_DIR: [f32; 2] = [0.10, -0.15];
/// Blur radius for shadow penumbra (world units).
pub(crate) const SHADOW_BLUR: f32 = 0.08;

/// Projection scale: 1 degree ≈ 10000 world units ≈ 111m per world unit
pub(crate) const PROJ_SCALE: f64 = 10000.0;

// Z-layers (all close to z=0, spaced to avoid z-fighting)
pub(crate) const Z_LANDUSE: f32 = -0.005;
pub(crate) const Z_LANDUSE_DETAIL: f32 = -0.0045; // commercial/industrial/residential (above base landuse)
pub(crate) const Z_PARK: f32 = -0.003;
pub(crate) const Z_WATER: f32 = -0.002;
pub(crate) const Z_HARDSCAPE: f32 = -0.001;
pub(crate) const Z_PATH_OUTLINE: f32 = 0.001;
pub(crate) const Z_PATH_FILL: f32 = 0.002;
pub(crate) const Z_ROAD_OUTLINE: f32 = 0.003;
pub(crate) const Z_ROAD_FILL: f32 = 0.004;
pub(crate) const Z_PARCEL: f32 = 0.005; // above roads, below buildings

pub(crate) const COLOR_PARCEL: [f32; 4] = [0.55, 0.48, 0.38, 0.4]; // warm brown semi-transparent

/// Project lat/lon to world coordinates (public wrapper for external use).
pub fn project_pub(lat: f64, lon: f64, center_lat: f64, center_lon: f64) -> [f32; 2] {
    project(lat, lon, center_lat, center_lon)
}

/// Convert world coordinates back to lat/lon.
pub fn unproject_pub(world_x: f32, world_y: f32, center_lat: f64, center_lon: f64) -> (f64, f64) {
    let lat = center_lat + (world_y as f64) / PROJ_SCALE;
    let cos_lat = (center_lat * std::f64::consts::PI / 180.0).cos();
    let lon = center_lon + (world_x as f64) / (PROJ_SCALE * cos_lat);
    (lat, lon)
}

/// Project lat/lon to world coordinates centered on (center_lat, center_lon).
pub(crate) fn project(lat: f64, lon: f64, center_lat: f64, center_lon: f64) -> [f32; 2] {
    let cos_lat = (center_lat * std::f64::consts::PI / 180.0).cos();
    let x = (lon - center_lon) * cos_lat * PROJ_SCALE;
    let y = (lat - center_lat) * PROJ_SCALE; // positive Y = north (camera handles orientation)
    [x as f32, y as f32]
}

/// Default API base URL for cached map data.
/// Default API base URL. Override via `api_base` in config.
pub const API_BASE: &str = "";

/// Build the Overpass query for a bounding box.
pub fn overpass_query(south: f64, west: f64, north: f64, east: f64) -> String {
    format!(
        r#"[out:json][timeout:90];
(
  way["natural"="water"]({s},{w},{n},{e});
  way["water"]({s},{w},{n},{e});
  way["waterway"="riverbank"]({s},{w},{n},{e});
  way["waterway"="dock"]({s},{w},{n},{e});
  way["waterway"="canal"]({s},{w},{n},{e});
  way["waterway"="river"]({s},{w},{n},{e});
  way["waterway"="stream"]({s},{w},{n},{e});
  way["landuse"]({s},{w},{n},{e});
  way["leisure"="park"]({s},{w},{n},{e});
  way["leisure"="garden"]({s},{w},{n},{e});
  way["natural"="wood"]({s},{w},{n},{e});
  way["building"]({s},{w},{n},{e});
  way["highway"]({s},{w},{n},{e});
  way["man_made"="bridge"]({s},{w},{n},{e});
  way["man_made"="pier"]({s},{w},{n},{e});
  way["man_made"="quay"]({s},{w},{n},{e});
  way["area:highway"]({s},{w},{n},{e});
  way["leisure"="playground"]({s},{w},{n},{e});
  way["leisure"="pitch"]({s},{w},{n},{e});
  way["place"="square"]({s},{w},{n},{e});
  way["tourism"="attraction"]["area"="yes"]({s},{w},{n},{e});
  way["amenity"="parking"]({s},{w},{n},{e});
  way["amenity"="fountain"]({s},{w},{n},{e});
  node["amenity"="fountain"]({s},{w},{n},{e});
  node["natural"="tree"]({s},{w},{n},{e});
  relation["natural"="water"]({s},{w},{n},{e});
  relation["waterway"="riverbank"]({s},{w},{n},{e});
  relation["waterway"="dock"]({s},{w},{n},{e});
  relation["water"]({s},{w},{n},{e});
  relation["building"]({s},{w},{n},{e});
  relation["leisure"="park"]({s},{w},{n},{e});
  relation["leisure"="garden"]({s},{w},{n},{e});
  relation["man_made"="bridge"]({s},{w},{n},{e});
  relation["man_made"="pier"]({s},{w},{n},{e});
  relation["landuse"]({s},{w},{n},{e});
);
out body;
>;
out skel qt;"#,
        s = south,
        w = west,
        n = north,
        e = east
    )
}

/// Fetch OSM data from API (native only).
#[cfg(not(target_arch = "wasm32"))]
pub fn fetch_osm_data(south: f64, west: f64, north: f64, east: f64) -> Result<MapData, String> {
    let api_url = format!(
        "{}/map/osm?south={}&west={}&north={}&east={}",
        API_BASE, south, west, north, east
    );

    let response = ureq::get(&api_url)
        .timeout(std::time::Duration::from_secs(10))
        .call()
        .map_err(|e| format!("API request failed: {}", e))?;

    let body = response
        .into_string()
        .map_err(|e| format!("Failed to read response: {}", e))?;

    parse_osm_json(&body, south, west, north, east)
}

/// Parse OSM JSON response into MapData. Uses bbox center for projection.
pub fn parse_osm_json(body: &str, south: f64, west: f64, north: f64, east: f64) -> Result<MapData, String> {
    let center_lat = (south + north) / 2.0;
    let center_lon = (west + east) / 2.0;
    parse_osm_json_centered(body, south, west, north, east, center_lat, center_lon)
}

/// Parse OSM JSON with an explicit projection center. Used by tile system
/// so all tiles share the same coordinate space.
pub fn parse_osm_json_centered(
    body: &str, south: f64, west: f64, north: f64, east: f64,
    center_lat: f64, center_lon: f64,
) -> Result<MapData, String> {

    let json: serde_json::Value =
        serde_json::from_str(body).map_err(|e| format!("Failed to parse JSON: {}", e))?;

    let elements = json["elements"]
        .as_array()
        .ok_or("No elements in response")?;


    // Phase 1: Build node lookup table and way lookup table
    let mut nodes: HashMap<u64, (f64, f64)> = HashMap::new();
    let mut way_nodes: HashMap<u64, Vec<u64>> = HashMap::new();

    for elem in elements {
        match elem["type"].as_str() {
            Some("node") => {
                if let (Some(id), Some(lat), Some(lon)) = (
                    elem["id"].as_u64(),
                    elem["lat"].as_f64(),
                    elem["lon"].as_f64(),
                ) {
                    nodes.insert(id, (lat, lon));
                }
            }
            Some("way") => {
                if let (Some(id), Some(node_ids)) = (elem["id"].as_u64(), elem["nodes"].as_array()) {
                    let ids: Vec<u64> = node_ids.iter().filter_map(|v| v.as_u64()).collect();
                    way_nodes.insert(id, ids);
                }
            }
            _ => {}
        }
    }


    // Helper: resolve a way's node IDs to projected coordinates
    let resolve_way = |way_id: u64| -> Vec<[f32; 2]> {
        way_nodes.get(&way_id).map_or_else(Vec::new, |nids| {
            nids.iter()
                .filter_map(|nid| {
                    let (lat, lon) = nodes.get(nid)?;
                    Some(project(*lat, *lon, center_lat, center_lon))
                })
                .collect()
        })
    };

    // Collect tree and fountain positions from tagged nodes
    let mut tree_positions: Vec<[f32; 2]> = Vec::new();
    let mut fountain_positions: Vec<[f32; 2]> = Vec::new();
    for elem in elements {
        if elem["type"].as_str() == Some("node") {
            let tags = &elem["tags"];
            if tags.is_null() {
                continue;
            }
            if let (Some(lat), Some(lon)) = (elem["lat"].as_f64(), elem["lon"].as_f64()) {
                if tags.get("natural").and_then(|v| v.as_str()) == Some("tree") {
                    tree_positions.push(project(lat, lon, center_lat, center_lon));
                }
                if tags.get("amenity").and_then(|v| v.as_str()) == Some("fountain") {
                    fountain_positions.push(project(lat, lon, center_lat, center_lon));
                }
            }
        }
    }

    // Phase 2: Process ways and relations into categorized geometry
    let mut water_ways: Vec<Vec<[f32; 2]>> = Vec::new();
    let mut water_lines: Vec<(Vec<[f32; 2]>, f32)> = Vec::new();
    let mut hardscape_ways: Vec<Vec<[f32; 2]>> = Vec::new();
    let mut landuse_ways: Vec<(Vec<[f32; 2]>, [f32; 4])> = Vec::new();
    let mut park_ways: Vec<(Vec<[f32; 2]>, Option<String>)> = Vec::new();
    let mut building_ways: Vec<(Vec<[f32; 2]>, Option<String>, f32)> = Vec::new();
    let mut road_ways: Vec<(Vec<[f32; 2]>, RoadType, Option<String>)> = Vec::new();

    // Helper closure to classify a polygon (coords + tags) into the right category
    let classify_and_push = |coords: Vec<[f32; 2]>,
                              tags: &serde_json::Value,
                              water_ways: &mut Vec<Vec<[f32; 2]>>,
                              water_lines: &mut Vec<(Vec<[f32; 2]>, f32)>,
                              hardscape_ways: &mut Vec<Vec<[f32; 2]>>,
                              landuse_ways: &mut Vec<(Vec<[f32; 2]>, [f32; 4])>,
                              park_ways: &mut Vec<(Vec<[f32; 2]>, Option<String>)>,
                              building_ways: &mut Vec<(Vec<[f32; 2]>, Option<String>, f32)>,
                              road_ways: &mut Vec<(Vec<[f32; 2]>, RoadType, Option<String>)>,
                              fountain_positions: &mut Vec<[f32; 2]>| {
        if coords.len() < 2 {
            return;
        }

        let name = tags.get("name").and_then(|v| v.as_str()).map(|s| s.to_string());

        // Detect fountain ways — render as water + add fountain geometry
        if tags.get("amenity").and_then(|v| v.as_str()) == Some("fountain") {
            if is_closed(&coords) && coords.len() >= 4 {
                let centroid = polygon_centroid(&coords);
                fountain_positions.push(centroid);
                water_ways.push(coords);
            }
            return;
        }

        if tags.get("highway").is_some() {
            let road_type = classify_road(tags["highway"].as_str().unwrap_or(""));
            road_ways.push((coords, road_type, name));
        } else if tags.get("natural").and_then(|v: &serde_json::Value| v.as_str()) == Some("water")
            || tags.get("waterway").is_some()
            || tags.get("water").is_some()
        {
            // Skip underground/covered waterways (tunnels, culverts)
            let is_underground = tags.get("tunnel").is_some()
                || tags.get("covered").and_then(|v| v.as_str()) == Some("yes")
                || tags.get("layer").and_then(|v| v.as_str())
                    .and_then(|l| l.parse::<i32>().ok())
                    .map_or(false, |l| l < 0);

            if is_closed(&coords) && coords.len() >= 4 && !is_underground {
                water_ways.push(coords);
            } else if coords.len() >= 2 && !is_underground {
                let waterway_type = tags.get("waterway").and_then(|v| v.as_str()).unwrap_or("");
                let width = match waterway_type {
                    "river" => 3.0,
                    "canal" => 1.8,
                    "stream" => 0.6,
                    _ => 1.2,
                };
                water_lines.push((coords, width));
            }
        } else if tags.get("leisure").is_some()
            || tags.get("natural").and_then(|v: &serde_json::Value| v.as_str()) == Some("wood")
        {
            if is_closed(&coords) && coords.len() >= 4 {
                park_ways.push((coords, name));
            }
        } else if tags.get("landuse").is_some() {
            if is_closed(&coords) && coords.len() >= 4 {
                landuse_ways.push((coords, classify_landuse(tags)));
            }
        } else if tags.get("man_made").and_then(|v| v.as_str()).map_or(false, |v| {
            matches!(v, "bridge" | "pier" | "quay")
        }) || tags.get("place").and_then(|v| v.as_str()) == Some("square")
           || tags.get("area:highway").is_some()
           || tags.get("amenity").and_then(|v| v.as_str()) == Some("parking")
        {
            if is_closed(&coords) && coords.len() >= 4 {
                hardscape_ways.push(coords);
            }
        } else if tags.get("building").is_some() {
            if is_closed(&coords) && coords.len() >= 4 {
                let height = extract_building_height(tags);
                building_ways.push((coords, name, height));
            }
        }
    };

    // Process simple ways
    for elem in elements {
        if elem["type"].as_str() != Some("way") {
            continue;
        }
        let tags = &elem["tags"];
        if tags.is_null() {
            continue;
        }
        let node_ids: &Vec<serde_json::Value> = match elem["nodes"].as_array() {
            Some(ids) => ids,
            None => continue,
        };
        let coords: Vec<[f32; 2]> = node_ids
            .iter()
            .filter_map(|id: &serde_json::Value| {
                let id = id.as_u64()?;
                let (lat, lon) = nodes.get(&id)?;
                Some(project(*lat, *lon, center_lat, center_lon))
            })
            .collect();

        classify_and_push(
            coords, tags,
            &mut water_ways, &mut water_lines, &mut hardscape_ways,
            &mut landuse_ways, &mut park_ways, &mut building_ways, &mut road_ways,
            &mut fountain_positions,
        );
    }

    // Process relations (multipolygons)
    let mut relation_count = 0u32;
    for elem in elements {
        if elem["type"].as_str() != Some("relation") {
            continue;
        }
        let tags = &elem["tags"];
        if tags.is_null() {
            continue;
        }
        let members = match elem["members"].as_array() {
            Some(m) => m,
            None => continue,
        };

        // Collect outer way members and assemble them into polygons
        let mut outer_rings: Vec<Vec<[f32; 2]>> = Vec::new();
        let mut pending_segments: Vec<Vec<[f32; 2]>> = Vec::new();

        for member in members {
            if member["type"].as_str() != Some("way") {
                continue;
            }
            let role = member["role"].as_str().unwrap_or("outer");
            if role == "inner" {
                continue; // Skip holes for now
            }
            let way_id = match member["ref"].as_u64() {
                Some(id) => id,
                None => continue,
            };
            let coords = resolve_way(way_id);
            if coords.len() < 2 {
                continue;
            }

            if is_closed(&coords) && coords.len() >= 4 {
                // This way is already a complete ring
                outer_rings.push(coords);
            } else {
                // Partial segment — try to join with pending segments
                pending_segments.push(coords);
            }
        }

        // Try to assemble pending segments into closed rings
        while !pending_segments.is_empty() {
            let mut ring = pending_segments.remove(0);
            let mut changed = true;
            while changed {
                changed = false;
                let mut i = 0;
                while i < pending_segments.len() {
                    let seg = &pending_segments[i];
                    let ring_start = ring[0];
                    let ring_end = ring[ring.len() - 1];
                    let seg_start = seg[0];
                    let seg_end = seg[seg.len() - 1];

                    let dist_sq = |a: [f32; 2], b: [f32; 2]| {
                        (a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1])
                    };

                    if dist_sq(ring_end, seg_start) < 0.001 {
                        // Append segment to end of ring
                        let mut seg = pending_segments.remove(i);
                        seg.remove(0); // remove duplicate junction point
                        ring.extend(seg);
                        changed = true;
                    } else if dist_sq(ring_end, seg_end) < 0.001 {
                        // Append reversed segment to end
                        let mut seg = pending_segments.remove(i);
                        seg.reverse();
                        seg.remove(0);
                        ring.extend(seg);
                        changed = true;
                    } else if dist_sq(ring_start, seg_end) < 0.001 {
                        // Prepend segment to start
                        let mut seg = pending_segments.remove(i);
                        seg.pop(); // remove duplicate junction point
                        seg.extend(ring);
                        ring = seg;
                        changed = true;
                    } else if dist_sq(ring_start, seg_start) < 0.001 {
                        // Prepend reversed segment
                        let mut seg = pending_segments.remove(i);
                        seg.reverse();
                        seg.pop();
                        seg.extend(ring);
                        ring = seg;
                        changed = true;
                    } else {
                        i += 1;
                    }
                }
            }

            if is_closed(&ring) && ring.len() >= 4 {
                outer_rings.push(ring);
            }
        }

        // Classify each assembled ring with the relation's tags
        for ring in outer_rings {
            relation_count += 1;
            classify_and_push(
                ring, tags,
                &mut water_ways, &mut water_lines, &mut hardscape_ways,
                &mut landuse_ways, &mut park_ways, &mut building_ways, &mut road_ways,
                &mut fountain_positions,
            );
        }
    }


    // Deduplicate buildings: only remove true duplicates (same building from way + relation)
    // Requires: centroid inside the other AND similar area (within 3x ratio)
    {
        let before = building_ways.len();
        let mut deduped: Vec<(Vec<[f32; 2]>, Option<String>, f32)> = Vec::new();
        for (coords, name, height) in building_ways.drain(..) {
            let centroid = polygon_centroid(&coords);
            let area = polygon_area(&coords).abs();
            let mut replaced = false;
            for existing in &mut deduped {
                let ec = polygon_centroid(&existing.0);
                let existing_area = polygon_area(&existing.0).abs();

                // Check if either centroid falls inside the other polygon
                let centroids_overlap = point_in_polygon(centroid, &existing.0)
                    || point_in_polygon(ec, &coords);

                // Only dedup if areas are similar (true duplicates, not neighbors)
                let area_ratio = if area > 1e-8 && existing_area > 1e-8 {
                    (area / existing_area).max(existing_area / area)
                } else {
                    f32::MAX
                };

                if centroids_overlap && area_ratio < 3.0 {
                    // Keep the taller one; if same height, keep larger footprint
                    if height > existing.2 || (height == existing.2 && coords.len() > existing.0.len()) {
                        *existing = (coords.clone(), name.clone(), height);
                    }
                    replaced = true;
                    break;
                }
            }
            if !replaced {
                deduped.push((coords, name, height));
            }
        }
        building_ways = deduped;
    }

    // Phase 3: Build GPU geometry in render order (back to front)
    let mut vertices: Vec<MapVertex> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    let mut shadow_vertices: Vec<MapVertex> = Vec::new();
    let mut shadow_indices: Vec<u32> = Vec::new();
    let mut labels: Vec<Label> = Vec::new();

    // Layer 0: Ground plane — solid land covering the entire bounding box (plus margin)
    let (map_min_x, map_min_y, map_max_x, map_max_y);
    {
        let (min_x, min_y, max_x, max_y) = compute_bounds_all(&water_ways, &building_ways, &road_ways, &park_ways, &hardscape_ways, center_lat, center_lon);
        map_min_x = min_x;
        map_min_y = min_y;
        map_max_x = max_x;
        map_max_y = max_y;
        // No ground plane — the clear color acts as the background.
        // This prevents the "tan frame" overlay issue in tile mode.
    }

    // Tile clipping bbox in world coordinates (prevents cross-tile overlay)
    let clip_min = project(south, west, center_lat, center_lon);
    let clip_max = project(north, east, center_lat, center_lon);
    let clip = (clip_min[0], clip_min[1], clip_max[0], clip_max[1]);

    let z_landuse = Z_LANDUSE;
    let z_park = Z_PARK;
    let z_water = Z_WATER;
    let z_hardscape = Z_HARDSCAPE;
    let z_road_outline = Z_ROAD_OUTLINE;
    let z_road_fill = Z_ROAD_FILL;
    let z_path_outline = Z_PATH_OUTLINE;
    let z_path_fill = Z_PATH_FILL;

    // Layer 1: Water polygons (clipped to tile bbox)
    for coords in &water_ways {
        let clipped = clip_polygon_to_bbox(coords, clip.0, clip.1, clip.2, clip.3);
        if clipped.len() >= 3 {
            triangulate_polygon_at_height(&clipped, z_water, COLOR_WATER, MAT_WATER, &mut vertices, &mut indices);
        }
    }

    // Layer 1b: Linear waterways (surface only — underground filtered out during classification)
    for (coords, width) in &water_lines {
        let start_vert = vertices.len();
        generate_line_geometry(coords, *width, COLOR_WATER, MAT_WATER, &mut vertices, &mut indices);
        for v in &mut vertices[start_vert..] {
            v.position[2] = z_water;
        }
    }

    // Layer 2: Landuse areas (clipped to tile bbox)
    for (coords, color) in &landuse_ways {
        let clipped = clip_polygon_to_bbox(coords, clip.0, clip.1, clip.2, clip.3);
        if clipped.len() >= 3 {
            triangulate_polygon_at_height(&clipped, z_landuse, *color, MAT_DEFAULT, &mut vertices, &mut indices);
        }
    }

    // Layer 2b: Hardscape (clipped to tile bbox)
    for coords in &hardscape_ways {
        let clipped = clip_polygon_to_bbox(coords, clip.0, clip.1, clip.2, clip.3);
        if clipped.len() >= 3 {
            triangulate_polygon_at_height(&clipped, z_hardscape, COLOR_SIDEWALK, MAT_DEFAULT, &mut vertices, &mut indices);
        }
    }

    // Layer 3: Parks (clipped to tile bbox)
    for (coords, name) in &park_ways {
        let clipped = clip_polygon_to_bbox(coords, clip.0, clip.1, clip.2, clip.3);
        if clipped.len() < 3 { continue; }
        triangulate_polygon_at_height(&clipped, z_park, COLOR_PARK, MAT_GRASS, &mut vertices, &mut indices);
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

    // Layer 3.5: Building shadows with soft edges
    // Shadow = footprint + offset roof connected, so it stays attached to the building base.
    for (coords, _, height) in &building_ways {
        let z = height / 11.1;
        if !is_closed(coords) || coords.len() < 4 {
            continue;
        }

        // Build a hull that covers both the footprint and the offset roof projection.
        // For each vertex: the shadow extends from the base vertex to base+offset.
        // We approximate this by emitting quads between each footprint edge and its
        // offset counterpart, plus filling both the footprint and offset polygons.
        let offset: Vec<[f32; 2]> = coords.iter()
            .map(|p| [p[0] + SHADOW_DIR[0] * z, p[1] + SHADOW_DIR[1] * z])
            .collect();
        let centroid = polygon_centroid(coords);

        // Outer penumbra (expanded from offset polygon)
        let outer = expand_polygon(&offset, centroid, SHADOW_BLUR * 2.0);
        triangulate_polygon_at_height(&outer, 0.001, COLOR_SHADOW_EDGE, MAT_DEFAULT, &mut shadow_vertices, &mut shadow_indices);
        // Mid penumbra
        let mid = expand_polygon(&offset, centroid, SHADOW_BLUR);
        triangulate_polygon_at_height(&mid, 0.0011, COLOR_SHADOW_MID, MAT_DEFAULT, &mut shadow_vertices, &mut shadow_indices);

        // Core shadow: fill the footprint
        triangulate_polygon_at_height(coords, 0.0012, COLOR_SHADOW_CORE, MAT_DEFAULT, &mut shadow_vertices, &mut shadow_indices);
        // Core shadow: fill the offset roof projection
        triangulate_polygon_at_height(&offset, 0.0012, COLOR_SHADOW_CORE, MAT_DEFAULT, &mut shadow_vertices, &mut shadow_indices);
        // Core shadow: connect footprint to offset with quads (the "sweep" between them)
        let pts = if coords.len() > 1 && coords[0] == coords[coords.len() - 1] {
            &coords[..coords.len() - 1]
        } else {
            coords.as_slice()
        };
        for i in 0..pts.len() {
            let j = (i + 1) % pts.len();
            let a = pts[i];
            let b = pts[j];
            let ao = [a[0] + SHADOW_DIR[0] * z, a[1] + SHADOW_DIR[1] * z];
            let bo = [b[0] + SHADOW_DIR[0] * z, b[1] + SHADOW_DIR[1] * z];
            let base = shadow_vertices.len() as u32;
            shadow_vertices.push(MapVertex::at_height(a[0], a[1], 0.0012, COLOR_SHADOW_CORE, MAT_DEFAULT));
            shadow_vertices.push(MapVertex::at_height(b[0], b[1], 0.0012, COLOR_SHADOW_CORE, MAT_DEFAULT));
            shadow_vertices.push(MapVertex::at_height(ao[0], ao[1], 0.0012, COLOR_SHADOW_CORE, MAT_DEFAULT));
            shadow_vertices.push(MapVertex::at_height(bo[0], bo[1], 0.0012, COLOR_SHADOW_CORE, MAT_DEFAULT));
            shadow_indices.extend_from_slice(&[base, base + 1, base + 2, base + 1, base + 3, base + 2]);
        }
    }

    // Layer 4: Buildings — 3D extruded with walls, roof, and outline
    for (coords, _name, height) in &building_ways {
        // Convert real-world height (meters) to world units
        // PROJ_SCALE maps 1 degree ≈ 10000 units ≈ 111km, so 1 unit ≈ 11.1m
        let z = height / 11.1;
        let is_skyscraper = *height >= SKYSCRAPER_HEIGHT;

        let (base_color, roof_mat, wall_mat) = if is_skyscraper {
            (COLOR_SKYSCRAPER, MAT_GLASS, MAT_GLASS_WALL)
        } else {
            (COLOR_BUILDING, MAT_BUILDING, MAT_BUILDING_WALL)
        };

        let color = [base_color[0], base_color[1], base_color[2], 1.0];

        // Walls
        extrude_walls_with_material(coords, z, color, wall_mat, &mut vertices, &mut indices);

        // Roof
        triangulate_polygon_at_height(coords, z, color, roof_mat, &mut vertices, &mut indices);

        // Roof edge outline (tiny z offset to avoid z-fighting with roof surface)
        generate_line_geometry_at_height(coords, 0.03, z + 0.005, COLOR_BUILDING_OUTLINE, MAT_DEFAULT, &mut vertices, &mut indices);
    }

    // Layer 5: Sidewalks / footpaths
    for (coords, road_type, name) in &road_ways {
        if !matches!(road_type, RoadType::Path) {
            continue;
        }
        let base_width = if name.is_some() { 0.25 } else { road_width(road_type) };
        let width = base_width + 0.04;
        let start_vert = vertices.len();
        generate_line_geometry(coords, width, COLOR_SIDEWALK_OUTLINE, MAT_COBBLESTONE, &mut vertices, &mut indices);
        for v in &mut vertices[start_vert..] {
            v.position[2] = z_path_outline;
        }
    }
    for (coords, road_type, name) in &road_ways {
        if !matches!(road_type, RoadType::Path) {
            continue;
        }
        let base_width = if name.is_some() { 0.25 } else { road_width(road_type) };
        let start_vert = vertices.len();
        generate_line_geometry(coords, base_width, COLOR_SIDEWALK, MAT_COBBLESTONE, &mut vertices, &mut indices);
        for v in &mut vertices[start_vert..] {
            v.position[2] = z_path_fill;
        }
    }

    // Layer 6: Road outlines (slightly wider, dark)
    for (coords, road_type, _) in &road_ways {
        if matches!(road_type, RoadType::Path) {
            continue;
        }
        let width = road_width(road_type) + 0.05;
        let start_vert = vertices.len();
        generate_line_geometry(coords, width, COLOR_ROAD_OUTLINE, MAT_ROAD, &mut vertices, &mut indices);
        for v in &mut vertices[start_vert..] {
            v.position[2] = z_road_outline;
        }
    }

    // Layer 7: Road fills (on top of outlines)
    for (coords, road_type, _) in &road_ways {
        if matches!(road_type, RoadType::Path) {
            continue;
        }
        let width = road_width(road_type);
        let color = match road_type {
            RoadType::Major => COLOR_ROAD_MAJOR,
            RoadType::Minor | RoadType::Residential => COLOR_ROAD_MINOR,
            RoadType::Rail => COLOR_RAIL,
            RoadType::Path => continue,
        };
        let start_vert = vertices.len();
        generate_line_geometry(coords, width, color, MAT_ROAD, &mut vertices, &mut indices);
        for v in &mut vertices[start_vert..] {
            v.position[2] = z_road_fill;
        }
    }

    // Extract road labels — place at regular intervals along each road segment
    let label_spacing = 12.0; // minimum world units between labels of the same name
    for (coords, road_type, name) in &road_ways {
        if let Some(name) = name {
            if matches!(road_type, RoadType::Path) {
                continue;
            }
            if coords.len() < 2 {
                continue;
            }

            // Walk the polyline, placing labels every `label_spacing` units
            let mut walked = 0.0f32;
            let mut next_label_at = label_spacing * 0.5; // first label at half spacing
            for i in 0..coords.len() - 1 {
                let dx = coords[i + 1][0] - coords[i][0];
                let dy = coords[i + 1][1] - coords[i][1];
                let seg_len = (dx * dx + dy * dy).sqrt();

                while walked + seg_len >= next_label_at && seg_len > 1e-8 {
                    let t = (next_label_at - walked) / seg_len;
                    let pos = [
                        coords[i][0] + dx * t,
                        coords[i][1] + dy * t,
                    ];
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
                        path: None,
                    });
                    next_label_at += label_spacing;
                }

                walked += seg_len;
            }
        }
    }

    // Building labels (only named buildings)
    for (coords, name, _height) in &building_ways {
        if let Some(name) = name {
            let center = polygon_centroid(coords);
            labels.push(Label {
                text: name.clone(),
                position: center,
                angle: 0.0,
                kind: LabelKind::Building,
                path: None,
            });
        }
    }

    // Tree shadows (into shadow buffers)
    for pos in &tree_positions {
        generate_tree_shadow(*pos, &mut shadow_vertices, &mut shadow_indices);
    }

    // Layer 8: Trees
    for pos in &tree_positions {
        generate_tree(*pos, &mut vertices, &mut indices);
    }

    // Layer 9: Fountains
    for pos in &fountain_positions {
        generate_fountain(*pos, z_water, &mut vertices, &mut indices);
    }

    // (clouds removed)

    Ok(MapData {
        vertices,
        indices,
        shadow_vertices,
        shadow_indices,
        labels,
        listings: Vec::new(),
        cars: Vec::new(),
        center_lat,
        center_lon,
    })
}

// --- Helpers ---

/// Expand a polygon outward from its centroid by a fixed offset distance.
pub(crate) fn expand_polygon(coords: &[[f32; 2]], centroid: [f32; 2], offset: f32) -> Vec<[f32; 2]> {
    coords.iter().map(|p| {
        let dx = p[0] - centroid[0];
        let dy = p[1] - centroid[1];
        let len = (dx * dx + dy * dy).sqrt();
        if len < 1e-8 {
            *p
        } else {
            [p[0] + dx / len * offset, p[1] + dy / len * offset]
        }
    }).collect()
}

/// Compute the bounding box of all geometry in world coordinates.
fn compute_bounds_all(
    water: &[Vec<[f32; 2]>],
    buildings: &[(Vec<[f32; 2]>, Option<String>, f32)],
    roads: &[(Vec<[f32; 2]>, RoadType, Option<String>)],
    parks: &[(Vec<[f32; 2]>, Option<String>)],
    hardscape: &[Vec<[f32; 2]>],
    _center_lat: f64,
    _center_lon: f64,
) -> (f32, f32, f32, f32) {
    let mut min_x = f32::MAX;
    let mut min_y = f32::MAX;
    let mut max_x = f32::MIN;
    let mut max_y = f32::MIN;

    let mut update = |coords: &[[f32; 2]]| {
        for p in coords {
            min_x = min_x.min(p[0]);
            min_y = min_y.min(p[1]);
            max_x = max_x.max(p[0]);
            max_y = max_y.max(p[1]);
        }
    };

    for c in water { update(c); }
    for (c, _, _) in buildings { update(c); }
    for (c, _, _) in roads { update(c); }
    for (c, _) in parks { update(c); }
    for c in hardscape { update(c); }

    (min_x, min_y, max_x, max_y)
}

/// Signed area of a polygon (shoelace formula).
/// Clip a polygon to a bounding box using Sutherland-Hodgman algorithm.
pub(crate) fn clip_polygon_to_bbox(coords: &[[f32; 2]], min_x: f32, min_y: f32, max_x: f32, max_y: f32) -> Vec<[f32; 2]> {
    let mut output = coords.to_vec();

    // Clip against each edge: left, right, bottom, top
    let edges: [(fn([f32;2], f32) -> bool, fn([f32;2], [f32;2], f32) -> [f32;2], f32); 4] = [
        (|p, v| p[0] >= v, |a, b, v| { let t = (v - a[0]) / (b[0] - a[0]); [v, a[1] + t * (b[1] - a[1])] }, min_x),
        (|p, v| p[0] <= v, |a, b, v| { let t = (v - a[0]) / (b[0] - a[0]); [v, a[1] + t * (b[1] - a[1])] }, max_x),
        (|p, v| p[1] >= v, |a, b, v| { let t = (v - a[1]) / (b[1] - a[1]); [a[0] + t * (b[0] - a[0]), v] }, min_y),
        (|p, v| p[1] <= v, |a, b, v| { let t = (v - a[1]) / (b[1] - a[1]); [a[0] + t * (b[0] - a[0]), v] }, max_y),
    ];

    for (inside, intersect, val) in &edges {
        if output.is_empty() { break; }
        let input = output;
        output = Vec::with_capacity(input.len() + 4);

        let n = input.len();
        for i in 0..n {
            let current = input[i];
            let prev = input[(i + n - 1) % n];
            let cur_in = inside(current, *val);
            let prev_in = inside(prev, *val);

            if prev_in && cur_in {
                output.push(current);
            } else if prev_in {
                output.push(intersect(prev, current, *val));
            } else if cur_in {
                output.push(intersect(prev, current, *val));
                output.push(current);
            }
        }
    }

    output
}

pub(crate) fn polygon_area(coords: &[[f32; 2]]) -> f32 {
    let n = coords.len();
    if n < 3 { return 0.0; }
    let mut area = 0.0f32;
    for i in 0..n {
        let j = (i + 1) % n;
        area += coords[i][0] * coords[j][1];
        area -= coords[j][0] * coords[i][1];
    }
    area * 0.5
}

/// Perimeter of a polygon.
fn polygon_perimeter(coords: &[[f32; 2]]) -> f32 {
    let n = coords.len();
    if n < 2 { return 0.0; }
    let mut perimeter = 0.0f32;
    for i in 0..n {
        let j = (i + 1) % n;
        let dx = coords[j][0] - coords[i][0];
        let dy = coords[j][1] - coords[i][1];
        perimeter += (dx * dx + dy * dy).sqrt();
    }
    perimeter
}

pub(crate) fn polygon_centroid(coords: &[[f32; 2]]) -> [f32; 2] {
    let n = coords.len() as f32;
    if n < 1.0 {
        return [0.0, 0.0];
    }
    let sum = coords.iter().fold([0.0f32, 0.0f32], |acc, p| {
        [acc[0] + p[0], acc[1] + p[1]]
    });
    [sum[0] / n, sum[1] / n]
}

/// Get the midpoint along a polyline (by arc length) and the road angle at that point.
/// The angle is normalized so text always reads left-to-right.
fn polyline_midpoint_and_angle(coords: &[[f32; 2]]) -> ([f32; 2], f32) {
    if coords.len() < 2 {
        return (coords.first().copied().unwrap_or([0.0, 0.0]), 0.0);
    }

    // Total length
    let mut total_len = 0.0f32;
    for i in 0..coords.len() - 1 {
        let dx = coords[i + 1][0] - coords[i][0];
        let dy = coords[i + 1][1] - coords[i][1];
        total_len += (dx * dx + dy * dy).sqrt();
    }

    // Walk to the midpoint
    let half = total_len * 0.5;
    let mut walked = 0.0f32;
    for i in 0..coords.len() - 1 {
        let dx = coords[i + 1][0] - coords[i][0];
        let dy = coords[i + 1][1] - coords[i][1];
        let seg_len = (dx * dx + dy * dy).sqrt();
        if walked + seg_len >= half && seg_len > 1e-8 {
            let t = (half - walked) / seg_len;
            let pos = [
                coords[i][0] + dx * t,
                coords[i][1] + dy * t,
            ];
            // Angle of the segment direction.
            // Negate because text pixel space has Y flipped vs world space.
            let mut angle = (-dy).atan2(dx);
            // Normalize so text reads left-to-right (angle in -π/2..π/2)
            if angle > std::f32::consts::FRAC_PI_2 {
                angle -= std::f32::consts::PI;
            } else if angle < -std::f32::consts::FRAC_PI_2 {
                angle += std::f32::consts::PI;
            }
            return (pos, angle);
        }
        walked += seg_len;
    }

    (coords[coords.len() / 2], 0.0)
}

/// Ray-casting point-in-polygon test.
fn point_in_polygon(point: [f32; 2], polygon: &[[f32; 2]]) -> bool {
    let (px, py) = (point[0], point[1]);
    let mut inside = false;
    let n = polygon.len();
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = (polygon[i][0], polygon[i][1]);
        let (xj, yj) = (polygon[j][0], polygon[j][1]);
        if ((yi > py) != (yj > py)) && (px < (xj - xi) * (py - yi) / (yj - yi) + xi) {
            inside = !inside;
        }
        j = i;
    }
    inside
}

pub(crate) fn is_closed(coords: &[[f32; 2]]) -> bool {
    if coords.len() < 3 {
        return false;
    }
    let first = coords[0];
    let last = coords[coords.len() - 1];
    (first[0] - last[0]).abs() < 1e-6 && (first[1] - last[1]).abs() < 1e-6
}

#[derive(Clone, Copy)]
pub(crate) enum RoadType {
    Major,
    Minor,
    Residential,
    Path,
    Rail,
}

pub(crate) fn classify_road(highway: &str) -> RoadType {
    match highway {
        "motorway" | "trunk" | "primary" | "motorway_link" | "trunk_link" | "primary_link" => {
            RoadType::Major
        }
        "secondary" | "tertiary" | "secondary_link" | "tertiary_link" => RoadType::Minor,
        "footway" | "cycleway" | "path" | "pedestrian" | "steps" | "service" | "track"
        | "corridor" | "bridleway" => RoadType::Path,
        _ => RoadType::Residential,
    }
}

pub(crate) fn road_width(road_type: &RoadType) -> f32 {
    match road_type {
        RoadType::Major => 0.70,
        RoadType::Minor => 0.45,
        RoadType::Residential => 0.30,
        RoadType::Path => 0.16,
        RoadType::Rail => 0.10,
    }
}

fn classify_landuse(tags: &serde_json::Value) -> [f32; 4] {
    match tags.get("landuse").and_then(|v| v.as_str()) {
        Some("residential") => COLOR_RESIDENTIAL,
        Some("commercial" | "retail") => COLOR_COMMERCIAL,
        Some("industrial") => COLOR_INDUSTRIAL,
        Some("forest" | "meadow" | "grass" | "recreation_ground") => COLOR_PARK,
        _ => COLOR_LAND,
    }
}

/// Extract building height from OSM tags (in meters).
fn extract_building_height(tags: &serde_json::Value) -> f32 {
    // Try explicit height tag first
    if let Some(h) = tags.get("height").and_then(|v| v.as_str()) {
        if let Ok(meters) = h.trim_end_matches(" m").parse::<f32>() {
            return meters;
        }
    }

    // Try building:levels (assume ~3.2m per level)
    if let Some(levels) = tags.get("building:levels").and_then(|v| v.as_str()) {
        if let Ok(n) = levels.parse::<f32>() {
            return n * 3.2;
        }
    }

    // Default height based on building type
    match tags.get("building").and_then(|v| v.as_str()) {
        Some("church" | "cathedral") => 20.0,
        Some("tower") => 30.0,
        Some("house" | "residential") => 9.0,
        Some("garage" | "shed" | "hut") => 3.5,
        _ => 10.0,
    }
}

/// Generate wall quads for a building extrusion (ground to height z).
fn extrude_walls(
    coords: &[[f32; 2]],
    z: f32,
    color: [f32; 4],
    vertices: &mut Vec<MapVertex>,
    indices: &mut Vec<u32>,
) {
    let pts = if coords.len() > 1 && coords[0] == coords[coords.len() - 1] {
        &coords[..coords.len() - 1]
    } else {
        coords
    };

    if pts.len() < 3 {
        return;
    }

    for i in 0..pts.len() {
        let j = (i + 1) % pts.len();
        let p0 = pts[i];
        let p1 = pts[j];

        // Compute wall normal for simple directional lighting
        let dx = p1[0] - p0[0];
        let dy = p1[1] - p0[1];
        let len = (dx * dx + dy * dy).sqrt();
        if len < 1e-8 {
            continue;
        }

        // Shade walls based on their facing direction (sun from top-left)
        let nx = -dy / len;
        let ny = dx / len;
        let sun_dot = (nx * 0.5 + ny * -0.7).max(0.0);
        let shade = 0.6 + sun_dot * 0.4;

        let wall_color = [
            color[0] * shade,
            color[1] * shade,
            color[2] * shade,
            1.0,
        ];

        let base = vertices.len() as u32;

        // Bottom-left, bottom-right, top-left, top-right
        vertices.push(MapVertex::at_height(p0[0], p0[1], 0.0, wall_color, MAT_BUILDING_WALL));
        vertices.push(MapVertex::at_height(p1[0], p1[1], 0.0, wall_color, MAT_BUILDING_WALL));
        vertices.push(MapVertex::at_height(p0[0], p0[1], z, wall_color, MAT_BUILDING_WALL));
        vertices.push(MapVertex::at_height(p1[0], p1[1], z, wall_color, MAT_BUILDING_WALL));

        indices.extend_from_slice(&[base, base + 1, base + 2, base + 1, base + 3, base + 2]);
    }
}

/// Generate wall quads with a specified material (for skyscrapers vs regular buildings).
pub(crate) fn extrude_walls_with_material(
    coords: &[[f32; 2]],
    z: f32,
    color: [f32; 4],
    material: f32,
    vertices: &mut Vec<MapVertex>,
    indices: &mut Vec<u32>,
) {
    let pts = if coords.len() > 1 && coords[0] == coords[coords.len() - 1] {
        &coords[..coords.len() - 1]
    } else {
        coords
    };

    if pts.len() < 3 {
        return;
    }

    for i in 0..pts.len() {
        let j = (i + 1) % pts.len();
        let p0 = pts[i];
        let p1 = pts[j];

        let dx = p1[0] - p0[0];
        let dy = p1[1] - p0[1];
        let len = (dx * dx + dy * dy).sqrt();
        if len < 1e-8 {
            continue;
        }

        let nx = -dy / len;
        let ny = dx / len;
        let sun_dot = (nx * 0.5 + ny * -0.7).max(0.0);
        let shade = 0.6 + sun_dot * 0.4;

        let wall_color = [
            color[0] * shade,
            color[1] * shade,
            color[2] * shade,
            1.0,
        ];

        let base = vertices.len() as u32;

        vertices.push(MapVertex::at_height(p0[0], p0[1], 0.0, wall_color, material));
        vertices.push(MapVertex::at_height(p1[0], p1[1], 0.0, wall_color, material));
        vertices.push(MapVertex::at_height(p0[0], p0[1], z, wall_color, material));
        vertices.push(MapVertex::at_height(p1[0], p1[1], z, wall_color, material));

        indices.extend_from_slice(&[base, base + 1, base + 2, base + 1, base + 3, base + 2]);
    }
}

/// Triangulate a polygon at a given height (for building roofs).
pub(crate) fn triangulate_polygon_at_height(
    coords: &[[f32; 2]],
    z: f32,
    color: [f32; 4],
    material: f32,
    vertices: &mut Vec<MapVertex>,
    indices: &mut Vec<u32>,
) {
    let pts = if coords.len() > 1 && coords[0] == coords[coords.len() - 1] {
        &coords[..coords.len() - 1]
    } else {
        coords
    };

    if pts.len() < 3 {
        return;
    }

    let flat: Vec<f64> = pts.iter().flat_map(|p| [p[0] as f64, p[1] as f64]).collect();
    let tri_indices = earcutr::earcut(&flat, &[], 2);

    if tri_indices.is_err() {
        return;
    }
    let tri_indices = tri_indices.unwrap();

    let base = vertices.len() as u32;

    for p in pts {
        vertices.push(MapVertex::at_height(p[0], p[1], z, color, material));
    }

    for idx in tri_indices {
        indices.push(base + idx as u32);
    }
}

/// Generate line geometry at a given height (for roof edge outlines).
pub(crate) fn generate_line_geometry_at_height(
    coords: &[[f32; 2]],
    width: f32,
    z: f32,
    color: [f32; 4],
    material: f32,
    vertices: &mut Vec<MapVertex>,
    indices: &mut Vec<u32>,
) {
    // Generate the line geometry normally, then patch the z values
    let start_vert = vertices.len();
    generate_line_geometry(coords, width, color, material, vertices, indices);
    for v in &mut vertices[start_vert..] {
        v.position[2] = z;
    }
}

/// Triangulate a closed polygon using earcut and append to the vertex/index buffers.
fn triangulate_polygon(
    coords: &[[f32; 2]],
    color: [f32; 4],
    material: f32,
    vertices: &mut Vec<MapVertex>,
    indices: &mut Vec<u32>,
) {
    // Remove the closing duplicate point
    let pts = if coords.len() > 1 && coords[0] == coords[coords.len() - 1] {
        &coords[..coords.len() - 1]
    } else {
        coords
    };

    if pts.len() < 3 {
        return;
    }

    // Flatten to f64 array for earcutr
    let flat: Vec<f64> = pts.iter().flat_map(|p| [p[0] as f64, p[1] as f64]).collect();
    let tri_indices = earcutr::earcut(&flat, &[], 2);

    if tri_indices.is_err() {
        return;
    }
    let tri_indices = tri_indices.unwrap();

    let base = vertices.len() as u32;

    for p in pts {
        vertices.push(MapVertex::new(p[0], p[1], color, material));
    }

    for idx in tri_indices {
        indices.push(base + idx as u32);
    }
}

/// Compute the perpendicular normal for a segment (p0 → p1), scaled by half_width.
pub(crate) fn segment_normal(p0: [f32; 2], p1: [f32; 2], half_w: f32) -> Option<[f32; 2]> {
    let dx = p1[0] - p0[0];
    let dy = p1[1] - p0[1];
    let len = (dx * dx + dy * dy).sqrt();
    if len < 1e-8 {
        return None;
    }
    Some([-dy / len * half_w, dx / len * half_w])
}

/// Generate thick line geometry with proper miter joins for a polyline.
/// Produces a continuous triangle strip with no gaps at corners.
pub(crate) fn generate_line_geometry(
    coords: &[[f32; 2]],
    width: f32,
    color: [f32; 4],
    material: f32,
    vertices: &mut Vec<MapVertex>,
    indices: &mut Vec<u32>,
) {
    if coords.len() < 2 {
        return;
    }

    let half_w = width * 0.5;

    // Compute offset points at each vertex using miter joins.
    // At each interior vertex, we average the normals of the two adjacent segments
    // and scale to preserve the correct width.
    let n = coords.len();
    let mut left_points: Vec<[f32; 2]> = Vec::with_capacity(n);
    let mut right_points: Vec<[f32; 2]> = Vec::with_capacity(n);

    for i in 0..n {
        let p = coords[i];

        if i == 0 {
            // Start cap: use first segment's normal
            if let Some(normal) = segment_normal(coords[0], coords[1], half_w) {
                left_points.push([p[0] + normal[0], p[1] + normal[1]]);
                right_points.push([p[0] - normal[0], p[1] - normal[1]]);
            } else {
                left_points.push(p);
                right_points.push(p);
            }
        } else if i == n - 1 {
            // End cap: use last segment's normal
            if let Some(normal) = segment_normal(coords[n - 2], coords[n - 1], half_w) {
                left_points.push([p[0] + normal[0], p[1] + normal[1]]);
                right_points.push([p[0] - normal[0], p[1] - normal[1]]);
            } else {
                left_points.push(p);
                right_points.push(p);
            }
        } else {
            // Interior vertex: compute miter join
            let n0 = segment_normal(coords[i - 1], coords[i], 1.0);
            let n1 = segment_normal(coords[i], coords[i + 1], 1.0);

            match (n0, n1) {
                (Some(na), Some(nb)) => {
                    // Average the two normals
                    let mx = na[0] + nb[0];
                    let my = na[1] + nb[1];
                    let mlen = (mx * mx + my * my).sqrt();

                    if mlen < 1e-6 {
                        // Normals cancel out (180° turn) — use first normal
                        left_points.push([p[0] + na[0] * half_w, p[1] + na[1] * half_w]);
                        right_points.push([p[0] - na[0] * half_w, p[1] - na[1] * half_w]);
                    } else {
                        // Miter vector (normalized)
                        let mx = mx / mlen;
                        let my = my / mlen;

                        // Scale factor: half_w / dot(miter, normal)
                        // This ensures the offset is correct at any angle
                        let dot = mx * na[0] + my * na[1];
                        let scale = if dot.abs() > 0.1 {
                            (half_w / dot).clamp(-half_w * 3.0, half_w * 3.0)
                        } else {
                            half_w // Fallback for very acute angles
                        };

                        left_points.push([p[0] + mx * scale, p[1] + my * scale]);
                        right_points.push([p[0] - mx * scale, p[1] - my * scale]);
                    }
                }
                _ => {
                    left_points.push(p);
                    right_points.push(p);
                }
            }
        }
    }

    // Build triangle strip from left/right offset points
    let base = vertices.len() as u32;

    for i in 0..n {
        vertices.push(MapVertex::new(left_points[i][0], left_points[i][1], color, material));
        vertices.push(MapVertex::new(right_points[i][0], right_points[i][1], color, material));
    }

    for i in 0..(n - 1) as u32 {
        let tl = base + i * 2;
        let tr = base + i * 2 + 1;
        let bl = base + (i + 1) * 2;
        let br = base + (i + 1) * 2 + 1;
        indices.extend_from_slice(&[tl, tr, bl, tr, br, bl]);
    }

    // Round end caps (semicircles at start and end)
    add_round_cap(coords[0], coords[1], half_w, color, material, vertices, indices);
    add_round_cap(coords[n - 1], coords[n - 2], half_w, color, material, vertices, indices);
}

/// Add a round cap (semicircle) at a line endpoint.
pub(crate) fn add_round_cap(
    point: [f32; 2],
    toward: [f32; 2],
    half_w: f32,
    color: [f32; 4],
    material: f32,
    vertices: &mut Vec<MapVertex>,
    indices: &mut Vec<u32>,
) {
    let dx = point[0] - toward[0];
    let dy = point[1] - toward[1];
    let len = (dx * dx + dy * dy).sqrt();
    if len < 1e-8 {
        return;
    }

    // Direction away from the line
    let dir_x = dx / len;
    let dir_y = dy / len;

    // Base angle of the cap direction
    let base_angle = dir_y.atan2(dir_x);
    let segments = 8;
    let base_idx = vertices.len() as u32;

    vertices.push(MapVertex::new(point[0], point[1], color, material));

    for s in 0..=segments {
        let angle = base_angle - std::f32::consts::PI * 0.5
            + std::f32::consts::PI * s as f32 / segments as f32;
        vertices.push(MapVertex::new(
            point[0] + angle.cos() * half_w,
            point[1] + angle.sin() * half_w,
            color,
            material,
        ));
    }

    for s in 0..segments as u32 {
        indices.extend_from_slice(&[base_idx, base_idx + 1 + s, base_idx + 2 + s]);
    }
}

// --- Tree geometry ---

pub(crate) const COLOR_TREE_LEAVES: [f32; 4] = [0.42, 0.50, 0.28, 1.0]; // olive green (row 2)
pub(crate) const COLOR_TREE_TRUNK: [f32; 4] = [0.48, 0.35, 0.25, 1.0]; // warm brown (row 3)

/// Generate a tree shadow — an ellipse on the ground with soft edges via vertex color gradient.
pub(crate) fn generate_tree_shadow(
    pos: [f32; 2],
    vertices: &mut Vec<MapVertex>,
    indices: &mut Vec<u32>,
) {
    // The tree's effective canopy center is around z=0.30, radius ~0.16
    let canopy_z = 0.30;
    let shadow_cx = pos[0] + SHADOW_DIR[0] * canopy_z;
    let shadow_cy = pos[1] + SHADOW_DIR[1] * canopy_z;
    let inner_radius = 0.14;
    let outer_radius = 0.24; // soft fade-out zone

    // Elongate shadow in the sun direction
    let stretch = 1.3;
    let sun_len = (SHADOW_DIR[0] * SHADOW_DIR[0] + SHADOW_DIR[1] * SHADOW_DIR[1]).sqrt();
    let sun_nx = SHADOW_DIR[0] / sun_len;
    let sun_ny = SHADOW_DIR[1] / sun_len;

    let segments = 12u32;
    let pi = std::f32::consts::PI;

    let stretched_offset = |angle: f32, radius: f32| -> (f32, f32) {
        let cx = angle.cos();
        let cy = angle.sin();
        let along_sun = cx * sun_nx + cy * sun_ny;
        let perp_sun = cx * (-sun_ny) + cy * sun_nx;
        let sx = along_sun * stretch * sun_nx + perp_sun * (-sun_ny);
        let sy = along_sun * stretch * sun_ny + perp_sun * sun_nx;
        (shadow_cx + sx * radius, shadow_cy + sy * radius)
    };

    let base = vertices.len() as u32;

    // Center vertex — darkest
    vertices.push(MapVertex::at_height(shadow_cx, shadow_cy, 0.001, COLOR_SHADOW_CORE, MAT_DEFAULT));

    // Inner ring — still dark
    for i in 0..=segments {
        let angle = 2.0 * pi * i as f32 / segments as f32;
        let (x, y) = stretched_offset(angle, inner_radius);
        vertices.push(MapVertex::at_height(x, y, 0.001, COLOR_SHADOW_MID, MAT_DEFAULT));
    }

    // Outer ring — fades to fully transparent
    for i in 0..=segments {
        let angle = 2.0 * pi * i as f32 / segments as f32;
        let (x, y) = stretched_offset(angle, outer_radius);
        vertices.push(MapVertex::at_height(x, y, 0.001, COLOR_SHADOW_EDGE, MAT_DEFAULT));
    }

    // Triangles: center fan to inner ring
    for i in 0..segments {
        indices.extend_from_slice(&[base, base + 1 + i, base + 2 + i]);
    }

    // Triangles: inner ring to outer ring (quad strip)
    let inner_start = base + 1;
    let outer_start = base + 1 + (segments + 1);
    for i in 0..segments {
        let i0 = inner_start + i;
        let i1 = inner_start + i + 1;
        let o0 = outer_start + i;
        let o1 = outer_start + i + 1;
        indices.extend_from_slice(&[i0, o0, i1, i1, o0, o1]);
    }
}

/// Generate a complete tree: brown trunk cylinder + 3 stacked/offset dark green spheres.
pub(crate) fn generate_tree(
    pos: [f32; 2],
    vertices: &mut Vec<MapVertex>,
    indices: &mut Vec<u32>,
) {
    // Deterministic random scale per tree based on position
    let hash = ((pos[0] * 127.1 + pos[1] * 311.7).sin() * 43758.5453).fract();
    let scale = 1.5 + hash * 0.7; // 1.5x to 2.2x size variation

    let trunk_radius = 0.03 * scale;
    let trunk_height = 0.38 * scale;

    // Trunk cylinder
    generate_cylinder(
        pos, trunk_radius, 0.0, trunk_height,
        COLOR_TREE_TRUNK, MAT_TREE_TRUNK,
        8, vertices, indices,
    );

    // 3 foliage spheres, stacked and offset — wide canopy
    let sphere_radius_bottom = 0.24 * scale;
    let sphere_radius_mid = 0.20 * scale;
    let sphere_radius_top = 0.15 * scale;

    // Offset variation per tree
    let hash2 = ((pos[0] * 269.3 + pos[1] * 183.1).sin() * 27183.2847).fract();
    let ox = (hash2 - 0.5) * 0.10 * scale;
    let oy = (hash - 0.5) * 0.10 * scale;

    // Bottom sphere — centered on trunk
    generate_sphere(
        [pos[0], pos[1]],
        0.20 * scale,
        sphere_radius_bottom,
        COLOR_TREE_LEAVES, MAT_TREE_LEAVES,
        8, 6, vertices, indices,
    );

    // Middle sphere — offset slightly
    generate_sphere(
        [pos[0] + 0.04 * scale + ox, pos[1] + 0.02 * scale + oy],
        0.32 * scale,
        sphere_radius_mid,
        COLOR_TREE_LEAVES, MAT_TREE_LEAVES,
        8, 6, vertices, indices,
    );

    // Top sphere — offset the other direction
    generate_sphere(
        [pos[0] - 0.02 * scale - ox, pos[1] + 0.01 * scale - oy],
        0.42 * scale,
        sphere_radius_top,
        COLOR_TREE_LEAVES, MAT_TREE_LEAVES,
        8, 6, vertices, indices,
    );
}

/// Generate a UV sphere centered at (cx, cy, cz) with given radius.
fn generate_sphere(
    pos: [f32; 2],
    center_z: f32,
    radius: f32,
    color: [f32; 4],
    material: f32,
    lon_segments: u32,
    lat_segments: u32,
    vertices: &mut Vec<MapVertex>,
    indices: &mut Vec<u32>,
) {
    let base = vertices.len() as u32;
    let pi = std::f32::consts::PI;

    // Generate vertices
    for lat in 0..=lat_segments {
        let theta = pi * lat as f32 / lat_segments as f32; // 0..PI
        let sin_theta = theta.sin();
        let cos_theta = theta.cos();

        for lon in 0..=lon_segments {
            let phi = 2.0 * pi * lon as f32 / lon_segments as f32; // 0..2PI
            let x = pos[0] + radius * sin_theta * phi.cos();
            let y = pos[1] + radius * sin_theta * phi.sin();
            let z = center_z + radius * cos_theta;

            // Vary color slightly based on height for natural look
            let shade = 0.85 + 0.15 * (cos_theta * 0.5 + 0.5);
            let shaded_color = [
                color[0] * shade,
                color[1] * shade,
                color[2] * shade,
                1.0,
            ];

            vertices.push(MapVertex::at_height(x, y, z, shaded_color, material));
        }
    }

    // Generate indices
    for lat in 0..lat_segments {
        for lon in 0..lon_segments {
            let current = base + lat * (lon_segments + 1) + lon;
            let next = current + lon_segments + 1;

            indices.extend_from_slice(&[current, next, current + 1]);
            indices.extend_from_slice(&[current + 1, next, next + 1]);
        }
    }
}

/// Generate a cylinder from z_bottom to z_top at the given position.
fn generate_cylinder(
    pos: [f32; 2],
    radius: f32,
    z_bottom: f32,
    z_top: f32,
    color: [f32; 4],
    material: f32,
    segments: u32,
    vertices: &mut Vec<MapVertex>,
    indices: &mut Vec<u32>,
) {
    let base = vertices.len() as u32;
    let pi = std::f32::consts::PI;

    // Generate wall vertices (bottom ring + top ring)
    for i in 0..=segments {
        let angle = 2.0 * pi * i as f32 / segments as f32;
        let x = pos[0] + radius * angle.cos();
        let y = pos[1] + radius * angle.sin();

        // Bottom vertex
        vertices.push(MapVertex::at_height(x, y, z_bottom, color, material));
        // Top vertex
        vertices.push(MapVertex::at_height(x, y, z_top, color, material));
    }

    // Generate wall quads
    for i in 0..segments {
        let bl = base + i * 2;
        let tl = bl + 1;
        let br = base + (i + 1) * 2;
        let tr = br + 1;

        indices.extend_from_slice(&[bl, br, tl, tl, br, tr]);
    }
}

// --- POI 3D icon geometry ---

/// Color constants for POI icons
const COLOR_POI_ORANGE: [f32; 4] = [0.93, 0.55, 0.15, 1.0];
const COLOR_POI_GREEN: [f32; 4] = [0.30, 0.65, 0.20, 1.0];
const COLOR_POI_BROWN: [f32; 4] = [0.55, 0.35, 0.20, 1.0];
const COLOR_POI_RED: [f32; 4] = [0.85, 0.20, 0.20, 1.0];
const COLOR_POI_WHITE: [f32; 4] = [0.95, 0.95, 0.95, 1.0];
const COLOR_POI_BLUE: [f32; 4] = [0.25, 0.45, 0.80, 1.0];
const COLOR_POI_YELLOW: [f32; 4] = [0.95, 0.85, 0.20, 1.0];
const COLOR_POI_PINK: [f32; 4] = [0.90, 0.40, 0.55, 1.0];
const COLOR_POI_GRAY: [f32; 4] = [0.60, 0.60, 0.60, 1.0];

/// Generate a 3D carrot icon (cone + green top)
pub(crate) fn generate_poi_carrot(
    pos: [f32; 2],
    vertices: &mut Vec<MapVertex>,
    indices: &mut Vec<u32>,
) {
    let s = 1.2;
    // Orange cone body (tapers down)
    let segs = 8u32;
    let base = vertices.len() as u32;
    let pi = std::f32::consts::PI;
    // Tip at bottom
    vertices.push(MapVertex::at_height(pos[0], pos[1], 0.15 * s, COLOR_POI_ORANGE, MAT_DEFAULT));
    for i in 0..=segs {
        let a = 2.0 * pi * i as f32 / segs as f32;
        let x = pos[0] + 0.3 * s * a.cos();
        let y = pos[1] + 0.3 * s * a.sin();
        vertices.push(MapVertex::at_height(x, y, 1.0 * s, COLOR_POI_ORANGE, MAT_DEFAULT));
    }
    for i in 0..segs {
        indices.extend_from_slice(&[base, base + 1 + i, base + 2 + i]);
    }
    // Green leafy top — small spheres
    generate_sphere([pos[0], pos[1]], 1.1 * s, 0.15 * s, COLOR_POI_GREEN, MAT_DEFAULT, 6, 4, vertices, indices);
    generate_sphere([pos[0] + 0.06 * s, pos[1]], 1.25 * s, 0.10 * s, COLOR_POI_GREEN, MAT_DEFAULT, 6, 4, vertices, indices);
}

/// Generate a 3D coffee cup icon
pub(crate) fn generate_poi_cafe(
    pos: [f32; 2],
    vertices: &mut Vec<MapVertex>,
    indices: &mut Vec<u32>,
) {
    let s = 1.0;
    // Cup body
    generate_cylinder(pos, 0.3 * s, 0.0, 0.7 * s, COLOR_POI_WHITE, MAT_DEFAULT, 8, vertices, indices);
    // Coffee inside (dark disc)
    generate_sphere(pos, 0.65 * s, 0.08 * s, COLOR_POI_BROWN, MAT_DEFAULT, 6, 3, vertices, indices);
}

/// Generate a 3D cross icon (hospital/medical)
pub(crate) fn generate_poi_hospital(
    pos: [f32; 2],
    vertices: &mut Vec<MapVertex>,
    indices: &mut Vec<u32>,
) {
    let s = 1.0;
    let h = 0.8 * s;
    let w = 0.12 * s;
    let l = 0.35 * s;
    // Vertical bar
    let base = vertices.len() as u32;
    vertices.push(MapVertex::at_height(pos[0] - w, pos[1] - l, 0.0, COLOR_POI_RED, MAT_DEFAULT));
    vertices.push(MapVertex::at_height(pos[0] + w, pos[1] - l, 0.0, COLOR_POI_RED, MAT_DEFAULT));
    vertices.push(MapVertex::at_height(pos[0] - w, pos[1] + l, 0.0, COLOR_POI_RED, MAT_DEFAULT));
    vertices.push(MapVertex::at_height(pos[0] + w, pos[1] + l, 0.0, COLOR_POI_RED, MAT_DEFAULT));
    vertices.push(MapVertex::at_height(pos[0] - w, pos[1] - l, h, COLOR_POI_RED, MAT_DEFAULT));
    vertices.push(MapVertex::at_height(pos[0] + w, pos[1] - l, h, COLOR_POI_RED, MAT_DEFAULT));
    vertices.push(MapVertex::at_height(pos[0] - w, pos[1] + l, h, COLOR_POI_RED, MAT_DEFAULT));
    vertices.push(MapVertex::at_height(pos[0] + w, pos[1] + l, h, COLOR_POI_RED, MAT_DEFAULT));
    // Front, back, top faces
    indices.extend_from_slice(&[base,base+1,base+4, base+1,base+5,base+4]);
    indices.extend_from_slice(&[base+2,base+6,base+3, base+3,base+6,base+7]);
    indices.extend_from_slice(&[base+4,base+5,base+6, base+5,base+7,base+6]);
    // Horizontal bar
    let base = vertices.len() as u32;
    vertices.push(MapVertex::at_height(pos[0] - l, pos[1] - w, 0.0, COLOR_POI_RED, MAT_DEFAULT));
    vertices.push(MapVertex::at_height(pos[0] + l, pos[1] - w, 0.0, COLOR_POI_RED, MAT_DEFAULT));
    vertices.push(MapVertex::at_height(pos[0] - l, pos[1] + w, 0.0, COLOR_POI_RED, MAT_DEFAULT));
    vertices.push(MapVertex::at_height(pos[0] + l, pos[1] + w, 0.0, COLOR_POI_RED, MAT_DEFAULT));
    vertices.push(MapVertex::at_height(pos[0] - l, pos[1] - w, h, COLOR_POI_RED, MAT_DEFAULT));
    vertices.push(MapVertex::at_height(pos[0] + l, pos[1] - w, h, COLOR_POI_RED, MAT_DEFAULT));
    vertices.push(MapVertex::at_height(pos[0] - l, pos[1] + w, h, COLOR_POI_RED, MAT_DEFAULT));
    vertices.push(MapVertex::at_height(pos[0] + l, pos[1] + w, h, COLOR_POI_RED, MAT_DEFAULT));
    indices.extend_from_slice(&[base,base+1,base+4, base+1,base+5,base+4]);
    indices.extend_from_slice(&[base+2,base+6,base+3, base+3,base+6,base+7]);
    indices.extend_from_slice(&[base+4,base+5,base+6, base+5,base+7,base+6]);
}

/// Generate a 3D book icon (school/education)
pub(crate) fn generate_poi_school(
    pos: [f32; 2],
    vertices: &mut Vec<MapVertex>,
    indices: &mut Vec<u32>,
) {
    let s = 1.0;
    // Book base (flat box)
    let hw = 0.3 * s;
    let hd = 0.22 * s;
    let h = 0.12 * s;
    let base = vertices.len() as u32;
    vertices.push(MapVertex::at_height(pos[0]-hw, pos[1]-hd, 0.0, COLOR_POI_BLUE, MAT_DEFAULT));
    vertices.push(MapVertex::at_height(pos[0]+hw, pos[1]-hd, 0.0, COLOR_POI_BLUE, MAT_DEFAULT));
    vertices.push(MapVertex::at_height(pos[0]-hw, pos[1]+hd, 0.0, COLOR_POI_BLUE, MAT_DEFAULT));
    vertices.push(MapVertex::at_height(pos[0]+hw, pos[1]+hd, 0.0, COLOR_POI_BLUE, MAT_DEFAULT));
    vertices.push(MapVertex::at_height(pos[0]-hw, pos[1]-hd, h, COLOR_POI_BLUE, MAT_DEFAULT));
    vertices.push(MapVertex::at_height(pos[0]+hw, pos[1]-hd, h, COLOR_POI_BLUE, MAT_DEFAULT));
    vertices.push(MapVertex::at_height(pos[0]-hw, pos[1]+hd, h, COLOR_POI_BLUE, MAT_DEFAULT));
    vertices.push(MapVertex::at_height(pos[0]+hw, pos[1]+hd, h, COLOR_POI_BLUE, MAT_DEFAULT));
    // All 6 faces
    indices.extend_from_slice(&[base,base+1,base+4, base+1,base+5,base+4]); // front
    indices.extend_from_slice(&[base+2,base+6,base+3, base+3,base+6,base+7]); // back
    indices.extend_from_slice(&[base+4,base+5,base+6, base+5,base+7,base+6]); // top
    indices.extend_from_slice(&[base,base+2,base+1, base+1,base+2,base+3]); // bottom
    indices.extend_from_slice(&[base,base+4,base+2, base+2,base+4,base+6]); // left
    indices.extend_from_slice(&[base+1,base+3,base+5, base+3,base+7,base+5]); // right
    // Pages (white stripe)
    let pw = 0.28 * s;
    let ph = 0.04 * s;
    let base = vertices.len() as u32;
    vertices.push(MapVertex::at_height(pos[0]-pw, pos[1]-hd-0.01*s, h*0.2, COLOR_POI_WHITE, MAT_DEFAULT));
    vertices.push(MapVertex::at_height(pos[0]+pw, pos[1]-hd-0.01*s, h*0.2, COLOR_POI_WHITE, MAT_DEFAULT));
    vertices.push(MapVertex::at_height(pos[0]-pw, pos[1]-hd-0.01*s, h*0.8, COLOR_POI_WHITE, MAT_DEFAULT));
    vertices.push(MapVertex::at_height(pos[0]+pw, pos[1]-hd-0.01*s, h*0.8, COLOR_POI_WHITE, MAT_DEFAULT));
    indices.extend_from_slice(&[base,base+1,base+2, base+1,base+3,base+2]);
}

/// Generate a 3D fork+knife icon (restaurant)
pub(crate) fn generate_poi_restaurant(
    pos: [f32; 2],
    vertices: &mut Vec<MapVertex>,
    indices: &mut Vec<u32>,
) {
    let s = 1.0;
    // Plate (flat disc)
    generate_cylinder(pos, 0.35 * s, 0.0, 0.04 * s, COLOR_POI_WHITE, MAT_DEFAULT, 10, vertices, indices);
    // Food ball on plate
    generate_sphere(pos, 0.06 * s, 0.12 * s, COLOR_POI_YELLOW, MAT_DEFAULT, 6, 4, vertices, indices);
}

/// Generate a generic 3D pin icon (default POI)
pub(crate) fn generate_poi_default(
    pos: [f32; 2],
    vertices: &mut Vec<MapVertex>,
    indices: &mut Vec<u32>,
) {
    let s = 0.8;
    // Stick
    generate_cylinder(pos, 0.03 * s, 0.0, 0.6 * s, COLOR_POI_GRAY, MAT_DEFAULT, 6, vertices, indices);
    // Ball on top
    generate_sphere(pos, 0.65 * s, 0.12 * s, COLOR_POI_PINK, MAT_DEFAULT, 6, 4, vertices, indices);
}

/// Generate the appropriate 3D POI icon based on the POI kind.
pub(crate) fn generate_poi_icon(
    pos: [f32; 2],
    kind: &str,
    vertices: &mut Vec<MapVertex>,
    indices: &mut Vec<u32>,
) {
    match kind {
        "supermarket" | "grocery" | "greengrocer" | "marketplace" =>
            generate_poi_carrot(pos, vertices, indices),
        "cafe" | "coffee" =>
            generate_poi_cafe(pos, vertices, indices),
        "hospital" | "clinic" | "doctors" | "pharmacy" | "dentist" =>
            generate_poi_hospital(pos, vertices, indices),
        "school" | "university" | "college" | "library" | "kindergarten" =>
            generate_poi_school(pos, vertices, indices),
        "restaurant" | "fast_food" | "food_court" | "bar" | "pub" | "biergarten" =>
            generate_poi_restaurant(pos, vertices, indices),
        _ => generate_poi_default(pos, vertices, indices),
    }
}

// --- Fountain geometry ---

pub(crate) const COLOR_FOUNTAIN: [f32; 4] = [0.85, 0.92, 0.98, 1.0];

// --- Cloud geometry (transparent spheres rendered via shadow pipeline) ---

// Cloud colors: white with alpha for transparent blending
pub(crate) const COLOR_CLOUD_CORE: [f32; 4] = [1.0, 1.0, 1.0, 0.18];
pub(crate) const COLOR_CLOUD_EDGE: [f32; 4] = [1.0, 1.0, 1.0, 0.06];

/// Generate volumetric cloud puffs around the map edges using clusters of spheres.
fn generate_edge_clouds(
    min_x: f32, min_y: f32, max_x: f32, max_y: f32,
    vertices: &mut Vec<MapVertex>,
    indices: &mut Vec<u32>,
) {
    let w = max_x - min_x;
    let h = max_y - min_y;
    let map_scale = w.max(h);

    // Sphere size relative to map
    let sphere_base = map_scale * 0.04;
    let num_per_edge = 14;
    let mut cloud_count = 0u32;

    for side in 0..4 {
        for i in 0..num_per_edge {
            let t = (i as f32 + 0.5) / num_per_edge as f32;
            let hash = ((side as f32 * 127.1 + i as f32 * 311.7).sin() * 43758.5453).fract();
            let hash2 = ((side as f32 * 269.3 + i as f32 * 183.1).sin() * 27183.2847).fract();
            let hash3 = ((side as f32 * 431.5 + i as f32 * 97.3).sin() * 31415.9265).fract();

            // Place clouds well inside the map edges
            let inset = map_scale * 0.15 * hash; // 0-15% inward from edge
            let (px, py) = match side {
                0 => (min_x + w * t, min_y + inset),
                1 => (max_x - inset, min_y + h * t),
                2 => (min_x + w * t, max_y - inset),
                _ => (min_x + inset, min_y + h * t),
            };

            let z = 0.3 + hash2 * 0.8;
            let r = sphere_base * (0.6 + hash * 0.8);

            // Each cloud puff = cluster of 3-4 overlapping spheres
            generate_cloud_sphere([px, py], z, r, vertices, indices);
            generate_cloud_sphere(
                [px + r * 0.5 * hash3, py + r * 0.3 * hash2],
                z + r * 0.2, r * 0.8, vertices, indices,
            );
            generate_cloud_sphere(
                [px - r * 0.4 * hash2, py - r * 0.3 * hash3],
                z + r * 0.35, r * 0.65, vertices, indices,
            );
            cloud_count += 3;
        }
    }

    // Extra large corner clusters
    let corners = [
        [min_x + w * 0.05, min_y + h * 0.05],
        [max_x - w * 0.05, min_y + h * 0.05],
        [max_x - w * 0.05, max_y - h * 0.05],
        [min_x + w * 0.05, max_y - h * 0.05],
    ];
    for (i, corner) in corners.iter().enumerate() {
        let hash = ((i as f32 * 457.3).sin() * 43758.5453).fract();
        let r = sphere_base * 1.5;
        generate_cloud_sphere(*corner, 0.4, r, vertices, indices);
        generate_cloud_sphere(
            [corner[0] + r * 0.4, corner[1] + r * 0.3],
            0.6, r * 0.9, vertices, indices,
        );
        generate_cloud_sphere(
            [corner[0] - r * 0.3, corner[1] - r * 0.2],
            0.8, r * 0.7, vertices, indices,
        );
        generate_cloud_sphere(
            [corner[0] + r * 0.1, corner[1] - r * 0.4],
            0.5, r * 0.85, vertices, indices,
        );
        cloud_count += 4;
    }

}

/// Generate a single cloud sphere with transparent white color.
fn generate_cloud_sphere(
    pos: [f32; 2],
    z: f32,
    radius: f32,
    vertices: &mut Vec<MapVertex>,
    indices: &mut Vec<u32>,
) {
    let base = vertices.len() as u32;
    let pi = std::f32::consts::PI;
    let lon_segments = 8u32;
    let lat_segments = 6u32;

    for lat in 0..=lat_segments {
        let theta = pi * lat as f32 / lat_segments as f32;
        let sin_theta = theta.sin();
        let cos_theta = theta.cos();

        for lon in 0..=lon_segments {
            let phi = 2.0 * pi * lon as f32 / lon_segments as f32;
            let x = pos[0] + radius * sin_theta * phi.cos();
            let y = pos[1] + radius * sin_theta * phi.sin();
            let vz = z + radius * cos_theta;

            // Alpha fades from core to edges
            let height_factor = (cos_theta * 0.5 + 0.5); // 1 at top, 0 at bottom
            let alpha = COLOR_CLOUD_CORE[3] * (0.4 + 0.6 * height_factor);
            let color = [1.0, 1.0, 1.0, alpha];

            vertices.push(MapVertex::at_height(x, y, vz, color, MAT_CLOUD));
        }
    }

    for lat in 0..lat_segments {
        for lon in 0..lon_segments {
            let current = base + lat * (lon_segments + 1) + lon;
            let next = current + lon_segments + 1;
            indices.extend_from_slice(&[current, next, current + 1]);
            indices.extend_from_slice(&[current + 1, next, next + 1]);
        }
    }
}

/// Generate an animated fountain: central jet + ring of spray arcs.
pub(crate) fn generate_fountain(
    pos: [f32; 2],
    water_z: f32,
    vertices: &mut Vec<MapVertex>,
    indices: &mut Vec<u32>,
) {
    let base_z = water_z + 0.001;
    let jet_height = 0.45;
    let jet_radius = 0.02;
    let spray_count = 8u32;
    let spray_height = 0.15;
    let spray_radius = 0.10; // how far spray arcs outward
    let pi = std::f32::consts::PI;

    // Central jet — thin cylinder going straight up
    generate_cylinder(
        pos, jet_radius, base_z, base_z + jet_height,
        COLOR_FOUNTAIN, MAT_FOUNTAIN,
        6, vertices, indices,
    );

    // Small sphere at the top of the jet
    generate_sphere(
        pos, base_z + jet_height,
        0.025,
        COLOR_FOUNTAIN, MAT_FOUNTAIN,
        6, 4, vertices, indices,
    );

    // Ring of spray jets — thin angled columns arcing outward
    for i in 0..spray_count {
        let angle = 2.0 * pi * i as f32 / spray_count as f32;
        let dx = angle.cos();
        let dy = angle.sin();

        // Each spray arc: from center base to an outer point at spray_height
        // We approximate the arc with a few segments
        let segments = 4u32;
        let base_idx = vertices.len() as u32;
        let half_w = 0.008;

        for s in 0..=segments {
            let t = s as f32 / segments as f32;
            // Parabolic arc: rises then falls
            let arc_z = base_z + spray_height * (4.0 * t * (1.0 - t));
            let out = spray_radius * t;
            let cx = pos[0] + dx * out;
            let cy = pos[1] + dy * out;

            // Two vertices per segment (left/right of the spray)
            let perp_x = -dy * half_w;
            let perp_y = dx * half_w;
            vertices.push(MapVertex::at_height(cx + perp_x, cy + perp_y, arc_z, COLOR_FOUNTAIN, MAT_FOUNTAIN));
            vertices.push(MapVertex::at_height(cx - perp_x, cy - perp_y, arc_z, COLOR_FOUNTAIN, MAT_FOUNTAIN));
        }

        // Connect the strip
        for s in 0..segments {
            let tl = base_idx + s * 2;
            let tr = tl + 1;
            let bl = tl + 2;
            let br = tl + 3;
            indices.extend_from_slice(&[tl, tr, bl, tr, br, bl]);
        }
    }

    // Splash ring at the base — a flat torus-like ring where spray hits the water
    let ring_inner = spray_radius - 0.02;
    let ring_outer = spray_radius + 0.02;
    let ring_segments = 16u32;
    let base_idx = vertices.len() as u32;

    for i in 0..=ring_segments {
        let angle = 2.0 * pi * i as f32 / ring_segments as f32;
        let cos_a = angle.cos();
        let sin_a = angle.sin();
        vertices.push(MapVertex::at_height(
            pos[0] + cos_a * ring_inner, pos[1] + sin_a * ring_inner,
            base_z + 0.005, COLOR_FOUNTAIN, MAT_FOUNTAIN,
        ));
        vertices.push(MapVertex::at_height(
            pos[0] + cos_a * ring_outer, pos[1] + sin_a * ring_outer,
            base_z + 0.005, COLOR_FOUNTAIN, MAT_FOUNTAIN,
        ));
    }

    for i in 0..ring_segments {
        let i0 = base_idx + i * 2;
        let i1 = i0 + 1;
        let o0 = i0 + 2;
        let o1 = i0 + 3;
        indices.extend_from_slice(&[i0, i1, o0, i1, o1, o0]);
    }
}

/// Public URL encoding wrapper for tile system.
#[cfg(not(target_arch = "wasm32"))]
pub fn urlencoded_pub(s: &str) -> String { urlencoded(s) }

/// Simple URL encoding for the Overpass query (native only).
#[cfg(not(target_arch = "wasm32"))]
fn urlencoded(s: &str) -> String {
    let mut result = String::with_capacity(s.len() * 2);
    for c in s.chars() {
        match c {
            'A'..='Z' | 'a'..='z' | '0'..='9' | '-' | '_' | '.' | '~' => result.push(c),
            ' ' => result.push('+'),
            _ => {
                let mut buf = [0u8; 4];
                let encoded = c.encode_utf8(&mut buf);
                for byte in encoded.bytes() {
                    result.push('%');
                    result.push_str(&format!("{:02X}", byte));
                }
            }
        }
    }
    result
}
