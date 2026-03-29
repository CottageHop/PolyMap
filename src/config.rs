use serde::{Deserialize, Serialize};

/// Configuration for initializing a PolyMap instance.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PolyMapConfig {
    pub canvas: Option<String>,
    pub center: Option<CenterConfig>,
    pub bbox: Option<BboxConfig>,
    pub zoom: Option<f32>,
    pub tilt: Option<f32>,
    pub data_url: Option<String>,
    pub api_base: Option<String>,
    pub api_key: Option<String>,
    pub pmtiles_url: Option<String>,
    pub parcels_url: Option<String>,
    pub colors: Option<ColorConfig>,
    pub layers: Option<LayerConfig>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CenterConfig {
    pub lat: f64,
    pub lon: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BboxConfig {
    pub south: f64,
    pub west: f64,
    pub north: f64,
    pub east: f64,
}

/// Customizable color palette. All values are [r, g, b, a] in 0.0-1.0 range.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ColorConfig {
    pub water: Option<[f32; 4]>,
    pub land: Option<[f32; 4]>,
    pub road: Option<[f32; 4]>,
    pub road_minor: Option<[f32; 4]>,
    pub road_outline: Option<[f32; 4]>,
    pub building: Option<[f32; 4]>,
    pub skyscraper: Option<[f32; 4]>,
    pub park: Option<[f32; 4]>,
    pub sidewalk: Option<[f32; 4]>,
}

/// Layer visibility toggles.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LayerConfig {
    pub buildings: Option<bool>,
    pub roads: Option<bool>,
    pub water: Option<bool>,
    pub parks: Option<bool>,
    pub trees: Option<bool>,
    pub shadows: Option<bool>,
    pub labels: Option<bool>,
    pub parcels: Option<bool>,
}

/// Runtime layer visibility state.
#[derive(Clone, Debug)]
pub struct LayerVisibility {
    pub buildings: bool,
    pub roads: bool,
    pub water: bool,
    pub parks: bool,
    pub trees: bool,
    pub shadows: bool,
    pub labels: bool,
    pub parcels: bool,
}

impl Default for LayerVisibility {
    fn default() -> Self {
        Self {
            buildings: true,
            roads: true,
            water: true,
            parks: true,
            trees: true,
            shadows: true,
            labels: true,
            parcels: true,
        }
    }
}

impl LayerVisibility {
    pub fn apply_config(&mut self, config: &LayerConfig) {
        if let Some(v) = config.buildings { self.buildings = v; }
        if let Some(v) = config.roads { self.roads = v; }
        if let Some(v) = config.water { self.water = v; }
        if let Some(v) = config.parks { self.parks = v; }
        if let Some(v) = config.trees { self.trees = v; }
        if let Some(v) = config.shadows { self.shadows = v; }
        if let Some(v) = config.labels { self.labels = v; }
        if let Some(v) = config.parcels { self.parcels = v; }
    }
}
