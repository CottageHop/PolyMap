/// Minimal hand-rolled protobuf decoder for Mapbox Vector Tiles (MVT).
/// No external protobuf dependencies — decodes the MVT spec directly.

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Tile {
    pub layers: Vec<Layer>,
}

#[derive(Debug, Clone)]
pub struct Layer {
    pub name: String,
    pub extent: u32,
    pub features: Vec<Feature>,
    pub keys: Vec<String>,
    pub values: Vec<Value>,
}

#[derive(Debug, Clone)]
pub enum Value {
    String(String),
    Float(f32),
    Double(f64),
    Int(i64),
    UInt(u64),
    SInt(i64),
    Bool(bool),
}

#[derive(Debug, Clone)]
pub struct Feature {
    pub id: u64,
    pub tags: Vec<u32>,
    pub geom_type: GeomType,
    pub geometry: Vec<Vec<[i32; 2]>>, // decoded rings / linestrings
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeomType {
    Unknown,
    Point,
    LineString,
    Polygon,
}

// ---------------------------------------------------------------------------
// Property access helpers
// ---------------------------------------------------------------------------

impl Feature {
    /// Look up a string property by key name.
    pub fn get_str<'a>(&self, layer: &'a Layer, key: &str) -> Option<&'a str> {
        for pair in self.tags.chunks_exact(2) {
            let key_idx = pair[0] as usize;
            let val_idx = pair[1] as usize;
            if layer.keys.get(key_idx).map(|k| k.as_str()) == Some(key) {
                if let Some(Value::String(s)) = layer.values.get(val_idx) {
                    return Some(s.as_str());
                }
            }
        }
        None
    }

    /// Look up a numeric property by key name, coercing to `f64`.
    pub fn get_f64(&self, layer: &Layer, key: &str) -> Option<f64> {
        for pair in self.tags.chunks_exact(2) {
            let key_idx = pair[0] as usize;
            let val_idx = pair[1] as usize;
            if layer.keys.get(key_idx).map(|k| k.as_str()) == Some(key) {
                match layer.values.get(val_idx) {
                    Some(Value::Float(v)) => return Some(*v as f64),
                    Some(Value::Double(v)) => return Some(*v),
                    Some(Value::Int(v)) => return Some(*v as f64),
                    Some(Value::UInt(v)) => return Some(*v as f64),
                    Some(Value::SInt(v)) => return Some(*v as f64),
                    Some(Value::Bool(v)) => return Some(if *v { 1.0 } else { 0.0 }),
                    _ => {}
                }
            }
        }
        None
    }
}

// ---------------------------------------------------------------------------
// Low-level protobuf primitives
// ---------------------------------------------------------------------------

/// A lightweight cursor over a byte slice.
struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.data.len() - self.pos
    }

    fn is_empty(&self) -> bool {
        self.pos >= self.data.len()
    }

    /// Read a base-128 varint, returning its u64 value.
    /// Returns None on truncated data.
    fn try_read_varint(&mut self) -> Option<u64> {
        let mut result: u64 = 0;
        let mut shift: u32 = 0;
        loop {
            if self.pos >= self.data.len() { return None; }
            let byte = self.data[self.pos];
            self.pos += 1;
            result |= ((byte & 0x7F) as u64) << shift;
            if byte & 0x80 == 0 {
                break;
            }
            shift += 7;
            if shift >= 64 { return None; }
        }
        Some(result)
    }

    fn read_varint(&mut self) -> u64 {
        self.try_read_varint().unwrap_or(0)
    }

    /// Read a protobuf field header, returning (field_number, wire_type).
    /// Returns None on truncated data.
    fn try_read_field_header(&mut self) -> Option<(u32, u32)> {
        let tag = self.try_read_varint()?;
        let field_number = (tag >> 3) as u32;
        let wire_type = (tag & 0x7) as u32;
        Some((field_number, wire_type))
    }

    fn read_field_header(&mut self) -> (u32, u32) {
        self.try_read_field_header().unwrap_or((0, 0))
    }

    /// Read `n` bytes and advance the cursor.
    fn read_bytes(&mut self, n: usize) -> &'a [u8] {
        if self.pos + n > self.data.len() {
            self.pos = self.data.len();
            return &[];
        }
        let slice = &self.data[self.pos..self.pos + n];
        self.pos += n;
        slice
    }

    /// Read a length-delimited field (returns the sub-slice).
    fn read_length_delimited(&mut self) -> &'a [u8] {
        let len = self.read_varint() as usize;
        self.read_bytes(len)
    }

    /// Read a 32-bit little-endian value.
    fn read_fixed32(&mut self) -> u32 {
        let bytes = self.read_bytes(4);
        u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
    }

    /// Read a 64-bit little-endian value.
    fn read_fixed64(&mut self) -> u64 {
        let bytes = self.read_bytes(8);
        u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3],
            bytes[4], bytes[5], bytes[6], bytes[7],
        ])
    }

    /// Skip over a field value based on its wire type.
    fn skip_field(&mut self, wire_type: u32) {
        match wire_type {
            0 => { self.read_varint(); }                    // varint
            1 => { self.read_bytes(8); }                    // 64-bit
            2 => { let _ = self.read_length_delimited(); }  // length-delimited
            5 => { self.read_bytes(4); }                    // 32-bit
            3 | 4 => {}  // start/end group (deprecated) — skip
            _ => {}     // unknown wire type — skip
        }
    }
}

// ---------------------------------------------------------------------------
// Packed repeated uint32 decoder
// ---------------------------------------------------------------------------

fn decode_packed_uint32(data: &[u8]) -> Vec<u32> {
    let mut cur = Cursor::new(data);
    let mut out = Vec::new();
    while !cur.is_empty() {
        out.push(cur.read_varint() as u32);
    }
    out
}

// ---------------------------------------------------------------------------
// Geometry command decoder
// ---------------------------------------------------------------------------

/// Zigzag-decode a u32 into an i32.
#[inline]
fn zigzag_decode(n: u32) -> i32 {
    ((n >> 1) as i32) ^ -((n & 1) as i32)
}

/// Decode MVT geometry commands into rings / linestrings.
///
/// Returns a `Vec<Vec<[i32; 2]>>` — each inner Vec is one ring (for
/// Polygon) or one linestring (for LineString) or one point (for Point).
pub fn decode_geometry(cmds: &[u32], _geom_type: GeomType) -> Vec<Vec<[i32; 2]>> {
    let mut rings: Vec<Vec<[i32; 2]>> = Vec::new();
    let mut current_ring: Vec<[i32; 2]> = Vec::new();
    let mut cx: i32 = 0;
    let mut cy: i32 = 0;
    let mut i = 0;

    while i < cmds.len() {
        let cmd_int = cmds[i];
        let command_id = cmd_int & 0x7;
        let count = (cmd_int >> 3) as usize;
        i += 1;

        match command_id {
            1 => {
                // MoveTo — starts a new ring / linestring / point
                for _ in 0..count {
                    // Each MoveTo with a fresh position starts a new ring
                    if !current_ring.is_empty() {
                        rings.push(std::mem::take(&mut current_ring));
                    }
                    let dx = zigzag_decode(cmds[i]);
                    let dy = zigzag_decode(cmds[i + 1]);
                    i += 2;
                    cx += dx;
                    cy += dy;
                    current_ring.push([cx, cy]);
                }
            }
            2 => {
                // LineTo — append to current ring
                for _ in 0..count {
                    let dx = zigzag_decode(cmds[i]);
                    let dy = zigzag_decode(cmds[i + 1]);
                    i += 2;
                    cx += dx;
                    cy += dy;
                    current_ring.push([cx, cy]);
                }
            }
            7 => {
                // ClosePath — close the current ring (repeat first point)
                if let Some(&first) = current_ring.first() {
                    current_ring.push(first);
                }
                rings.push(std::mem::take(&mut current_ring));
            }
            _ => {
                // Unknown command — skip (shouldn't happen in valid tiles)
            }
        }
    }

    // Flush any remaining points (e.g. unclosed linestrings or points).
    if !current_ring.is_empty() {
        rings.push(current_ring);
    }

    rings
}

// ---------------------------------------------------------------------------
// Message decoders
// ---------------------------------------------------------------------------

fn decode_value(data: &[u8]) -> Value {
    let mut cur = Cursor::new(data);
    // A Value message has exactly one field set. We default to an empty string
    // in case the message is somehow empty.
    let mut result = Value::String(String::new());

    while !cur.is_empty() {
        let (field, wire) = cur.read_field_header();
        match field {
            1 => {
                // string_value (wire type 2)
                let bytes = cur.read_length_delimited();
                result = Value::String(String::from_utf8_lossy(bytes).into_owned());
            }
            2 => {
                // float_value (wire type 5 — 32-bit)
                let bits = cur.read_fixed32();
                result = Value::Float(f32::from_bits(bits));
            }
            3 => {
                // double_value (wire type 1 — 64-bit)
                let bits = cur.read_fixed64();
                result = Value::Double(f64::from_bits(bits));
            }
            4 => {
                // int_value (varint, signed)
                let v = cur.read_varint();
                result = Value::Int(v as i64);
            }
            5 => {
                // uint_value (varint)
                let v = cur.read_varint();
                result = Value::UInt(v);
            }
            6 => {
                // sint_value (varint, zigzag-encoded)
                let v = cur.read_varint();
                // zigzag decode for i64
                let decoded = ((v >> 1) as i64) ^ -((v & 1) as i64);
                result = Value::SInt(decoded);
            }
            7 => {
                // bool_value (varint)
                let v = cur.read_varint();
                result = Value::Bool(v != 0);
            }
            _ => cur.skip_field(wire),
        }
    }

    result
}

fn decode_feature(data: &[u8]) -> Feature {
    let mut cur = Cursor::new(data);
    let mut id: u64 = 0;
    let mut tags: Vec<u32> = Vec::new();
    let mut geom_type = GeomType::Unknown;
    let mut raw_geometry: Vec<u32> = Vec::new();

    while !cur.is_empty() {
        let (field, wire) = cur.read_field_header();
        match field {
            1 => {
                // id (uint64, varint)
                id = cur.read_varint();
            }
            2 => {
                // tags (packed repeated uint32)
                let bytes = cur.read_length_delimited();
                tags = decode_packed_uint32(bytes);
            }
            3 => {
                // type (GeomType enum, varint)
                let v = cur.read_varint();
                geom_type = match v {
                    1 => GeomType::Point,
                    2 => GeomType::LineString,
                    3 => GeomType::Polygon,
                    _ => GeomType::Unknown,
                };
            }
            4 => {
                // geometry (packed repeated uint32)
                let bytes = cur.read_length_delimited();
                raw_geometry = decode_packed_uint32(bytes);
            }
            _ => cur.skip_field(wire),
        }
    }

    let geometry = decode_geometry(&raw_geometry, geom_type);

    Feature {
        id,
        tags,
        geom_type,
        geometry,
    }
}

fn decode_layer(data: &[u8]) -> Layer {
    let mut cur = Cursor::new(data);
    let mut name = String::new();
    let mut extent: u32 = 4096; // default per spec
    let mut features: Vec<Feature> = Vec::new();
    let mut keys: Vec<String> = Vec::new();
    let mut values: Vec<Value> = Vec::new();

    while !cur.is_empty() {
        let (field, wire) = cur.read_field_header();
        match field {
            15 => {
                // version (uint32, varint)
                let _version = cur.read_varint();
            }
            1 => {
                // name (string)
                let bytes = cur.read_length_delimited();
                name = String::from_utf8_lossy(bytes).into_owned();
            }
            2 => {
                // features (repeated Feature message)
                let bytes = cur.read_length_delimited();
                features.push(decode_feature(bytes));
            }
            3 => {
                // keys (repeated string)
                let bytes = cur.read_length_delimited();
                keys.push(String::from_utf8_lossy(bytes).into_owned());
            }
            4 => {
                // values (repeated Value message)
                let bytes = cur.read_length_delimited();
                values.push(decode_value(bytes));
            }
            5 => {
                // extent (uint32, varint)
                extent = cur.read_varint() as u32;
            }
            _ => cur.skip_field(wire),
        }
    }

    Layer {
        name,
        extent,
        features,
        keys,
        values,
    }
}

// ---------------------------------------------------------------------------
// Top-level decoder
// ---------------------------------------------------------------------------

/// Decode a Mapbox Vector Tile from raw protobuf bytes.
pub fn decode_tile(data: &[u8]) -> Tile {
    let mut cur = Cursor::new(data);
    let mut layers: Vec<Layer> = Vec::new();

    while !cur.is_empty() {
        let (field, wire) = cur.read_field_header();
        match field {
            3 => {
                // layers (repeated Layer message)
                let bytes = cur.read_length_delimited();
                layers.push(decode_layer(bytes));
            }
            _ => cur.skip_field(wire),
        }
    }

    Tile { layers }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: encode a varint into a byte vec.
    fn encode_varint(mut val: u64) -> Vec<u8> {
        let mut out = Vec::new();
        loop {
            let mut byte = (val & 0x7F) as u8;
            val >>= 7;
            if val != 0 {
                byte |= 0x80;
            }
            out.push(byte);
            if val == 0 {
                break;
            }
        }
        out
    }

    /// Helper: encode a protobuf tag (field number + wire type).
    fn encode_tag(field: u32, wire_type: u32) -> Vec<u8> {
        encode_varint(((field as u64) << 3) | (wire_type as u64))
    }

    /// Helper: wrap bytes as a length-delimited field.
    fn encode_length_delimited(field: u32, data: &[u8]) -> Vec<u8> {
        let mut out = encode_tag(field, 2);
        out.extend_from_slice(&encode_varint(data.len() as u64));
        out.extend_from_slice(data);
        out
    }

    /// Helper: encode a varint field.
    fn encode_varint_field(field: u32, val: u64) -> Vec<u8> {
        let mut out = encode_tag(field, 0);
        out.extend_from_slice(&encode_varint(val));
        out
    }

    /// Zigzag encode an i32 for geometry commands.
    fn zigzag_encode(n: i32) -> u32 {
        ((n << 1) ^ (n >> 31)) as u32
    }

    #[test]
    fn test_zigzag_decode() {
        assert_eq!(zigzag_decode(0), 0);
        assert_eq!(zigzag_decode(1), -1);
        assert_eq!(zigzag_decode(2), 1);
        assert_eq!(zigzag_decode(3), -2);
        assert_eq!(zigzag_decode(4), 2);
    }

    #[test]
    fn test_varint_roundtrip() {
        for &v in &[0u64, 1, 127, 128, 300, 16384, u64::MAX] {
            let encoded = encode_varint(v);
            let mut cur = Cursor::new(&encoded);
            assert_eq!(cur.read_varint(), v);
            assert!(cur.is_empty());
        }
    }

    #[test]
    fn test_decode_geometry_polygon() {
        // A simple triangle: MoveTo(10,10), LineTo(20,0), LineTo(0,20), ClosePath
        let cmds: Vec<u32> = vec![
            // MoveTo count=1
            (1 << 3) | 1,
            zigzag_encode(10),
            zigzag_encode(10),
            // LineTo count=2
            (2 << 3) | 2,
            zigzag_encode(20),
            zigzag_encode(0),
            zigzag_encode(0),
            zigzag_encode(20),
            // ClosePath count=1
            (1 << 3) | 7,
        ];
        let rings = decode_geometry(&cmds, GeomType::Polygon);
        assert_eq!(rings.len(), 1);
        // Should be: [10,10], [30,10], [30,30], [10,10] (closed)
        assert_eq!(rings[0].len(), 4);
        assert_eq!(rings[0][0], [10, 10]);
        assert_eq!(rings[0][1], [30, 10]);
        assert_eq!(rings[0][2], [30, 30]);
        assert_eq!(rings[0][3], [10, 10]); // closed
    }

    #[test]
    fn test_decode_geometry_point() {
        let cmds: Vec<u32> = vec![
            (1 << 3) | 1, // MoveTo count=1
            zigzag_encode(25),
            zigzag_encode(17),
        ];
        let rings = decode_geometry(&cmds, GeomType::Point);
        assert_eq!(rings.len(), 1);
        assert_eq!(rings[0], vec![[25, 17]]);
    }

    #[test]
    fn test_decode_geometry_multipoint() {
        // Two points via MoveTo count=2
        let cmds: Vec<u32> = vec![
            (2 << 3) | 1, // MoveTo count=2
            zigzag_encode(5),
            zigzag_encode(10),
            zigzag_encode(3),
            zigzag_encode(4),
        ];
        let rings = decode_geometry(&cmds, GeomType::Point);
        assert_eq!(rings.len(), 2);
        assert_eq!(rings[0], vec![[5, 10]]);
        assert_eq!(rings[1], vec![[8, 14]]);
    }

    #[test]
    fn test_decode_tile_roundtrip() {
        // Build a minimal tile with one layer, one feature (a point).

        // Value message: string_value = "park"
        let value_msg = encode_length_delimited(1, b"park");

        // Geometry: MoveTo(100, 200)
        let mut geom_packed = Vec::new();
        geom_packed.extend_from_slice(&encode_varint(((1u64) << 3) | 1)); // MoveTo count=1
        geom_packed.extend_from_slice(&encode_varint(zigzag_encode(100) as u64));
        geom_packed.extend_from_slice(&encode_varint(zigzag_encode(200) as u64));

        // Tags: [0, 0] (key_idx=0, val_idx=0)
        let mut tags_packed = Vec::new();
        tags_packed.extend_from_slice(&encode_varint(0));
        tags_packed.extend_from_slice(&encode_varint(0));

        // Feature message
        let mut feature_msg = Vec::new();
        feature_msg.extend_from_slice(&encode_varint_field(1, 42)); // id=42
        feature_msg.extend_from_slice(&encode_length_delimited(2, &tags_packed)); // tags
        feature_msg.extend_from_slice(&encode_varint_field(3, 1)); // type=Point
        feature_msg.extend_from_slice(&encode_length_delimited(4, &geom_packed)); // geometry

        // Layer message
        let mut layer_msg = Vec::new();
        layer_msg.extend_from_slice(&encode_varint_field(15, 2)); // version=2
        layer_msg.extend_from_slice(&encode_length_delimited(1, b"places")); // name
        layer_msg.extend_from_slice(&encode_length_delimited(2, &feature_msg)); // feature
        layer_msg.extend_from_slice(&encode_length_delimited(3, b"class")); // keys[0]
        layer_msg.extend_from_slice(&encode_length_delimited(4, &value_msg)); // values[0]
        layer_msg.extend_from_slice(&encode_varint_field(5, 4096)); // extent

        // Tile message
        let tile_data = encode_length_delimited(3, &layer_msg);

        let tile = decode_tile(&tile_data);
        assert_eq!(tile.layers.len(), 1);

        let layer = &tile.layers[0];
        assert_eq!(layer.name, "places");
        assert_eq!(layer.extent, 4096);
        assert_eq!(layer.keys, vec!["class"]);
        assert_eq!(layer.features.len(), 1);

        let feat = &layer.features[0];
        assert_eq!(feat.id, 42);
        assert_eq!(feat.geom_type, GeomType::Point);
        assert_eq!(feat.geometry.len(), 1);
        assert_eq!(feat.geometry[0], vec![[100, 200]]);
        assert_eq!(feat.tags, vec![0, 0]);

        // Test property helpers
        assert_eq!(feat.get_str(layer, "class"), Some("park"));
        assert_eq!(feat.get_str(layer, "missing"), None);
    }

    #[test]
    fn test_decode_value_types() {
        // float_value (field 2, wire type 5)
        let mut data = encode_tag(2, 5);
        data.extend_from_slice(&f32::to_le_bytes(3.14));
        match decode_value(&data) {
            Value::Float(v) => assert!((v - 3.14).abs() < 0.001),
            other => panic!("expected Float, got {:?}", other),
        }

        // double_value (field 3, wire type 1)
        let mut data = encode_tag(3, 1);
        data.extend_from_slice(&f64::to_le_bytes(2.718281828));
        match decode_value(&data) {
            Value::Double(v) => assert!((v - 2.718281828).abs() < 1e-9),
            other => panic!("expected Double, got {:?}", other),
        }

        // int_value (field 4, varint)
        let data = encode_varint_field(4, (-42i64 as u64));
        match decode_value(&data) {
            Value::Int(v) => assert_eq!(v, -42),
            other => panic!("expected Int, got {:?}", other),
        }

        // uint_value (field 5, varint)
        let data = encode_varint_field(5, 999);
        match decode_value(&data) {
            Value::UInt(v) => assert_eq!(v, 999),
            other => panic!("expected UInt, got {:?}", other),
        }

        // bool_value (field 7, varint)
        let data = encode_varint_field(7, 1);
        match decode_value(&data) {
            Value::Bool(v) => assert!(v),
            other => panic!("expected Bool, got {:?}", other),
        }

        // sint_value (field 6, varint, zigzag)
        // zigzag_encode(-99) for i64: ((-99) << 1) ^ ((-99) >> 63) = 197
        let zigzag_val = ((-99i64 as u64) << 1) ^ ((-99i64 >> 63) as u64);
        let data = encode_varint_field(6, zigzag_val);
        match decode_value(&data) {
            Value::SInt(v) => assert_eq!(v, -99),
            other => panic!("expected SInt, got {:?}", other),
        }
    }

    #[test]
    fn test_feature_get_f64() {
        // Build a layer with a numeric value
        let layer = Layer {
            name: "test".into(),
            extent: 4096,
            features: vec![],
            keys: vec!["height".into(), "name".into()],
            values: vec![Value::Double(42.5), Value::String("tower".into())],
        };
        let feat = Feature {
            id: 1,
            tags: vec![0, 0, 1, 1], // height->42.5, name->"tower"
            geom_type: GeomType::Point,
            geometry: vec![vec![[0, 0]]],
        };
        assert_eq!(feat.get_f64(&layer, "height"), Some(42.5));
        assert_eq!(feat.get_str(&layer, "name"), Some("tower"));
        assert_eq!(feat.get_f64(&layer, "name"), None);
        assert_eq!(feat.get_f64(&layer, "missing"), None);
    }
}
