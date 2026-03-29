//! WASM-compatible PMTiles v3 reader using HTTP range requests.
//!
//! Port of mappy's native `pmtiles.rs` for async WASM usage.
//! Uses `wasm_fetch_bytes` for HTTP range requests and caches
//! the header, root directory, and leaf directories in a
//! thread-local singleton.

#[cfg(target_arch = "wasm32")]
mod inner {
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::io::{self, Cursor, Read};

    use flate2::read::GzDecoder;

    /// PMTiles v3 header is exactly 127 bytes.
    const HEADER_SIZE: u64 = 127;

    // ── Types (identical to mappy) ──────────────────────────────────────

    #[derive(Debug, Clone, Copy, PartialEq)]
    #[repr(u8)]
    pub enum Compression {
        Unknown = 0,
        None = 1,
        Gzip = 2,
        Brotli = 3,
        Zstd = 4,
    }

    impl From<u8> for Compression {
        fn from(v: u8) -> Self {
            match v {
                1 => Compression::None,
                2 => Compression::Gzip,
                3 => Compression::Brotli,
                4 => Compression::Zstd,
                _ => Compression::Unknown,
            }
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq)]
    #[repr(u8)]
    pub enum TileType {
        Unknown = 0,
        Mvt = 1,
        Png = 2,
        Jpeg = 3,
        Webp = 4,
        Avif = 5,
    }

    impl From<u8> for TileType {
        fn from(v: u8) -> Self {
            match v {
                1 => TileType::Mvt,
                2 => TileType::Png,
                3 => TileType::Jpeg,
                4 => TileType::Webp,
                5 => TileType::Avif,
                _ => TileType::Unknown,
            }
        }
    }

    #[derive(Debug)]
    pub struct Header {
        pub root_dir_offset: u64,
        pub root_dir_length: u64,
        pub metadata_offset: u64,
        pub metadata_length: u64,
        pub leaf_dir_offset: u64,
        pub leaf_dir_length: u64,
        pub tile_data_offset: u64,
        pub tile_data_length: u64,
        pub num_addressed_tiles: u64,
        pub num_tile_entries: u64,
        pub num_tile_contents: u64,
        pub clustered: bool,
        pub internal_compression: Compression,
        pub tile_compression: Compression,
        pub tile_type: TileType,
        pub min_zoom: u8,
        pub max_zoom: u8,
    }

    #[derive(Debug, Clone)]
    pub struct DirEntry {
        pub tile_id: u64,
        pub offset: u64,
        pub length: u32,
        pub run_length: u32,
    }

    // ── Cached state ────────────────────────────────────────────────────

    struct CachedState {
        url: String,
        header: Header,
        root_dir: Vec<DirEntry>,
        /// Leaf directories keyed by (offset, length) within the leaf section.
        leaf_cache: HashMap<(u64, u32), Vec<DirEntry>>,
    }

    thread_local! {
        static STATE: RefCell<Option<CachedState>> = RefCell::new(None);
    }

    // ── Parsing (identical to mappy) ────────────────────────────────────

    fn parse_header(buf: &[u8]) -> Option<Header> {
        if buf.len() < HEADER_SIZE as usize {
            return None;
        }
        let r = |off: usize| u64::from_le_bytes(buf[off..off + 8].try_into().unwrap());
        Some(Header {
            root_dir_offset: r(8),
            root_dir_length: r(16),
            metadata_offset: r(24),
            metadata_length: r(32),
            leaf_dir_offset: r(40),
            leaf_dir_length: r(48),
            tile_data_offset: r(56),
            tile_data_length: r(64),
            num_addressed_tiles: r(72),
            num_tile_entries: r(80),
            num_tile_contents: r(88),
            clustered: buf[96] == 1,
            internal_compression: Compression::from(buf[97]),
            tile_compression: Compression::from(buf[98]),
            tile_type: TileType::from(buf[99]),
            min_zoom: buf[100],
            max_zoom: buf[101],
        })
    }

    fn read_varint(cursor: &mut Cursor<&[u8]>) -> io::Result<u64> {
        let mut result: u64 = 0;
        let mut shift = 0;
        let mut buf = [0u8; 1];
        loop {
            cursor.read_exact(&mut buf)?;
            let byte = buf[0];
            result |= ((byte & 0x7F) as u64) << shift;
            if byte & 0x80 == 0 {
                break;
            }
            shift += 7;
            if shift >= 64 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Varint too long",
                ));
            }
        }
        Ok(result)
    }

    fn parse_directory(data: &[u8]) -> io::Result<Vec<DirEntry>> {
        let mut cursor = Cursor::new(data);
        let num_entries = read_varint(&mut cursor)? as usize;

        // Read tile_ids (delta-encoded)
        let mut tile_ids = Vec::with_capacity(num_entries);
        let mut last_id: u64 = 0;
        for _ in 0..num_entries {
            let delta = read_varint(&mut cursor)?;
            last_id += delta;
            tile_ids.push(last_id);
        }

        // Read run_lengths
        let mut run_lengths = Vec::with_capacity(num_entries);
        for _ in 0..num_entries {
            run_lengths.push(read_varint(&mut cursor)? as u32);
        }

        // Read lengths
        let mut lengths = Vec::with_capacity(num_entries);
        for _ in 0..num_entries {
            lengths.push(read_varint(&mut cursor)? as u32);
        }

        // Read offsets (delta-encoded for runs with length > 0)
        let mut offsets = Vec::with_capacity(num_entries);
        let mut last_offset: u64 = 0;
        for i in 0..num_entries {
            let v = read_varint(&mut cursor)?;
            if v == 0 && i > 0 {
                // Offset continues from previous entry's end
                last_offset += lengths[i - 1] as u64;
            } else {
                last_offset = v - 1;
            }
            offsets.push(last_offset);
        }

        let entries = (0..num_entries)
            .map(|i| DirEntry {
                tile_id: tile_ids[i],
                offset: offsets[i],
                length: lengths[i],
                run_length: run_lengths[i],
            })
            .collect();

        Ok(entries)
    }

    fn decompress_and_parse_dir(
        data: &[u8],
        compression: Compression,
    ) -> io::Result<Vec<DirEntry>> {
        let decompressed = match compression {
            Compression::Gzip => {
                let mut decoder = GzDecoder::new(data);
                let mut out = Vec::new();
                decoder.read_to_end(&mut out)?;
                out
            }
            Compression::None => data.to_vec(),
            other => {
                return Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    format!("Unsupported compression: {:?}", other),
                ));
            }
        };
        parse_directory(&decompressed)
    }

    fn decompress_tile(data: &[u8], compression: Compression) -> Option<Vec<u8>> {
        match compression {
            Compression::Gzip => {
                let mut decoder = GzDecoder::new(data);
                let mut out = Vec::new();
                decoder.read_to_end(&mut out).ok()?;
                Some(out)
            }
            Compression::None => Some(data.to_vec()),
            _ => None,
        }
    }

    /// Convert z/x/y to a Hilbert TileID (PMTiles v3 spec).
    fn zxy_to_tileid(z: u8, x: u32, y: u32) -> u64 {
        if z == 0 {
            return 0;
        }
        let base_id: u64 = (0..z as u64).map(|i| 4u64.pow(i as u32)).sum();
        let n = 1u32 << z;
        let mut rx: u32;
        let mut ry: u32;
        let mut d: u64 = 0;
        let mut s = n / 2;
        let mut tx = x;
        let mut ty = y;
        while s > 0 {
            rx = if (tx & s) > 0 { 1 } else { 0 };
            ry = if (ty & s) > 0 { 1 } else { 0 };
            d += s as u64 * s as u64 * ((3 * rx) ^ ry) as u64;
            // Rotate
            if ry == 0 {
                if rx == 1 {
                    tx = s * 2 - 1 - tx;
                    ty = s * 2 - 1 - ty;
                }
                std::mem::swap(&mut tx, &mut ty);
            }
            s /= 2;
        }
        base_id + d
    }

    // ── External fetch (defined elsewhere in the WASM crate) ────────────

    use crate::wasm_fetch_bytes;

    /// Fetch a byte range from the given URL.
    async fn fetch_range(url: &str, start: u64, end: u64) -> Option<Vec<u8>> {
        wasm_fetch_bytes(url, start, end).await
    }

    // ── Initialization ──────────────────────────────────────────────────

    /// Ensure the header and root directory are loaded for `url`.
    /// If the URL differs from the cached one, re-initialise.
    async fn ensure_init(url: &str) -> bool {
        let already = STATE.with(|s| {
            s.borrow()
                .as_ref()
                .map_or(false, |st| st.url == url)
        });
        if already {
            return true;
        }

        let header_bytes = match fetch_range(url, 0, HEADER_SIZE).await {
            Some(b) if b.len() == HEADER_SIZE as usize => b,
            _ => return false,
        };

        // Verify magic bytes "PMTiles" + version 3
        if &header_bytes[0..7] != b"PMTiles" || header_bytes[7] != 3 {
            return false;
        }

        let header = match parse_header(&header_bytes) {
            Some(h) => h,
            None => return false,
        };

        let root_bytes = match fetch_range(
            url,
            header.root_dir_offset,
            header.root_dir_offset + header.root_dir_length,
        )
        .await
        {
            Some(b) => b,
            None => return false,
        };

        let root_dir = match decompress_and_parse_dir(&root_bytes, header.internal_compression) {
            Ok(d) => d,
            Err(_) => return false,
        };

        STATE.with(|s| {
            *s.borrow_mut() = Some(CachedState {
                url: url.to_string(),
                header,
                root_dir,
                leaf_cache: HashMap::new(),
            });
        });

        true
    }

    // ── Tile lookup ─────────────────────────────────────────────────────

    /// Search `dir` for `tile_id`, returning the matching `DirEntry` if found.
    fn search_dir(dir: &[DirEntry], tile_id: u64) -> Option<DirEntry> {
        let m = dir.partition_point(|e| e.tile_id <= tile_id);
        if m == 0 {
            return None;
        }
        let entry = &dir[m - 1];

        if entry.run_length == 0 {
            // Leaf directory pointer -- always follow
            return Some(entry.clone());
        }

        if tile_id >= entry.tile_id && tile_id < entry.tile_id + entry.run_length as u64 {
            return Some(entry.clone());
        }

        None
    }

    /// Recursively resolve `tile_id` through root and leaf directories,
    /// fetching leaf directories and tile data over HTTP as needed.
    async fn find_tile(url: &str, tile_id: u64) -> Option<Vec<u8>> {
        // Start searching in the root directory.
        let entry = STATE.with(|s| {
            let st = s.borrow();
            let st = st.as_ref()?;
            search_dir(&st.root_dir, tile_id)
        })?;

        resolve_entry(url, tile_id, entry).await
    }

    /// Given a matched `DirEntry`, either fetch the tile data directly or
    /// follow the leaf directory chain.
    fn resolve_entry(
        url: &str,
        tile_id: u64,
        entry: DirEntry,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Option<Vec<u8>>> + '_>> {
        Box::pin(async move {
            if entry.run_length == 0 {
                // Leaf directory -- fetch, parse, and recurse.
                let leaf_entry =
                    fetch_leaf_and_search(url, entry.offset, entry.length, tile_id).await?;
                // Recurse (leaf entries should not themselves be leaf pointers in
                // practice, but handle it just in case).
                return resolve_entry(url, tile_id, leaf_entry).await;
            }

            // Direct tile data reference.
            let (tile_data_offset, tile_compression) = STATE.with(|s| {
                let st = s.borrow();
                let st = st.as_ref()?;
                Some((st.header.tile_data_offset, st.header.tile_compression))
            })?;

            let abs_start = tile_data_offset + entry.offset;
            let abs_end = abs_start + entry.length as u64;
            let raw = fetch_range(url, abs_start, abs_end).await?;

            decompress_tile(&raw, tile_compression)
        })
    }

    /// Fetch (or retrieve from cache) a leaf directory, then search it.
    async fn fetch_leaf_and_search(
        url: &str,
        offset: u64,
        length: u32,
        tile_id: u64,
    ) -> Option<DirEntry> {
        // Check cache first.
        let cached = STATE.with(|s| {
            let st = s.borrow();
            let st = st.as_ref()?;
            st.leaf_cache.get(&(offset, length)).and_then(|dir| search_dir(dir, tile_id))
        });
        if let Some(entry) = cached {
            return Some(entry);
        }

        // Fetch and parse.
        let (leaf_dir_offset, internal_compression) = STATE.with(|s| {
            let st = s.borrow();
            let st = st.as_ref()?;
            Some((st.header.leaf_dir_offset, st.header.internal_compression))
        })?;

        let abs_start = leaf_dir_offset + offset;
        let abs_end = abs_start + length as u64;
        let raw = fetch_range(url, abs_start, abs_end).await?;
        let dir = decompress_and_parse_dir(&raw, internal_compression).ok()?;

        let result = search_dir(&dir, tile_id);

        // Store in cache.
        STATE.with(|s| {
            if let Some(ref mut st) = *s.borrow_mut() {
                st.leaf_cache.insert((offset, length), dir);
            }
        });

        result
    }

    // ── Public API ──────────────────────────────────────────────────────

    /// Fetch a single tile by z/x/y from a PMTiles v3 archive at `url`.
    ///
    /// The first call for a given URL will read the 127-byte header and
    /// root directory. Subsequent calls reuse the cached state.
    ///
    /// Returns `None` if the tile does not exist, the archive cannot be
    /// read, or the network request fails.
    pub async fn get_tile(url: &str, z: u8, x: u32, y: u32) -> Option<Vec<u8>> {
        if !ensure_init(url).await {
            return None;
        }
        let tile_id = zxy_to_tileid(z, x, y);
        find_tile(url, tile_id).await
    }

    // ── Tests ───────────────────────────────────────────────────────────

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_tileid_z0() {
            assert_eq!(zxy_to_tileid(0, 0, 0), 0);
        }

        #[test]
        fn test_tileid_z1() {
            let id = zxy_to_tileid(1, 0, 0);
            assert_eq!(id, 1); // base=1, hilbert(0,0)=0 -> 1
        }

        #[test]
        fn test_parse_header_magic() {
            let mut buf = vec![0u8; 127];
            buf[0..7].copy_from_slice(b"PMTiles");
            buf[7] = 3;
            let h = parse_header(&buf);
            assert_eq!(h.min_zoom, 0);
            assert_eq!(h.max_zoom, 0);
            assert_eq!(h.tile_type, TileType::Unknown);
        }

        #[test]
        fn test_varint_roundtrip() {
            // Encode 300 as varint: 300 = 0b100101100
            // byte 0: 0b0_0101100 | 0x80 = 0xAC
            // byte 1: 0b0_0000010 = 0x02
            let data: &[u8] = &[0xAC, 0x02];
            let mut cursor = Cursor::new(data);
            assert_eq!(read_varint(&mut cursor).unwrap(), 300);
        }
    }
}

// Re-export the public API so callers can use `pmtiles::get_tile(...)`.
#[cfg(target_arch = "wasm32")]
pub use inner::get_tile;

// Stub for non-WASM targets so that `mod pmtiles` compiles everywhere.
#[cfg(not(target_arch = "wasm32"))]
pub async fn get_tile(_url: &str, _z: u8, _x: u32, _y: u32) -> Option<Vec<u8>> {
    None
}
