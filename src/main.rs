fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    println!("PolyMap — Tile-based dynamic loading");
    println!("Pan around to explore! Tiles load automatically.");

    // Start with no data — tile manager will load tiles as the camera moves
    polymap::run_app(None);
}
