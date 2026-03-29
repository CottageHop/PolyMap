# Contributing to PolyMap

Thanks for your interest in contributing! Here's how to get started.

## Development Setup

1. Install [Rust](https://rustup.rs/) (stable toolchain)
2. Install [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/): `cargo install wasm-pack`
3. Install the WASM target: `rustup target add wasm32-unknown-unknown`

```bash
git clone https://github.com/CottageHop/PolyMap.git
cd PolyMap
make build-wasm
make serve
```

## Project Structure

- **`src/`** — Core Rust library. Most rendering and tile logic lives here.
- **`js/src/`** — JavaScript wrapper that bridges the WASM module to a friendly API.
- **`polymap-worker/`** — Web Worker crate for off-main-thread tile processing. Shares `mapdata.rs`, `mvt.rs`, `mvt_convert.rs`, and `pmtiles.rs` with the main crate.
- **`web/`** — Demo page for local development.

## Building

```bash
make build-wasm    # Build WASM module (output in web/pkg/)
make build-js      # Build JS package (output in js/pkg/)
make serve         # Build + serve demo at localhost:8080
make clean         # Remove all build artifacts
```

## Keeping the Worker in Sync

The `polymap-worker/` crate shares source files with the main crate. After changing any of these files in `src/`, copy them to `polymap-worker/src/`:

- `mapdata.rs`
- `mvt.rs`
- `mvt_convert.rs`
- `pmtiles.rs`

## Code Style

- Follow standard Rust formatting: `cargo fmt`
- Run clippy: `cargo clippy --target wasm32-unknown-unknown`
- Keep GPU buffer creation going through `gpu::safe_buffer()` for WebGPU compatibility

## Pull Requests

1. Fork the repo and create a feature branch
2. Make your changes with clear commit messages
3. Ensure `cargo check --target wasm32-unknown-unknown` passes
4. Open a PR with a description of what changed and why

## Areas for Contribution

- **Browser compatibility** — testing and fixes for Safari, Firefox, mobile browsers
- **Performance** — reducing GPU memory, faster MVT decoding, smarter tile caching
- **Rendering** — new material types, better shadows, water effects
- **Labels** — curved text along roads, better collision avoidance
- **Documentation** — examples, tutorials, API docs

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
