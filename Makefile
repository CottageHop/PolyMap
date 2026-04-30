.PHONY: build run build-wasm build-worker build-js serve clean check

# Native build
build:
	cargo build --release

# Native run
run:
	cargo run --release

# WASM build — full renderer (default features)
build-wasm:
	wasm-pack build --release --target web --out-dir web/pkg
	@echo "WASM size: $$(du -h web/pkg/polymap_bg.wasm | cut -f1)"
	@if command -v wasm-opt > /dev/null 2>&1; then \
		wasm-opt -Oz --enable-bulk-memory --enable-nontrapping-float-to-int \
			web/pkg/polymap_bg.wasm -o web/pkg/polymap_bg.wasm && \
		echo "After wasm-opt: $$(du -h web/pkg/polymap_bg.wasm | cut -f1)"; \
	fi
	@gzip -9 -k -f web/pkg/polymap_bg.wasm 2>/dev/null && \
		echo "Gzipped: $$(du -h web/pkg/polymap_bg.wasm.gz | cut -f1)" || true

# WASM build — slim worker (worker feature only, no wgpu/winit/fontdue).
# Same crate, different feature set; output is renamed to keep the existing
# tile-worker.js paths working.
build-worker:
	wasm-pack build --release --target web --out-dir web/worker-pkg \
		-- --no-default-features --features worker
	@mv web/worker-pkg/polymap_bg.wasm web/worker-pkg/polymap_worker_bg.wasm
	@mv web/worker-pkg/polymap.js web/worker-pkg/polymap_worker.js
	@mv web/worker-pkg/polymap.d.ts web/worker-pkg/polymap_worker.d.ts 2>/dev/null || true
	@mv web/worker-pkg/polymap_bg.wasm.d.ts web/worker-pkg/polymap_worker_bg.wasm.d.ts 2>/dev/null || true
	@# Rewrite the JS shim's reference to the wasm filename.
	@sed -i.bak "s|polymap_bg\.wasm|polymap_worker_bg.wasm|g" web/worker-pkg/polymap_worker.js && rm web/worker-pkg/polymap_worker.js.bak
	@echo "Worker WASM size: $$(du -h web/worker-pkg/polymap_worker_bg.wasm | cut -f1)"

# Build JS library (copies WASM pkg into JS package)
build-js: build-wasm
	cp -r web/pkg js/pkg
	@echo "JS package ready at js/"

# Download OSM tiles for a region (default: NYC/Manhattan area)
# Requires: pmtiles CLI (brew install pmtiles or go install github.com/protomaps/go-pmtiles/cmd/pmtiles@latest)
PMTILES_BUILD ?= 20260430
# ~9×13 km from Midtown South down through Lower Manhattan + Battery Park,
# plus western Brooklyn (Park Slope, Brooklyn Heights), DUMBO, Williamsburg,
# Long Island City. With MAX_TILES = 32 in tiles.rs, only ~32 tiles are GPU-
# resident at once regardless of bbox size; bigger bbox just gives more area
# to pan through. Override:
#   make download-tiles PMTILES_BBOX=-74.05,40.68,-73.90,40.82
PMTILES_BBOX ?= -74.04,40.66,-73.93,40.78
download-tiles:
	@if ! command -v pmtiles > /dev/null 2>&1; then \
		echo "Error: pmtiles CLI not found. Install with: brew install pmtiles"; exit 1; \
	fi
	pmtiles extract https://build.protomaps.com/$(PMTILES_BUILD).pmtiles web/tiles.pmtiles \
		--bbox="$(PMTILES_BBOX)"
	@echo "Tiles downloaded: $$(du -h web/tiles.pmtiles | cut -f1)"

# Build everything needed for the web demo
build-web: build-wasm build-worker
	@echo "Web demo ready — run 'make serve' to start"

# Serve WASM locally on port 8080
serve: build-web
	@if [ ! -f web/tiles.pmtiles ]; then \
		echo "No tiles found. Downloading NYC area..."; \
		$(MAKE) download-tiles; \
	fi
	@echo "Serving at http://localhost:8080"
	cd web && python3 server.py 8080

# Type-check both feature combinations from one crate
check:
	cargo check --lib --target wasm32-unknown-unknown
	cargo check --lib --no-default-features --features worker --target wasm32-unknown-unknown

# Clean all build artifacts
clean:
	cargo clean
	rm -rf web/pkg web/worker-pkg js/pkg
