.PHONY: build run build-wasm build-worker build-js serve clean check

# Native build
build:
	cargo build --release

# Native run
run:
	cargo run --release

# WASM build
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

# Build the Web Worker WASM module
build-worker:
	cd polymap-worker && wasm-pack build --release --target web --out-dir pkg
	mkdir -p web/worker-pkg
	cp polymap-worker/pkg/polymap_worker_bg.wasm web/worker-pkg/
	cp polymap-worker/pkg/polymap_worker.js web/worker-pkg/
	@echo "Worker WASM copied to web/worker-pkg/"

# Build JS library (copies WASM pkg into JS package)
build-js: build-wasm
	cp -r web/pkg js/pkg
	@echo "JS package ready at js/"

# Download OSM tiles for a region (default: NYC/Manhattan area)
# Requires: pmtiles CLI (brew install pmtiles or go install github.com/protomaps/go-pmtiles/cmd/pmtiles@latest)
PMTILES_BUILD ?= 20260325
PMTILES_BBOX ?= -74.05,40.68,-73.90,40.82
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
	cd web && python3 -m http.server 8080

# Type-check both crates
check:
	cargo check --lib --target wasm32-unknown-unknown
	cd polymap-worker && cargo check --lib --target wasm32-unknown-unknown

# Sync shared source files from src/ to polymap-worker/src/
sync-worker:
	@for f in mapdata.rs mvt.rs mvt_convert.rs pmtiles.rs; do \
		cp src/$$f polymap-worker/src/$$f; \
	done
	@echo "Worker synced"

# Clean all build artifacts
clean:
	cargo clean
	cd polymap-worker && cargo clean
	rm -rf web/pkg web/worker-pkg js/pkg
