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

# Build JS library (copies WASM pkg into JS package)
build-js: build-wasm
	cp -r web/pkg js/pkg
	@echo "JS package ready at js/"

# Serve WASM locally on port 8080
serve: build-wasm
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
	rm -rf web/pkg js/pkg
