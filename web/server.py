"""Minimal static file server with HTTP Range request support.

Python's built-in http.server ignores Range headers, which breaks
PMTiles (it relies on Range requests to read individual tiles from
a single archive file). This server handles Range requests correctly.
"""

import http.server
import io
import os


class RangeHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        path = self.translate_path(self.path)
        range_header = self.headers.get('Range')

        # Only handle Range requests for existing files
        if not os.path.isfile(path) or not range_header or not range_header.startswith('bytes='):
            return super().do_GET()

        file_size = os.path.getsize(path)
        range_spec = range_header[6:]  # strip "bytes="
        parts = range_spec.split('-')
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if parts[1] else file_size - 1
        end = min(end, file_size - 1)
        length = end - start + 1

        with open(path, 'rb') as f:
            f.seek(start)
            data = f.read(length)

        self.send_response(206)
        self.send_header('Content-Type', self.guess_type(path))
        self.send_header('Content-Length', str(length))
        self.send_header('Content-Range', f'bytes {start}-{end}/{file_size}')
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Expose-Headers', 'Content-Range, Accept-Ranges')
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Headers', 'Range')
        self.send_header('Access-Control-Expose-Headers', 'Content-Range, Accept-Ranges')
        self.end_headers()

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()


if __name__ == '__main__':
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    with http.server.HTTPServer(('', port), RangeHTTPRequestHandler) as server:
        print(f'Serving at http://localhost:{port}')
        server.serve_forever()
