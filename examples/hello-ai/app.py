"""A deliberately dependency-free HTTP service.

The standard library only: the point of the example is to demonstrate the
packaging and air-gapped install flow, not to exercise a web framework. Adding
one would mean the example could not be built on a machine without a wheel
cache, which would undercut the thing it is meant to show.
"""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("PORT", "8000"))


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - name fixed by BaseHTTPRequestHandler
        if self.path == "/health":
            self._json(200, {"status": "healthy"})
        elif self.path == "/":
            self._json(
                200,
                {
                    "service": "hello-ai",
                    "message": "Running offline.",
                    "network": "not required",
                },
            )
        else:
            self._json(404, {"error": "not found"})

    def _json(self, status: int, payload: dict[str, str]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: object) -> None:
        # Quiet by default; the container runtime captures stdout.
        pass


if __name__ == "__main__":
    print(f"hello-ai listening on :{PORT}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()  # noqa: S104
