"""RAG application skeleton.

Deliberately thin: the example is about the packaging problem, not about
retrieval quality. What matters here is that every address it talks to points
inside the deployment, so the whole stack runs with no route to the outside.
"""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("PORT", "8080"))

DEPENDENCIES = {
    "llm": os.environ.get("LLM_BASE_URL", "http://vllm:8000/v1"),
    "vectors": os.environ.get("QDRANT_URL", "http://qdrant:6333"),
    "cache": os.environ.get("REDIS_URL", "redis://redis:6379/0"),
}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - name fixed by the base class
        if self.path == "/health":
            self._json(200, {"status": "healthy"})
        elif self.path == "/":
            self._json(
                200,
                {
                    "service": "rag-stack",
                    "dependencies": DEPENDENCIES,
                    "network": "all addresses are internal to the deployment",
                },
            )
        else:
            self._json(404, {"error": "not found"})

    def _json(self, status: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: object) -> None:
        pass


if __name__ == "__main__":
    print(f"rag-stack listening on :{PORT}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()  # noqa: S104
