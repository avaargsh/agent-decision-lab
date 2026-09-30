from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .service import DecisionService


def build_http_server(
    service: DecisionService,
    *,
    host: str = "127.0.0.1",
    port: int = 8080,
) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def _write_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/health":
                self.send_error(404)
                return
            self._write_json(200, {"status": "ok"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/decision":
                self.send_error(404)
                return

            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length))
                result = service.decide(payload)
                self._write_json(200, result)
            except (KeyError, ValueError, json.JSONDecodeError) as exc:
                self._write_json(400, {"error": str(exc)})

        def log_message(self, format: str, *args: Any) -> None:
            return

    return ThreadingHTTPServer((host, port), Handler)


def serve(
    service: DecisionService,
    *,
    host: str = "127.0.0.1",
    port: int = 8080,
) -> None:
    server = build_http_server(service, host=host, port=port)
    server.serve_forever()
