import json
import threading
from urllib.request import urlopen

from decision_lab.http import build_http_server
from decision_lab.server import build_service


def test_health_endpoint_reports_ready() -> None:
    server = build_http_server(
        build_service("demo"),
        host="127.0.0.1",
        port=0,
    )
    thread = threading.Thread(
        target=server.serve_forever,
        daemon=True,
    )
    thread.start()

    try:
        port = server.server_address[1]
        with urlopen(
            f"http://127.0.0.1:{port}/health",
            timeout=2,
        ) as response:
            assert response.status == 200
            assert json.load(response) == {"status": "ok"}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
