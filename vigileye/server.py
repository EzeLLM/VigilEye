"""Minimal JSON HTTP API (stdlib only). Binds to 127.0.0.1 by default; set VIGILEYE_API_TOKEN to require a bearer token."""

from __future__ import annotations

import hmac
import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .config import Config


def make_handler(app):
    token = os.environ.get("VIGILEYE_API_TOKEN", "")

    class Handler(BaseHTTPRequestHandler):
        server_version = "VigilEye/2"

        def log_message(self, fmt, *args):  # route through stderr quietly
            if not app.quiet:
                super().log_message(fmt, *args)

        def _send(self, code: int, payload) -> None:
            data = json.dumps(payload, default=str).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _authed(self) -> bool:
            if not token:
                return True
            got = self.headers.get("Authorization", "")
            return hmac.compare_digest(got, f"Bearer {token}")

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}") if n else {}

        def do_GET(self):
            if self.path == "/health":
                return self._send(200, {"ok": True})
            if not self._authed():
                return self._send(401, {"error": "unauthorized"})
            try:
                if self.path == "/v1/lenses":
                    from .analysis import list_lenses
                    return self._send(200, [{"name": l.name, "title": l.title, "description": l.description} for l in list_lenses()])
                if self.path == "/v1/collectors":
                    from .collectors import list_collectors
                    return self._send(200, list_collectors())
                if self.path == "/v1/providers":
                    return self._send(200, app.router.describe())
                if self.path == "/v1/cases":
                    return self._send(200, app.store.list_cases())
                m = re.fullmatch(r"/v1/cases/([^/]+)/reports", self.path)
                if m:
                    return self._send(200, app.store.list_reports(m.group(1)))
                m = re.fullmatch(r"/v1/reports/(\d+)", self.path)
                if m:
                    rep = app.store.get_report(int(m.group(1)))
                    return self._send(200, rep.to_dict()) if rep else self._send(404, {"error": "not found"})
                return self._send(404, {"error": "not found"})
            except Exception as e:
                return self._send(500, {"error": str(e)})

        def do_POST(self):
            if not self._authed():
                return self._send(401, {"error": "unauthorized"})
            try:
                body = self._body()
                if self.path in ("/v1/collect", "/v1/run"):
                    targets = body.get("targets") or []
                    if not targets:
                        return self._send(400, {"error": "targets required"})
                    case = body.get("case") or re.sub(r"[^a-z0-9]+", "-", targets[0].lower()).strip("-")
                    counts = app.collect(targets, case, int(body.get("limit", 100)), body.get("opts") or {})
                    if self.path == "/v1/collect":
                        return self._send(200, {"case": case, "counts": counts})
                    body["case"] = case
                if self.path in ("/v1/analyze", "/v1/run"):
                    rid, report, paths = app.analyze(body["case"], lens=body.get("lens"), query=body.get("query", ""),
                                                     vision=body.get("vision"), ach=body.get("ach"))
                    return self._send(200, {"report_id": rid, "files": [str(p) for p in paths], "report": report.to_dict()})
                return self._send(404, {"error": "not found"})
            except KeyError as e:
                return self._send(400, {"error": f"missing field {e}"})
            except Exception as e:
                return self._send(500, {"error": str(e)})

    return Handler


def serve(config: Config, host: str = "127.0.0.1", port: int = 8765, store_path: str | None = None) -> None:
    from .cli import App

    app = App(config, store_path=store_path, quiet=False)
    httpd = ThreadingHTTPServer((host, port), make_handler(app))
    print(f"VigilEye API listening on http://{host}:{port}  (auth: {'bearer token' if os.environ.get('VIGILEYE_API_TOKEN') else 'none'})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
