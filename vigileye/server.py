"""Minimal JSON HTTP API (stdlib only). Binds to 127.0.0.1 by default; set VIGILEYE_API_TOKEN to require a bearer token."""

from __future__ import annotations

import hmac
import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

from .config import Config


_TLP = ("clear", "green", "amber", "red")


def _opt_bool(v):
    """JSON body flag -> True/False/None (None keeps the configured default)."""
    if v is None or isinstance(v, bool):
        return v
    if isinstance(v, str):
        low = v.strip().lower()
        if low in ("1", "true", "yes", "on"):
            return True
        if low in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"expected a boolean, got {v!r}")
    return bool(v)


def _formats(v):
    if v is None:
        return None
    from .report import FORMATS

    items = v.split(",") if isinstance(v, str) else list(v)
    out = [str(f).strip().lower() for f in items if str(f).strip()]
    bad = [f for f in out if f not in FORMATS]
    if bad:
        raise ValueError(f"unknown report format(s) {bad}; choose from {list(FORMATS)}")
    return out or None


def _tlp(v, app):
    t = str(v or app.config.get("output.tlp", "clear") or "clear").strip().lower().removeprefix("tlp:")
    if t not in _TLP:
        raise ValueError(f"unknown TLP level {t!r}; expected one of {list(_TLP)}")
    return t


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
            url = urlsplit(self.path)
            path, qs = url.path, parse_qs(url.query)
            if path == "/health":
                return self._send(200, {"ok": True})
            if not self._authed():
                return self._send(401, {"error": "unauthorized"})
            try:
                if path == "/v1/lenses":
                    from .analysis import list_lenses
                    return self._send(200, [{"name": l.name, "title": l.title, "description": l.description} for l in list_lenses()])
                if path == "/v1/collectors":
                    from .collectors import list_collectors
                    return self._send(200, list_collectors())
                if path == "/v1/providers":
                    return self._send(200, app.router.describe())
                if path == "/v1/cases":
                    return self._send(200, app.store.list_cases())
                m = re.fullmatch(r"/v1/cases/([^/]+)/reports", path)
                if m:
                    return self._send(200, app.store.list_reports(m.group(1)))
                m = re.fullmatch(r"/v1/reports/(\d+)/(stix|navigator)", path)
                if m:  # STIX 2.1 bundle (?tlp=clear|green|amber|red) or ATT&CK Navigator layer
                    from .export import report_to_navigator_layer, report_to_stix
                    rep = app.store.get_report(int(m.group(1)))
                    if rep is None:
                        return self._send(404, {"error": "not found"})
                    if m.group(2) == "stix":
                        try:
                            tlp = _tlp((qs.get("tlp") or [None])[0], app)
                        except ValueError as e:
                            return self._send(400, {"error": str(e)})
                        return self._send(200, report_to_stix(rep, tlp=tlp))
                    return self._send(200, report_to_navigator_layer(rep))
                m = re.fullmatch(r"/v1/reports/(\d+)", path)
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
                opts: dict = {}
                if self.path in ("/v1/analyze", "/v1/run"):  # validate options before collecting or calling a model
                    try:
                        opts = {"vision": _opt_bool(body.get("vision")), "ach": _opt_bool(body.get("ach")),
                                "verify": _opt_bool(body.get("verify")), "critic": _opt_bool(body.get("critic")),
                                "ground_cves": _opt_bool(body.get("ground_cves")),
                                "formats": _formats(body.get("formats")),
                                "tlp": _tlp(body.get("tlp"), app) if body.get("tlp") else None}
                    except ValueError as e:
                        return self._send(400, {"error": str(e)})
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
                                                     **opts)
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
