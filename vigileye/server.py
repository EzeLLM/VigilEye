"""Minimal JSON HTTP API (stdlib only).

Security model: binds to 127.0.0.1 by default; set VIGILEYE_API_TOKEN to require a bearer token
(compared in constant time). POST bodies must be ``application/json``; without a token, requests
whose ``Host`` is not a loopback name, an IP literal or a name in VIGILEYE_API_ALLOWED_HOSTS, or
that carry a foreign ``Origin``, are refused (CSRF and DNS-rebinding protection). The API never reads local files on a client's behalf: ``file:``
targets and lens file paths are refused, and every collect/run/investigate target must pass
:func:`vigileye.policy.check_target` (``mode="api"``), which refuses URLs on non-public hosts.
"""

from __future__ import annotations

import hmac
import json
import os
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlsplit

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


def _lens(v):
    """Built-in lens names only: a lens *path* would make the API read a local file."""
    if v is None or v == "":
        return None
    from .analysis.lenses import list_lenses

    names = {l.name for l in list_lenses()}
    name = str(v).strip().lower()
    if name not in names:
        raise ValueError(f"unknown lens {v!r}; the API accepts the built-in lenses {sorted(names)}")
    return name


def _policy_gate(specs, field: str = "targets"):
    """``(normalized specs, None)`` or ``(None, 400 payload)``: every spec through ``check_target(mode="api")``."""
    from .policy import check_targets

    if isinstance(specs, str):
        specs = [specs]
    if not isinstance(specs, list) or not specs or not all(isinstance(t, str) for t in specs):
        return None, {"error": f"{field} required (a spec string or a list of spec strings)"}
    ok, rejected = check_targets(specs, mode="api", allow_files=False)
    if rejected:
        why = "; ".join(f"{r['spec']}: {r['reason']}" for r in rejected)
        return None, {"error": f"{field} refused by policy: {why}", "rejected": rejected}
    return ok, None


def _tlp(v, app):
    t = str(v or app.config.get("output.tlp", "clear") or "clear").strip().lower().removeprefix("tlp:")
    if t not in _TLP:
        raise ValueError(f"unknown TLP level {t!r}; expected one of {list(_TLP)}")
    return t


_LOOPBACK_NAMES = {"localhost", "127.0.0.1", "::1", "localhost.localdomain"}


def _hostname(value: str) -> str:
    """``Host`` / ``Origin`` value -> bare lower-case host name (port, brackets, trailing dot dropped)."""
    v = str(value or "").strip().lower()
    if "://" in v:
        v = urlsplit(v).netloc
    v = v.rsplit("@", 1)[-1]
    if v.startswith("["):
        v = v[1:].split("]", 1)[0]
    elif v.count(":") == 1:
        v = v.split(":", 1)[0]
    return v.rstrip(".")


def _is_ip(host: str) -> bool:
    import ipaddress

    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def _allowed_hosts() -> set[str]:
    """Host names (beyond loopback names and IP literals) the API answers to without a token:
    ``VIGILEYE_API_ALLOWED_HOSTS`` (comma-separated), e.g. a Docker service name."""
    extra = os.environ.get("VIGILEYE_API_ALLOWED_HOSTS", "")
    return _LOOPBACK_NAMES | {_hostname(h) for h in extra.split(",") if h.strip()}


def make_handler(app):
    token = os.environ.get("VIGILEYE_API_TOKEN", "")
    allowed_hosts = _allowed_hosts()

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

        def _browser_refusal(self) -> str | None:
            """Why a request looks like it came from a web page rather than an API client.

            Without a token, any page the analyst opens could otherwise drive the API: a CORS
            "simple" POST (``text/plain`` body, no preflight) is a CSRF, and a DNS-rebinding name
            that resolves to 127.0.0.1 makes same-origin reads possible. So the ``Host`` must be a
            loopback name, an IP literal (rebinding needs a domain name) or a name listed in
            ``VIGILEYE_API_ALLOWED_HOSTS``, and a foreign ``Origin`` is refused. With a token, a page
            cannot attach the ``Authorization`` header without a preflight this server never grants.
            """
            if token:
                return None
            host = _hostname(self.headers.get("Host", ""))
            if not host or not (host in allowed_hosts or _is_ip(host)):
                return "unexpected Host header (set VIGILEYE_API_ALLOWED_HOSTS or VIGILEYE_API_TOKEN)"
            origin = self.headers.get("Origin")
            if origin is not None:
                oh = _hostname(origin)
                if origin.strip().lower() == "null" or not (oh in allowed_hosts or (_is_ip(oh) and oh == host)):
                    return "cross-origin requests are refused"
            return None

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}") if n else {}

        def do_GET(self):
            url = urlsplit(self.path)
            path, qs = url.path, parse_qs(url.query)
            if path == "/health":
                return self._send(200, {"ok": True})
            refusal = self._browser_refusal()
            if refusal:
                return self._send(403, {"error": refusal})
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
                m = re.fullmatch(r"/v1/cases/([^/]+)/evidence(/verify)?", path)
                if m:  # evidence manifest, or chain + document verification (?anchor=<earlier chain head>)
                    case = unquote(m.group(1))
                    if m.group(2):
                        return self._send(200, app.ledger.verify_case(case, anchor=(qs.get("anchor") or [None])[0]))
                    return self._send(200, app.ledger.manifest(case))
                m = re.fullmatch(r"/v1/cases/([^/]+)/reports", path)
                if m:
                    return self._send(200, app.store.list_reports(unquote(m.group(1))))
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
            refusal = self._browser_refusal()
            if refusal:
                return self._send(403, {"error": refusal})
            if not self._authed():
                return self._send(401, {"error": "unauthorized"})
            ctype = (self.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
            if ctype != "application/json":  # forces a CORS preflight for browser callers
                return self._send(415, {"error": "Content-Type must be application/json"})
            try:
                try:
                    body = self._body()
                except ValueError as e:
                    return self._send(400, {"error": f"invalid JSON body: {e}"})
                if not isinstance(body, dict):
                    return self._send(400, {"error": "the JSON body must be an object"})
                opts: dict = {}
                lens = None
                if self.path in ("/v1/analyze", "/v1/run", "/v1/investigate"):  # validate before any work
                    try:
                        lens = _lens(body.get("lens"))
                        opts = {"vision": _opt_bool(body.get("vision")), "ach": _opt_bool(body.get("ach")),
                                "verify": _opt_bool(body.get("verify")), "critic": _opt_bool(body.get("critic")),
                                "ground_cves": _opt_bool(body.get("ground_cves")),
                                "formats": _formats(body.get("formats")),
                                "tlp": _tlp(body.get("tlp"), app) if body.get("tlp") else None}
                    except ValueError as e:
                        return self._send(400, {"error": str(e)})
                if self.path == "/v1/investigate":
                    from .investigate import Investigation

                    question = str(body.get("question") or "").strip()
                    if not question:
                        return self._send(400, {"error": "question and seeds required"})
                    seeds, refused = _policy_gate(body.get("seeds"), "seeds")
                    if refused:
                        return self._send(400, refused)
                    try:
                        inv = Investigation(
                            app, case=body.get("case") or re.sub(r"[^a-z0-9]+", "-", question.lower()).strip("-")[:60],
                            question=question, lens=lens or app.config.get("analysis.lens", "general"),
                            seeds=seeds, rounds=min(int(body.get("rounds", 3)), 5),
                            max_new_targets=min(int(body.get("max_new", 4)), 8),
                            limit_per_target=min(int(body.get("limit", 50)), 200),
                            max_docs=min(int(body.get("max_docs", 2000)), 5000),
                            seed_mode="api", allow_files=False, opts=body.get("opts") or {}, analyze_kwargs=opts)
                    except (KeyError, TypeError, ValueError) as e:
                        return self._send(400, {"error": str(e)})
                    res = inv.run()
                    return self._send(200 if res.report is not None else 422, res.to_dict())
                if self.path == "/v1/ask":
                    question, case = str(body.get("question") or "").strip(), body.get("case")
                    if not isinstance(case, str) or not case.strip() or not question:
                        return self._send(400, {"error": "'case' and 'question' are required"})
                    try:
                        k = int(body.get("k", 12))
                        verify = _opt_bool(body.get("verify"))
                    except (TypeError, ValueError) as e:
                        return self._send(400, {"error": f"invalid option: {e}"})
                    if not 1 <= k <= 100:
                        return self._send(400, {"error": "'k' must be between 1 and 100"})
                    if not app.store.count_documents(case):
                        return self._send(404, {"error": f"case '{case}' has no documents"})
                    ans = app.ask(case, question, k=k, verify=True if verify is None else verify,
                                  platform=body.get("platform"))
                    return self._send(200, ans.to_dict())
                if self.path in ("/v1/collect", "/v1/run"):
                    targets, refused = _policy_gate(body.get("targets"))
                    if refused:
                        return self._send(400, refused)
                    try:
                        limit = int(body.get("limit", 100))
                    except (TypeError, ValueError) as e:
                        return self._send(400, {"error": f"invalid limit: {e}"})
                    case = body.get("case") or re.sub(r"[^a-z0-9]+", "-", targets[0].lower()).strip("-")
                    counts = app.collect(targets, case, limit, body.get("opts") or {}, trust="api")
                    if self.path == "/v1/collect":
                        return self._send(200, {"case": case, "counts": counts})
                    body["case"] = case
                if self.path in ("/v1/analyze", "/v1/run"):
                    rid, report, paths = app.analyze(body["case"], lens=lens, query=body.get("query", ""), **opts)
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
    if not os.environ.get("VIGILEYE_API_TOKEN") and host not in ("127.0.0.1", "localhost", "::1"):
        print(f"warning: serving on {host} without VIGILEYE_API_TOKEN: anyone who can reach this port can "
              "collect, analyze and read every case. Set VIGILEYE_API_TOKEN or bind to 127.0.0.1.",
              file=sys.stderr)
    httpd = ThreadingHTTPServer((host, port), make_handler(app))
    print(f"VigilEye API listening on http://{host}:{port}  (auth: {'bearer token' if os.environ.get('VIGILEYE_API_TOKEN') else 'none'})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
