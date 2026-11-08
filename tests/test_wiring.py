"""Wiring of chain of custody, cited Q&A, the investigation loop and the target policy / SSRF guard into
the CLI, the HTTP API, the collectors, the router and the engine (offline: mock provider, fake resolvers)."""

import hashlib
import json
import sqlite3
import threading
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from http.server import ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

from vigileye.analysis import AnalysisEngine
from vigileye.analysis.enrich import describe_media
from vigileye.cli import App, main
from vigileye.collectors import get_collector
from vigileye.config import load_config
from vigileye.llm import LLMRouter
from vigileye.llm.router import LLMError
from vigileye.models import Document, Media
from vigileye.policy import PolicyError
from vigileye.report import render_html, render_markdown
from vigileye.server import make_handler

ROOT = Path(__file__).resolve().parents[1]
OFFLINE = str(ROOT / "config" / "offline.yaml")
T0 = datetime(2026, 11, 1, tzinfo=timezone.utc)
PUBLIC = lambda host, port: ["93.184.216.34"]  # noqa: E731 - every name "resolves" to a public address
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


def _corpus(tmp_path: Path) -> Path:
    rows = [{"id": f"p{i}", "platform": "file", "source": "demo", "author": f"user{i}",
             "created_at": (T0 + timedelta(hours=i)).isoformat(),
             "text": ["Attackers exploited CVE-2024-3400 against GlobalProtect gateways and dropped a backdoor.",
                      "Patch your Palo Alto firewalls now. T1190 initial access observed.",
                      "The farmers market runs as usual this weekend."][i % 3] + f" Post {i}."}
            for i in range(6)]
    p = tmp_path / "posts.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return p


def _tamper(db_path: Path, case: str) -> str:
    db = sqlite3.connect(db_path)
    fp, js = db.execute('SELECT fingerprint, json FROM documents WHERE "case"=? ORDER BY fingerprint LIMIT 1',
                        (case,)).fetchone()
    doc = json.loads(js)
    doc["text"] += " (quietly edited)"
    db.execute('UPDATE documents SET json=? WHERE "case"=? AND fingerprint=?', (json.dumps(doc), case, fp))
    db.commit()
    db.close()
    return fp


# ---------------------------------------------------------------------------------------------- CLI
def test_cli_custody_verify_tamper_manifest_and_log(tmp_path, capsys):
    store = tmp_path / "cases.db"
    base = ["--config", OFFLINE, "--store", str(store), "-q"]
    assert main(base + ["collect", f"file:{_corpus(tmp_path)}", "--case", "c"]) == 0
    assert main(base + ["analyze", "--case", "c", "--out", str(tmp_path / "out"), "--formats", "md,html"]) == 0
    md = next((tmp_path / "out").glob("*.md")).read_text(encoding="utf-8")
    assert "**Chain of custody:** chain head " in md and "6/6 analysed documents hash-stamped" in md
    assert "--case c --anchor" in md and "- **custody:**" not in md  # rendered once, not as a raw stats dump
    capsys.readouterr()

    assert main(base + ["evidence", "verify", "--case", "c"]) == 0
    res = json.loads(capsys.readouterr().out)
    assert res["ok"] and res["documents_checked"] == 6 and res["reports"]["checked"] == 1

    manifest = tmp_path / "m" / "c.json"
    assert main(base + ["evidence", "manifest", "--case", "c", "--out", str(manifest)]) == 0
    head = capsys.readouterr().out.split("chain_head=")[1].strip()
    digest, name = (manifest.parent / "c.json.sha256").read_text().split()
    assert name == "c.json" and digest == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert main(base + ["evidence", "log", "--case", "c"]) == 0
    log = capsys.readouterr().out.splitlines()
    assert [ln.split()[2] for ln in log] == ["collect", "analyze", "export", "manifest"]

    fp = _tamper(store, "c")
    assert main(base + ["evidence", "verify", "--case", "c", "--anchor", head]) == 1
    res = json.loads(capsys.readouterr().out)
    assert not res["ok"] and res["mismatched"] == [fp] and res["unlogged"] == [fp] and res["chain"]["ok"]
    assert main(base + ["evidence", "log", "--case", "nope"]) == 1


def test_cli_export_is_logged_and_unchanged_recollection_keeps_stamp(tmp_path, capsys):
    store, corpus = tmp_path / "cases.db", _corpus(tmp_path)
    base = ["--config", OFFLINE, "--store", str(store), "-q"]
    assert main(base + ["collect", f"file:{corpus}", "--case", "c"]) == 0
    app = App(load_config(OFFLINE), store_path=str(store), quiet=True)
    first = app.ledger.stamps("c")
    assert main(base + ["collect", f"file:{corpus}", "--case", "c"]) == 0
    assert app.ledger.stamps("c") == first  # re-collected unchanged items keep their first collected_at
    assert main(base + ["export", "--case", "c", "--out", str(tmp_path / "e.jsonl")]) == 0
    entry = app.ledger.entries("c")[-1]
    assert entry["action"] == "export" and entry["details"]["files"][0]["name"] == "e.jsonl"
    assert app.ledger.verify_case("c")["ok"]


def test_cli_ask_returns_verified_citations(tmp_path, capsys):
    base = ["--config", OFFLINE, "--store", str(tmp_path / "cases.db"), "-q"]
    assert main(base + ["collect", f"file:{_corpus(tmp_path)}", "--case", "c"]) == 0
    capsys.readouterr()
    assert main(base + ["ask", "--case", "c", "--json", "Was CVE-2024-3400 exploited against GlobalProtect?"]) == 0
    ans = json.loads(capsys.readouterr().out)
    assert ans["meta"]["status"] == "answered" and not ans["unanswerable"] and ans["citations"]
    assert set(ans["citations"]) <= set(ans["retrieved"]) and ans["meta"]["retrieval"]["mode"] == "hybrid"
    assert not any(f"user{i}" in json.dumps(ans) for i in range(6))  # authors never leave the store
    assert main(base + ["ask", "--case", "c", "--no-verify", "What", "about", "GlobalProtect?"]) == 0
    assert "Verified citations: [file:" in capsys.readouterr().out
    assert main(base + ["ask", "--case", "empty", "anything?"]) == 1


def test_cli_investigate_offline_and_amended_report_verifies(tmp_path, capsys):
    store = tmp_path / "cases.db"
    base = ["--config", OFFLINE, "--store", str(store), "-q"]
    assert main(base + ["investigate", "What is happening to GlobalProtect gateways?", "--case", "inv", "--yes",
                        "--seed", f"file:{_corpus(tmp_path)}", "--seed", "web:http://169.254.169.254/latest/",
                        "--rounds", "2", "--out", str(tmp_path / "out"), "--formats", "md,json"]) == 0
    out = capsys.readouterr().out
    assert "stopped: planner_done after 1 planner round(s)" in out
    md = next((tmp_path / "out").glob("*.md")).read_text(encoding="utf-8")
    assert "## Collection plan & provenance" in md and "cloud metadata address" in md
    assert [p.suffix for p in (tmp_path / "out" / "interim" / "round-0").iterdir()] == [".json"]  # not overwritten
    app = App(load_config(OFFLINE), store_path=str(store), quiet=True)
    actions = [e["action"] for e in app.ledger.entries("inv")]
    assert actions[-2:] == ["amend", "export"]  # the provenance rewrite is itself a custody event
    res = app.ledger.verify_case("inv")
    assert res["ok"] and res["reports"]["mismatched"] == [] and res["reports"]["checked"] == 2
    # a silent in-place edit (no custody entry) is caught
    rid = int(app.ledger.entries("inv")[-2]["details"]["report_id"])
    rep = app.store.get_report(rid)
    rep.bluf = "rewritten"
    assert app.store.update_report(rid, rep) and app.ledger.verify_case("inv")["reports"]["mismatched"] == [rid]
    assert main(base + ["investigate", "q?", "--case", "x", "--yes", "--seed", "file:/nope/missing.jsonl"]) == 1


def test_custody_can_be_switched_off(tmp_path):
    cfg = load_config(OFFLINE)
    cfg.set("evidence.custody", False)
    app = App(cfg, store_path=str(tmp_path / "s.db"), quiet=True)
    app.collect([f"file:{_corpus(tmp_path)}"], "c")
    assert app.ledger.entries("c") == [] and app.ledger.verify_case("c")["unstamped"]


# ---------------------------------------------------------------------------------------------- HTTP API
@pytest.fixture()
def api(tmp_path):
    cfg = load_config(OFFLINE)
    cfg.set("output.dir", str(tmp_path / "exp"))
    app = App(cfg, store_path=str(tmp_path / "s.db"), quiet=True)
    app.collect([f"file:{_corpus(tmp_path)}"], "srv")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
    threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"

    def call(path, body=None, raw=None):
        data = raw if raw is not None else (None if body is None else json.dumps(body).encode())
        req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    yield app, call
    httpd.shutdown()
    httpd.server_close()


@pytest.mark.parametrize("path,field", [("/v1/collect", "targets"), ("/v1/run", "targets"),
                                        ("/v1/investigate", "seeds")])
@pytest.mark.parametrize("spec,why", [("file:/etc/passwd", "file: targets are disabled"),
                                      ("web:http://127.0.0.1:8765/v1/cases", "loopback"),
                                      ("rss:http://169.254.169.254/latest/meta-data/", "cloud metadata"),
                                      ("web:file:///etc/passwd", "non-http(s) scheme"),
                                      ("/etc/passwd.json", "file: targets are disabled")])
def test_api_targets_go_through_the_policy(api, path, field, spec, why):
    app, call = api
    code, body = call(path, {field: ["hn:search:ok", spec], "case": "srv", "question": "q?"})
    assert code == 400 and why in body["error"] and body["rejected"][0]["spec"] == spec
    assert why in body["rejected"][0]["reason"]
    assert app.store.count_documents("srv") == 6  # nothing was collected


def test_api_rejects_lens_paths_bad_bodies_and_missing_targets(api, tmp_path):
    _, call = api
    lens = tmp_path / "secret.yaml"
    lens.write_text("name: x\nfocus: top secret\n", encoding="utf-8")
    for path in ("/v1/analyze", "/v1/run", "/v1/investigate"):
        code, body = call(path, {"case": "srv", "lens": str(lens), "targets": ["hn:search:x"], "seeds": ["hn:search:x"],
                                 "question": "q?"})
        assert code == 400 and "built-in lenses" in body["error"], path
    assert call("/v1/collect", {"targets": []})[0] == 400
    assert call("/v1/collect", {"targets": [{"spec": "x"}]})[0] == 400
    assert call("/v1/investigate", {"seeds": ["hn:search:x"]})[0] == 400
    assert call("/v1/collect", raw=b"{not json")[0] == 400
    assert call("/v1/collect", raw=b"[1, 2]")[0] == 400


def test_api_ask_and_evidence_routes(api):
    app, call = api
    code, ans = call("/v1/ask", {"case": "srv", "question": "Was CVE-2024-3400 exploited against GlobalProtect?"})
    assert code == 200 and ans["meta"]["status"] == "answered" and ans["citations"]
    assert call("/v1/ask", {"case": "srv"})[0] == 400
    assert call("/v1/ask", {"case": "srv", "question": "q", "k": 0})[0] == 400
    assert call("/v1/ask", {"case": "srv", "question": "q", "verify": "maybe"})[0] == 400
    assert call("/v1/ask", {"case": "nope", "question": "q"})[0] == 404

    code, rep = call("/v1/analyze", {"case": "srv", "lens": "cyber", "formats": ["json"]})
    assert code == 200 and rep["report"]["stats"]["custody"]["documents_stamped"] == 6
    code, manifest = call("/v1/cases/srv/evidence")
    assert code == 200 and manifest["summary"] == {"documents": 6, "verified": 6, "mismatch": 0, "unstamped": 0}
    head = manifest["chain"]["head"]
    code, res = call(f"/v1/cases/srv/evidence/verify?anchor={head}")
    assert code == 200 and res["ok"] and res["chain"]["anchor_found"]
    assert not call("/v1/cases/srv/evidence/verify?anchor=" + "0" * 64)[1]["ok"]
    assert call("/v1/cases/srv%20x/evidence")[1]["case"] == "srv x"  # case names are URL-decoded


# ---------------------------------------------------------------------------------------------- SSRF guard
def _web(handler, **cfg):
    col = get_collector("web", {"min_interval": 0, **cfg}, http=httpx.Client(transport=httpx.MockTransport(handler)))
    col.resolver = PUBLIC
    return col


def test_web_collector_refuses_private_hosts_and_redirect_hops():
    seen = []

    def handler(request):
        seen.append(str(request.url))
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        if request.url.host == "news.example":
            return httpx.Response(302, headers={"location": "http://169.254.169.254/latest/meta-data/"})
        return httpx.Response(200, text="<html><title>ok</title><p>hi</p></html>",
                              headers={"content-type": "text/html"})

    with pytest.raises(PolicyError, match="loopback"):
        list(_web(handler).collect("http://127.0.0.1:8080/admin"))
    assert seen == []  # refused before any request
    with pytest.raises(PolicyError, match="cloud metadata"):
        list(_web(handler).collect("https://news.example/story"))
    assert not any("169.254" in u for u in seen)  # the redirect hop was never fetched
    lab = _web(handler, allow_private=True)
    assert [d.title for d in lab.collect("http://10.0.0.5/")] == ["ok"]
    col = _web(handler)
    col.resolver = lambda host, port: ["10.0.0.7"]  # public-looking name, private answer (fail closed)
    with pytest.raises(PolicyError, match="resolves to a non-public address"):
        list(col.collect("https://intranet.example/"))


def test_collector_clients_get_the_request_hook_unless_opted_out():
    from vigileye.collectors.base import collector_config

    assert get_collector("rss", {}).http.event_hooks["request"]
    assert not get_collector("rss", {"allow_private": True}).http.event_hooks["request"]
    section = {"http": {"timeout": 5}, "web": {"allow_private": True}}
    assert collector_config(section, "web")["allow_private"] and "allow_private" not in collector_config(section, "rss")
    hook = get_collector("rss", {}).http.event_hooks["request"][0]
    with pytest.raises(PolicyError, match="private"):
        hook(httpx.Request("GET", "http://192.168.1.1/feed"))


def test_image_refs_refuse_private_urls_and_unrequested_local_files(tmp_path):
    ref = LLMRouter._image_ref
    with pytest.raises(LLMError, match="non-public image URL"):
        ref("http://169.254.169.254/latest/meta-data/x.png", inline=False)
    assert ref("https://cdn.example/a.png", inline=False, resolver=PUBLIC) == "https://cdn.example/a.png"
    assert ref("http://10.0.0.1/a.png", inline=False, allow_private=True) == "http://10.0.0.1/a.png"
    img, secret = tmp_path / "a.png", tmp_path / "id_rsa"
    img.write_bytes(PNG)
    secret.write_text("PRIVATE KEY")
    with pytest.raises(LLMError, match="unsupported image reference"):
        ref(str(img), inline=True)  # a path from collected content is never read
    assert ref(str(img), inline=True, allow_local=True).startswith("data:image/png;base64,")
    with pytest.raises(LLMError, match="not an image"):
        ref(str(secret), inline=True, allow_local=True)
    for bad in ("file:///etc/passwd", "gopher://x/", "/etc/hostname"):
        with pytest.raises(LLMError):
            ref(bad, inline=True)


def test_describe_media_reads_local_images_only_for_file_documents(tmp_path):
    img = tmp_path / "a.png"
    img.write_bytes(PNG)
    router = LLMRouter.from_config(load_config(OFFLINE))
    web_doc = Document(id="w", platform="reddit", source="r/x", text="x", media=[Media(url=str(img), kind="image")])
    file_doc = Document(id="f", platform="file", source="demo", text="x", media=[Media(url=str(img), kind="image")])
    assert describe_media([web_doc, file_doc], router) == 1
    assert file_doc.media[0].description and web_doc.media[0].description is None
    assert "unsupported image reference" in web_doc.meta["vision_errors"][0]
    assert not LLMRouter.from_config(load_config(OFFLINE)).allow_private_urls
    cfg = load_config(OFFLINE)
    cfg.set("collectors.web.allow_private", True)
    assert LLMRouter.from_config(cfg).allow_private_urls


# ---------------------------------------------------------------------------------------------- engine
def _selection_docs() -> list[Document]:
    docs = [Document(id=f"n{i}", platform="hn", source="hn", text=f"Weather and sports chatter number {i}. " * 40,
                     created_at=T0 + timedelta(hours=i), metrics={"score": 500 + i}) for i in range(20)]
    docs.append(Document(id="needle", platform="hn", source="hn", created_at=T0 - timedelta(days=30),
                         text="Attackers exploited CVE-2024-3400 against GlobalProtect gateways. " * 20,
                         metrics={"score": 0}))
    return docs


def test_engine_relevance_aware_selection_keeps_the_relevant_document():
    cfg = load_config(OFFLINE)
    engine = AnalysisEngine(LLMRouter.from_config(cfg), cfg)
    docs = _selection_docs()
    _, dropped = engine._select_and_chunk(docs, 1500, 2)
    assert "needle" in {d.id for d in dropped}  # engagement/recency alone drops it
    chunks, dropped = engine._select_and_chunk(docs, 1500, 2, query="Was CVE-2024-3400 exploited on GlobalProtect?")
    assert "needle" in {d.id for c in chunks for d in c} and dropped
    engine.settings["relevance_selection"] = False
    _, dropped = engine._select_and_chunk(docs, 1500, 2, query="Was CVE-2024-3400 exploited on GlobalProtect?")
    assert "needle" in {d.id for d in dropped}


def test_engine_gap_message_names_relevance_when_a_query_is_set():
    raw = load_config(OFFLINE).raw
    raw["analysis"] = {**raw["analysis"], "chunk_tokens": 1500, "max_chunks": 2, "signals": False, "verify": False}
    from vigileye.config import Config

    cfg = Config(raw=raw)
    report = AnalysisEngine(LLMRouter.from_config(cfg), cfg).run(_selection_docs(), query="CVE-2024-3400 GlobalProtect")
    assert any("relevant to the query" in g for g in report.gaps)
    assert "needle" not in json.dumps(report.meta.get("dropped_doc_ids", []))


def test_custody_line_renders_in_html_and_is_absent_without_custody():
    from vigileye.models import Report

    def rep(**stats):
        return Report(case="c", lens="general", query="q", bluf="b", key_judgments=[], body_markdown="", stats=stats)

    r = rep(custody={"chain_head": "ab" * 32, "chain_entries": 3, "chain_ok": True})
    assert "Chain of custody" in render_html(r) and "ab" * 32 in render_markdown(r)
    assert "Chain of custody" not in render_markdown(rep())
