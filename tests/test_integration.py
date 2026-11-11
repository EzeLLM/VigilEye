"""Integration of the evidence/quality layers into the engine, CLI, server and renderers (offline, mock provider)."""

import json
import threading
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from http.server import ThreadingHTTPServer
from pathlib import Path

import yaml

from vigileye.analysis import AnalysisEngine
from vigileye.cli import App, main
from vigileye.config import Config, load_config
from vigileye.export import TLP, validate_bundle
from vigileye.llm import LLMRouter
from vigileye.models import Document, Finding, Report
from vigileye.report import render_html, render_markdown
from vigileye.server import make_handler

ROOT = Path(__file__).resolve().parents[1]
OFFLINE = str(ROOT / "config" / "offline.yaml")
T0 = datetime(2026, 11, 1, tzinfo=timezone.utc)

KEV = {"catalogVersion": "2026.11.06", "dateReleased": "2026-11-06T19:44:45Z", "count": 1,
       "vulnerabilities": [{"cveID": "CVE-2024-3400", "vendorProject": "Palo Alto Networks", "product": "PAN-OS",
                            "vulnerabilityName": "PAN-OS Command Injection", "dateAdded": "2024-04-12",
                            "shortDescription": "Command injection.", "requiredAction": "Patch.",
                            "dueDate": "2024-04-19", "knownRansomwareCampaignUse": "Unknown", "notes": ""}]}


def _rows() -> list[dict]:
    """Organic posts, a 10-post copypasta flood, and a 5-account ring co-sharing the same links."""
    rows = []
    for i in range(12):
        rows.append({"id": f"o{i}", "platform": "reddit", "source": "r/netsec", "author": f"organic_{i}",
                     "created_at": (T0 + timedelta(hours=7 * i)).isoformat(),
                     "text": f"Topic {i}: " + ["CVE-2024-3400 is exploited against GlobalProtect (T1190).",
                                               "Saw beaconing to evil-update[.]xyz after initial access.",
                                               "Transit vote forum notes from the council meeting.",
                                               "Weekend weather looks fine for the market."][i % 4] + f" {i * 37}"})
    appeal = "Everyone must share this now before it is deleted, the transit measure will triple your taxes overnight"
    for i in range(10):
        rows.append({"id": f"c{i}", "platform": ["mastodon", "bluesky"][i % 2], "source": "feed",
                     "author": f"cp_{i}", "created_at": (T0 + timedelta(days=2, minutes=13 * i)).isoformat(),
                     "text": appeal.replace("Everyone", ["Everyone", "Everybody", "Folks"][i % 3]) + f" @u{i}"})
    for occ in range(4):
        for a in range(5):
            rows.append({"id": f"r{occ}{a}", "platform": "bluesky", "source": "feed", "author": f"ring_{a}",
                         "created_at": (T0 + timedelta(days=1, hours=6 * occ, seconds=3 * a)).isoformat(),
                         "text": f"Look {occ}{a} https://mirror.example/story/{occ}?utm_source=x"})
    return rows


def _corpus(tmp_path: Path) -> Path:
    p = tmp_path / "corpus.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in _rows()), encoding="utf-8")
    return p


def _docs() -> list[Document]:
    return [Document(id=r["id"], platform=r["platform"], source=r["source"], author=r["author"], text=r["text"],
                     created_at=datetime.fromisoformat(r["created_at"])) for r in _rows()]


def _offline_cfg(**analysis) -> Config:
    raw = yaml.safe_load(Path(OFFLINE).read_text())
    raw["analysis"] = {**raw.get("analysis", {}), **analysis}
    return Config(raw=raw)


# ---------------------------------------------------------------------------------------------- CLI end to end
def test_cli_run_writes_all_formats_and_new_sections(tmp_path, capsys):
    base = ["--config", OFFLINE, "--store", str(tmp_path / "s.db"), "-q"]
    out = tmp_path / "out"
    assert main(base + ["run", f"file:{_corpus(tmp_path)}", "--case", "it", "--lens", "influence", "--ach",
                        "--formats", "md,json,html,stix,navigator", "--tlp", "amber", "--out", str(out)]) == 0
    names = sorted(p.name.split(".", 1)[1] for p in out.iterdir())
    assert names == ["html", "json", "layer.json", "md", "stix.json"]
    md = next(out.glob("*.md")).read_text()
    for title in ("## Evidence Signals", "## Citation verification", "## Analytic review"):
        assert title in md
        assert md.index(title) < md.index("## Methodology")
    assert "sig:copypasta#1" in md and "sig:coordination#1" in md
    bundle = json.loads(next(out.glob("*.stix.json")).read_text())
    assert validate_bundle(bundle) == []
    assert any(o["type"] == "report" for o in bundle["objects"])
    assert any(o["id"] == TLP["amber"] for o in bundle["objects"])
    assert not any(h in json.dumps(bundle) for h in ("ring_", "organic_", "cp_"))
    layer = json.loads(next(out.glob("*.layer.json")).read_text())
    assert any(t["techniqueID"] == "T1190" for t in layer["techniques"])
    html = next(out.glob("*.html")).read_text()
    assert '<section class="extra">' in html and "| ID | Kind" not in html
    report = json.loads(next(f for f in out.glob("*.json") if f.name.count(".") == 1).read_text())
    assert report["stats"]["collapsed_docs"] >= 9 and "citation_verification" in report["stats"]
    assert report["meta"]["coordination_graph"]["nodes"]

    capsys.readouterr()
    assert main(base + ["show", "1", "--format", "stix", "--tlp", "green"]) == 0
    assert any(o["id"] == TLP["green"] for o in json.loads(capsys.readouterr().out)["objects"])
    assert main(base + ["show", "1", "--format", "navigator"]) == 0
    assert json.loads(capsys.readouterr().out)["versions"]["layer"]
    assert main(base + ["signals", "--case", "it"]) == 0
    assert "sig:copypasta#1" in capsys.readouterr().out
    assert main(base + ["signals", "--case", "it", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["clusters"]


def test_cli_flags_toggle_layers_and_bad_format_fails_fast(tmp_path):
    store = tmp_path / "s.db"
    base = ["--config", OFFLINE, "--store", str(store), "-q"]
    corpus = _corpus(tmp_path)
    assert main(base + ["run", f"file:{corpus}", "--case", "bad", "--formats", "md,stx"]) == 1
    assert not App(load_config(OFFLINE), store_path=str(store), quiet=True).store.get_documents("bad")  # nothing collected

    assert main(base + ["run", f"file:{corpus}", "--case", "f", "--no-verify", "--critic", "--formats", "json",
                        "--out", str(tmp_path / "o")]) == 0
    rep = json.loads(next((tmp_path / "o").glob("*.json")).read_text())
    assert "citation_verification" not in rep["stats"]
    assert rep["meta"]["analytic_review"]["llm"] is True
    assert rep["meta"]["settings"]["verify"] is False and rep["meta"]["settings"]["critic_llm"] is True


def test_cli_ground_cves_reads_offline_cache(tmp_path):
    cache = tmp_path / "refdata"
    cache.mkdir()
    (cache / "kev.json").write_text(json.dumps(KEV))
    raw = yaml.safe_load(Path(OFFLINE).read_text())
    raw["intel"] = {"offline": True, "cache_dir": str(cache)}
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump(raw))
    base = ["--config", str(cfg), "--store", str(tmp_path / "s.db"), "-q"]
    assert main(base + ["run", f"file:{_corpus(tmp_path)}", "--case", "g", "--lens", "cyber", "--ground-cves",
                        "--formats", "md,json", "--out", str(tmp_path / "o")]) == 0
    assert "## Reference facts" in next((tmp_path / "o").glob("*.md")).read_text()
    rep = json.loads(next((tmp_path / "o").glob("*.json")).read_text())
    assert rep["meta"]["reference_grounding"]["in_kev"] == 1
    assert any(s["id"] == "ref:cisa-kev" for s in rep["sources"])


# ---------------------------------------------------------------------------------------------- engine settings
def test_engine_layers_can_be_disabled():
    cfg = _offline_cfg(signals=False, verify=False, critique=False)
    report = AnalysisEngine(LLMRouter.from_config(cfg), cfg).run(_docs(), lens="influence", case="off")
    assert report.meta["computed_signals"] == [] and report.stats["collapsed_docs"] == 0
    assert "citation_verification" not in report.stats and "analytic_review" not in report.meta
    assert not report.meta.get("extra_sections")
    assert report.meta["temporal_profile"]["total"] == len(_docs())


def test_engine_defaults_run_every_layer_without_errors():
    cfg = _offline_cfg()
    report = AnalysisEngine(LLMRouter.from_config(cfg), cfg).run(_docs(), lens="influence", ach=True, case="on")
    ids = [s["id"] for s in report.meta["extra_sections"]]
    assert ids == ["evidence_signals", "citation_verification", "analytic_review"]
    assert report.meta["errors"] == [] and report.meta["degraded"] is False
    st = report.stats
    assert st["analyzed_documents"] + st["dropped_docs"] + st["collapsed_docs"] == st["unique_documents"]
    assert "volume_profile" in st and st["signals"]["coordination"] >= 1
    assert any(e.startswith("sig:") for e in report.meta["ach"]["evidence_items"].values())


# ---------------------------------------------------------------------------------------------- renderers (H0)
def test_extra_sections_render_before_methodology_with_tables():
    r = Report(case="c", lens="general", query="q", bluf="b", key_judgments=[Finding(title="t", summary="s")],
               body_markdown="",
               meta={"extra_sections": [{"id": "x", "title": "Extra <One>", "markdown": "| A | B |\n|---|---|\n| 1 \\| 2 | <i> |"},
                                        {"id": "empty", "title": "Skipped", "markdown": "  "}]})
    md = render_markdown(r)
    assert md.index("## Extra <One>") < md.index("## Methodology") and "## Skipped" not in md
    html = render_html(r)
    assert "<h2>Extra &lt;One&gt;</h2>" in html and "<td>1 | 2</td><td>&lt;i&gt;</td>" in html


# ---------------------------------------------------------------------------------------------- HTTP API
def test_server_export_routes_and_option_validation(tmp_path):
    cfg = load_config(OFFLINE)
    cfg.set("output.dir", str(tmp_path / "exp"))
    app = App(cfg, store_path=str(tmp_path / "s.db"), quiet=True)
    app.collect([f"file:{_corpus(tmp_path)}"], "srv")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"

    def call(path, body=None):
        req = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    try:
        code, body = call("/v1/analyze", {"case": "srv", "formats": ["md", "stix"], "tlp": "green", "verify": False})
        assert code == 200 and sorted(Path(f).name.split(".", 1)[1] for f in body["files"]) == ["md", "stix.json"]
        assert "citation_verification" not in body["report"]["stats"]
        rid = body["report_id"]
        code, bundle = call(f"/v1/reports/{rid}/stix?tlp=amber")
        assert code == 200 and any(o["id"] == TLP["amber"] for o in bundle["objects"])
        assert call(f"/v1/reports/{rid}/stix?tlp=purple")[0] == 400
        assert call(f"/v1/reports/{rid}/navigator")[0] == 200
        assert call("/v1/reports/9999/stix")[0] == 404
        assert call("/v1/analyze", {"case": "srv", "formats": "md,bogus"})[0] == 400
        assert call("/v1/analyze", {"case": "srv", "verify": "maybe"})[0] == 400
    finally:
        httpd.shutdown()
        httpd.server_close()
