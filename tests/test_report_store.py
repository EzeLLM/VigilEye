import json
from datetime import datetime, timezone

from vigileye.models import Document, Entity, Finding, Report
from vigileye.report import load_report, render_html, render_json, render_markdown, write_report
from vigileye.store import CaseStore


def _docs():
    return [
        Document(id="t3_a", platform="reddit", source="r/osint", text="first", created_at="2026-01-01T00:00:00Z"),
        Document(id="t3_b", platform="reddit", source="r/osint", text="second", created_at="2026-01-03T00:00:00Z"),
        Document(id="99", platform="hn", source="front", text="third", created_at="2026-01-02T00:00:00Z",
                 metrics={"points": 5}),
    ]


def _report():
    return Report(
        case="Op Test", lens="threat", query="q <b>", bluf="Bottom line <script>alert(1)</script>",
        key_judgments=[Finding(title="KJ1", summary="Actor active", confidence="high", likelihood="likely",
                               evidence=["reddit:t3_a"])],
        body_markdown="## Detail\n\n**bold** and *it* `code` [ok](https://x.org) [bad](javascript:alert(1))\n\n"
                      "- one\n- two\n\n1. a\n2. b\n\n<script>evil()</script>",
        entities=[Entity(name="ACME", type="org", mentions=3), Entity(name="Bob", type="person", mentions=9)],
        indicators=[{"type": "url", "value": "http://evil.example.com/x", "count": 2},
                    {"type": "ipv4", "value": "10.0.0.1", "count": 1}, {"type": "hashtag", "value": "#a.b"}],
        assumptions=["Sources are authentic"], gaps=["No telemetry"],
        hypotheses=[{"hypothesis": "H1", "likelihood": "likely", "consistent": ["reddit:t3_a"]}, "H2 bare"],
        sources=[{"id": "reddit:t3_a", "platform": "reddit", "source": "r/osint", "grade": "B2",
                  "url": "https://reddit.com/x"}],
        stats={"documents": 3, "llm_usage": {"totals": {"input_tokens": 10, "output_tokens": 4}}},
        created_at=datetime(2026, 2, 3, 4, 5, 6, tzinfo=timezone.utc),
    )


def test_store_documents_dedupe_and_roundtrip(tmp_path):
    s = CaseStore(tmp_path / "sub" / "cases.db")
    assert s.add_documents("c1", _docs()) == 3
    assert s.add_documents("c1", _docs()) == 0
    assert s.count_documents("c1") == 3
    docs = s.get_documents("c1")
    assert [d.ref for d in docs] == ["reddit:t3_b", "hn:99", "reddit:t3_a"]
    assert docs[1].metrics == {"points": 5} and docs[1].created_at.tzinfo is not None
    assert len(s.get_documents("c1", platform="reddit")) == 2
    assert len(s.get_documents("c1", since=datetime(2026, 1, 2, tzinfo=timezone.utc))) == 2
    assert len(s.get_documents("c1", limit=1)) == 1
    assert s.export_jsonl("c1", tmp_path / "out.jsonl") == 3
    assert len((tmp_path / "out.jsonl").read_text().splitlines()) == 3
    s.cache_put("k", {"a": 1})
    assert s.cache_get("k") == {"a": 1} and s.cache_get("missing") is None
    s.close()


def test_store_reports_and_cases(tmp_path):
    s = CaseStore(tmp_path / "cases.db")
    s.ensure_case("Op Test", "desc")
    s.add_documents("Op Test", _docs())
    rid = s.save_report(_report())
    got = s.get_report(rid)
    assert got.to_dict() == _report().to_dict()
    assert s.list_reports("Op Test")[0]["bluf"].startswith("Bottom line")
    cases = s.list_cases()
    assert cases[0]["name"] == "Op Test" and cases[0]["documents"] == 3 and cases[0]["reports"] == 1
    assert cases[0]["description"] == "desc" and cases[0]["last_activity"]
    assert s.get_report(9999) is None
    s.delete_case("Op Test")
    assert s.list_cases() == [] and s.count_documents("Op Test") == 0
    s.close()


def test_markdown_render_and_defang():
    md = render_markdown(_report())
    assert "UNCLASSIFIED // OPEN SOURCE" in md
    assert "hxxp://evil[.]example[.]com/x" in md and "10[.]0[.]0[.]1" in md
    assert "evil.example.com" not in md and "#a.b" in md
    assert "1. **KJ1**" in md and "[reddit:t3_a]" in md
    assert "## Analysis of Competing Hypotheses" in md and "H2 bare" in md
    assert md.index("Bob") < md.index("ACME")
    assert "input_tokens=10" in md and "human review" in md


def test_json_and_html_render():
    r = _report()
    assert json.loads(render_json(r))["case"] == "Op Test"
    h = render_html(r)
    assert "<script>" not in h and "&lt;script&gt;" in h
    assert "prefers-color-scheme" in h and "@media print" in h
    assert "<strong>bold</strong>" in h and "<em>it</em>" in h and "<code>code</code>" in h
    assert '<a href="https://x.org"' in h and 'href="javascript' not in h
    assert "<ul>" in h and "<ol>" in h and "<h3>Detail</h3>" in h
    assert "hxxp://evil[.]example[.]com/x" in h


def test_write_and_load_report(tmp_path):
    r = _report()
    paths = write_report(r, tmp_path / "a" / "b")
    assert [p.suffix for p in paths] == [".md", ".json", ".html"] and all(p.exists() for p in paths)
    assert paths[0].name == "op-test-threat-20260203-040506.md"
    loaded = load_report(paths[1])
    assert loaded.to_dict() == r.to_dict()
    assert write_report(r, tmp_path, formats=("md",), basename="x")[0].name == "x.md"
