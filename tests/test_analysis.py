"""Offline tests for the analysis engine (mock provider only, no network)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from vigileye.analysis import (
    AnalysisEngine,
    chunk_documents,
    default_source_grade,
    extract_indicators,
    get_lens,
    list_lenses,
    load_lens,
    merge_entities,
    normalize_confidence,
    normalize_wep,
    refang,
    wep_for_probability,
)
from vigileye.analysis.chunking import render_chunk
from vigileye.config import Config
from vigileye.llm import LLMRouter
from vigileye.models import Document, Entity, Finding, Media, Report
from vigileye.tokens import count_tokens

MOCK_CONFIG = {
    "providers": {"mock": {"base_url": "mock://"}},
    "roles": {
        "analyst": [{"provider": "mock", "model": "mock-analyst", "context_window": 32000}],
        "extractor": [{"provider": "mock", "model": "mock-extractor", "context_window": 16000}],
        "vision": [{"provider": "mock", "model": "mock-vision", "context_window": 8000}],
        "synthesizer": [{"provider": "mock", "model": "mock-synth", "context_window": 32000}],
    },
    "llm": {"cache": False},
    "analysis": {"chunk_tokens": 400, "max_chunks": 24, "max_concurrency": 3},
}


def _docs(n: int = 6) -> list[Document]:
    base = datetime(2026, 11, 1, 12, 0, tzinfo=timezone.utc)
    texts = [
        "Operators of the Lockbit affiliate posted that CVE-2024-3400 is being exploited against Palo Alto devices. "
        "C2 at 185.220.101[.]45 and hxxps://evil-update[.]xyz/payload.bin ; technique T1190 and T1059.001.",
        "Hash of the dropper: 44d88612fea8a8f36de82e1238573e8f and "
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b856 #ransomware #infosec",
        "Community members discuss $TSLA and $GME volatility after the Reuters report on supply chains.",
        "Donations requested at bc1qar0srrr7xfkvy5l643lydnw9re59gtzzwf5mdq and 0x52908400098527886E0F7030069857D2E4169EE7.",
        "Contact me at someone@example.com or +1 555 123 4567 for the leak (should NOT be extracted).",
        "Moderators of r/netsec announced new rules about posting unverified breach claims; NATO and CISA were mentioned.",
    ]
    docs = []
    for i in range(n):
        docs.append(Document(
            id=f"t3_{i:03d}", platform="reddit", source="r/netsec", text=texts[i % len(texts)],
            title=f"Post {i}", author=f"user{i}", url=f"https://www.reddit.com/r/netsec/comments/{i}",
            created_at=base + timedelta(hours=i), metrics={"score": 10 * i, "num_comments": i},
            media=[Media(url=f"https://i.redd.it/img{i}.png")] if i == 0 else [],
        ))
    return docs


@pytest.fixture()
def router() -> LLMRouter:
    return LLMRouter.from_config(Config(raw=MOCK_CONFIG))


# --------------------------------------------------------------------------- tradecraft

def test_wep_and_confidence():
    assert wep_for_probability(0.03) == "almost no chance"
    assert wep_for_probability(0.5) == "roughly even chance"
    assert wep_for_probability(0.7) == "likely"
    assert wep_for_probability(90) == "very likely"
    assert wep_for_probability(0.99) == "almost certain"
    assert normalize_wep("Highly likely") == "very likely"
    assert normalize_wep("we judge this probable") == "likely"
    assert normalize_wep("nonsense words") is None
    assert normalize_confidence("HIGH confidence") == "high"
    assert normalize_confidence("medium") == "moderate"
    assert normalize_confidence(None) == "moderate"
    assert normalize_confidence("weak") == "low"
    assert default_source_grade("reddit") == "F6"
    assert default_source_grade("rss")[0] in "ABC"


# --------------------------------------------------------------------------- indicators

def test_refang():
    assert refang("hxxps://bad[.]example[.]com/x") == "https://bad.example.com/x"
    assert refang("10[.]0[.]0[.]1") == "10.0.0.1"


def test_extract_indicators_including_defanged():
    docs = _docs()
    inds = extract_indicators(docs)
    by_type: dict[str, set[str]] = {}
    for r in inds:
        by_type.setdefault(r["type"], set()).add(r["value"])
        assert r["count"] >= 1 and r["docs"]
        assert "defanged" in r

    assert "185.220.101.45" in by_type["ipv4"]
    assert "https://evil-update.xyz/payload.bin" in by_type["url"]
    assert "evil-update.xyz" in by_type["domain"]
    assert "CVE-2024-3400" in by_type["cve"]
    assert {"T1190", "T1059.001"} <= by_type["attack_technique"]
    assert "44d88612fea8a8f36de82e1238573e8f" in by_type["md5"]
    assert any(len(v) == 64 for v in by_type["sha256"])
    assert "bc1qar0srrr7xfkvy5l643lydnw9re59gtzzwf5mdq" in by_type["btc"]
    assert "0x52908400098527886e0f7030069857d2e4169ee7" in by_type["eth"]
    assert {"#ransomware", "#infosec"} <= by_type["hashtag"]
    assert {"$TSLA", "$GME"} <= by_type["cashtag"]

    ip = next(r for r in inds if r["value"] == "185.220.101.45")
    assert ip["was_defanged"] is True
    assert ip["defanged"] == "185[.]220[.]101[.]45"
    assert ip["docs"] == ["reddit:t3_000"]

    # PII guardrail: no e-mails / phone numbers, and no email-derived domains.
    flat = " ".join(r["value"] for r in inds)
    assert "someone@example.com" not in flat
    assert "example.com" not in {r["value"] for r in inds if r["type"] == "domain"}
    assert "555" not in flat
    assert not any(r["type"] in ("email", "phone") for r in inds)


def test_extract_indicators_dedupe_counts():
    text = "1.2.3.4 then again 1.2.3[.]4 and 8.8.8.8; see CVE-2023-1234 and cve-2023-1234"
    inds = extract_indicators(text, doc_id="web:x")
    ip = next(r for r in inds if r["value"] == "1.2.3.4")
    assert ip["count"] == 2 and ip["docs"] == ["web:x"]
    cve = [r for r in inds if r["type"] == "cve"]
    assert len(cve) == 1 and cve[0]["count"] == 2
    assert not any(r["type"] == "domain" and r["value"].endswith(".txt") for r in extract_indicators("see notes.txt"))


# --------------------------------------------------------------------------- chunking

def test_chunking_respects_budget_and_order():
    docs = _docs(12)
    chunks = chunk_documents(docs, 120)
    assert len(chunks) > 1
    assert [d.id for c in chunks for d in c] == [d.id for d in docs]
    for c in chunks:
        if len(c) > 1:
            assert sum(count_tokens(d.render()) for d in c) <= 120


def test_chunking_truncates_oversized_doc():
    big = Document(id="big", platform="web", source="example", text="word " * 5000)
    chunks = chunk_documents([big, *_docs(2)], 200)
    assert chunks[0] == [big]
    assert count_tokens(render_chunk(chunks[0], 200)) <= 220


# --------------------------------------------------------------------------- lenses

def test_lens_registry():
    names = {lens.name for lens in list_lenses()}
    assert {"general", "threat", "influence", "narrative", "cyber", "market", "crisis", "community"} <= names
    for lens in list_lenses():
        assert lens.title and lens.focus and lens.key_questions
    assert get_lens("cyber").name == "cyber"
    assert get_lens(None).name == "general"
    with pytest.raises(KeyError):
        get_lens("does-not-exist")


def test_load_custom_lens(tmp_path):
    p = tmp_path / "elections.yaml"
    p.write_text(
        "name: elections\ntitle: Election Integrity\ndescription: d\nfocus: Look for voting misinformation.\n"
        "key_questions:\n  - Is misinformation spreading?\n",
        encoding="utf-8",
    )
    lens = load_lens(p)
    assert lens.name == "elections" and lens.key_questions == ["Is misinformation spreading?"]
    assert get_lens(str(p)).title == "Election Integrity"


def test_merge_entities():
    merged = merge_entities([
        Entity("CISA", "organization", 2),
        Entity("cisa", "unknown", 1, aliases=["Cybersecurity and Infrastructure Security Agency"]),
        Entity("Cybersecurity and Infrastructure Security Agency", "organization", 1),
        Entity("NATO", "organization", 1),
    ])
    assert len(merged) == 2
    cisa = merged[0]
    assert cisa.name == "CISA" and cisa.mentions == 4 and cisa.type == "organization"
    assert any("Cybersecurity" in a for a in cisa.aliases)


# --------------------------------------------------------------------------- end-to-end

def test_engine_end_to_end_mock(router):
    cfg = Config(raw=MOCK_CONFIG)
    docs = _docs(8)
    docs.append(docs[0])  # duplicate
    events: list[str] = []
    report = AnalysisEngine(router, cfg).run(
        docs, lens="cyber", query="What threats are discussed?", case="unit",
        use_vision=True, ach=True, progress=events.append,
    )
    assert isinstance(report, Report)
    assert report.case == "unit" and report.lens == "cyber"
    assert report.bluf
    assert report.body_markdown
    assert report.key_judgments and all(isinstance(f, Finding) for f in report.key_judgments)
    known = {f"reddit:{d.id}" for d in docs}
    for f in report.key_judgments:
        assert f.confidence in ("low", "moderate", "high")
        assert set(f.evidence) <= known
    assert report.hypotheses and all("likelihood" in h for h in report.hypotheses)
    assert report.indicators and any(i["type"] == "cve" for i in report.indicators)
    assert report.sources and {"id", "platform", "source", "url", "author", "created_at", "grade"} <= set(report.sources[0])
    assert all(s["grade"] == "F6" for s in report.sources)
    st = report.stats
    assert st["documents"] == 9 and st["unique_documents"] == 8 and st["duplicates_removed"] == 1
    assert st["chunks"] >= 1 and st["dropped_docs"] == 0
    assert st["platforms"] == {"reddit": 8}
    assert st["time_range"]["start"] and st["time_range"]["end"]
    assert "llm_usage" in st and st["images_described"] == 1
    assert isinstance(report.meta["errors"], list)
    assert events and events[-1].startswith("done")
    # Report must serialise.
    d = report.to_dict()
    assert Report.from_dict(d).bluf == report.bluf


def test_engine_max_chunks_drops_docs(router):
    cfg = Config(raw={**MOCK_CONFIG, "analysis": {"chunk_tokens": 600, "max_chunks": 2}})
    docs = _docs(10)
    for d in docs:
        d.text = (d.text + " ") * 8  # ~250+ tokens each -> far more than 2 chunks
    report = AnalysisEngine(router, cfg).run(docs, lens="general", extract_entities=False)
    assert report.stats["chunks"] <= 2
    assert report.stats["dropped_docs"] > 0
    assert report.stats["analyzed_documents"] + report.stats["dropped_docs"] == 10
    assert any("not analysed" in g for g in report.gaps)


def test_engine_degrades_when_llm_fails(router):
    class Broken:
        def __init__(self, inner):
            self.inner = inner

        def chat_json(self, *a, **k):
            raise RuntimeError("endpoint down")

        def describe_image(self, *a, **k):
            raise RuntimeError("no vision")

        def __getattr__(self, name):
            return getattr(self.inner, name)

    report = AnalysisEngine(Broken(router), Config(raw=MOCK_CONFIG)).run(_docs(4), use_vision=True)
    assert isinstance(report, Report)
    assert report.meta["errors"] and report.meta["degraded"]
    assert report.gaps
    assert report.bluf
    assert report.indicators  # regex enrichment still works offline


def test_engine_empty_input(router):
    report = AnalysisEngine(router, Config(raw=MOCK_CONFIG)).run([])
    assert report.stats["documents"] == 0 and report.gaps
