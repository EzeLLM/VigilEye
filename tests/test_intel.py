"""Offline tests for vigileye.intel: KEV/EPSS grounding and warninglists (injected fetchers, tmp_path, no network)."""
from __future__ import annotations

import copy
import json
import os
import time
from datetime import datetime, timedelta, timezone

import pytest

from vigileye.analysis import extract_indicators, summarize_indicators
from vigileye.intel import (
    BUILTIN_LISTS,
    EPSS_URL,
    EXPLOIT_RE,
    KEV_URL,
    RefCache,
    Warninglist,
    annotate_indicators,
    apply_grounding,
    epss_scores,
    exploitation_claim_check,
    ground_cves,
    load_kev,
    load_misp_warninglists,
    lookup_cve,
    match,
    reference_facts_block,
    refresh,
)
from vigileye.models import Document, Finding, Report

# Mirrors the live feed's schema (fields as of catalog 2026.11.06); values are test data.
KEV_FIXTURE = {
    "title": "CISA Catalog of Known Exploited Vulnerabilities",
    "catalogVersion": "2026.11.06",
    "dateReleased": "2026-11-06T19:44:45.1797Z",
    "count": 2,
    "vulnerabilities": [
        {"cveID": "CVE-2024-3400", "vendorProject": "Palo Alto Networks", "product": "PAN-OS",
         "vulnerabilityName": "Palo Alto Networks PAN-OS Command Injection Vulnerability", "dateAdded": "2024-04-12",
         "shortDescription": "Command injection in GlobalProtect.", "requiredAction": "Apply mitigations.",
         "dueDate": "2024-04-19", "knownRansomwareCampaignUse": "Unknown", "notes": "", "cwes": ["CWE-77"]},
        {"cveID": "CVE-2021-44228", "vendorProject": "Apache", "product": "Log4j2",
         "vulnerabilityName": "Apache Log4j2 Remote Code Execution Vulnerability", "dateAdded": "2021-12-10",
         "shortDescription": "JNDI lookup RCE.", "requiredAction": "Apply updates.", "dueDate": "2021-12-24",
         "knownRansomwareCampaignUse": "Known", "notes": "", "cwes": ["CWE-20", "CWE-400", "CWE-502"]},
    ],
}
EPSS_ROWS = {
    "CVE-2024-3400": {"cve": "CVE-2024-3400", "epss": "0.943510000", "percentile": "0.999570000", "date": "2026-11-07"},
    "CVE-2021-44228": {"cve": "CVE-2021-44228", "epss": "0.999990000", "percentile": "1.000000000", "date": "2026-11-07"},
    "CVE-2023-99999": {"cve": "CVE-2023-99999", "epss": "0.002000000", "percentile": "0.361130000", "date": "2026-11-07"},
}


class StubFetch:
    """Stands in for the network: KEV bulk feed and the EPSS API, recording every call."""

    def __init__(self, fail: bool = False):
        self.calls: list[tuple[str, dict | None]] = []
        self.fail = fail

    def __call__(self, url: str, params: dict | None):
        self.calls.append((url, copy.deepcopy(params)))
        if self.fail:
            raise ConnectionError("network down")
        if url == KEV_URL:
            return copy.deepcopy(KEV_FIXTURE)
        if url == EPSS_URL:
            ids = (params or {}).get("cve", "").split(",")
            data = [EPSS_ROWS[c] for c in ids if c in EPSS_ROWS]
            return {"status": "OK", "status-code": 200, "version": "1.0", "total": len(data), "data": data}
        raise AssertionError(f"unexpected url {url}")

    def urls(self) -> list[str]:
        return [u for u, _ in self.calls]


def _cache(tmp_path, fetch=None, **kw) -> RefCache:
    return RefCache(tmp_path / "refdata", fetch=fetch or StubFetch(), **kw)


def _cve(value: str, count: int = 1) -> dict:
    return {"type": "cve", "value": value, "defanged": value, "count": count, "docs": ["reddit:t3_000"]}


@pytest.fixture(autouse=True)
def _online_env(monkeypatch):
    monkeypatch.delenv("VIGILEYE_OFFLINE", raising=False)


# --------------------------------------------------------------------------- (1) RefCache
def test_refcache_fetches_once_then_serves_cache(tmp_path):
    fetch = StubFetch()
    cache = _cache(tmp_path, fetch)
    first = cache.get_json("kev", KEV_URL)
    second = cache.get_json("kev", KEV_URL)
    assert first == second == KEV_FIXTURE
    assert fetch.urls() == [KEV_URL]
    assert cache.path("kev") == tmp_path / "refdata" / "kev.json" and cache.path("kev").is_file()
    assert not [p for p in (tmp_path / "refdata").iterdir() if p.suffix == ".tmp"]  # atomic write cleaned up
    # A second cache instance (new process) reads the file without fetching.
    assert _cache(tmp_path, fetch).get_json("kev", KEV_URL)["catalogVersion"] == "2026.11.06"
    assert len(fetch.calls) == 1


def test_refcache_ttl_expiry_refetches(tmp_path):
    fetch = StubFetch()
    cache = _cache(tmp_path, fetch, ttl_hours=1)
    cache.get_json("kev", KEV_URL)
    old = time.time() - 2 * 3600
    os.utime(cache.path("kev"), (old, old))
    cache.get_json("kev", KEV_URL)
    assert len(fetch.calls) == 2


def test_refcache_offline_empty_returns_none_without_fetch(tmp_path, monkeypatch):
    fetch = StubFetch()
    assert _cache(tmp_path, fetch, offline=True).get_json("kev", KEV_URL) is None
    monkeypatch.setenv("VIGILEYE_OFFLINE", "1")
    assert _cache(tmp_path, fetch).get_json("kev", KEV_URL) is None
    assert fetch.calls == []


def test_refcache_offline_serves_stale_cache(tmp_path):
    _cache(tmp_path).get_json("kev", KEV_URL)
    cache = _cache(tmp_path, StubFetch(fail=True), offline=True, ttl_hours=0)
    assert cache.get_json("kev", KEV_URL)["count"] == 2 and cache.errors == []


def test_refcache_fetch_error_returns_stale_cache(tmp_path):
    _cache(tmp_path).get_json("kev", KEV_URL)
    broken = StubFetch(fail=True)
    cache = _cache(tmp_path, broken, ttl_hours=0)
    assert cache.get_json("kev", KEV_URL)["catalogVersion"] == "2026.11.06"
    assert broken.calls and "stale cache" in cache.errors[0]
    assert _cache(tmp_path / "empty", StubFetch(fail=True)).get_json("kev", KEV_URL) is None


def test_refcache_rejects_malformed_payload(tmp_path):
    cache = RefCache(tmp_path, fetch=lambda url, params: {"oops": True})
    assert load_kev(cache) == {}
    assert not cache.path("kev").exists() and "malformed" in cache.errors[0]


def test_refcache_from_config_variants(tmp_path):
    for cfg in ({"cache_dir": str(tmp_path), "offline": True}, {"intel": {"cache_dir": str(tmp_path), "offline": True}}):
        cache = RefCache.from_config(cfg)
        assert cache.root == tmp_path and cache.offline


# --------------------------------------------------------------------------- (2) KEV
def test_load_kev_parses_live_schema(tmp_path):
    kev = load_kev(_cache(tmp_path))
    assert kev["catalogVersion"] == "2026.11.06" and kev["count"] == 2
    assert set(kev["by_cve"]) == {"CVE-2024-3400", "CVE-2021-44228"}
    entry = kev["by_cve"]["CVE-2024-3400"]
    assert entry["dateAdded"] == "2024-04-12" and entry["knownRansomwareCampaignUse"] == "Unknown"
    assert load_kev(_cache(tmp_path / "none", offline=True)) == {}


def test_refresh_forces_download(tmp_path):
    fetch = StubFetch()
    cache = _cache(tmp_path, fetch)
    load_kev(cache)
    res = refresh(cache)
    assert res["catalogVersion"] == "2026.11.06" and res["count"] == 2 and res["refreshed"]
    assert fetch.urls() == [KEV_URL, KEV_URL]
    offline = refresh(_cache(tmp_path, StubFetch(fail=True), offline=True))
    assert offline["count"] == 2 and not offline["refreshed"]


# --------------------------------------------------------------------------- (3) EPSS
def test_epss_scores_parse_string_numbers_and_batch(tmp_path):
    fetch = StubFetch()
    cache = _cache(tmp_path, fetch)
    scores = epss_scores(["cve-2024-3400", "CVE-2021-44228", "CVE-2023-99999", "not-a-cve; DROP"], cache, batch=2)
    assert scores["CVE-2024-3400"] == {"epss": pytest.approx(0.94351), "percentile": pytest.approx(0.99957),
                                       "date": "2026-11-07"}
    assert isinstance(scores["CVE-2023-99999"]["epss"], float)
    sent = [p["cve"] for u, p in fetch.calls if u == EPSS_URL]
    assert sent == ["CVE-2021-44228,CVE-2023-99999", "CVE-2024-3400"]  # sorted, batched, only valid CVE ids
    assert epss_scores(["CVE-2024-3400"], cache)["CVE-2024-3400"]["date"] == "2026-11-07"
    assert len(fetch.calls) == 2  # that exact single-id batch was already cached


# --------------------------------------------------------------------------- (4) ground_cves
def test_ground_cves_annotates_records_and_facts(tmp_path):
    inds = [_cve("CVE-2024-3400", 3), _cve("CVE-2023-99999"), {"type": "domain", "value": "evil-update.xyz"}]
    g = ground_cves(inds, cache=_cache(tmp_path))
    listed, unlisted = inds[0], inds[1]
    assert listed["kev"]["dateAdded"] == "2024-04-12" and listed["kev"]["vendorProject"] == "Palo Alto Networks"
    assert set(listed["kev"]) == {"dateAdded", "dueDate", "knownRansomwareCampaignUse", "vendorProject", "product",
                                  "vulnerabilityName", "cwes"}
    assert listed["epss"]["epss"] == pytest.approx(0.94351)
    assert "kev" in unlisted and unlisted["kev"] is None and unlisted["epss"]["epss"] == pytest.approx(0.002)
    assert "kev" not in inds[2]
    assert g["kev_version"] == "2026.11.06" and g["checked"] == 2 and g["in_kev"] == 1 and g["errors"] == []
    assert g["facts"][0] == ("CVE-2024-3400: in CISA KEV since 2024-04-12 (Palo Alto Networks PAN-OS; "
                             "ransomware use: Unknown); EPSS 0.94 (pctl 0.99)")
    assert any("CVE-2023-99999: NOT in CISA KEV (catalog 2026.11.06); EPSS 0.002" in f for f in g["facts"])
    assert [s["id"] for s in g["sources"]] == ["ref:cisa-kev", "ref:first-epss"]
    assert g["sources"][0]["grade"] == "B1" and g["sources"][1]["grade"] == "B2"
    block = reference_facts_block(g)
    assert block.startswith("AUTHORITATIVE REFERENCE FACTS (machine-retrieved; do not contradict; "
                            "cite [ref:cisa-kev] / [ref:first-epss]):\n- CVE-2024-3400")
    assert "not proof" in block and reference_facts_block(g, limit=1).count("\n- ") == 2


def test_ground_cves_no_cves_never_fetches(tmp_path):
    fetch = StubFetch()
    g = ground_cves([{"type": "ipv4", "value": "185.220.101.45"}], cache=_cache(tmp_path, fetch))
    assert fetch.calls == [] and g["checked"] == 0 and g["facts"] == [] and reference_facts_block(g) == ""


def test_ground_cves_offline_without_cache_degrades(tmp_path):
    inds = [_cve("CVE-2024-3400")]
    g = ground_cves(inds, config={"intel": {"cache_dir": str(tmp_path), "offline": True}})
    assert "kev" not in inds[0] and "epss" not in inds[0]
    assert g["checked"] == 0 and g["in_kev"] == 0 and g["sources"] == [] and g["errors"]
    assert g["facts"] == ["CVE-2024-3400: CISA KEV status unknown (catalog unavailable)"]
    assert reference_facts_block(g).startswith("AUTHORITATIVE REFERENCE FACTS (machine-retrieved; do not contradict):")


def test_ground_cves_offline_with_seeded_kev_only_cites_kev(tmp_path):
    (tmp_path / "kev.json").write_text(json.dumps(KEV_FIXTURE))
    inds = [_cve("CVE-2021-44228"), _cve("CVE-2024-3400")]
    g = ground_cves(inds, config={"cache_dir": str(tmp_path), "offline": True})
    assert g["in_kev"] == 2 and g["epss_scored"] == 0 and [s["id"] for s in g["sources"]] == ["ref:cisa-kev"]
    assert "epss" not in inds[0] and inds[0]["kev"]["knownRansomwareCampaignUse"] == "Known"
    assert "cite [ref:cisa-kev]):" in reference_facts_block(g) and "first-epss" not in reference_facts_block(g)


def test_lookup_cve(tmp_path):
    res = lookup_cve("cve-2021-44228", _cache(tmp_path))
    assert res["in_kev"] and res["kev"]["product"] == "Log4j2" and res["epss"]["percentile"] == 1.0
    with pytest.raises(ValueError):
        lookup_cve("CVE-xyz", _cache(tmp_path))


# --------------------------------------------------------------------------- (5) claim check
def _report(*kjs: Finding, bluf: str = "Assessment.") -> Report:
    return Report(case="t", lens="cyber", query="q", bluf=bluf, key_judgments=list(kjs), body_markdown="",
                  sources=[{"id": "reddit:t3_000", "grade": "F6", "cited": 1}])


def test_exploit_regex():
    for s in ("is actively exploited", "exploitation observed", "used in the wild", "a 0-day", "zero-day",
              "attackers are exploiting it"):
        assert EXPLOIT_RE.search(s), s
    assert not EXPLOIT_RE.search("the flaw is unexploited and patched")
    from vigileye.intel.refdata import _negated

    assert _negated("There is no evidence CVE-2024-3400 has been exploited.")
    assert _negated("CVE-2024-3400 hasn't been exploited in the wild.")
    assert not _negated("No patch exists; CVE-2024-3400 is actively exploited.")
    assert not _negated("Although not patched, CVE-2024-3400 is exploited.")


def test_exploitation_claim_check_corroborates_and_flags(tmp_path):
    g = ground_cves([_cve("CVE-2024-3400"), _cve("CVE-2023-99999")], cache=_cache(tmp_path))
    kj1 = Finding(title="PAN-OS flaw under attack", summary="CVE-2024-3400 is actively exploited [reddit:t3_000].",
                  confidence="moderate", evidence=["reddit:t3_000"])
    kj2 = Finding(title="Obscure bug", summary="Posters claim CVE-2023-99999 is exploited in the wild.",
                  confidence="low", evidence=["reddit:t3_001"])
    kj3 = Finding(title="Unrelated", summary="Moderators changed the rules.", evidence=["reddit:t3_002"])
    report = _report(kj1, kj2, kj3)
    checks = exploitation_claim_check(report, g)
    by = {(c["target"], c["cve"]): c["status"] for c in checks}
    assert by == {("KJ1", "CVE-2024-3400"): "corroborated", ("KJ2", "CVE-2023-99999"): "unconfirmed"}
    assert kj1.evidence == ["reddit:t3_000", "ref:cisa-kev"] and "ref:cisa-kev" not in kj2.evidence
    gaps = [g_ for g_ in report.gaps if "CVE-2023-99999" in g_]
    assert len(gaps) == 1 and "not corroborated" in gaps[0] and "0.002" in gaps[0]
    assert report.meta["reference_checks"] == checks
    exploitation_claim_check(report, g)  # idempotent
    assert kj1.evidence.count("ref:cisa-kev") == 1 and len(report.gaps) == 1


def test_exploitation_claim_check_negation_and_unknown_cve(tmp_path):
    g = ground_cves([_cve("CVE-2024-3400")], cache=_cache(tmp_path))
    kj = Finding(title="Low risk", summary="There is no evidence CVE-2024-3400 has been exploited.")
    report = _report(kj, bluf="Actors are exploiting CVE-2020-1472 according to one post.")
    checks = exploitation_claim_check(report, g)
    assert {(c["target"], c["status"]) for c in checks} == {("KJ1", "contradicted"), ("BLUF", "unchecked")}
    assert "ref:cisa-kev" not in kj.evidence
    assert any("listed it as known exploited since 2024-04-12" in x for x in report.gaps)
    assert any("CVE-2020-1472" in x and "none of the collected documents" in x for x in report.gaps)


def test_apply_grounding_sources_meta_and_section(tmp_path):
    g = ground_cves([_cve("CVE-2024-3400")], cache=_cache(tmp_path))
    kj = Finding(title="Exploited", summary="CVE-2024-3400 is a zero-day used against firewalls.")
    report = _report(kj)
    apply_grounding(report, g)
    apply_grounding(report, g)  # idempotent
    refs = [s for s in report.sources if s["id"].startswith("ref:")]
    assert [s["id"] for s in refs] == ["ref:cisa-kev", "ref:first-epss"]
    assert refs[0]["cited"] == 1 and refs[1]["cited"] == 0
    assert report.meta["reference_grounding"] == {"kev_version": "2026.11.06",
                                                  "kev_date_released": "2026-11-06T19:44:45.1797Z",
                                                  "checked": 1, "in_kev": 1, "epss_scored": 1, "errors": []}
    sections = [s for s in report.meta["extra_sections"] if s["id"] == "reference_facts"]
    assert len(sections) == 1 and sections[0]["title"] == "Reference facts"
    assert "KJ1, CVE-2024-3400: corroborated" in sections[0]["markdown"]
    assert Report.from_dict(json.loads(json.dumps(report.to_dict()))).meta["reference_grounding"]["in_kev"] == 1


# --------------------------------------------------------------------------- (6) warninglists
def test_builtin_warninglist_matches():
    assert match("10.1.2.3", "ipv4") == ["rfc1918-and-bogons"]
    assert match("fe80::1", "ipv6") == ["rfc1918-and-bogons"]
    assert match("8.8.8.8", "ipv4") == ["public-dns-resolvers"]
    assert match("2001:4860:4860:0:0:0:0:8888", "ipv6") == ["public-dns-resolvers"]
    assert match("sub.example.com", "domain") == ["reserved-test-domains"]
    assert match("hxxps://bit[.]ly/x", "url") == ["link-shorteners"]
    assert match("https://attack.mitre.org/techniques/T1190/", "url") == ["security-references"]
    assert match("old.reddit.com", "domain") == ["major-platforms-cdn"]
    assert match("evil-update.xyz", "domain") == []
    assert match("185.220.101.45", "ipv4") == []
    assert match("notexample.com", "domain") == []  # suffix match is on label boundaries
    assert match("http://192.168.1.1/admin", "url") == ["rfc1918-and-bogons"]
    assert {wl.name for wl in BUILTIN_LISTS} == {"rfc1918-and-bogons", "public-dns-resolvers", "reserved-test-domains",
                                                 "link-shorteners", "major-platforms-cdn", "security-references"}


def test_warninglist_types_and_matching_attributes():
    sub = Warninglist("tlds", [".zip"], type="substring", matching_attributes=["domain"])
    assert sub.matches("files.zip", "domain") and not sub.matches("files.zip", "ipv4")
    rx = Warninglist("rx", ["/^cdn[0-9]+\\.example\\.net$/i"], type="regex")
    assert rx.matches("CDN12.example.net", "domain") and not rx.matches("cdn.example.net", "domain")
    with pytest.raises(ValueError):
        Warninglist("bad", ["x"], type="fuzzy")
    assert Warninglist("c", ["1.2.3.4"], type="cidr", category="false-positive").category == "false_positive"


# --------------------------------------------------------------------------- (7) MISP loader
def test_load_misp_warninglists_layout(tmp_path):
    specs = {
        "s": {"name": "Strings", "type": "string", "list": ["1.2.3.4"], "matching_attributes": ["ip-dst", "ip-src"]},
        "sub": {"name": "Substrings", "type": "substring", "list": [".internal"], "category": "known"},
        "host": {"name": "Hosts", "type": "hostname", "list": ["corp.example"], "matching_attributes": ["hostname"]},
        "cidr": {"name": "Networks", "type": "cidr", "list": ["203.0.112.0/24", "2001:db9::/32"],
                 "category": "false-positive", "matching_attributes": ["ip-src", "ip-dst"]},
        "rx": {"name": "Regexes", "type": "regex", "list": ["^update[0-9]+\\.vendor\\.com$"]},
    }
    for slug, data in specs.items():
        (tmp_path / "lists" / slug).mkdir(parents=True)
        (tmp_path / "lists" / slug / "list.json").write_text(json.dumps({"version": 1, "description": "t", **data}))
    (tmp_path / "lists" / "broken").mkdir()
    (tmp_path / "lists" / "broken" / "list.json").write_text("{not json")
    (tmp_path / "lists" / "weird").mkdir()
    (tmp_path / "lists" / "weird" / "list.json").write_text(json.dumps({"name": "W", "type": "ssdeep", "list": ["a"]}))
    lists = load_misp_warninglists(tmp_path)
    assert sorted(wl.type for wl in lists) == sorted(["string", "substring", "hostname", "cidr", "regex"])
    assert {wl.name: wl.category for wl in lists}["Networks"] == "false_positive"
    assert match("1.2.3.4", "ipv4", lists) == ["Strings"]
    assert match("db.internal", "domain", lists) == ["Substrings"]
    assert match("hxxps://vpn.corp[.]example/login", "url", lists) == ["Hosts"]
    assert match("203.0.112.9", "ipv4", lists) == ["Networks"]
    assert match("2001:db9::5", "ipv6", lists) == ["Networks"]
    assert match("update7.vendor.com", "domain", lists) == ["Regexes"]
    # Flat folder of *.json works too; a missing directory yields nothing.
    flat = tmp_path / "flat"
    flat.mkdir()
    (flat / "dns.json").write_text(json.dumps(specs["s"]))
    assert [wl.name for wl in load_misp_warninglists(flat)] == ["Strings"]
    assert load_misp_warninglists(tmp_path / "missing") == []


# --------------------------------------------------------------------------- (8) annotate
def _docs() -> list[Document]:
    base = datetime(2026, 11, 1, 12, 0, tzinfo=timezone.utc)
    texts = [
        "Operators of the Lockbit affiliate posted that CVE-2024-3400 is being exploited against Palo Alto devices. "
        "C2 at 185.220.101[.]45 and hxxps://evil-update[.]xyz/payload.bin ; technique T1190 and T1059.001.",
        "Internal scan hit 10.1.2.3 and resolvers 8.8.8.8; writeup at hxxps://bit[.]ly/x and https://example.com/doc",
        "Moderators of r/netsec announced new rules about posting unverified breach claims; NATO and CISA were mentioned.",
    ]
    return [Document(id=f"t3_{i:03d}", platform="reddit", source="r/netsec", text=t, title=f"Post {i}",
                     author=f"user{i}", url=f"https://www.reddit.com/r/netsec/comments/{i}",
                     created_at=base + timedelta(hours=i)) for i, t in enumerate(texts)]


def test_annotate_indicators_marks_benign_not_drop():
    inds = extract_indicators(_docs())
    n_before = len(inds)
    hits = annotate_indicators(inds)
    assert hits >= 4 and len(inds) == n_before
    by = {r["value"]: r for r in inds}
    assert not by["185.220.101.45"].get("benign") and "warninglists" not in by["185.220.101.45"]
    assert not by["evil-update.xyz"].get("benign")
    assert by["10.1.2.3"]["benign"] and by["10.1.2.3"]["warninglists"] == ["rfc1918-and-bogons"]
    assert by["8.8.8.8"]["benign"] and by["bit.ly"]["warninglists"] == ["link-shorteners"]
    assert by["https://bit.ly/x"]["benign"] and by["example.com"]["benign"]
    top = summarize_indicators(inds)["top"]
    assert [r["value"] for r in top["ipv4"]] == ["185[.]220[.]101[.]45"]
    assert annotate_indicators(inds) == hits and by["10.1.2.3"]["warninglists"] == ["rfc1918-and-bogons"]
    known_only = extract_indicators(_docs())
    annotate_indicators(known_only, benign_categories=["false_positive"])
    assert {r["value"]: r for r in known_only}["bit.ly"].get("benign") is None


# --------------------------------------------------------------------------- (9) engine hook
def _engine_hooked() -> bool:
    from vigileye.analysis import engine

    return "ground_cves" in engine.DEFAULT_SETTINGS


@pytest.mark.skipif(not _engine_hooked(), reason="engine integration hook not applied yet")
def test_engine_grounding_offline(tmp_path):
    from vigileye.analysis import AnalysisEngine
    from vigileye.config import Config
    from vigileye.llm import LLMRouter

    mock = {
        "providers": {"mock": {"base_url": "mock://"}},
        "roles": {r: [{"provider": "mock", "model": f"mock-{r}", "context_window": 32000}]
                  for r in ("analyst", "extractor", "vision", "synthesizer")},
        "llm": {"cache": False},
        "analysis": {"chunk_tokens": 400, "max_chunks": 24, "max_concurrency": 1, "ground_cves": True},
        "intel": {"offline": True, "cache_dir": str(tmp_path)},
    }
    (tmp_path / "kev.json").write_text(json.dumps(KEV_FIXTURE))  # pre-seeded cache; offline: no fetch possible

    class Recorder:
        def __init__(self, inner):
            self.inner, self.prompts = inner, []

        def chat_json(self, messages, role="extractor"):
            self.prompts.append("\n".join(m["content"] for m in messages if isinstance(m.get("content"), str)))
            return self.inner.chat_json(messages, role=role)

        def __getattr__(self, name):
            return getattr(self.inner, name)

    cfg = Config(raw=mock)
    router = Recorder(LLMRouter.from_config(cfg))
    report = AnalysisEngine(router, cfg).run(_docs(), lens="cyber", query="What is exploited?", case="intel")
    assert report.meta["reference_grounding"]["in_kev"] == 1
    assert any("AUTHORITATIVE REFERENCE FACTS" in p for p in router.prompts)
    cve = next(r for r in report.indicators if r["type"] == "cve")
    assert cve["kev"]["dateAdded"] == "2024-04-12"
    assert next(r for r in report.indicators if r["value"] == "10.1.2.3")["benign"]
    assert any(s["id"] == "ref:cisa-kev" for s in report.sources)
