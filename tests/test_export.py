"""Offline tests for the STIX 2.1 and ATT&CK Navigator exporters (no network, mock provider only)."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from vigileye.config import Config
from vigileye.export import (
    STIX_NS,
    TLP,
    TLP_OBJECTS,
    canonical_json,
    deterministic_id,
    report_to_navigator_layer,
    report_to_stix,
    sco_for_indicator,
    stix_ts,
    validate_bundle,
    write_navigator,
    write_stix,
)
from vigileye.models import Document, Entity, Finding, Report

SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b856"
MOCK_CONFIG = {
    "providers": {"mock": {"base_url": "mock://"}},
    "roles": {r: [{"provider": "mock", "model": f"mock-{r}", "context_window": 32000}]
              for r in ("analyst", "extractor", "vision", "synthesizer")},
    "llm": {"cache": False},
    "analysis": {"chunk_tokens": 400, "max_chunks": 24, "max_concurrency": 3},
}


def _report(**over) -> Report:
    kw = dict(
        case="Op Test", lens="threat", query="q <b>", bluf="LockBit affiliates exploit CVE-2024-3400; user1 boasted.",
        key_judgments=[
            Finding(title="KJ1", summary="Actor active, per user1.", confidence="high", likelihood="likely",
                    evidence=["reddit:t3_a"], tags=["ransomware"]),
            Finding(title="KJ2", summary="Exploit scope unclear", confidence="low", likelihood=None,
                    evidence=["hn:99"]),
        ],
        body_markdown="## Detail\n\nThe affiliate (posting as u/user1) shared tooling.",
        entities=[Entity(name="ACME", type="org", mentions=3), Entity(name="Bob", type="person", mentions=9),
                  Entity(name="LockBit", type="threat_actor", mentions=5, aliases=["LockBit 3.0", "@lockbitsupp"]),
                  Entity(name="user1", type="organization", mentions=2),
                  Entity(name="r/netsec", type="community"), Entity(name="Kyiv", type="location")],
        indicators=[
            {"type": "cve", "value": "CVE-2024-3400", "count": 3, "docs": ["reddit:t3_a"],
             "kev": {"dateAdded": "2024-04-12", "vulnerabilityName": "PAN-OS Command Injection"},
             "epss": {"epss": 0.94, "percentile": 0.99, "date": "2026-11-01"}},
            {"type": "attack_technique", "value": "T1190", "count": 4, "docs": ["reddit:t3_a", "hn:99"]},
            {"type": "attack_technique", "value": "T1059.001", "count": 1, "docs": ["hn:99"]},
            {"type": "attack_tactic", "value": "TA0001", "count": 2, "docs": ["hn:99"]},
            {"type": "domain", "value": "evil-update.xyz", "count": 2, "docs": ["reddit:t3_a"]},
            {"type": "url", "value": "http://evil.example.com/x", "count": 2, "docs": ["reddit:t3_a"]},
            {"type": "ipv4", "value": "185.220.101.45", "count": 1, "docs": ["hn:99"]},
            {"type": "sha256", "value": SHA256, "count": 1, "docs": ["hn:99"]},
            {"type": "hashtag", "value": "#ransomware", "count": 5},
            {"type": "cashtag", "value": "$TSLA", "count": 1},
            {"type": "domain", "value": "github.com", "count": 7, "benign": True},
            {"type": "ipv4", "value": "10.0.0.1", "count": 1, "warninglists": ["RFC 1918 private ranges"]},
            {"type": "url", "value": "https://mastodon.social/@someone/1234", "count": 1},
        ],
        assumptions=["Sources are authentic"], gaps=["No telemetry"],
        hypotheses=[{"hypothesis": "H1 organic", "likelihood": "likely", "consistent": ["reddit:t3_a"]}, "H2 bare"],
        sources=[{"id": "reddit:t3_a", "platform": "reddit", "source": "r/osint", "grade": "B2",
                  "url": "https://reddit.com/x", "author": "user1"},
                 {"id": "hn:99", "platform": "hn", "source": "front", "grade": "C3",
                  "url": "https://bsky.app/profile/alice.example/post/1", "author": "alice.example"}],
        stats={"documents": 3},
        created_at=datetime(2026, 2, 3, 4, 5, 6, 789000, tzinfo=timezone.utc),
    )
    kw.update(over)
    return Report(**kw)


def _by_type(bundle: dict, t: str) -> list[dict]:
    return [o for o in bundle["objects"] if o["type"] == t]


# --------------------------------------------------------------------------- (1) ids
def test_deterministic_ids_and_timestamps():
    ns = uuid.UUID("00abedb4-aa42-466c-9c01-fed23315a9b7")
    expected = "domain-name--" + str(uuid.uuid5(ns, '{"value":"evil-update.xyz"}'))
    assert deterministic_id("domain-name", {"value": "evil-update.xyz"}) == expected
    assert STIX_NS == uuid.UUID("00abedb4-aa42-466c-9c01-fed23315a9b7")
    assert canonical_json({"b": 1, "a": "é"}) == '{"a":"é","b":1}'
    assert stix_ts(datetime(2026, 2, 3, 4, 5, 6, 789123, tzinfo=timezone.utc)) == "2026-02-03T04:05:06.789Z"
    assert stix_ts(datetime(2026, 2, 3, 6, 5, 6, tzinfo=timezone(timedelta(hours=2)))) == "2026-02-03T04:05:06.000Z"
    assert sco_for_indicator({"type": "domain", "value": "evil-update[.]xyz"})["id"] == expected
    assert sco_for_indicator({"type": "hashtag", "value": "#x"}) is None

    b1, b2 = report_to_stix(_report()), report_to_stix(_report())
    assert canonical_json(b1) == canonical_json(b2)  # fully idempotent, ids included
    ids = {o["id"] for o in b1["objects"]}
    assert expected in ids
    assert {o["id"] for o in _by_type(b1, "report")} == {o["id"] for o in _by_type(b2, "report")}
    assert [o["id"] for o in _by_type(b1, "note")] == [o["id"] for o in _by_type(b2, "note")]


# --------------------------------------------------------------------------- (2) content
def test_report_bundle_content_and_validity():
    bundle = report_to_stix(_report())
    assert validate_bundle(bundle) == []
    (rep,) = _by_type(bundle, "report")
    assert rep["report_types"] == ["threat-report"] and rep["labels"] == ["threat"]
    assert rep["published"] == rep["created"] == "2026-02-03T04:05:06.789Z"
    assert rep["id"] == deterministic_id("report", {"name": "op test: threat assessment",
                                                    "published": "2026-02-03T04:05:06.789Z"})
    refs = {r.get("external_id"): r for r in rep["external_references"]}
    assert refs["reddit:t3_a"] == {"source_name": "reddit", "url": "https://reddit.com/x",
                                   "external_id": "reddit:t3_a", "description": "Admiralty B2"}
    assert "url" not in refs["hn:99"]  # profile URL that names the author is dropped

    kjs = [n for n in _by_type(bundle, "note") if n.get("x_vigileye_section") == "key_judgment"]
    kj1 = next(n for n in kjs if n["abstract"] == "KJ1")
    assert kj1["confidence"] == 85 and kj1["likelihood"] == 70 and kj1["note_types"] == ["assessment"]
    assert kj1["x_vigileye_estimative_term"] == "likely" and kj1["x_vigileye_confidence_term"] == "high"
    assert kj1["x_vigileye_evidence"] == ["reddit:t3_a"] and "[reddit:t3_a]" in kj1["content"]
    kj2 = next(n for n in kjs if n["abstract"] == "KJ2")
    assert kj2["confidence"] == 15 and "likelihood" not in kj2

    (vuln,) = _by_type(bundle, "vulnerability")
    assert vuln["name"] == "CVE-2024-3400"
    assert vuln["external_references"][0]["source_name"] == "cve"
    assert vuln["external_references"][0]["external_id"] == "CVE-2024-3400"
    assert vuln["x_opencti_cisa_kev"] is True and vuln["x_opencti_epss_score"] == 0.94
    assert vuln["x_opencti_epss_percentile"] == 0.99
    aps = {a["x_mitre_id"]: a for a in _by_type(bundle, "attack-pattern")}
    assert set(aps) == {"T1190", "T1059.001"}
    assert aps["T1190"]["id"] == deterministic_id("attack-pattern", {"x_mitre_id": "t1190"})  # pycti recipe
    assert aps["T1190"]["external_references"][0] == {
        "source_name": "mitre-attack", "external_id": "T1190", "url": "https://attack.mitre.org/techniques/T1190/"}
    (f,) = _by_type(bundle, "file")
    assert f["hashes"] == {"SHA-256": SHA256}
    assert {o["value"] for o in _by_type(bundle, "domain-name")} == {"evil-update.xyz"}
    assert [o["value"] for o in _by_type(bundle, "ipv4-addr")] == ["185.220.101.45"]
    assert [o["value"] for o in _by_type(bundle, "url")] == ["http://evil.example.com/x"]
    (actor,) = _by_type(bundle, "intrusion-set")
    assert actor["name"] == "LockBit" and actor["aliases"] == ["LockBit 3.0"]  # '@handle' alias dropped
    groups = {i["name"]: i["identity_class"] for i in _by_type(bundle, "identity")}
    assert groups == {"VigilEye": "system", "ACME": "organization", "r/netsec": "group"}

    # KJ1 cites reddit:t3_a -> linked to the objects whose indicator docs include that ref.
    linked = set(kj1["object_refs"])
    assert rep["id"] in linked and vuln["id"] in linked and aps["T1190"]["id"] in linked
    assert aps["T1059.001"]["id"] not in linked
    sections = {n.get("x_vigileye_section") for n in _by_type(bundle, "note")}
    assert sections == {"body", "key_judgment", "assumptions", "gaps", "ach"}
    ach = next(n for n in _by_type(bundle, "note") if n.get("x_vigileye_section") == "ach")
    assert [h["hypothesis"] for h in ach["x_vigileye_ach"]] == ["H1 organic", "H2 bare"]
    assert set(rep["object_refs"]) <= {o["id"] for o in bundle["objects"]}


# --------------------------------------------------------------------------- (3) scope guard
def test_scope_guard_no_people_no_authors():
    bundle = report_to_stix(_report())
    text = json.dumps(bundle)
    assert not any(o.get("identity_class") == "individual" for o in bundle["objects"])
    assert not any(o.get("name") == "Bob" for o in bundle["objects"]) and "Bob" not in text
    assert not _by_type(bundle, "location")
    authors = ["user1", "alice.example"]
    for ident in _by_type(bundle, "identity"):
        assert not any(a in json.dumps(ident) for a in authors)
    # Authors never appear anywhere: free text is pseudonymised, deterministically per case.
    assert not any(a in text for a in authors)
    (rep,) = _by_type(bundle, "report")
    assert "author-" in rep["description"] and "LockBit affiliates" in rep["description"]
    body = next(n for n in _by_type(bundle, "note") if n.get("x_vigileye_section") == "body")
    assert "u/author-" in body["content"]
    other = report_to_stix(_report(case="Other case"))
    assert _by_type(other, "report")[0]["description"] != rep["description"]  # pseudonyms are case-salted
    assert "mastodon.social/@someone" not in text  # account URLs are never exported
    bad = {"type": "bundle", "id": bundle["id"], "objects": [
        {**_by_type(bundle, "identity")[0], "identity_class": "individual"}] + bundle["objects"][:2]}
    assert any("individual" in e for e in validate_bundle(bad))


# --------------------------------------------------------------------------- (4) exclusions
def test_benign_warninglisted_and_social_tags_excluded():
    bundle = report_to_stix(_report())
    text = json.dumps(bundle)
    assert "github.com" not in text and "10.0.0.1" not in text
    assert "#ransomware" not in [o.get("value") for o in bundle["objects"]] and "$TSLA" not in text
    assert "TA0001" not in text
    # enrich marks RFC 1918 space scope='non-routable'; WS5 adds warninglists. Either one suppresses the SCO.
    rep = _report(indicators=[{"type": "ipv4", "value": "10.0.0.1", "scope": "non-routable"},
                              {"type": "ipv4", "value": "10.0.0.1", "warninglists": ["rfc1918"]}])
    assert not _by_type(report_to_stix(rep), "ipv4-addr")


# --------------------------------------------------------------------------- (5) patterns
def test_indicator_patterns_and_escaping():
    rep = _report(indicators=_report().indicators + [
        {"type": "url", "value": "http://evil.example.com/a'b\\c", "count": 1, "docs": ["hn:99"]}])
    bundle = report_to_stix(rep, emit_indicator_patterns=True)
    assert validate_bundle(bundle) == []
    inds = {i["pattern"]: i for i in _by_type(bundle, "indicator")}
    assert "[domain-name:value = 'evil-update.xyz']" in inds
    assert f"[file:hashes.'SHA-256' = '{SHA256}']" in inds
    assert "[url:value = 'http://evil.example.com/a\\'b\\\\c']" in inds
    ind = inds["[domain-name:value = 'evil-update.xyz']"]
    assert ind["pattern_type"] == "stix" and ind["valid_from"] == "2026-02-03T04:05:06.789Z"
    rels = [r for r in _by_type(bundle, "relationship") if r["source_ref"] == ind["id"]]
    dom = _by_type(bundle, "domain-name")[0]
    assert len(rels) == 1 and rels[0]["relationship_type"] == "based-on" and rels[0]["target_ref"] == dom["id"]
    assert not _by_type(report_to_stix(rep), "indicator")


# --------------------------------------------------------------------------- (6) TLP
def test_tlp_marking_on_every_object():
    bundle = report_to_stix(_report(), tlp="amber")
    assert validate_bundle(bundle) == []
    amber = TLP["amber"]
    assert TLP_OBJECTS["amber"] in bundle["objects"]
    for o in bundle["objects"]:
        if o["type"] not in ("marking-definition", "bundle"):
            assert amber in o["object_marking_refs"], o["id"]
            assert len(o["object_marking_refs"]) == 2  # TLP + machine-assisted statement
    statement = [o for o in bundle["objects"] if o.get("definition_type") == "statement"]
    assert len(statement) == 1 and "Machine-assisted" in statement[0]["definition"]["statement"]
    assert TLP["clear"] == TLP["white"]
    assert TLP_OBJECTS["green"]["created"] == "2017-01-20T00:00:00.000Z"
    with pytest.raises(ValueError):
        report_to_stix(_report(), tlp="purple")
    assert report_to_stix(_report(), tlp="TLP:RED")["objects"][0]["id"] == TLP["red"]


def test_validate_bundle_catches_broken_bundles():
    bundle = report_to_stix(_report())
    rep = dict(_by_type(bundle, "report")[0], object_refs=["malware--00000000-0000-4000-8000-000000000000"],
               modified="2026-02-03 04:05")
    broken = {**bundle, "objects": [o for o in bundle["objects"] if o["type"] != "report"] + [rep, rep]}
    errs = validate_bundle(broken)
    assert any("unresolved" in e for e in errs) and any("duplicate" in e for e in errs)
    assert any("RFC 3339" in e for e in errs)
    assert validate_bundle({"type": "nope"}) == ["not a STIX bundle (type != 'bundle')"]


# --------------------------------------------------------------------------- (7) stix2 oracle
def test_stix2_library_parses_bundle():
    stix2 = pytest.importorskip("stix2")
    for tlp in ("clear", "amber"):
        parsed = stix2.parse(report_to_stix(_report(), tlp=tlp, emit_indicator_patterns=True), allow_custom=True)
        assert parsed.type == "bundle"
    assert stix2.parse(report_to_stix(_mock_report(), emit_indicator_patterns=True), allow_custom=True).objects
    # Our SCO ids equal the ones the reference library derives itself.
    assert sco_for_indicator({"type": "sha256", "value": SHA256})["id"] == stix2.File(hashes={"SHA-256": SHA256}).id
    assert sco_for_indicator({"type": "url", "value": "http://x.org/é"})["id"] == stix2.URL(value="http://x.org/é").id


# --------------------------------------------------------------------------- (8) navigator
def test_navigator_layer(tmp_path):
    layer = report_to_navigator_layer(_report())
    json.dumps(layer)
    assert layer["versions"]["layer"] == "4.5" and layer["domain"] == "enterprise-attack"
    assert layer["sorting"] == 3 and layer["gradient"]["maxValue"] == 4
    assert {"name": "case", "value": "Op Test"} in layer["metadata"]
    tech = {t["techniqueID"]: t for t in layer["techniques"]}
    assert set(tech) == {"T1190", "T1059.001", "T1059"}  # parent stub shows the sub-technique; TA ignored
    assert tech["T1190"]["score"] == 4 and tech["T1190"]["comment"] == "cited in: reddit:t3_a, hn:99"
    assert tech["T1190"]["metadata"] == [{"name": "doc", "value": "reddit:t3_a"}, {"name": "doc", "value": "hn:99"}]
    assert tech["T1059.001"]["score"] == 1 and "score" not in tech["T1059"] and tech["T1059"]["showSubtechniques"]
    many = _report(indicators=[{"type": "attack_technique", "value": "T1566", "count": 30,
                                "docs": [f"reddit:t3_{i}" for i in range(15)]}])
    t = report_to_navigator_layer(many, name="L", attack_version="19")["techniques"][0]
    assert len(t["metadata"]) == 10 and t["comment"].endswith("(+5 more)")
    empty = report_to_navigator_layer(_report(indicators=[{"type": "domain", "value": "a.xyz"}]))
    assert empty["techniques"] == [] and empty["gradient"]["maxValue"] > empty["gradient"]["minValue"]
    json.dumps(empty)
    p = write_navigator(_report(), tmp_path)
    assert p.name.endswith(".layer.json") and json.loads(p.read_text())["versions"]["layer"] == "4.5"


# --------------------------------------------------------------------------- (9) end to end
def _docs(n: int = 6) -> list[Document]:
    base = datetime(2026, 11, 1, 12, 0, tzinfo=timezone.utc)
    texts = [
        "Lockbit affiliate says CVE-2024-3400 is exploited. C2 185.220.101[.]45 and hxxps://evil-update[.]xyz/p.bin; "
        "technique T1190 and T1059.001.",
        f"Hash of the dropper: 44d88612fea8a8f36de82e1238573e8f and {SHA256} #ransomware",
        "Moderators of r/netsec announced new rules; NATO and CISA were mentioned. See https://github.com/x/y",
    ]
    return [Document(id=f"t3_{i:03d}", platform="reddit", source="r/netsec", text=texts[i % len(texts)],
                     title=f"Post {i}", author=f"user{i}", url=f"https://www.reddit.com/r/netsec/comments/{i}",
                     created_at=base + timedelta(hours=i)) for i in range(n)]


def _mock_report() -> Report:
    from vigileye.analysis import AnalysisEngine
    from vigileye.llm import LLMRouter

    cfg = Config(raw=MOCK_CONFIG)
    return AnalysisEngine(LLMRouter.from_config(cfg), cfg).run(_docs(), lens="cyber", query="threats?", case="e2e",
                                                               ach=True)


def test_mock_engine_report_exports(tmp_path):
    report = _mock_report()
    bundle = report_to_stix(report, emit_indicator_patterns=True)
    assert validate_bundle(bundle) == []
    assert _by_type(bundle, "vulnerability") and _by_type(bundle, "attack-pattern")
    assert "github.com" not in json.dumps(bundle)  # benign domains stay out
    assert not any(f"user{i}" in json.dumps(bundle) for i in range(6))
    p = write_stix(report, tmp_path / "sub" / "out.stix.json", tlp="green")
    assert validate_bundle(json.loads(p.read_text())) == [] and TLP["green"] in p.read_text()
    assert write_stix(report, tmp_path).name.endswith(".stix.json")
    layer = report_to_navigator_layer(report)
    assert {t["techniqueID"] for t in layer["techniques"]} >= {"T1190", "T1059.001"}


def test_write_report_stix_and_navigator_formats(tmp_path):
    from vigileye.report import write_report

    try:
        paths = write_report(_mock_report(), tmp_path, formats=("stix", "navigator"))
    except ValueError as e:
        if "unknown report format" in str(e):
            pytest.skip("report.write_report has no stix/navigator renderer yet (integration pending)")
        raise
    names = sorted(p.name for p in paths)
    assert names[0].endswith(".layer.json") and names[1].endswith(".stix.json")
    stix = next(p for p in paths if p.name.endswith(".stix.json"))
    assert validate_bundle(json.loads(stix.read_text())) == []
