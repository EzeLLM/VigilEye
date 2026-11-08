"""Offline tests for evidence preservation and the chain-of-custody ledger (vigileye/evidence.py)."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import unicodedata
from datetime import datetime, timedelta, timezone

import pytest

from vigileye.analysis import AnalysisEngine
from vigileye.config import Config
from vigileye.evidence import (
    CONTENT_FIELDS,
    GENESIS,
    STAMP_FIELDS,
    WAYBACK_API,
    EvidenceLedger,
    canonical_json,
    collect_details,
    content_hash,
    content_record,
    file_hashes,
    format_entry,
    mask_accounts,
    raw_hash,
    report_sha256,
    stamp,
    wayback_lookup,
)
from vigileye.llm import LLMRouter
from vigileye.models import Document, Media
from vigileye.store import CaseStore

T0 = datetime(2026, 11, 8, 9, 0, tzinfo=timezone.utc)
MOCK_CONFIG = {
    "providers": {"mock": {"base_url": "mock://"}},
    "roles": {r: [{"provider": "mock", "model": f"mock-{r}", "context_window": 32000}]
              for r in ("analyst", "extractor", "vision", "synthesizer")},
    "llm": {"cache": False},
    "analysis": {"chunk_tokens": 400, "max_chunks": 24, "max_concurrency": 2},
}


class Clock:
    """Deterministic clock: each call advances one second."""

    def __init__(self, start: datetime = T0):
        self.t = start

    def __call__(self) -> datetime:
        self.t += timedelta(seconds=1)
        return self.t


def _doc(i: int = 0, **kw) -> Document:
    base = dict(id=f"t3_{i:03d}", platform="reddit", source="r/osint", title=f"Post {i}",
                text=f"Public post number {i} about CVE-2024-3400 exploitation.", author=f"user{i}",
                url=f"https://www.reddit.com/r/osint/comments/{i}", created_at=T0 - timedelta(hours=i),
                metrics={"score": i}, raw={"name": f"t3_{i:03d}", "score": i})
    base.update(kw)
    return Document(**base)


def _hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _collect(store: CaseStore, ledger: EvidenceLedger, case: str, docs, spec: str = "reddit:r/osint") -> dict:
    """The App.collect integration recipe: stamp -> add_documents -> custody 'collect' entry."""
    docs = stamp(docs, spec=spec, collected_at=T0, existing=ledger.stamps(case))
    new = store.add_documents(case, docs)
    return ledger.record(case, "collect", subject=spec, details=collect_details(docs, new=new))


def _tamper(store: CaseStore, sql: str, args=()) -> None:
    with store._lock, store._db:
        store._db.execute(sql, args)


@pytest.fixture()
def env(tmp_path):
    store = CaseStore(tmp_path / "cases.db")
    ledger = EvidenceLedger(store, clock=Clock())
    yield store, ledger
    store.close()


# --------------------------------------------------------------------------- hashes
def test_content_hash_is_canonical_and_reproducible():
    d = Document(id="1", platform="p", source="s", text="t")
    expected = '{"created_at":null,"id":"1","media":[],"platform":"p","source":"s","text":"t","title":"","url":null}'
    assert canonical_json(content_record(d)) == expected
    assert content_hash(d) == _hex(expected)  # reproducible by a third party with only hashlib + json
    assert set(content_record(d)) == set(CONTENT_FIELDS)
    assert content_hash(d.to_dict()) == content_hash(d)  # dict input (e.g. a JSONL export line)

    a = _doc(text=unicodedata.normalize("NFD", "Café crème"))
    b = _doc(text=unicodedata.normalize("NFC", "Café crème"))
    assert a.text != b.text and content_hash(a) == content_hash(b)
    # same instant in another offset; author/metrics/tags/meta/raw are not evidentiary content
    plus2 = _doc(created_at=(T0).astimezone(timezone(timedelta(hours=2))), author="someone-else",
                 metrics={"score": 999}, tags=["#x"], meta={"k": 1}, raw={"other": True})
    assert content_hash(plus2) == content_hash(_doc())
    for change in ({"text": "edited"}, {"url": "https://example.org/x"}, {"title": "T"},
                   {"media": [Media(url="https://i.example/1.png")]}, {"created_at": T0 + timedelta(seconds=1)}):
        assert content_hash(_doc(**change)) != content_hash(_doc()), change


def test_hashes_survive_the_store_round_trip(tmp_path):
    raw = {"when": datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc), "pair": (1, 2), 7: "int key",
           "nan": float("nan"), "nested": {"name": unicodedata.normalize("NFD", "Zoë"), "list": [None, True, 1.5]}}
    d = _doc(raw=raw, media=[Media(url="https://i.example/a.png"), {"url": "https://i.example/b.png"}],
             created_at="2026-11-01T10:00:00+05:30")
    c0, r0 = content_hash(d), raw_hash(d)
    s = CaseStore(tmp_path / "c.db")
    s.add_documents("c", [d])
    back = s.get_documents("c")[0]
    assert back.raw != raw  # the store really changed the Python types ...
    assert (content_hash(back), raw_hash(back)) == (c0, r0)  # ... but not the hashes
    assert raw_hash(_doc(raw={})) == raw_hash(_doc(raw=None)) == _hex("{}")
    s.close()


# --------------------------------------------------------------------------- stamping
def test_stamp_fields_and_never_overwrite():
    docs = stamp([_doc(0), _doc(1)], spec="hn:search:ransomware", collected_at=T0)
    ev = docs[0].meta["evidence"]
    assert tuple(ev) == STAMP_FIELDS
    assert ev["content_sha256"] == content_hash(docs[0]) and ev["raw_sha256"] == raw_hash(docs[0])
    assert ev["collected_at"] == "2026-11-08T09:00:00+00:00" and ev["collector"] == "hackernews"
    assert ev["target_spec"] == "hn:search:ransomware" and ev["tool_version"].startswith("vigileye/")
    assert stamp([_doc()], spec="bogus spec", collector="custom", tool_version="x/1")[0].meta["evidence"][
        "collector"] == "custom"

    # an existing stamp is never overwritten, even after the content changed
    docs[0].text = "changed"
    again = stamp(docs, spec="other:spec", collected_at=T0 + timedelta(days=1))
    assert again[0].meta["evidence"] is ev and again[0].meta["evidence"]["target_spec"] == "hn:search:ransomware"
    with pytest.raises(ValueError):
        stamp([_doc()], spec="x", collected_at="not a date")


def test_stamp_carries_forward_unchanged_items_only():
    first = stamp([_doc(0)], spec="reddit:r/osint", collected_at=T0)[0]
    existing = {first.fingerprint(): first.meta["evidence"]}
    later = T0 + timedelta(days=2)
    same = stamp([_doc(0)], spec="reddit:r/osint", collected_at=later, existing=existing)[0]
    assert same.meta["evidence"] == first.meta["evidence"]  # first collected_at survives a re-collection
    bumped = stamp([_doc(0, raw={"name": "t3_000", "score": 50})], spec="reddit:r/osint", collected_at=later,
                   existing=existing)[0]
    assert bumped.meta["evidence"]["collected_at"] == later.isoformat()
    assert bumped.meta["evidence"]["content_sha256"] == first.meta["evidence"]["content_sha256"]


# --------------------------------------------------------------------------- ledger
def test_chain_genesis_linkage_and_independent_recomputation(env):
    store, ledger = env
    a1 = ledger.record("A", "collect", subject="reddit:r/osint", details={"collected": 2, "new": 2})
    b1 = ledger.record("B", "note", details={"text": "unrelated case"})
    a2 = ledger.record("A", "analyze", subject="report:1", details={"report_id": 1}, actor="analyst-7")
    assert a1["prev_hash"] == GENESIS and b1["prev_hash"] == GENESIS  # genesis per case
    assert a2["prev_hash"] == a1["entry_hash"] and a2["seq"] > b1["seq"] > a1["seq"]
    # recompute entry_hash from the documented formula with the stdlib only
    e = ledger.entries("A")[1]
    body = json.dumps({k: e[k] for k in ("case", "at", "action", "actor", "subject", "details")},
                      sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    assert _hex(e["prev_hash"] + body) == e["entry_hash"] == a2["entry_hash"]
    assert e["at"] == "2026-11-08T09:00:03+00:00" and e["actor"] == "analyst-7"
    v = ledger.verify_chain("A")
    assert v == {"ok": True, "entries": 2, "broken_at": None, "head": a2["entry_hash"], "reason": None}
    assert ledger.head("A") == a2["entry_hash"] and ledger.head("nope") is None
    assert ledger.verify_chain("nope") == {"ok": True, "entries": 0, "broken_at": None, "head": None, "reason": None}
    for bad in ("", "Has Space", "UPPER", 3):
        with pytest.raises(ValueError):
            ledger.record("A", bad)
    with pytest.raises(TypeError):
        ledger.record("A", "note", details=["not", "a", "mapping"])
    assert len(ledger.entries("A")) == 2  # failed calls appended nothing


def test_append_only_and_tamper_evidence(env):
    store, ledger = env
    entries = [ledger.record("A", "note", details={"n": i}) for i in range(4)]
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        _tamper(store, "UPDATE custody SET actor='mallory' WHERE seq=?", (entries[1]["seq"],))
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        _tamper(store, "DELETE FROM custody WHERE seq=?", (entries[3]["seq"],))
    assert ledger.verify_chain("A")["ok"]

    # someone with raw database access drops the guards: every kind of edit is detected
    _tamper(store, "DROP TRIGGER custody_no_update")
    _tamper(store, "DROP TRIGGER custody_no_delete")
    anchor = ledger.head("A")
    _tamper(store, "UPDATE custody SET details_json=? WHERE seq=?", ('{"n":99}', entries[1]["seq"]))
    v = ledger.verify_chain("A")
    assert not v["ok"] and v["broken_at"] == entries[1]["seq"] and "entry_hash" in v["reason"]
    _tamper(store, "UPDATE custody SET details_json=? WHERE seq=?", ('{"n":1}', entries[1]["seq"]))
    assert ledger.verify_chain("A")["ok"]  # restored byte-for-byte content verifies again

    _tamper(store, "DELETE FROM custody WHERE seq=?", (entries[2]["seq"],))  # a gap in the middle
    v = ledger.verify_chain("A")
    assert not v["ok"] and v["broken_at"] == entries[3]["seq"] and "link" in v["reason"]
    _tamper(store, "DELETE FROM custody WHERE seq=?", (entries[3]["seq"],))  # now a clean truncation
    assert ledger.verify_chain("A")["ok"]  # a bare chain cannot see a truncated tail ...
    v = ledger.verify_chain("A", anchor=anchor)  # ... an external anchor can
    assert not v["ok"] and v["anchor_found"] is False and "anchor" in v["reason"]
    assert ledger.verify_chain("A", anchor=entries[1]["entry_hash"])["anchor_found"] is True


def test_verify_case_detects_edits_restamps_and_raw_changes(env):
    store, ledger = env
    docs = [_doc(i) for i in range(3)]
    _collect(store, ledger, "op", docs)
    store.add_documents("op", [_doc(9)])  # bypasses stamping: legacy/unlogged path
    res = ledger.verify_case("op")
    assert res["ok"] and res["documents_checked"] == 3 and res["ledger_bound"]
    assert res["unstamped"] == [_doc(9).fingerprint()] and not res["mismatched"] and not res["unlogged"]

    fp0, fp1, fp2 = (d.fingerprint() for d in docs)

    def edit(fp, fn):
        (js,) = store._db.execute('SELECT json FROM documents WHERE "case"=? AND fingerprint=?', ("op", fp)).fetchone()
        d = json.loads(js)
        fn(d)
        _tamper(store, 'UPDATE documents SET json=? WHERE "case"=? AND fingerprint=?', (json.dumps(d), "op", fp))

    edit(fp0, lambda d: d.update(text="Fabricated quote."))
    edit(fp1, lambda d: d.update(text="Edited and re-stamped.") or d["meta"]["evidence"].update(
        content_sha256=content_hash(Document.from_dict(d))))
    edit(fp2, lambda d: d["raw"].update(score=10_000))
    res = ledger.verify_case("op")
    assert not res["ok"] and res["chain"]["ok"]
    assert res["mismatched"] == [fp0] and res["raw_mismatched"] == [fp2]
    assert res["unlogged"] == sorted([fp0, fp1])  # neither edited text was ever logged at collection


def test_recollection_keeps_ledger_binding(env):
    store, ledger = env
    _collect(store, ledger, "op", [_doc(0), _doc(1)])
    _collect(store, ledger, "op", [_doc(0), _doc(1, text="Post edited by its author upstream.")])
    res = ledger.verify_case("op")
    assert res["ok"] and res["documents_checked"] == 2 and not res["unlogged"]
    collects = [e for e in ledger.entries("op") if e["action"] == "collect"]
    assert [e["details"]["new"] for e in collects] == [2, 0] and len(collects[1]["details"]["items"]) == 2
    assert collects[1]["details"]["batch_sha256"] == _hex(canonical_json(collects[1]["details"]["items"]))


def test_report_hash_round_trip_and_report_check(tmp_path):
    store = CaseStore(tmp_path / "cases.db")
    ledger = EvidenceLedger(store, clock=Clock())
    _collect(store, ledger, "op", [_doc(i) for i in range(6)])
    docs = store.get_documents("op")
    report = AnalysisEngine(LLMRouter.from_config(Config(raw=MOCK_CONFIG)), Config(raw=MOCK_CONFIG)).run(
        docs, lens="cyber", query="Is it exploited?", case="op")
    report.stats["custody"] = ledger.custody_summary("op", docs)  # Methodology shows the chain head
    assert report.stats["custody"] == {"chain_head": ledger.head("op"), "chain_entries": 1, "chain_ok": True,
                                       "documents": 6, "documents_stamped": 6}
    rid = store.save_report(report)
    assert report_sha256(store.get_report(rid)) == report_sha256(report)  # stable through the store
    ledger.record("op", "analyze", subject=f"report:{rid}",
                  details={"report_id": rid, "lens": report.lens, "report_sha256": report_sha256(report)})
    out = tmp_path / "out.md"
    out.write_text("# report\n", encoding="utf-8")
    files = file_hashes([out])
    assert files == [{"name": "out.md", "bytes": 9, "sha256": _hex("# report\n")}]
    ledger.record("op", "export", subject=f"report:{rid}", details={"report_id": rid, "files": files})
    res = ledger.verify_case("op")
    assert res["ok"] and res["reports"] == {"checked": 1, "mismatched": [], "missing": []}

    (js,) = store._db.execute("SELECT json FROM reports WHERE id=?", (rid,)).fetchone()
    tampered = json.loads(js)
    tampered["bluf"] = "Rewritten bottom line."
    _tamper(store, "UPDATE reports SET json=? WHERE id=?", (json.dumps(tampered), rid))
    res = ledger.verify_case("op")
    assert not res["ok"] and res["reports"]["mismatched"] == [rid]
    store.close()


def test_manifest_and_sidecar_are_verifiable_and_people_free(env, tmp_path):
    store, ledger = env
    masto = _doc(5, platform="mastodon", id="masto_social_1", source="#osint", author="alice",
                 url="https://mastodon.social/@alice/1")
    _collect(store, ledger, "op", [_doc(1), _doc(0), masto])
    head = ledger.head("op")
    res = ledger.write_manifest("op", tmp_path / "evidence" / "op.manifest.json")
    path = tmp_path / "evidence" / "op.manifest.json"
    data = path.read_bytes()
    assert res["sha256"] == hashlib.sha256(data).hexdigest()
    assert (tmp_path / "evidence" / "op.manifest.json.sha256").read_text() == f"{res['sha256']}  op.manifest.json\n"
    m = json.loads(data)
    assert m == res["manifest"] and m["case"] == "op" and m["chain"] == {"head": head, "entries": 1, "ok": True}
    assert m["generated_at"].endswith("+00:00") and m["tool_version"].startswith("vigileye/")
    assert m["summary"] == {"documents": 3, "verified": 3, "mismatch": 0, "unstamped": 0}
    assert [d["ref"] for d in m["documents"]] == ["mastodon:masto_social_1", "reddit:t3_000", "reddit:t3_001"]
    row = m["documents"][1]
    assert set(row) == {"ref", "fingerprint", "url", "collected_at", "content_sha256", "raw_sha256", "status"}
    assert row["content_sha256"] == content_hash(_doc(0)) and row["collected_at"] == T0.isoformat()
    assert m["documents"][0]["url"] == "https://mastodon.social/@[account]/1"
    assert "alice" not in data.decode() and "user0" not in data.decode()  # no author strings anywhere
    last = ledger.entries("op")[-1]  # the manifest itself is a logged handling event
    assert last["action"] == "manifest" and last["details"]["sha256"] == res["sha256"]
    assert last["details"]["chain_head"] == head
    assert ledger.manifest("nobody")["documents"] == []


def test_determinism_across_databases(tmp_path):
    heads, manifests = [], []
    for name in ("a.db", "b.db"):
        store = CaseStore(tmp_path / name)
        ledger = EvidenceLedger(store, clock=Clock())
        _collect(store, ledger, "op", [_doc(i) for i in range(4)])
        ledger.record("op", "note", details={"z": 1, "a": [3, 2, 1], "é": "x"})
        heads.append(ledger.head("op"))
        manifests.append(json.dumps(ledger.manifest("op")))
        store.close()
    assert heads[0] == heads[1] and manifests[0] == manifests[1]


def test_shared_file_concurrent_writers(tmp_path):
    db = tmp_path / "cases.db"
    with EvidenceLedger(str(db)) as one, EvidenceLedger(db) as two:  # two connections, one file
        def work(ledger, tag):
            for i in range(15):
                ledger.record("busy", "note", subject=f"{tag}-{i}")

        threads = [threading.Thread(target=work, args=(lg, f"{tag}{k}"))
                   for k in range(2) for tag, lg in (("one", one), ("two", two))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        v = one.verify_chain("busy")
        assert v["ok"] and v["entries"] == 60 and two.head("busy") == v["head"]


def test_masking_and_log_lines():
    assert mask_accounts("mastodon:@user@infosec.exchange") == "mastodon:@[account]@infosec.exchange"
    assert mask_accounts("bluesky:@someone.bsky.social") == "bluesky:@[account]"
    assert mask_accounts("https://bsky.app/profile/someone.bsky.social/post/3k") == \
        "https://bsky.app/profile/[account]/post/3k"
    assert mask_accounts("reddit:u/someone") == "reddit:u/[account]" and mask_accounts("hn:user:pg") == \
        "hn:user:[account]"
    for keep in ("reddit:r/osint", "mastodon:#osint@infosec.exchange", "hn:search:menu/ransomware",
                 "https://www.reddit.com/r/osint/comments/1", "https://news.ycombinator.com/item?id=1", None):
        assert mask_accounts(keep) == keep
    line = format_entry({"seq": 3, "at": "2026-11-08T09:00:00+00:00", "action": "collect", "actor": "vigileye",
                         "subject": "reddit:u/someone", "entry_hash": "ab" * 32,
                         "details": {"collected": 2, "new": 1, "batch_sha256": "f" * 64, "items": [["x", "y"]]}})
    assert line == ("#3 2026-11-08T09:00:00+00:00 collect   vigileye reddit:u/[account] collected=2 new=1 "
                    "[abababababababab]")


def test_wayback_lookup_uses_injected_fetch_only():
    calls = []

    def fetch(url):
        calls.append(url)
        return {"archived_snapshots": {"closest": {"available": True, "status": "200", "timestamp": "20240102030405",
                                                   "url": "http://web.archive.org/web/20240102030405/https://x.org/a"}}}

    res = wayback_lookup("https://x.org/a?b=1&c=2", fetch=fetch)
    assert calls == [WAYBACK_API + "https%3A%2F%2Fx.org%2Fa%3Fb%3D1%26c%3D2"]
    assert res["archived"] and res["timestamp"] == "2024-01-02T03:04:05+00:00" and res["status"] == "200"
    assert res["snapshot_url"].startswith("http://web.archive.org/web/") and res["error"] is None
    assert wayback_lookup("https://x.org/b", fetch=lambda u: '{"archived_snapshots": {}}')["archived"] is False

    def boom(url):
        raise OSError("offline")

    res = wayback_lookup("https://x.org/c", fetch=boom)
    assert res["archived"] is False and "offline" in res["error"]
    res = wayback_lookup("file:///etc/passwd", fetch=fetch)
    assert res["error"] == "not an http(s) URL" and len(calls) == 1
