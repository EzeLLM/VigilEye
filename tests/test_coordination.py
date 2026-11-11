"""Offline, seeded tests for content-level coordination detection (no LLM, no network)."""
from __future__ import annotations

import importlib
import json
import random
import sys
import time

import pytest

import vigileye.analysis.coordination as coord
from vigileye.analysis.coordination import (
    DETECTOR,
    canonical_url,
    case_key,
    co_share_edges,
    components,
    coordination_graph,
    detect_coordination,
    display_object,
    estimate_window,
    null_distribution,
    pseudonymize,
    share_events,
    share_objects,
)
from vigileye.models import Document

T0 = 1_767_571_200.0          # 2026-01-05 00:00 UTC
H = 3600.0
BURSTS = (3 * H, 13 * H, 22 * H, 31 * H, 41 * H)   # five ring bursts across two days
RING_URL = "https://www.news-site.example/story/{k}/?utm_source=ring&id={k}#top"


class Corpus:
    def __init__(self, seed: int = 1):
        self.rng = random.Random(seed)
        self.docs: list[Document] = []

    def add(self, platform: str, author: str | None, t: float | None, text: str, **kw) -> Document:
        d = Document(id=f"p{len(self.docs) + 1}", platform=platform, source="test", text=text, author=author,
                     created_at=None if t is None else T0 + t, **kw)
        self.docs.append(d)
        return d

    def ring(self, *, gap: float | None = None, text=None, bursts=BURSTS, n: int = 6, prefix: str = "ring"):
        for k, bt in enumerate(bursts):
            for a in range(n):
                plat = "bluesky" if a % 2 == 0 else "mastodon"
                author = f"{prefix}_{a}" + ("" if plat == "bluesky" else "@social.example")
                t = bt + (a * gap if gap else self.rng.uniform(0, 20))
                self.add(plat, author, t, text(k) if text else f"must read {RING_URL.format(k=k)}")
        return self

    def organic(self, n: int = 60, posts: int = 15):
        for o in range(n):
            plat = ("bluesky", "mastodon", "reddit")[o % 3]
            for _ in range(posts):
                url = f"https://blog{self.rng.randrange(200)}.example/p/{self.rng.randrange(3000)}"
                self.add(plat, f"organic_{o}", self.rng.uniform(0, 48 * H), f"reading {url}")
        return self

    def viral(self, n: int = 40, url: str = "https://viral.example/big-news"):
        for o in range(n):
            self.add(("bluesky", "mastodon", "reddit")[o % 3], f"organic_{o}", 20 * H + self.rng.uniform(0, 600),
                     f"wow {url}")
        return self


def _events(docs, case="t", clusters=None):
    return share_events(docs, key=case_key(case), cluster_of=coord._cluster_index(docs, clusters or []))


# ----------------------------------------------------------------------------------- (1) URLs
def test_canonical_url_strips_tracking_fragment_www_and_sorts():
    assert canonical_url("HTTPS://WWW.Example.com/a/b/?utm_source=x&b=2&fbclid=abc&a=1#frag") == \
        "https://example.com/a/b?a=1&b=2"
    assert canonical_url("http://example.com:80/x/?gclid=1&igshid=2&mc_cid=3&mc_eid=4&ref=5&ref_src=6&si=7") == \
        "http://example.com/x"
    assert canonical_url("https://example.com:8443/") == "https://example.com:8443"
    assert canonical_url("https://user:pw@Example.com/p?UTM_Medium=1&q=") == "https://example.com/p?q="
    assert canonical_url("not a url") == "not a url"
    assert canonical_url("https://[::1]:8080/x") == "https://[::1]:8080/x"


def test_share_objects_urls_text_and_tagseq():
    d = Document(id="1", platform="bluesky", source="s", author="a", created_at=T0,
                 url="https://bsky.app/profile/a/post/1",
                 text="see https://www.x.example/a?utm_campaign=1. and hxxps://evil[.]example/p "
                      "https://bsky.app/profile/b https://cdn.bsky.app/img.png #One #two #Three")
    objs = share_objects(d, cluster_of={"bluesky:1": "nd4"})
    assert objs == {"URL:https://x.example/a", "URL:https://evil.example/p", "TEXT:nd4", "TAGSEQ:one|two|three"}
    m = Document(id="2", platform="mastodon", source="s", author="a", created_at=T0,
                 url="https://social.example/@a/2", text="https://social.example/@b/9 https://other.example/@c/7")
    assert share_objects(m) == {"URL:https://other.example/@c/7"}
    assert display_object("URL:https://other.example/@c/7") == "URL:https://other.example/@[account]/7"
    assert display_object("URL:https://x.example/user/bob/posts") == "URL:https://x.example/user/[account]/posts"
    assert share_objects(m, min_tagseq=0) == {"URL:https://other.example/@c/7"}


# ---------------------------------------------------------------------------- (2) pseudonyms
def test_pseudonyms_stable_per_key_and_distinct_across_cases(monkeypatch):
    monkeypatch.delenv(coord.KEY_ENV, raising=False)
    k1, k2 = case_key("case-a"), case_key("case-b")
    p = pseudonymize("bluesky", "ring_0", k1)
    assert p == pseudonymize("bluesky", "@Ring_0", k1) and p.startswith("acct_") and len(p) == 15
    assert p != pseudonymize("bluesky", "ring_0", k2)
    assert p != pseudonymize("mastodon", "ring_0", k1)
    assert case_key("case-a", secret="s3cret") != k1
    monkeypatch.setenv(coord.KEY_ENV, "s3cret")
    assert case_key("case-a") == case_key("case-a", secret="s3cret")


def test_signals_contain_no_raw_handles():
    c = Corpus(1).ring().organic().viral()
    blob = json.dumps(detect_coordination(c.docs, clusters=[], case="x"))
    blob += json.dumps(coordination_graph(c.docs, clusters=[], case="x"))
    assert "ring_" not in blob and "organic_" not in blob and "social.example" not in blob


def test_public_allowlist_shows_handle_verbatim_others_pseudonymized():
    c = Corpus(2).ring().organic(posts=10)
    for d in c.docs:
        if d.author == "ring_0":
            d.author = "NewsDesk.example"
    sig = detect_coordination(c.docs, clusters=[], public_accounts=["bluesky:@newsdesk.example"])
    accts = sig[0]["params"]["accounts"]
    assert "bluesky:NewsDesk.example" in accts and sig[0]["params"]["public_accounts"] == ["bluesky:NewsDesk.example"]
    assert sum(a.startswith("acct_") for a in accts) == 5
    graph = coordination_graph(c.docs, clusters=[], public_accounts=["newsdesk.example"])
    assert [n["id"] for n in graph["nodes"] if n["public"]] == ["bluesky:NewsDesk.example"]


# --------------------------------------------------------------------------- (3) ring vs organic
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_synthetic_ring_is_the_only_significant_component(seed):
    c = Corpus(seed).ring().organic().viral()
    sig = detect_coordination(c.docs, clusters=[], case="ring-test")
    significant = [s for s in sig if s["params"]["significant"]]
    assert len(significant) == 1
    s = significant[0]
    assert s["id"] == "sig:coordination#1" and s["kind"] == "coordination" and s["detector"] == DETECTOR
    assert s["params"]["size"] == 6 and s["value"] == 15.0
    assert s["p_value"] <= 0.05 and s["z"] > 3 and s["baseline"] < s["value"]
    assert s["params"]["platforms"] == {"bluesky": 3, "mastodon": 3}
    assert s["params"]["n_shared_objects"] == 5 and s["params"]["object_kinds"] == {"URL": 5}
    assert [o["shares"] for o in s["params"]["shared_objects"]] == [6] * 5
    assert all("viral" not in o["object"] for o in s["params"]["shared_objects"])
    assert s["params"]["edge_symmetry"] == 1.0 and 0 < s["params"]["avg_time_delta_s"] < 20
    ring_refs = [d.ref for d in c.docs[:30]]
    assert len(s["doc_refs"]) == 20 and set(s["doc_refs"]) <= set(ring_refs)
    assert set(ring_refs[:6]) & set(s["doc_refs"]) and set(ring_refs[-6:]) & set(s["doc_refs"])  # all bursts
    assert "6 accounts repeatedly co-shared 5 objects within 60s" in s["summary"]
    assert "across bluesky, mastodon" in s["summary"]
    # the viral URL alone links organic accounts only at weight 1 -> no component of its own
    edges, _ = co_share_edges(_events(c.docs), window_s=60)
    viral = [e for e in edges.values() if "URL:https://viral.example/big-news" in e["objects"]]
    assert viral and all(e["weight"] == 1 for e in viral)
    assert [len(x) for x in components(edges, min_weight=2)] == [6]


def test_detection_is_deterministic_and_order_independent():
    c = Corpus(4).ring().organic().viral()
    a = detect_coordination(c.docs, clusters=[], n_iter=60)
    b = detect_coordination(list(reversed(c.docs)), clusters=[], n_iter=60)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


# ------------------------------------------------------------------------------- (4) organic
@pytest.mark.parametrize("seed", [5, 6])
def test_organic_corpus_has_no_significant_signals(seed):
    c = Corpus(seed).organic().viral()
    assert not [s for s in detect_coordination(c.docs, clusters=[]) if s["params"]["significant"]]


def test_chance_components_are_not_significant():
    # Dense organic co-sharing from a small pool forms chance components; the max-statistic
    # null must not call them coordinated.
    rng = random.Random(11)
    docs = [Document(id=f"d{a}_{k}", platform="mastodon", source="s", author=f"user{a}",
                     created_at=T0 + rng.uniform(0, 6 * H), text=f"https://pool.example/{rng.randrange(25)}")
            for a in range(30) for k in range(12)]
    sig = detect_coordination(docs, clusters=[], window_s=300, n_iter=100)
    assert sig, "fixture should produce chance components"
    assert not any(s["params"]["significant"] for s in sig)
    assert all("not significant" in s["summary"] for s in sig)


# --------------------------------------------------------------------- (5) gaps beyond window
def test_ring_with_five_minute_gaps_forms_no_component():
    c = Corpus(7).ring(gap=300).organic(posts=5)
    edges, _ = co_share_edges(_events(c.docs), window_s=60)
    assert components(edges, min_weight=2) == []
    assert detect_coordination(c.docs, clusters=[], window_s=60) == []


# ---------------------------------------------------------------------- (6) viral objects skipped
def test_objects_above_max_object_shares_are_skipped_and_listed():
    c = Corpus(8).ring().organic(posts=8).viral(n=40)
    _, skipped = co_share_edges(_events(c.docs), window_s=60, max_object_shares=30)
    assert skipped == ["URL:https://viral.example/big-news"]
    sig = detect_coordination(c.docs, clusters=[], max_object_shares=30)
    assert sig[0]["params"]["skipped_viral_objects"] == [{"object": "URL:https://viral.example/big-news", "shares": 40}]
    assert sig[0]["params"]["n_skipped_viral_objects"] == 1
    graph = coordination_graph(c.docs, clusters=[], max_object_shares=30)
    assert graph["params"]["skipped_viral_objects"][0]["shares"] == 40


# ------------------------------------------------------------------------- (7) TEXT objects
def test_text_objects_from_ws2_clusters_produce_edges():
    c = Corpus(9)
    c.ring(text=lambda k: f"template message number {k} with no links at all")
    clusters = [{"cluster_id": f"nd{k + 1}", "method": "minhash" if k % 2 else "exact",
                 "members": [d.ref for d in c.docs if f"number {k} " in d.text]} for k in range(5)]
    clusters.append({"cluster_id": "nd9", "method": "tagseq", "members": [d.ref for d in c.docs]})
    events = _events(c.docs, clusters=clusters)
    assert {e[1] for e in events} == {f"TEXT:nd{k}" for k in range(1, 6)}
    edges, _ = co_share_edges(events, window_s=60)
    assert len(edges) == 15 and all(e["weight"] == 5 for e in edges.values())
    c.organic(posts=10)
    sig = detect_coordination(c.docs, clusters=clusters)
    assert sig[0]["params"]["object_kinds"] == {"TEXT": 5} and sig[0]["params"]["significant"]


def test_one_post_with_url_and_text_counts_as_one_occasion():
    # Share-button copies: same URL *and* same text in a single co-post must not reach weight 2.
    docs = [Document(id=f"s{i}", platform="mastodon", source="s", author=f"fan{i}", created_at=T0 + i,
                     text="Big story https://paper.example/story") for i in range(4)]
    clusters = [{"cluster_id": "nd1", "method": "exact", "members": [d.ref for d in docs]}]
    edges, _ = co_share_edges(_events(docs, clusters=clusters), window_s=60)
    assert all(len(e["objects"]) == 2 and e["weight"] == 1 for e in edges.values())
    assert detect_coordination(docs, clusters=clusters) == []


# ---------------------------------------------------------------------------- (8) performance
def test_null_model_on_2000_events_is_fast():
    c = Corpus(10).ring().organic(n=120, posts=16).viral(n=60)
    events = _events(c.docs)
    assert len(events) >= 2000
    t0 = time.perf_counter()
    sig = detect_coordination(c.docs, clusters=[], n_iter=50)
    assert time.perf_counter() - t0 < 2.0
    assert sig and sig[0]["params"]["n_iter"] == 50


def test_work_cap_reduces_permutations_deterministically(monkeypatch):
    c = Corpus(1).ring().organic().viral()
    monkeypatch.setattr(coord, "MAX_NULL_WORK", 10_000)
    a = detect_coordination(c.docs, clusters=[], n_iter=200)
    assert a[0]["params"]["n_iter"] == 39
    assert json.dumps(a) == json.dumps(detect_coordination(c.docs, clusters=[], n_iter=200))


# ------------------------------------------------------------------------------- (9) robustness
def test_docs_without_author_or_time_are_ignored():
    c = Corpus(12)
    c.add("reddit", None, 10, "https://a.example/x")
    c.add("reddit", "someone", None, "https://a.example/x")
    for i in range(5):
        c.add("reddit", "[deleted]", 20 + i, f"https://a.example/{i}")
    events = _events(c.docs)
    assert events == []
    assert detect_coordination(c.docs, clusters=[]) == []
    assert detect_coordination([], clusters=None) == []
    assert coordination_graph([], clusters=[])["nodes"] == []


# ----------------------------------------------------------------------------- null model API
def test_null_distribution_member_and_max_statistics():
    c = Corpus(1).ring().organic().viral()
    events = _events(c.docs)
    handles = [("bluesky", f"ring_{i}") for i in (0, 2, 4)]
    handles += [("mastodon", f"ring_{i}@social.example") for i in (1, 3, 5)]
    ring = {pseudonymize(p, a, case_key("t")) for p, a in handles}
    member = null_distribution(events, ring, n_iter=30, seed=3)
    assert len(member) == 30 and member == null_distribution(events, ring, n_iter=30, seed=3)
    assert max(member) < 15
    maxed = null_distribution(events, None, n_iter=30, seed=3)
    assert maxed == null_distribution(events, None, n_iter=30, seed=3) != null_distribution(events, None, n_iter=30)
    assert sum(m >= 15 for m in maxed) <= 1          # the observed ring (15 repeated pairs) is rare under the null
    # the within-object null cannot see a ring that is the only sharer of its content
    obj = null_distribution(events, ring, n_iter=10, null_model="object")
    assert obj == [15] * 10
    with pytest.raises(ValueError):
        null_distribution(events, ring, n_iter=1, null_model="bogus")


def test_object_null_model_is_reported_in_params():
    c = Corpus(1).ring().organic().viral()
    sig = detect_coordination(c.docs, clusters=[], null_model="object", n_iter=20)
    assert sig[0]["params"]["null_model"] == "object" and not sig[0]["params"]["significant"]
    assert sig[0]["params"]["stratum_s"] is None


def test_auto_window_and_no_null_run():
    c = Corpus(1).ring().organic().viral()
    w = estimate_window(_events(c.docs))
    assert 1.0 <= w <= 20.0
    assert estimate_window([]) == 60.0
    sig = detect_coordination(c.docs, clusters=[], window_s="auto")
    assert sig and sig[0]["params"]["window_mode"] == "auto" and sig[0]["params"]["window_s"] == w
    none = detect_coordination(c.docs, clusters=[], n_iter=0)
    assert none[0]["p_value"] is None and none[0]["baseline"] is None and "no null model" in none[0]["summary"]
    with pytest.raises(ValueError):
        detect_coordination(c.docs, window_s=0)


def test_max_signals_and_numbering():
    c = Corpus(13).ring().ring(prefix="crew", bursts=(5 * H, 15 * H, 27 * H, 37 * H), n=4).organic()
    sig = detect_coordination(c.docs, clusters=[])
    assert [s["id"] for s in sig] == ["sig:coordination#1", "sig:coordination#2"]
    assert sorted(s["params"]["size"] for s in sig) == [4, 6]
    assert len(detect_coordination(c.docs, clusters=[], max_signals=1)) == 1


# ---------------------------------------------------------------------------------- graph
def test_coordination_graph_shape():
    c = Corpus(1).ring().organic().viral()
    g = coordination_graph(c.docs, clusters=[], n_iter=999, alpha=0.1)   # detector-only kwargs ignored
    assert g["params"]["n_nodes"] == 6 and g["params"]["n_edges"] == 15 and g["params"]["n_components"] == 1
    assert all(n["id"].startswith("acct_") and n["shares"] == 5 and n["component"] == 1 for n in g["nodes"])
    assert {n["platform"] for n in g["nodes"]} == {"bluesky", "mastodon"}
    assert all(e["weight"] == 5 and e["source"] < e["target"] and e["avg_delta_s"] < 20 for e in g["edges"])


# ------------------------------------------------------------------------- WS2 integration
def test_compute_signals_runs_the_coordination_detector():
    signals = pytest.importorskip("vigileye.analysis.signals")
    c = Corpus(1).ring().organic().viral()
    res = signals.compute_signals(c.docs, case="ws2", settings={
        "coordination": {"window_s": 60, "min_weight": 2, "n_iter": 200, "public_accounts": []}})
    coordination = [s for s in res["signals"] if s["kind"] == "coordination"]
    assert res["detectors"]["coordination"] == len(coordination) >= 1
    assert coordination[0]["params"]["significant"] and not [e for e in res["errors"] if "coordination" in e]


def test_fallbacks_without_ws2(monkeypatch):
    monkeypatch.setitem(sys.modules, "vigileye.analysis.signals", None)
    try:
        importlib.reload(coord)
        d = Document(id="1", platform="mastodon", source="s", author="a", created_at=T0, text="#a #b #c x")
        assert coord.share_objects(d) == {"TAGSEQ:a|b|c"}
        assert coord._cluster_index([d], None) == {}
        c = Corpus(1).ring().organic().viral()
        sig = coord.detect_coordination(c.docs, clusters=None, n_iter=50)
        assert sig[0]["id"] == "sig:coordination#1" and sig[0]["params"]["significant"]
    finally:
        monkeypatch.undo()
        importlib.reload(coord)
