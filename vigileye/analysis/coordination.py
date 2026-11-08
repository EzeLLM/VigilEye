"""Content-level coordination detection with a permutation null model (pseudonymised).

What it measures
----------------
Accounts that *repeatedly* share the same content objects within a short time window, in
the style of coordination-network-toolkit, CooRTweet and CooRnet:

* objects are multimodal: ``URL:<canonical url>``, ``TEXT:<near-duplicate cluster id>``
  (exact/minhash clusters from :func:`vigileye.analysis.signals.near_duplicate_clusters`)
  and ``TAGSEQ:a|b|c`` (an ordered run of hashtags);
* two accounts are linked when they shared one object within ``window_s`` of each other.
  The edge **weight** is the number of distinct co-share occasions,
  ``min(distinct objects, distinct post pairs)``. One post that carries a URL *and* its
  share-button text therefore counts once, and so does one object re-posted many times;
* connected components over edges of weight ``>= min_weight`` are candidate networks.
  Only components of at least ``min_component_size`` accounts are reported.

Null model
----------
The surveyed tools stop at the network, so breaking-news virality looks like coordination.
Each candidate network here is tested against label permutations. A permutation keeps the
timing of every object's shares (its burst shape) and every account's number of posts. It
shuffles only *which account made which post*:

* ``null_model="block"`` (default): post authorship is permuted within strata of one
  platform and one ``stratum_s`` time block (one day), so each account's daily activity
  and the platform mix stay fixed. A post keeps all of its objects together. Finer blocks
  also control for time of day, but they lose power on campaign accounts that post only
  during their bursts. ``stratum_s=None`` uses one block per platform.
* ``null_model="object"``: account labels are permuted among the share events of each
  object (a within-object design). It controls for audience overlap (who shares what) and
  tests only timing. It cannot see a network that is the only sharer of its content,
  because then every permutation reproduces the observed edges. That is why it is not the
  default.

Candidate networks are *selected* from the data. Testing a component only against its own
members' chance co-sharing would make chance clusters look significant. The statistic is
therefore a cluster-mass maximum (Maris & Oostenveld 2007): for each permutation, the
largest number of repeated pairs inside any component of ``>= min_component_size``
accounts. ``p = (1 + #(null >= observed)) / (1 + n_iter)`` then controls the family-wise
error over all reported networks.

Privacy
-------
Accounts are HMAC-SHA256 pseudonymised with a per-case key (:func:`case_key`), so
pseudonyms do not link across cases. Set ``$VIGILEYE_PSEUDONYM_KEY``: with the built-in
default key, anyone can reverse a pseudonym by hashing candidate handles. Only aggregates
leave this module: counts, pseudonyms, doc refs, and shared objects with profile-like URL
paths masked. Raw handles appear only for accounts that the analyst allowlists as public
institutional or media accounts (``public_accounts``). The module measures how clusters of
accounts behave around content and builds no per-person profile. It never puts collected
text into a prompt.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import math
import os
import random
import re
import statistics
from array import array
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Iterable
from urllib.parse import unquote_plus, urlsplit

from ..models import Document
from .chunking import doc_ref
from .enrich import refang

log = logging.getLogger("vigileye.analysis")

COORD_VERSION = "1"
DETECTOR = f"coordination@{COORD_VERSION}"
KEY_ENV = "VIGILEYE_PSEUDONYM_KEY"
DEFAULT_KEY = "vigileye-local"
#: Upper bound on (slot pairs + label moves) x permutations per detect run. Above it the
#: number of permutations shrinks deterministically (never below 39, so p can reach 0.025).
MAX_NULL_WORK = 6_000_000
_MIN_CAPPED_ITER = 39
_NULL_MODELS = ("block", "object")

try:  # WS2: shared signal record and hashtag-sequence keys
    from .signals import hashtag_sequence_key as _tagseq_key
    from .signals import make_signal
except ImportError:  # WS2 not installed: minimal compatible stand-ins
    _TAG_RE = re.compile(r"(?<![\w&#/])#([^\W\d]\w{0,138})")

    def _tagseq_key(doc: Document, min_len: int = 3) -> tuple[str, ...] | None:
        tags = [m.group(1).casefold() for m in _TAG_RE.finditer(re.sub(r"\S+://\S+", " ", doc.text or ""))]
        key = tuple(dict.fromkeys(tags))
        return key if len(key) >= max(1, int(min_len)) else None

    def make_signal(kind: str, n: int, summary: str, *, value: float, doc_refs: list[str],
                    baseline: float | None = None, z: float | None = None, p_value: float | None = None,
                    params: dict | None = None, detector: str = DETECTOR) -> dict:
        return {"id": f"sig:{kind}#{int(n)}", "kind": kind, "summary": " ".join(str(summary).split())[:400],
                "value": float(value), "baseline": baseline, "z": z, "p_value": p_value,
                "doc_refs": list(dict.fromkeys(doc_refs or [])), "params": dict(params or {}),
                "detector": detector}


# --------------------------------------------------------------------------- pseudonyms
_PLACEHOLDER_AUTHORS = frozenset({"", "[deleted]", "[removed]", "deleted", "removed", "anonymous", "unknown",
                                  "none", "null"})


def _norm_account(account: Any) -> str:
    return str(account or "").strip().lstrip("@").casefold()


def case_key(case: str, secret: str | None = None) -> bytes:
    """Per-case HMAC key: ``sha256(secret or $VIGILEYE_PSEUDONYM_KEY or 'vigileye-local' + NUL + case)``."""
    material = secret or os.environ.get(KEY_ENV) or DEFAULT_KEY
    return hashlib.sha256(f"{material}\x00{case}".encode("utf-8")).digest()


def pseudonymize(platform: str, account: str, key: bytes) -> str:
    """``acct_`` + the first 10 hex chars of HMAC-SHA256(key, platform NUL account).

    The account is normalised first (trimmed, leading ``@`` dropped, casefolded).
    """
    msg = f"{(platform or '').strip().lower()}\x00{_norm_account(account)}".encode("utf-8")
    return "acct_" + hmac.new(key, msg, hashlib.sha256).hexdigest()[:10]


def _public_labels(docs: Iterable[Document], key: bytes, public_accounts: Iterable[str] | None) -> dict[str, str]:
    """Pseudonym -> ``platform:handle`` for allowlisted public accounts (``platform:handle`` or bare handle)."""
    pairs, anywhere = set(), set()
    for entry in public_accounts or []:
        plat, sep, handle = str(entry).strip().partition(":")
        if sep and plat and handle:
            pairs.add((plat.lower(), _norm_account(handle)))
        elif _norm_account(entry):
            anywhere.add(_norm_account(entry))
    out: dict[str, str] = {}
    if not pairs and not anywhere:
        return out
    for d in docs:
        acct = _norm_account(d.author)
        if acct and ((d.platform.lower(), acct) in pairs or acct in anywhere):
            out.setdefault(pseudonymize(d.platform, d.author, key), f"{d.platform}:{str(d.author).strip()}")
    return out


# --------------------------------------------------------------------------- share objects
_URL_RE = re.compile(r"\bhttps?://[^\s<>\"'`\])}|]+", re.I)
_URL_TRAIL = ".,;:!?'\"*_~"
_TRACKING = frozenset({"fbclid", "gclid", "igshid", "mc_cid", "mc_eid", "ref", "ref_src", "si"})
_PLATFORM_HOSTS = {
    "reddit": ("reddit.com", "redd.it", "redditmedia.com", "redditstatic.com"),
    "bluesky": ("bsky.app", "bsky.social", "bsky.network"),
    "hackernews": ("ycombinator.com",),
}
# Profile-like URL path segments, masked whenever an object is shown (the object id itself stays exact).
_PROFILE_PATH = re.compile(r"(/@)[^/?#]+|(/(?:u|user|users|profile|people|channel|c|author)/)[^/?#]+"
                           r"|(profile\.php\?)[^#]*", re.I)


def _tracking(param: str) -> bool:
    name = unquote_plus(param.split("=", 1)[0]).strip().lower()
    return name.startswith("utm_") or name in _TRACKING


def canonical_url(url: str) -> str:
    """Canonical form for matching shared links.

    Lowercases scheme and host, strips ``www.``, user info, default ports, the fragment,
    tracking parameters (``utm_*``, ``fbclid``, ``gclid``, ``igshid``, ``mc_cid``, ``mc_eid``,
    ``ref``, ``ref_src``, ``si``) and the trailing ``/``, and sorts the remaining query
    parameters. Unparseable input is returned stripped.
    """
    s = (url or "").strip()
    try:
        p = urlsplit(s)
        port = p.port
    except ValueError:
        return s
    scheme, host = p.scheme.lower(), (p.hostname or "").rstrip(".")
    if not scheme or not host:
        return s
    if host.startswith("www."):
        host = host[4:]
    if ":" in host:
        host = f"[{host}]"
    if port is not None and (scheme, port) not in (("http", 80), ("https", 443)):
        host = f"{host}:{port}"
    query = sorted(q for q in p.query.split("&") if q and not _tracking(q))
    return f"{scheme}://{host}{p.path.rstrip('/')}" + ("?" + "&".join(query) if query else "")


def _host(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").rstrip(".")
    except ValueError:
        return ""


def _own_hosts(doc: Document) -> tuple[str, ...]:
    hosts = list(_PLATFORM_HOSTS.get(doc.platform, ()))
    if doc.url:
        h = _host(canonical_url(doc.url))
        if h:
            hosts.append(h)
    return tuple(hosts)


def display_object(obj: str) -> str:
    """Object id as shown in outputs: profile-like URL path segments, and the account part of
    links on known social hosts (``x.com/<handle>/status/1``, ``t.me/<channel>/45``), are masked
    as ``[account]``."""
    if not obj.startswith("URL:"):
        return obj
    from ..policy import mask_profile_urls  # lazy: policy imports the collectors package

    obj = _PROFILE_PATH.sub(lambda m: (m.group(1) or m.group(2) or m.group(3)) + "[account]", obj)
    return mask_profile_urls(obj)


def share_objects(doc: Document, *, cluster_of: dict[str, str] | None = None, min_tagseq: int = 3) -> set[str]:
    """Content objects a document shares.

    * ``URL:<canonical>`` for outbound links in the text (the title only when the text is empty,
      since Reddit comments carry the thread title), excluding ``doc.url`` itself and the
      platform's own hosts (and the host of ``doc.url``, e.g. the Mastodon instance);
    * ``TEXT:<cluster_id>`` when the doc ref is in ``cluster_of`` (an exact/minhash cluster);
    * ``TAGSEQ:a|b|c`` for an ordered run of at least ``min_tagseq`` hashtags (``0`` disables).
    """
    out: set[str] = set()
    own = _own_hosts(doc)
    self_url = canonical_url(doc.url) if doc.url else None
    for raw in _URL_RE.findall(refang(doc.text or doc.title or "")):
        u = canonical_url(raw.rstrip(_URL_TRAIL))
        host = _host(u)
        if host and u != self_url and not any(host == h or host.endswith("." + h) for h in own):
            out.add("URL:" + u)
    cid = (cluster_of or {}).get(doc_ref(doc))
    if cid:
        out.add(f"TEXT:{cid}")
    if min_tagseq and int(min_tagseq) > 0:
        seq = _tagseq_key(doc, int(min_tagseq))
        if seq:
            out.add("TAGSEQ:" + "|".join(seq))
    return out


def _cluster_index(docs: list[Document], clusters: list[dict] | None) -> dict[str, str]:
    """Doc ref -> cluster id for exact/minhash clusters. ``None`` computes them with WS2 if installed."""
    if clusters is None:
        try:
            from .signals import near_duplicate_clusters
        except ImportError:
            clusters = []
        else:
            clusters = near_duplicate_clusters(docs)
    out: dict[str, str] = {}
    for c in clusters or []:
        if c.get("method") in ("exact", "minhash") and c.get("cluster_id"):
            for ref in c.get("members") or []:
                out.setdefault(str(ref), str(c["cluster_id"]))
    return out


def share_events(docs: list[Document], *, key: bytes, cluster_of: dict[str, str] | None = None,
                 min_tagseq: int = 3) -> list[tuple[str, str, float, str]]:
    """``(pseudonym, object, epoch_s, doc_ref)`` per shared object, sorted by time.

    Documents without an author (or with a placeholder such as ``[deleted]``) or without
    ``created_at`` are skipped, as are repeated doc refs.
    """
    out: list[tuple[str, str, float, str]] = []
    seen: set[str] = set()
    for d in docs or []:
        ref = doc_ref(d)
        if ref in seen or d.created_at is None or _norm_account(d.author) in _PLACEHOLDER_AUTHORS:
            continue
        seen.add(ref)
        acct, t = pseudonymize(d.platform, d.author, key), d.created_at.timestamp()
        out.extend((acct, obj, t, ref) for obj in sorted(share_objects(d, cluster_of=cluster_of,
                                                                         min_tagseq=min_tagseq)))
    out.sort(key=lambda e: (e[2], e[3], e[1]))
    return out


def estimate_window(events: list[tuple[str, str, float, str]], *, q: float = 0.1, p: float = 0.5,
                    min_s: float = 1.0, max_s: float = 3600.0, default: float = 60.0) -> float:
    """CooRnet-style data-driven co-share window.

    Ranks objects with at least two shares by the delay of their second share, keeps the
    quickest ``q`` fraction, and returns the median time these objects took to gather ``p``
    of their shares, clamped to ``[min_s, max_s]``.
    """
    by_obj: dict[str, list[float]] = defaultdict(list)
    for _, obj, t, _ in events:
        by_obj[obj].append(t)
    series = sorted((sorted(ts) for ts in by_obj.values() if len(ts) >= 2), key=lambda ts: (ts[1] - ts[0], len(ts)))
    if not series:
        return float(default)
    quick = series[: max(1, math.ceil(q * len(series)))]
    w = statistics.median(ts[max(2, math.ceil(p * len(ts))) - 1] - ts[0] for ts in quick)
    return float(min(max(w, min_s), max_s))


# --------------------------------------------------------------------------- co-share index
class _Index:
    """Share slots in canonical order with every within-window slot pair precomputed.

    Timestamps do not change under the null model, so the in-window slot pairs are computed
    once; a permutation changes only which account label sits in each slot.
    """

    def __init__(self, events: Iterable[tuple[str, str, float, str]], window_s: float, max_object_shares: int):
        by_obj: dict[str, list[tuple]] = defaultdict(list)
        for ev in events:
            by_obj[ev[1]].append(ev)
        self.skipped = sorted(((o, len(v)) for o, v in by_obj.items() if len(v) > max_object_shares),
                              key=lambda x: (-x[1], x[0]))
        skip = {o for o, _ in self.skipped}
        self.accts: list[str] = []
        self.acct_platform: list[str] = []
        self.refs: list[str] = []
        self.objs: list[str] = []
        self.slot_acct: list[int] = []
        self.slot_doc: list[int] = []
        self.slot_time: list[float] = []
        self.doc_acct: list[int] = []
        self.doc_time: list[float] = []
        self.obj_ranges: list[tuple[int, int]] = []
        acct_ix: dict[str, int] = {}
        doc_ix: dict[str, int] = {}
        for obj in sorted(o for o in by_obj if o not in skip):
            self.objs.append(obj)
            start = len(self.slot_acct)
            for acct, _, t, ref in sorted(by_obj[obj], key=lambda e: (e[2], e[3], e[0])):
                a = acct_ix.setdefault(acct, len(self.accts))
                if a == len(self.accts):
                    self.accts.append(acct)
                    self.acct_platform.append(ref.split(":", 1)[0])
                d = doc_ix.setdefault(ref, len(self.refs))
                if d == len(self.refs):
                    self.refs.append(ref)
                    self.doc_acct.append(a)
                    self.doc_time.append(float(t))
                self.slot_acct.append(a)
                self.slot_doc.append(d)
                self.slot_time.append(float(t))
            self.obj_ranges.append((start, len(self.slot_acct)))
        self.obj_ix = {o: i for i, o in enumerate(self.objs)}
        # In-window slot pairs: (slot i, slot j, object, post-pair id).
        self.p1, self.p2, self.pobj, self.pdp = array("l"), array("l"), array("l"), array("l")
        dp_ix: dict[tuple[int, int], int] = {}
        st, sd = self.slot_time, self.slot_doc
        for oi, (start, end) in enumerate(self.obj_ranges):
            for i in range(start, end):
                ti, di = st[i], sd[i]
                j = i + 1
                while j < end and st[j] - ti <= window_s:
                    dj = sd[j]
                    if dj != di:
                        self.p1.append(i)
                        self.p2.append(j)
                        self.pobj.append(oi)
                        self.pdp.append(dp_ix.setdefault((di, dj) if di < dj else (dj, di), len(dp_ix)))
                    j += 1

    def work(self) -> int:
        return len(self.p1) + len(self.slot_acct) + len(self.refs)


def _edge_details(ix: _Index) -> dict[tuple[str, str], dict]:
    """Observed edges ``(a, b)`` with ``a < b`` and their objects, deltas, refs, weight and symmetry."""
    edges: dict[tuple[str, str], dict] = {}
    for s1, s2, o, dp in zip(ix.p1, ix.p2, ix.pobj, ix.pdp):
        a, b = ix.slot_acct[s1], ix.slot_acct[s2]
        if a == b:
            continue
        (na, ra), (nb, rb) = sorted(((ix.accts[a], ix.refs[ix.slot_doc[s1]]), (ix.accts[b], ix.refs[ix.slot_doc[s2]])))
        e = edges.get((na, nb))
        if e is None:
            e = edges[(na, nb)] = {"objects": set(), "deltas": [], "refs": set(),
                                   "_dp": set(), "_sides": (set(), set())}
        e["objects"].add(ix.objs[o])
        if dp not in e["_dp"]:
            e["_dp"].add(dp)
            e["deltas"].append(abs(ix.slot_time[s2] - ix.slot_time[s1]))
        e["refs"].update((ra, rb))
        e["_sides"][0].add(ra)
        e["_sides"][1].add(rb)
    for e in edges.values():
        dps, (sa, sb) = e.pop("_dp"), e.pop("_sides")
        e["n_doc_pairs"] = len(dps)
        e["weight"] = min(len(e["objects"]), len(dps))
        e["symmetry"] = min(len(sa), len(sb)) / max(len(sa), len(sb))
    return edges


def co_share_edges(events: list[tuple[str, str, float, str]], *, window_s: float = 60.0,
                   max_object_shares: int = 500) -> tuple[dict[tuple[str, str], dict], list[str]]:
    """Co-share edges between accounts that shared an object within ``window_s`` seconds.

    Returns ``(edges, skipped)``. ``edges[(a, b)]`` (``a < b``) holds ``objects`` (set),
    ``deltas`` (seconds, one per distinct post pair), ``refs`` (set), ``n_doc_pairs``,
    ``weight`` (``min(len(objects), n_doc_pairs)``) and ``symmetry`` (CooRTweet-style
    ``min/max`` of the posts each side contributed). Objects with more than
    ``max_object_shares`` shares are viral rather than coordinated and cost O(n^2), so
    they are skipped and listed in ``skipped`` (most shared first).
    """
    ix = _Index(events, float(window_s), int(max_object_shares))
    return _edge_details(ix), [o for o, _ in ix.skipped]


def _union_find(pairs: Iterable[tuple[Any, Any]]) -> dict[Any, list]:
    parent: dict[Any, Any] = {}

    def find(x: Any) -> Any:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    groups: dict[Any, list] = defaultdict(list)
    for x in list(parent):
        groups[find(x)].append(x)
    return groups


def components(edges: dict, *, min_weight: int = 2) -> list[list[str]]:
    """Connected components (sorted member lists) over edges of weight ``>= min_weight``, largest first."""
    keep = [k for k, e in edges.items() if e.get("weight", len(e.get("objects") or ())) >= int(min_weight)]
    comps = [sorted(g) for g in _union_find(keep).values()]
    return sorted(comps, key=lambda c: (-len(c), c[0]))


# --------------------------------------------------------------------------- null model
def _qualifying(ix: _Index, lab: list[int], min_weight: int) -> list[tuple[int, int]]:
    """Account-id pairs whose co-share weight under the slot labels ``lab`` is ``>= min_weight``."""
    n = max(1, len(ix.accts))
    acc: dict[int, Any] = {}
    for s1, s2, o, dp in zip(ix.p1, ix.p2, ix.pobj, ix.pdp):
        a, b = lab[s1], lab[s2]
        if a == b:
            continue
        k = a * n + b if a < b else b * n + a
        r = acc.get(k)
        if r is None:
            acc[k] = (o, dp)                        # first co-share: a cheap tuple
        elif type(r) is tuple:
            acc[k] = [{r[0], o}, {r[1], dp}]   # second co-share: upgrade to sets
        else:
            r[0].add(o)
            r[1].add(dp)
    if min_weight <= 1:
        keys = list(acc)
    else:
        keys = [k for k, r in acc.items()
                if type(r) is not tuple and len(r[0]) >= min_weight and len(r[1]) >= min_weight]
    return [divmod(k, n) for k in keys]


def _statistic(pairs: list[tuple[int, int]], members: set[int] | None, min_component_size: int) -> int:
    """Repeated pairs among ``members``; with ``members=None`` the largest count inside any component."""
    if members is not None:
        return sum(1 for a, b in pairs if a in members and b in members)
    if not pairs:
        return 0
    root_of = {x: r for r, xs in _union_find(pairs).items() for x in xs}
    sizes = Counter(root_of.values())
    mass = Counter(root_of[a] for a, _ in pairs)
    return max((m for r, m in mass.items() if sizes[r] >= min_component_size), default=0)


def _null(ix: _Index, members: set[int] | None, *, min_weight: int, min_component_size: int, n_iter: int,
          seed: int, null_model: str, stratum_s: float | None) -> list[int]:
    if null_model not in _NULL_MODELS:
        raise ValueError(f"null_model must be one of {_NULL_MODELS}, got {null_model!r}")
    strata: list[tuple[list[int], list[int]]] = []
    if null_model == "block":
        groups: dict[tuple, list[int]] = defaultdict(list)
        for d, (ref, t) in enumerate(zip(ix.refs, ix.doc_time)):
            block = math.floor(t / stratum_s) if stratum_s and stratum_s > 0 else 0
            groups[(ref.split(":", 1)[0], block)].append(d)
        for k in sorted(groups):
            ds = sorted(groups[k], key=lambda d: (ix.doc_time[d], ix.refs[d]))
            if len(ds) > 1:
                strata.append((ds, [ix.doc_acct[d] for d in ds]))
    out: list[int] = []
    for i in range(max(0, int(n_iter))):
        rng = random.Random(int(seed) + i)
        if null_model == "block":
            doc_lab = list(ix.doc_acct)
            for ds, labs in strata:
                labs = labs[:]
                rng.shuffle(labs)
                for d, a in zip(ds, labs):
                    doc_lab[d] = a
            lab = [doc_lab[d] for d in ix.slot_doc]
        else:
            lab = list(ix.slot_acct)
            for start, end in ix.obj_ranges:
                if end - start > 1:
                    seg = lab[start:end]
                    rng.shuffle(seg)
                    lab[start:end] = seg
        out.append(_statistic(_qualifying(ix, lab, min_weight), members, min_component_size))
    return out


def null_distribution(events: list[tuple[str, str, float, str]], members: set[str] | None = None, *,
                      window_s: float = 60.0, min_weight: int = 2, n_iter: int = 200, seed: int = 7,
                      max_object_shares: int = 500, min_component_size: int = 3, null_model: str = "block",
                      stratum_s: float | None = 86400.0) -> list[int]:
    """Permutation null of the repeated-pair statistic, one value per iteration.

    Iteration ``i`` uses ``random.Random(seed + i)`` and reassigns account labels (see the
    module docstring for ``null_model`` and ``stratum_s``). With ``members`` the statistic is
    the number of member pairs with co-share weight ``>= min_weight``: use it for an account
    set fixed *before* looking at the data. With ``members=None`` the statistic is the
    largest such count inside any component of ``>= min_component_size`` accounts. That is
    the selection-aware reference :func:`detect_coordination` uses.
    """
    ix = _Index(events, float(window_s), int(max_object_shares))
    ids = None if members is None else {i for i, a in enumerate(ix.accts) if a in set(members)}
    return _null(ix, ids, min_weight=max(1, int(min_weight)), min_component_size=int(min_component_size),
                 n_iter=n_iter, seed=seed, null_model=null_model, stratum_s=stratum_s)


# --------------------------------------------------------------------------- detection
def _window(window_s: Any, events: list) -> tuple[float, str]:
    if isinstance(window_s, str) and window_s.strip().lower() == "auto":
        return estimate_window(events), "auto"
    w = float(window_s)
    if not w > 0:
        raise ValueError("window_s must be > 0 or 'auto'")
    return w, "fixed"


def _secs(x: float) -> str:
    return f"{x:.0f}s" if x >= 10 or abs(x - round(x)) < 0.05 else f"{x:.1f}s"


def _spread(items: list, k: int) -> list:
    """At most ``k`` items evenly spaced over ``items`` (keeps the first and last), so citations span the timeline."""
    if k <= 1:
        return list(items[: max(k, 0)])
    if len(items) <= k:
        return list(items)
    return [items[round(i * (len(items) - 1) / (k - 1))] for i in range(k)]


def _iso(t: float | None) -> str | None:
    return datetime.fromtimestamp(t, tz=timezone.utc).isoformat() if t is not None else None


def _prepare(docs: Iterable[Document], clusters: list[dict] | None, case: str, secret: str | None,
             public_accounts: Iterable[str] | None, window_s: Any, max_object_shares: int):
    docs = list(docs or [])
    key = case_key(case, secret)
    events = share_events(docs, key=key, cluster_of=_cluster_index(docs, clusters))
    window, mode = _window(window_s, events)
    ix = _Index(events, window, int(max_object_shares))
    return ix, window, mode, _public_labels(docs, key, public_accounts)


def detect_coordination(docs: list[Document], *, clusters: list[dict] | None = None, case: str = "default",
                        window_s: float | str = 60.0, min_weight: int = 2, min_component_size: int = 3,
                        n_iter: int = 200, seed: int = 7, alpha: float = 0.05, max_signals: int = 10,
                        secret: str | None = None, public_accounts: list[str] | None = None,
                        max_object_shares: int = 500, null_model: str = "block",
                        stratum_s: float | None = 86400.0) -> list[dict]:
    """``coordination`` signals: account networks that repeatedly co-share content within ``window_s``.

    ``clusters`` are WS2 near-duplicate clusters (for ``TEXT:`` objects). ``None`` computes
    them when :mod:`vigileye.analysis.signals` is installed, and ``[]`` disables them.
    ``window_s="auto"`` uses :func:`estimate_window`. One signal is emitted per component of
    ``>= min_component_size`` accounts: significant ones first, then by p, z and size, at
    most ``max_signals``, numbered ``sig:coordination#1``... Fields:
    ``value`` = repeated pairs inside the component, ``baseline`` = null mean, ``z``
    = ``(value - mean) / max(std, floor)``. The floor is the standard deviation of a single
    exceedance in ``n_iter`` draws, so a null that is always zero gives a large but finite z.
    ``p_value`` = ``(1 + #(null >= value)) / (1 + n_iter)``. ``params`` holds aggregates
    only, and accounts appear as pseudonyms unless allowlisted in ``public_accounts``.
    """
    min_weight, min_component_size = max(1, int(min_weight)), max(2, int(min_component_size))
    if null_model not in _NULL_MODELS:
        raise ValueError(f"null_model must be one of {_NULL_MODELS}, got {null_model!r}")
    ix, window, mode, public = _prepare(docs, clusters, case, secret, public_accounts, window_s, max_object_shares)
    edges = _edge_details(ix)
    comps = [c for c in components(edges, min_weight=min_weight) if len(c) >= min_component_size]
    if not comps:
        return []

    n_req = max(0, int(n_iter))
    work = max(1, ix.work())
    n_eff = n_req if n_req * work <= MAX_NULL_WORK else max(min(n_req, _MIN_CAPPED_ITER), MAX_NULL_WORK // work)
    if n_eff < n_req:
        log.info("coordination: null model capped at %d of %d permutations (%d slot pairs)", n_eff, n_req, len(ix.p1))
    null = _null(ix, None, min_weight=min_weight, min_component_size=min_component_size, n_iter=n_eff, seed=seed,
                 null_model=null_model, stratum_s=stratum_s)
    mean = statistics.fmean(null) if null else None
    std = statistics.pstdev(null) if null else None
    floor = math.sqrt(n_eff) / (n_eff + 1) if n_eff else None   # sqrt(q(1-q)), q = 1/(n+1)

    acct_id = {a: i for i, a in enumerate(ix.accts)}
    posts = Counter(ix.doc_acct)
    ref_time = dict(zip(ix.refs, ix.doc_time))
    skipped = [{"object": display_object(o), "shares": n} for o, n in ix.skipped[:10]]
    comp_no = {m: n for n, c in enumerate(comps) for m in c}
    intra_of: dict[int, list[dict]] = defaultdict(list)
    for (a, _), e in edges.items():
        if a in comp_no and e["weight"] >= min_weight:
            intra_of[comp_no[a]].append(e)
    records = []
    for n_comp, comp in enumerate(comps):
        intra = intra_of[n_comp]
        obs = len(intra)
        p = (1 + sum(1 for x in null if x >= obs)) / (1 + n_eff) if n_eff else None
        z = (obs - mean) / max(std, floor) if n_eff else None
        significant = p is not None and p < float(alpha)
        objects = sorted(set().union(*(e["objects"] for e in intra)))
        ids = {acct_id[m] for m in comp}
        shares = {o: sum(1 for s in range(*ix.obj_ranges[ix.obj_ix[o]]) if ix.slot_acct[s] in ids) for o in objects}
        top = sorted(objects, key=lambda o: (-shares[o], o))[:5]
        deltas = [x for e in intra for x in e["deltas"]]
        avg = statistics.fmean(deltas) if deltas else 0.0
        refs = sorted(set().union(*(e["refs"] for e in intra)), key=lambda r: (ref_time.get(r, 0.0), r))
        platforms = Counter(ix.acct_platform[acct_id[m]] for m in comp)
        plats = sorted(platforms, key=lambda x: (-platforms[x], x))
        where = ("across " + ", ".join(plats)) if len(plats) > 1 else f"on {plats[0]}"
        stats_txt = (f"repeated pairs {obs} vs null {mean:.1f}+/-{std:.1f} (p={p:.3g}"
                     + ("" if significant else ", not significant") + ")") if n_eff else \
            f"repeated pairs {obs} (no null model run)"
        summary = (f"{len(comp)} accounts repeatedly co-shared {len(objects)} objects within {_secs(window)} "
                   f"(avg delta {_secs(avg)}) {where}; {stats_txt}")
        accounts = sorted(comp, key=lambda m: (-posts[acct_id[m]], public.get(m, m)))
        params = {
            "size": len(comp),
            "accounts": [public.get(m, m) for m in accounts[:50]],
            "public_accounts": sorted(public[m] for m in comp if m in public),
            "platforms": {x: platforms[x] for x in plats},
            "shared_objects": [{"object": display_object(o), "shares": shares[o]} for o in top],
            "n_shared_objects": len(objects),
            "object_kinds": dict(sorted(Counter(o.split(":", 1)[0] for o in objects).items())),
            "avg_time_delta_s": round(avg, 2),
            "edge_symmetry": round(statistics.fmean(e["symmetry"] for e in intra), 3),
            "first_seen": _iso(min(map(ref_time.get, refs))),
            "last_seen": _iso(max(map(ref_time.get, refs))),
            "window_s": window,
            "window_mode": mode,
            "min_weight": min_weight,
            "min_component_size": min_component_size,
            "n_iter": n_eff,
            "seed": int(seed),
            "null_model": null_model,
            "stratum_s": stratum_s if null_model == "block" else None,
            "null_statistic": "max repeated pairs in any component",
            "null_std": round(std, 3) if std is not None else None,
            "alpha": float(alpha),
            "significant": significant,
            "skipped_viral_objects": skipped,
            "n_skipped_viral_objects": len(ix.skipped),
            "pseudonym_keyed": bool(secret or os.environ.get(KEY_ENV)),
        }
        records.append((not significant, p if p is not None else 1.0, -(z or 0.0), -obs, -len(comp), accounts[0],
                        summary, obs, mean, z, p, refs, params))
    records.sort(key=lambda r: r[:6])
    return [make_signal("coordination", n, summary, value=float(obs),
                        baseline=round(mean, 3) if mean is not None else None,
                        z=round(z, 3) if z is not None else None, p_value=p, doc_refs=_spread(refs, 20), params=params,
                        detector=DETECTOR)
            for n, (*_, summary, obs, mean, z, p, refs, params) in enumerate(records[: max(0, int(max_signals))], 1)]


def coordination_graph(docs: list[Document], *, clusters: list[dict] | None = None, case: str = "default",
                       window_s: float | str = 60.0, min_weight: int = 2, secret: str | None = None,
                       public_accounts: list[str] | None = None, max_object_shares: int = 500,
                       **_detector_only: Any) -> dict:
    """Pseudonymised co-share network over edges of weight ``>= min_weight`` (for later GEXF export).

    ``{'nodes': [{'id', 'platform', 'shares', 'component', 'public'}], 'edges': [{'source',
    'target', 'weight', 'avg_delta_s', 'symmetry'}], 'params': {...}}``. ``component`` is the
    1-based rank of the node's component by size. Detector-only keywords (``n_iter``,
    ``alpha``...) are accepted and ignored, so ``**settings['coordination']`` can be passed.
    """
    min_weight = max(1, int(min_weight))
    ix, window, mode, public = _prepare(docs, clusters, case, secret, public_accounts, window_s, max_object_shares)
    edges = {k: e for k, e in _edge_details(ix).items() if e["weight"] >= min_weight}
    comp_of = {m: n for n, c in enumerate(components(edges, min_weight=min_weight), 1) for m in c}
    acct_id = {a: i for i, a in enumerate(ix.accts)}
    posts = Counter(ix.doc_acct)
    label = lambda m: public.get(m, m)  # noqa: E731
    nodes = [{"id": label(m), "platform": ix.acct_platform[acct_id[m]],
              "shares": posts[acct_id[m]], "component": comp_of[m], "public": m in public}
             for m in comp_of]
    out_edges = [{"source": src, "target": dst, "weight": e["weight"],
                  "avg_delta_s": round(statistics.fmean(e["deltas"]), 2), "symmetry": round(e["symmetry"], 3)}
                 for (a, b), e in edges.items() for src, dst in [sorted((label(a), label(b)))]]
    return {
        "nodes": sorted(nodes, key=lambda n: (n["component"], n["id"])),
        "edges": sorted(out_edges, key=lambda e: (e["source"], e["target"])),
        "params": {"window_s": window, "window_mode": mode, "min_weight": min_weight,
                   "max_object_shares": int(max_object_shares), "n_nodes": len(nodes), "n_edges": len(out_edges),
                   "n_components": len(set(comp_of.values())),
                   "skipped_viral_objects": [{"object": display_object(o), "shares": n} for o, n in ix.skipped[:10]],
                   "pseudonym_keyed": bool(secret or os.environ.get(KEY_ENV)), "detector": DETECTOR},
    }


__all__ = [
    "COORD_VERSION", "DETECTOR", "KEY_ENV", "MAX_NULL_WORK", "case_key", "pseudonymize", "canonical_url",
    "display_object", "share_objects", "share_events", "estimate_window", "co_share_edges", "components",
    "null_distribution", "detect_coordination", "coordination_graph",
]
