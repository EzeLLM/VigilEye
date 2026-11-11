"""Computed evidence signals: deterministic measurements that judgments can cite.

The influence, narrative and community lenses used to ask the model to *infer*
copypasta, bursts and coordination from chunks it happened to read. This module
adds a measured evidence layer instead (no LLM involved):

* one shared **signal record** (plain dict, see :func:`make_signal`) that every
  detector emits and that judgments cite exactly like documents, e.g.
  ``[sig:copypasta#1]``;
* **near-duplicate / copypasta clustering** (:func:`near_duplicate_clusters`):
  exact matches on normalised text, MinHash-LSH near duplicates verified by
  Jaccard similarity, and OSoMe-style hashtag-sequence traces;
* **collapse** of templated floods to one representative plus a copy count
  (:func:`collapse_near_duplicates`) so one template cannot dominate the map
  phase or the token budget;
* an orchestrator (:func:`compute_signals`) that also runs the temporal-burst and
  coordination detectors when those modules are installed.

Privacy: clusters and signals carry document refs and *counts* of accounts only,
never author names; normalisation strips @handles, ``u/`` names, e-mail
addresses and URLs from the text samples. Samples are collected, untrusted text
and must only be shown to a model inside a block framed as data.
"""
from __future__ import annotations

import hashlib
import heapq
import logging
import math
import operator
import re
import struct
import unicodedata
from collections import defaultdict
from datetime import timezone
from itertools import combinations
from typing import Any, Iterable

from ..models import Document
from .chunking import doc_ref

log = logging.getLogger("vigileye.analysis")

SIGNALS_VERSION = "1"
DETECTOR = f"signals@{SIGNALS_VERSION}"

_KIND_RE = re.compile(r"^[a-z0-9_.-]+$")
_METHOD_RANK = {"exact": 0, "minhash": 1, "tagseq": 2}
_MAX64 = (1 << 64) - 1


# --------------------------------------------------------------------------- signal record
def _opt_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _one_line(text: Any, limit: int) -> str:
    s = " ".join(str(text or "").split())
    return s if len(s) <= limit else s[: limit - 1].rstrip() + "…"


def make_signal(kind: str, n: int, summary: str, *, value: float, doc_refs: list[str],
                baseline: float | None = None, z: float | None = None, p_value: float | None = None,
                params: dict | None = None, detector: str = DETECTOR) -> dict:
    """Build the shared signal record. ``id`` is ``sig:<kind>#<n>`` (citable like a doc ref).

    Non-finite numbers become ``None`` so the record always serialises to valid JSON.
    """
    if not _KIND_RE.match(kind or ""):
        raise ValueError(f"signal kind must match [a-z0-9_.-]+, got {kind!r}")
    return {
        "id": f"sig:{kind}#{int(n)}",
        "kind": kind,
        "summary": _one_line(summary, 400),
        "value": _opt_float(value) or 0.0,
        "baseline": _opt_float(baseline),
        "z": _opt_float(z),
        "p_value": _opt_float(p_value),
        "doc_refs": list(dict.fromkeys(str(r) for r in (doc_refs or []))),
        "params": dict(params or {}),
        "detector": str(detector),
    }


# --------------------------------------------------------------------------- normalisation
_URL_RE = re.compile(r"(?:\b(?:https?|hxxps?|ftp)://|\bwww\.)\S+", re.I)
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_HANDLE_RE = re.compile(r"(?<![\w@])@[\w.-]+(?:@[\w-]+(?:\.[\w-]+)+)?")   # @user, @a.bsky.social, @u@host
_REDDIT_USER_RE = re.compile(r"(?<![\w/])/?u/[\w-]{2,}", re.I)
# Mirrors enrich._HASHTAG_RE, but also accepts one-letter tags.
_HASHTAG_RE = re.compile(r"(?<![\w&#/])#([^\W\d]\w{0,138})")


class _StripTable(dict):
    """Lazy ``str.translate`` table: punctuation/symbols/controls -> space, invisible format chars -> removed."""

    def __missing__(self, cp: int) -> str | None:
        ch = chr(cp)
        cat = unicodedata.category(ch)
        if cat == "Cf" or 0xFE00 <= cp <= 0xFE0F or 0xE0100 <= cp <= 0xE01EF or cp == 0x20E3:
            rep: str | None = None          # ZWJ/ZWSP/bidi marks, variation selectors, keycap: evasion noise
        elif cat[0] in "PS" or cat in ("Cc", "Cs", "Co", "Cn"):
            rep = " "                        # punctuation, emoji & other symbols (So/Sk/Sm/Sc), controls
        else:
            rep = ch
        self[cp] = rep
        return rep


_STRIP = _StripTable()


def normalize_text(text: str) -> str:
    """Canonical form for duplicate detection.

    NFKC (folds fullwidth / math-bold evasion), casefold, then drop URLs, e-mail
    addresses, @handles and ``u/`` names, emoji and other symbols, punctuation and
    invisible format characters; whitespace is collapsed. Idempotent.
    """
    if not text:
        return ""
    s = unicodedata.normalize("NFKC", str(text)).casefold()
    s = _URL_RE.sub(" ", s)
    s = _EMAIL_RE.sub(" ", s)
    s = _HANDLE_RE.sub(" ", s)
    s = _REDDIT_USER_RE.sub(" ", s)
    return " ".join(s.translate(_STRIP).split())


def _shingles(norm: str, k: int) -> set[str]:
    words = norm.split()
    if len(words) >= k:
        return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}
    if len(norm) <= 5:
        return {norm} if norm else set()
    return {norm[i:i + 5] for i in range(len(norm) - 4)}


def _h64(token: str) -> int:
    return int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "little")


def _bottom_k(sh: set[str], k: int) -> set[str]:
    """Consistent sample of a large shingle set: the ``k`` shingles with the smallest blake2b-64 hash.

    Near-identical long texts keep near-identical samples, so Jaccard is preserved
    while MinHash cost and memory stay bounded for long articles.
    """
    return sh if k <= 0 or len(sh) <= k else set(heapq.nsmallest(k, sh, key=_h64))


def shingles(text: str, k: int = 5) -> set[str]:
    """Word ``k``-gram shingles of the normalised text; character 5-grams when it has fewer than ``k`` words."""
    return _shingles(normalize_text(text), max(1, int(k)))


# --------------------------------------------------------------------------- MinHash
class MinHasher:
    """MinHash signatures over ``num_perm`` independent, seeded 64-bit hash functions.

    Each distinct token is hashed once with SHAKE-128 keyed by ``seed``; the XOF
    stream yields one 64-bit value per permutation and the signature is the
    element-wise minimum (computed in C via ``zip``/``min``). This replaces the
    textbook ``(a*x+b) mod (2**61-1)`` loop, which costs ~25 µs per token in pure
    Python against ~6 µs here, and uses fully random rather than 2-universal hash
    functions. Signatures are stable across processes and Python versions
    (no reliance on ``hash()``).
    """

    def __init__(self, num_perm: int = 128, seed: int = 1):
        if int(num_perm) < 1:
            raise ValueError("num_perm must be >= 1")
        self.num_perm = int(num_perm)
        self.seed = int(seed)
        self._salt = hashlib.blake2b(f"vigileye.minhash:{self.seed}".encode(), digest_size=16).digest()
        self._unpack = struct.Struct(f"<{self.num_perm}Q").unpack
        self._nbytes = 8 * self.num_perm

    def signature(self, tokens: Iterable[str]) -> tuple[int, ...]:
        salt, unpack, nbytes = self._salt, self._unpack, self._nbytes
        rows = [unpack(hashlib.shake_128(salt + t.encode("utf-8")).digest(nbytes)) for t in set(tokens)]
        if not rows:
            return (_MAX64,) * self.num_perm
        return tuple(map(min, zip(*rows)))

    @staticmethod
    def jaccard(a: tuple[int, ...], b: tuple[int, ...]) -> float:
        """Estimated Jaccard similarity: the share of positions where two signatures agree."""
        if len(a) != len(b):
            raise ValueError("signatures differ in length")
        return sum(map(operator.eq, a, b)) / len(a) if a else 0.0


# --------------------------------------------------------------------------- hashtags
def hashtag_sequence_key(doc: Document, min_len: int = 3) -> tuple[str, ...] | None:
    """Ordered, casefolded hashtags of ``doc.text`` (repeats dropped), or ``None`` if fewer than ``min_len``.

    Falls back to ``doc.tags`` entries written as hashtags (``#name``, as the Mastodon
    and Bluesky collectors store them), so RSS categories or Reddit flair never count.
    """
    text = _URL_RE.sub(" ", unicodedata.normalize("NFKC", doc.text or ""))
    tags = [m.group(1).casefold() for m in _HASHTAG_RE.finditer(text)]
    if len(set(tags)) < min_len:
        tags = [unicodedata.normalize("NFKC", str(t)).lstrip("#").casefold()
                for t in (doc.tags or []) if str(t).startswith("#") and len(str(t)) > 1]
    key = tuple(dict.fromkeys(tags))
    return key if len(key) >= max(1, int(min_len)) else None


# --------------------------------------------------------------------------- clustering
def _chrono_key(d: Document) -> tuple:
    """Earliest first; ties -> most engaged; then ref. Undated documents sort last."""
    ts = d.created_at.timestamp() if d.created_at else 0.0
    return (d.created_at is None, ts, -d.engagement(), doc_ref(d))


def _iso(dt) -> str | None:
    return dt.astimezone(timezone.utc).isoformat() if dt else None


def _record(method: str, docs: list[Document], sample: str, params: dict, **extra: Any) -> dict:
    ds = sorted(docs, key=_chrono_key)
    dated = [d.created_at for d in ds if d.created_at]
    first, last = (min(dated), max(dated)) if dated else (None, None)
    return {
        "cluster_id": "",
        "method": method,
        "representative": doc_ref(ds[0]),
        "members": [doc_ref(d) for d in ds],
        "size": len(ds),
        "first_seen": _iso(first),
        "last_seen": _iso(last),
        "span_seconds": (last - first).total_seconds() if dated else None,
        "n_platforms": len({d.platform for d in ds}),
        "n_sources": len({(d.platform, d.source) for d in ds}),
        "n_accounts": len({(d.platform, d.author) for d in ds if d.author}),
        "sample": sample[:200],
        **extra,
        "params": dict(params),
    }


def _content_norm(d: Document) -> str:
    """Normalised body text; the title is used only when the body is empty (Reddit comment titles
    are the thread title, so preferring them would cluster whole threads)."""
    return normalize_text(d.text) or normalize_text(d.title)


def near_duplicate_clusters(docs: list[Document], *, threshold: float = 0.8, num_perm: int = 128, bands: int = 32,
                            min_chars: int = 40, min_size: int = 2, shingle_k: int = 3, min_exact_chars: int = 20,
                            tagseq_min_len: int = 3, max_shingles: int = 512, seed: int = 1) -> list[dict]:
    """Cluster exact copies, near duplicates and shared hashtag sequences.

    1. **exact**: sha256 of :func:`normalize_text` (texts of at least ``min_exact_chars``).
    2. **minhash**: texts of at least ``min_chars`` are shingled into word ``shingle_k``-grams
       and MinHash-LSH banded (``bands`` x ``num_perm // bands`` rows); each candidate pair is
       verified by the *exact* Jaccard of its shingle sets (>= ``threshold``) and merged by
       union-find (single linkage). Word 3-grams are the default because with 5-grams one
       substituted word in a 30-word post already drops Jaccard to ~0.68. Texts with more than
       ``max_shingles`` shingles are represented by a consistent bottom-k sample of them.
    3. **tagseq**: documents sharing the same ordered run of >= ``tagseq_min_len`` hashtags,
       kept only when the run spans more than one text cluster (otherwise it adds nothing).

    Returns cluster dicts (``cluster_id`` ``nd1``...) sorted by ``(-size, first_seen,
    representative)``. The result is independent of input order. ``n_accounts`` is a count;
    author names never leave this function.
    """
    if not 0.0 < float(threshold) <= 1.0:
        raise ValueError("threshold must be in (0, 1]")
    num_perm = max(1, int(num_perm))
    bands = max(1, min(int(bands), num_perm))
    rows = num_perm // bands
    min_size = max(2, int(min_size))
    params = {"threshold": float(threshold), "num_perm": num_perm, "bands": bands, "rows": rows,
              "shingle_k": int(shingle_k), "min_chars": int(min_chars), "min_exact_chars": int(min_exact_chars),
              "max_shingles": int(max_shingles), "seed": int(seed), "verify": "exact_jaccard", "detector": DETECTOR}

    items: dict[str, Document] = {}
    for d in docs or []:
        items.setdefault(doc_ref(d), d)

    # Pass 1: units of identical normalised text, in canonical (hash) order.
    units: dict[str, list[Document]] = defaultdict(list)
    unit_text: dict[str, str] = {}
    for d in items.values():
        norm = _content_norm(d)
        if len(norm) < int(min_exact_chars):
            continue
        key = hashlib.sha256(norm.encode("utf-8")).hexdigest()
        units[key].append(d)
        unit_text[key] = norm
    keys = sorted(units)
    parent = list(range(len(keys)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    # Pass 2: MinHash LSH over units long enough to be meaningful.
    hasher = MinHasher(num_perm, seed)
    shingle_sets: dict[int, set[str]] = {}
    buckets: dict[tuple, list[int]] = defaultdict(list)
    for i, key in enumerate(keys):
        text = unit_text[key]
        if len(text) < int(min_chars):
            continue
        sh = _bottom_k(_shingles(text, max(1, int(shingle_k))), int(max_shingles))
        if not sh:
            continue
        shingle_sets[i] = sh
        sig = hasher.signature(sh)
        for b in range(bands):
            buckets[(b, sig[b * rows:(b + 1) * rows])].append(i)

    def try_union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri == rj:
            return
        a, b = shingle_sets[i], shingle_sets[j]
        inter = len(a & b)
        if inter and inter / (len(a) + len(b) - inter) >= threshold:
            parent[max(ri, rj)] = min(ri, rj)

    for members in buckets.values():
        if len(members) < 2:
            continue
        if len(members) <= 16:
            for i, j in combinations(members, 2):
                try_union(i, j)
        else:  # bound the work in giant buckets: star from the first member plus a chain
            for n in range(1, len(members)):
                try_union(members[0], members[n])
                try_union(members[n - 1], members[n])

    components: dict[int, list[int]] = defaultdict(list)
    for i in range(len(keys)):
        components[find(i)].append(i)

    clusters: list[dict] = []
    comp_of: dict[str, str] = {}
    for root, idxs in components.items():
        cdocs = [d for i in idxs for d in units[keys[i]]]
        for d in cdocs:
            comp_of[doc_ref(d)] = f"u{root}"
        if len(cdocs) < min_size:
            continue
        rep = min(cdocs, key=_chrono_key)
        clusters.append(_record("exact" if len(idxs) == 1 else "minhash", cdocs, _content_norm(rep), params))

    # Pass 3: hashtag-sequence traces.
    seqs: dict[tuple[str, ...], list[Document]] = defaultdict(list)
    for d in items.values():
        key = hashtag_sequence_key(d, tagseq_min_len)
        if key:
            seqs[key].append(d)
    for seq, ds in seqs.items():
        if len(ds) < min_size or len({comp_of.get(doc_ref(d), doc_ref(d)) for d in ds}) < 2:
            continue
        clusters.append(_record("tagseq", ds, " ".join(f"#{t}" for t in seq), params, hashtags=list(seq)))

    clusters.sort(key=lambda c: (-c["size"], c["first_seen"] is None, c["first_seen"] or "",
                                 c["representative"], _METHOD_RANK[c["method"]]))
    for n, c in enumerate(clusters, 1):
        c["cluster_id"] = f"nd{n}"
    return clusters


def collapse_near_duplicates(docs: list[Document], clusters: list[dict], *,
                             min_size: int = 3) -> tuple[list[Document], int]:
    """Keep one representative per exact/minhash cluster of ``>= min_size`` documents.

    The representative gets ``metrics['near_dup_copies'] = size - 1`` (rendered in the
    prompt header) and ``meta['near_dup_cluster']``; every other member gets
    ``meta['near_dup_of']`` and ``meta['near_dup_cluster']`` and is left out of the
    returned list. Hashtag-sequence clusters are never collapsed (different texts).
    Stale marks from an earlier call are cleared first. Returns ``(kept_docs, n_collapsed)``
    with ``kept_docs`` in input order.
    """
    by_ref: dict[str, Document] = {}
    for d in docs:
        by_ref.setdefault(doc_ref(d), d)
        d.meta.pop("near_dup_of", None)
        d.meta.pop("near_dup_cluster", None)
        d.metrics.pop("near_dup_copies", None)
    removed: set[str] = set()
    for c in clusters or []:
        if c.get("method") not in ("exact", "minhash") or int(c.get("size", 0)) < int(min_size):
            continue
        rep = by_ref.get(c.get("representative"))
        if rep is None or doc_ref(rep) in removed:
            continue
        cid = c.get("cluster_id")
        rep.metrics["near_dup_copies"] = int(c["size"]) - 1
        rep.meta["near_dup_cluster"] = cid
        for ref in c.get("members") or []:
            d = by_ref.get(ref)
            if d is None or d is rep or ref in removed:
                continue
            d.meta["near_dup_of"] = doc_ref(rep)
            d.meta["near_dup_cluster"] = cid
            removed.add(ref)
    kept, seen = [], set()
    for d in docs:
        ref = doc_ref(d)
        if ref in removed or ref in seen:
            continue
        seen.add(ref)
        kept.append(d)
    return kept, len(removed)


# --------------------------------------------------------------------------- signals
def _span_text(seconds: float | None) -> str:
    if seconds is None:
        return ""
    s = int(round(seconds))
    if s < 60:
        return f"{s}s"
    m, h, d = s // 60, s // 3600, s // 86400
    if s < 3600:
        return f"{m}m"
    if s < 86400:
        return f"{h}h {m % 60}m"
    return f"{d}d {h % 24}h"


def copypasta_signals(clusters: list[dict], *, min_size: int = 3, max_signals: int = 20) -> list[dict]:
    """One ``copypasta`` signal per cluster of ``>= min_size`` documents (largest first), ``value`` = size."""
    out: list[dict] = []
    for c in clusters or []:
        if int(c.get("size", 0)) < int(min_size):
            continue
        if len(out) >= int(max_signals):
            break
        size, method = int(c["size"]), c.get("method", "")
        span = _span_text(c.get("span_seconds"))
        where = (f"from {c.get('n_accounts', 0)} account(s) on {c.get('n_platforms', 0)} platform(s) / "
                 f"{c.get('n_sources', 0)} source(s)" + (f" within {span}" if span else ""))
        sample = _one_line(c.get("sample", ""), 90)
        if method == "tagseq":
            summary = f"{size} posts with differing text share the hashtag sequence {sample} {where}"
        else:
            kind = "identical" if method == "exact" else (
                f"near-identical (Jaccard >= {c.get('params', {}).get('threshold', 0.8):g})")
            summary = f'{size} {kind} posts {where}; sample: "{sample}"'
        params = {k: c.get(k) for k in ("cluster_id", "method", "size", "representative", "first_seen", "last_seen",
                                        "span_seconds", "n_platforms", "n_sources", "n_accounts")}
        params.update(c.get("params") or {})
        out.append(make_signal("copypasta", len(out) + 1, summary, value=float(size),
                               doc_refs=list(c.get("members") or []), params=params))
    return out


def _run_detector(name: str, module: str, func: str, call, errors: list[str], skipped: list[str]) -> list[dict]:
    """Import ``vigileye.analysis.<module>.<func>`` and run it; a missing module is skipped silently."""
    qual = f"{__package__}.{module}"
    try:
        mod = __import__(qual, fromlist=[func])
        fn = getattr(mod, func)
    except ModuleNotFoundError as e:
        if e.name in (qual, None) or qual.startswith(str(e.name) + "."):
            skipped.append(name)
            return []
        errors.append(f"signals: {name} detector import failed: {type(e).__name__}: {e}")
        return []
    except (ImportError, AttributeError) as e:
        errors.append(f"signals: {name} detector unavailable: {type(e).__name__}: {e}")
        return []
    try:
        res = call(fn)
    except Exception as e:  # noqa: BLE001 - a detector must never break the run
        log.warning("%s detector failed: %s", name, e)
        errors.append(f"signals: {name} detector failed: {type(e).__name__}: {e}")
        return []
    if isinstance(res, dict):
        errors.extend(str(x) for x in res.get("errors") or [])
        res = res.get("signals") or []
    good = [s for s in (res or []) if isinstance(s, dict) and str(s.get("id", "")).startswith("sig:")]
    if len(good) != len(res or []):
        errors.append(f"signals: {name} detector returned {len(res or []) - len(good)} malformed signal(s)")
    return good


def detector_options(value: Any) -> dict:
    """A detector setting (``analysis.bursts`` / ``analysis.coordination``) as kwargs.

    ``None`` and ``True`` mean "on, with defaults"; a mapping is passed through; anything else
    raises :class:`ValueError` (``False`` is handled by callers: it disables the detector).
    """
    if value is None or value is True:
        return {}
    if isinstance(value, dict):
        return dict(value)
    raise ValueError(f"expected true/false or a mapping of options, got {type(value).__name__} {value!r}")


def compute_signals(docs: list[Document], *, settings: dict | None = None, case: str = "default") -> dict:
    """Run every available deterministic detector over ``docs`` (no LLM).

    ``settings`` is typically the engine's settings dict; keys used: ``near_dup_threshold``
    (0.8), ``near_dup_min_size`` (3), ``near_dup`` (extra kwargs for
    :func:`near_duplicate_clusters`), ``max_copypasta_signals`` (20), ``bursts`` and
    ``coordination`` (kwargs for those detectors; ``False`` disables one).

    Returns ``{"signals", "clusters", "errors", "detectors": {name: n_signals}, "skipped",
    "version"}``. Every detector runs in its own ``try`` so a failure is reported in
    ``errors`` and never raised.
    """
    s = dict(settings or {})
    docs = list(docs or [])
    errors: list[str] = []
    skipped: list[str] = []
    signals: list[dict] = []
    clusters: list[dict] = []
    detectors = {"copypasta": 0, "burst": 0, "coordination": 0}
    try:
        clusters = near_duplicate_clusters(docs, threshold=float(s.get("near_dup_threshold", 0.8)),
                                           **dict(s.get("near_dup") or {}))
        cp = copypasta_signals(clusters, min_size=int(s.get("near_dup_min_size", 3)),
                               max_signals=int(s.get("max_copypasta_signals", 20)))
        signals += cp
        detectors["copypasta"] = len(cp)
    except Exception as e:  # noqa: BLE001
        log.warning("copypasta detector failed: %s", e)
        errors.append(f"signals: copypasta detector failed: {type(e).__name__}: {e}")

    plan = (
        ("burst", "temporal", "detect_bursts", "bursts", {}),
        ("coordination", "coordination", "detect_coordination", "coordination", {"case": case}),
    )
    for name, module, func, key, extra in plan:
        opts = s.get(key, {})
        if opts is False:
            skipped.append(name)
            continue
        try:
            kwargs = {**extra, **detector_options(opts)}
        except ValueError as e:
            errors.append(f"signals: {name} detector not run: {e}")
            skipped.append(name)
            continue
        found = _run_detector(name, module, func, lambda fn, kw=kwargs: fn(docs, clusters=clusters, **kw),
                              errors, skipped)
        seen = {x["id"] for x in signals}
        fresh = [x for x in found if x["id"] not in seen]
        if len(fresh) != len(found):
            errors.append(f"signals: {name} detector returned {len(found) - len(fresh)} duplicate signal id(s)")
        signals += fresh
        detectors[name] = len(fresh)
    return {"signals": signals, "clusters": clusters, "errors": errors, "detectors": detectors,
            "skipped": skipped, "version": SIGNALS_VERSION}


# --------------------------------------------------------------------------- rendering
def signals_for_refs(signals: list[dict], refs: set[str], limit: int = 8) -> list[dict]:
    """Signals whose documents overlap ``refs`` (e.g. one chunk), most overlap first, stable otherwise."""
    refs = set(refs or ())
    scored = []
    for i, sig in enumerate(signals or []):
        hit = len(refs.intersection(sig.get("doc_refs") or ()))
        if hit:
            scored.append((-hit, i, sig))
    return [sig for _, _, sig in sorted(scored, key=lambda t: t[:2])[: max(0, int(limit))]]


def _num(x: Any) -> str:
    if x is None:
        return ""
    f = float(x)
    return str(int(f)) if f.is_integer() and abs(f) < 1e15 else f"{f:.3g}"


def _docs_text(refs: list[str], n: int = 3) -> str:
    refs = list(refs or [])
    more = f" (+{len(refs) - n} more)" if len(refs) > n else ""
    return ", ".join(refs[:n]) + more


def render_signals_block(signals: list[dict], limit: int = 15) -> str:
    """Prompt block, one line per signal::

        [sig:copypasta#1] (copypasta) <summary> | value=10 z=3.1 p=0.002 | docs: a, b, c (+7 more)

    Statistics that are ``None`` are omitted. Summaries are single-lined and bounded.
    """
    lines = []
    for sig in (signals or [])[: max(0, int(limit))]:
        stats = [f"value={_num(sig.get('value'))}"]
        for label, key in (("baseline", "baseline"), ("z", "z"), ("p", "p_value")):
            if sig.get(key) is not None:
                stats.append(f"{label}={_num(sig[key])}")
        lines.append(f"[{sig.get('id')}] ({sig.get('kind')}) {_one_line(sig.get('summary'), 300)} | "
                     f"{' '.join(stats)} | docs: {_docs_text(sig.get('doc_refs'))}")
    return "\n".join(lines)


def _md_cell(x: Any) -> str:
    return " ".join(str(x).split()).replace("|", "\\|")


def signals_markdown(signals: list[dict], limit: int = 40) -> str:
    """Markdown table ``ID | Kind | Summary | Value | Baseline | z | p | Docs`` for reports and the CLI."""
    signals = list(signals or [])
    if not signals:
        return "_No computed signals._"
    lines = ["_Deterministic measurements computed without an LLM; judgments cite them as `[sig:<kind>#<n>]`._", "",
             "| ID | Kind | Summary | Value | Baseline | z | p | Docs |", "|---|---|---|---|---|---|---|---|"]
    for sig in signals[: max(0, int(limit))]:
        cells = [sig.get("id"), sig.get("kind"), _one_line(sig.get("summary"), 300), _num(sig.get("value")),
                 _num(sig.get("baseline")), _num(sig.get("z")), _num(sig.get("p_value")),
                 _docs_text(sig.get("doc_refs"))]
        lines.append("| " + " | ".join(_md_cell(c) for c in cells) + " |")
    if len(signals) > limit:
        lines += ["", f"_{len(signals) - limit} more signal(s) not shown; see `meta.computed_signals`._"]
    return "\n".join(lines)


__all__ = [
    "SIGNALS_VERSION", "DETECTOR", "make_signal", "normalize_text", "shingles", "MinHasher",
    "hashtag_sequence_key", "near_duplicate_clusters", "collapse_near_duplicates", "copypasta_signals",
    "compute_signals", "signals_for_refs", "render_signals_block", "signals_markdown",
]
