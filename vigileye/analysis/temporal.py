"""Temporal burst and anomaly detection: when attention spiked, on what, against which baseline.

Deterministic, pure stdlib, no LLM. For each *target* (overall volume, a hashtag, an
outbound link host, a WS2 near-duplicate cluster) the documents are binned on a UTC-aligned
grid and scored with a **rolling robust z-score** (median / MAD of the preceding bins, so
one spike cannot inflate its own baseline and there is no look-ahead). A run of hot bins is
reported as a burst only when it is also **confirmed by Kleinberg's two-state automaton**
over the target's inter-arrival gaps (targets with at least ``kleinberg_min_events``
events), which filters noise in sparse series.

Each burst becomes a citable signal ``sig:burst#n`` (the WS2 record from
:func:`vigileye.analysis.signals.make_signal`) whose params carry aggregate cues the
critic and ACH can use to weigh an *organic* surge against an *injected* one:
``n_accounts`` (a count, never names), ``pre_burst_events`` (did the target exist before
it spiked?), ``platforms`` and ``at_window_edge`` (a spike at the end of the collection
window may be a recency artefact of the collector).

Privacy and safety: author names never leave this module (only distinct-account counts);
summaries contain only computed numbers, casefolded ``#hashtags``, defanged host names that
passed a strict hostname check and validated cluster ids, never collected free text.
"""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Iterable
from urllib.parse import urlsplit

from ..models import Document
from .chunking import doc_ref
from .enrich import defang, refang

TEMPORAL_VERSION = "1"
DETECTOR = f"temporal@{TEMPORAL_VERSION}"

try:  # WS2: the shared signal record
    from .signals import make_signal
except ImportError:  # pragma: no cover - WS2 not installed: minimal compatible stand-in
    def make_signal(kind: str, n: int, summary: str, *, value: float, doc_refs: list[str],
                    baseline: float | None = None, z: float | None = None, p_value: float | None = None,
                    params: dict | None = None, detector: str = DETECTOR) -> dict:
        return {"id": f"sig:{kind}#{int(n)}", "kind": kind, "summary": " ".join(str(summary).split())[:400],
                "value": float(value), "baseline": baseline, "z": z, "p_value": p_value,
                "doc_refs": list(dict.fromkeys(doc_refs or [])), "params": dict(params or {}),
                "detector": detector}

BIN_CHOICES = (300, 900, 3600, 10800, 21600, 86400)
TARGET_KINDS = ("volume", "hashtag", "domain", "near_dup")
_KIND_RANK = {k: i for i, k in enumerate(TARGET_KINDS)}
_MAD_SCALE = 1.4826            # MAD -> sigma for a normal distribution
_BLOCKS = "▁▂▃▄▅▆▇█"
_MAX_REFS = 15
MAX_BINS = 5000                # finer grids are coarsened, e.g. when one bad timestamp says 1970

# Same hashtag grammar as WS2 (signals._HASHTAG_RE): a letter first, so "#1" and "#2024" are not tags.
_TAG_RE = re.compile(r"(?<![\w&#/])#([^\W\d]\w{0,138})")
_STRIP_URL_RE = re.compile(r"(?:\b(?:https?|hxxps?|ftp)://|\bwww\.)\S+", re.I)
_LINK_RE = re.compile(r"\bhttps?://[^\s<>\"'`\])}|]+", re.I)
_LINK_TRAIL = ".,;:!?'\"*_~"
_HOST_OK = re.compile(r"^[\w-]+(?:\.[\w-]+)+$")
_CLUSTER_ID_OK = re.compile(r"^[\w.-]{1,40}$")
_PLATFORM_HOSTS = {
    "reddit": ("reddit.com", "redd.it", "redditmedia.com", "redditstatic.com"),
    "bluesky": ("bsky.app", "bsky.social", "bsky.network"),
    "hackernews": ("ycombinator.com",),
}
_PLACEHOLDER_AUTHORS = frozenset({"", "[deleted]", "[removed]", "deleted", "removed", "anonymous", "unknown"})


# --------------------------------------------------------------------------- time helpers
def _ts(dt: datetime) -> float:
    """Epoch seconds; naive datetimes are taken as UTC."""
    return (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).timestamp()


def _dt(t: float) -> datetime:
    return datetime.fromtimestamp(t, tz=timezone.utc)


def _iso(t: float) -> str:
    return _dt(t).isoformat()


def _unit(bin_s: int) -> str:
    """Bin width as a rate unit: ``h``, ``3 h``, ``day``, ``15 min``."""
    for size, one, many in ((86400, "day", "days"), (3600, "h", "h"), (60, "min", "min")):
        if bin_s % size == 0:
            k = bin_s // size
            return one if k == 1 else f"{k} {many}"
    return f"{bin_s} s"


def _duration(seconds: float) -> str:
    s = int(round(seconds))
    if s < 60:
        return "under a minute"
    if s < 3600:
        return f"{s // 60} min"
    if s < 86400:
        h, m = divmod(s // 60, 60)
        return f"{h} h" + (f" {m} min" if m else "")
    d, h = divmod(s // 3600, 24)
    return f"{d} d" + (f" {h} h" if h else "")


def _when(a: float, b: float) -> str:
    da, db = _dt(a), _dt(b)
    if da.date() == db.date():
        return f"{da:%Y-%m-%d %H:%M}-{db:%H:%M}Z"
    return f"{da:%Y-%m-%d %H:%M}Z to {db:%Y-%m-%d %H:%M}Z"


def _rate(count: float, bin_s: int) -> str:
    """``1/h`` for one-unit bins, ``3 per 3 h`` otherwise."""
    unit = _unit(bin_s)
    return f"{count:g}/{unit}" if " " not in unit else f"{count:g} per {unit}"


def _dated(docs: Iterable[Document] | None) -> list[tuple[float, str, Document]]:
    """``(epoch, ref, doc)`` for dated documents, de-duplicated by ref, sorted by time then ref."""
    seen: dict[str, Document] = {}
    for d in docs or []:
        if d is None or d.created_at is None:
            continue
        seen.setdefault(doc_ref(d), d)
    return sorted(((_ts(d.created_at), ref, d) for ref, d in seen.items()), key=lambda r: (r[0], r[1]))


# --------------------------------------------------------------------------- binning
def auto_bin_seconds(span_s: float, target_bins: int = 96) -> int:
    """Smallest bin in ``BIN_CHOICES`` (5 min ... 1 day) giving at most ``target_bins`` bins over ``span_s``.

    Spans too long for daily bins at ``target_bins`` still get daily bins (more of them).
    """
    try:
        span = float(span_s)
    except (TypeError, ValueError):
        span = 0.0
    if not math.isfinite(span) or span <= 0:
        return BIN_CHOICES[0]
    need = span / max(1, int(target_bins))
    return next((b for b in BIN_CHOICES if b >= need), BIN_CHOICES[-1])


def bin_counts(times: list[datetime], *, bin_s: int, start: datetime, end: datetime) -> list[int]:
    """Counts per bin ``[start + i*bin_s, start + (i+1)*bin_s)``; the bin holding ``end`` is the last one.

    Times outside ``[start, end]`` and ``None`` entries are ignored. ``end < start`` gives ``[]``.
    """
    b = int(bin_s)
    if b <= 0:
        raise ValueError("bin_s must be a positive number of seconds")
    s0, s1 = _ts(start), _ts(end)
    if s1 < s0:
        return []
    counts = [0] * (int((s1 - s0) // b) + 1)
    for t in times or []:
        if t is None:
            continue
        x = _ts(t)
        if s0 <= x <= s1:
            counts[int((x - s0) // b)] += 1
    return counts


def _grid(t_first: float, t_last: float, bin_s: int | None) -> tuple[int, float, int]:
    """``(bin_s, origin, n_bins)`` of an epoch-aligned UTC grid covering ``[t_first, t_last]``."""
    b = int(bin_s) if bin_s else auto_bin_seconds(t_last - t_first)
    if b <= 0:
        raise ValueError("bin_s must be a positive number of seconds")
    span = t_last - t_first
    if span / b >= MAX_BINS - 2:   # alignment adds at most two bins
        b = next((c for c in BIN_CHOICES if c > b and span / c < MAX_BINS - 2),
                 max(b, math.ceil(span / (MAX_BINS - 2) / 86400) * 86400))
    origin = math.floor(t_first / b) * b
    return b, origin, int((t_last - origin) // b) + 1


# --------------------------------------------------------------------------- robust z
def _med(sorted_vals: list[float]) -> float:
    n = len(sorted_vals)
    mid = n // 2
    return sorted_vals[mid] if n % 2 else (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0


def _med_mad(vals: list[float]) -> tuple[float, float]:
    m = _med(sorted(vals))
    return m, _med(sorted(abs(v - m) for v in vals))


def _robust_baseline(series: list[float], window: int, min_mad: float) -> tuple[list[float], list[float]]:
    """``(z, median)`` per bin; see :func:`rolling_robust_z`."""
    xs = [float(x) for x in series or []]
    if not xs:
        return [], []
    w = max(1, int(window))
    floor = max(float(min_mad), 1e-9)
    g_med, g_mad = _med_mad(xs)
    zs: list[float] = []
    meds: list[float] = []
    for t, x in enumerate(xs):
        m, d = (g_med, g_mad) if t < w else _med_mad(xs[t - w:t])
        zs.append((x - m) / (_MAD_SCALE * max(d, floor)))
        meds.append(m)
    return zs, meds


def rolling_robust_z(series: list[int], window: int = 24, min_mad: float = 0.5) -> list[float]:
    """Robust z-score of each bin against the median / MAD of the ``window`` bins before it.

    ``z_t = (x_t - median(x[t-window:t])) / (1.4826 * max(MAD, min_mad))``. The current bin is
    never part of its own baseline and later bins never are (no look-ahead). Warm-up bins
    (``t < window``) have no full history and use the whole-series median / MAD, which a
    single spike cannot move. The ``min_mad`` floor (and a 1e-9 hard floor) means a flat
    series scores 0 everywhere and nothing ever divides by zero.
    """
    return _robust_baseline(series, window, min_mad)[0]


# --------------------------------------------------------------------------- Kleinberg
def kleinberg_bursts(times_s: list[float], *, s: float = 2.0, gamma: float = 1.0,
                     span_s: float | None = None) -> list[dict]:
    """Kleinberg (2002) two-state burst automaton, decoded exactly with Viterbi.

    Inter-arrival gaps are exponential with rate ``a0 = n/T`` in the base state and
    ``a1 = s*a0`` in the burst state; entering the burst state costs ``gamma * ln n`` and
    leaving it is free (``n`` = number of gaps). ``T`` is ``span_s`` when given (the
    observation window, so a target that only exists inside a spike is still measured
    against the whole collection) and otherwise the span of ``times_s``.

    Returns ``[{"start_idx", "end_idx", "level": 1}]``: inclusive indices into the
    *sorted* times of each maximal run of burst-state gaps. Fewer than two finite times
    or a non-positive ``T`` give ``[]``.
    """
    if not float(s) > 1.0:
        raise ValueError("s must be > 1")
    if float(gamma) < 0:
        raise ValueError("gamma must be >= 0")
    ts = sorted(t for t in (float(x) for x in times_s or []) if math.isfinite(t))
    n = len(ts) - 1
    if n < 1:
        return []
    span = (ts[-1] - ts[0]) if span_s is None else float(span_s)
    if not (math.isfinite(span) and span > 0):
        return []
    a0 = n / span
    a1 = float(s) * a0
    l0, l1 = math.log(a0), math.log(a1)
    up = float(gamma) * math.log(n)
    c0, c1 = 0.0, up                    # the automaton starts in the base state
    back0, back1 = bytearray(n), bytearray(n)
    for k in range(n):
        x = ts[k + 1] - ts[k]
        if c1 < c0:
            n0, back0[k] = c1, 1
        else:
            n0 = c0
        if c0 + up < c1:
            n1 = c0 + up
        else:
            n1, back1[k] = c1, 1
        c0, c1 = n0 + a0 * x - l0, n1 + a1 * x - l1
    state = 1 if c1 < c0 else 0
    states = bytearray(n)
    for k in range(n - 1, -1, -1):
        states[k] = state
        state = back1[k] if state else back0[k]
    out: list[dict] = []
    k = 0
    while k < n:
        if states[k]:
            a = k
            while k + 1 < n and states[k + 1]:
                k += 1
            out.append({"start_idx": a, "end_idx": k + 1, "level": 1})
        k += 1
    return out


# --------------------------------------------------------------------------- targets
def _hashtags(doc: Document) -> set[str]:
    """Casefolded hashtags in the text (the title only when the text is empty) and ``#``-prefixed ``doc.tags``.

    Bare tags (RSS categories, Reddit flair, HN item types) are not hashtags and are skipped,
    as in WS2's hashtag-sequence keys.
    """
    text = _STRIP_URL_RE.sub(" ", unicodedata.normalize("NFKC", doc.text or doc.title or ""))
    out = {m.group(1).casefold() for m in _TAG_RE.finditer(text)}
    for t in doc.tags or []:
        m = _TAG_RE.fullmatch(unicodedata.normalize("NFKC", str(t)).strip())
        if m:
            out.add(m.group(1).casefold())
    return out


def _host(url: str) -> str:
    try:
        host = (urlsplit(url).hostname or "").rstrip(".")
    except ValueError:
        return ""
    return host[4:] if host.startswith("www.") else host


def _link_hosts(doc: Document) -> set[str]:
    """Outbound link hosts in the text (title only when the text is empty), defanged notation included.

    The platform's own hosts and the host of ``doc.url`` (a Mastodon instance, a news site's
    own links) are excluded; only names passing a strict hostname check are kept.
    """
    own = _PLATFORM_HOSTS.get(doc.platform, ()) + ((_host(doc.url),) if doc.url else ())
    own = tuple(h for h in own if h)
    out: set[str] = set()
    for raw in _LINK_RE.findall(refang(doc.text or doc.title or "")):
        host = _host(raw.rstrip(_LINK_TRAIL))
        if host and _HOST_OK.match(host) and not any(host == h or host.endswith("." + h) for h in own):
            out.add(host)
    return out


def _target_members(dated: list[tuple[float, str, Document]], kinds: tuple[str, ...],
                    clusters: list[dict] | None) -> dict[tuple[str, str], list[int]]:
    """``(kind, target) -> ascending indices into dated``."""
    members: dict[tuple[str, str], list[int]] = defaultdict(list)
    if "volume" in kinds:
        members[("volume", "all")] = list(range(len(dated)))
    want_tags, want_hosts = "hashtag" in kinds, "domain" in kinds
    if want_tags or want_hosts:
        for i, (_, _, d) in enumerate(dated):
            if want_tags:
                for tag in _hashtags(d):
                    members[("hashtag", "#" + tag)].append(i)
            if want_hosts:
                for host in _link_hosts(d):
                    members[("domain", host)].append(i)
    if "near_dup" in kinds and clusters:
        pos = {ref: i for i, (_, ref, _) in enumerate(dated)}
        for n, c in enumerate(clusters, 1):
            if not isinstance(c, dict) or c.get("method", "minhash") not in ("exact", "minhash"):
                continue
            cid = str(c.get("cluster_id") or "")
            cid = cid if _CLUSTER_ID_OK.match(cid) else f"cluster{n}"
            idx = sorted({pos[str(r)] for r in c.get("members") or [] if str(r) in pos})
            if idx:
                members[("near_dup", cid)] = idx
    return members


def _label(kind: str, target: str) -> str:
    if kind == "volume":
        return "All collected posts"
    if kind == "domain":
        return f"Links to {defang(target, 'domain')}"
    if kind == "near_dup":
        return f"Near-duplicate cluster {target}"
    return target


def _runs(counts: list[int], zs: list[float], z_threshold: float, min_events: int) -> list[tuple[int, int]]:
    """Maximal ``[a, b]`` (inclusive) runs of bins with ``z >= z_threshold`` and ``count >= min_events``."""
    out: list[tuple[int, int]] = []
    a = None
    for t, (c, z) in enumerate(zip(counts, zs)):
        hot = c >= min_events and z >= z_threshold
        if hot and a is None:
            a = t
        elif not hot and a is not None:
            out.append((a, t - 1))
            a = None
    if a is not None:
        out.append((a, len(counts) - 1))
    return out


def _accounts(docs: Iterable[Document]) -> int:
    return len({(d.platform, str(d.author).strip().casefold()) for d in docs
                if d.author and str(d.author).strip().casefold() not in _PLACEHOLDER_AUTHORS})


# --------------------------------------------------------------------------- detector
def detect_bursts(docs: list[Document], *, clusters: list[dict] | None = None,
                  targets: tuple[str, ...] = TARGET_KINDS, bin_s: int | None = None, min_events: int = 5,
                  z_threshold: float = 3.5, max_targets_per_kind: int = 50, max_signals: int = 20,
                  z_window: int = 24, min_mad: float = 0.5, kleinberg_s: float = 2.0,
                  kleinberg_gamma: float = 1.0, kleinberg_min_events: int = 10,
                  include_left_censored: bool = False) -> list[dict]:
    """Burst signals (kind ``burst``) for overall volume, hashtags, link hosts and near-duplicate clusters.

    * Dated documents only (undated ones are ignored), de-duplicated by ref, on a UTC grid of
      ``bin_s`` seconds (default :func:`auto_bin_seconds` of the span) aligned to the epoch;
      a grid of more than ``MAX_BINS`` bins is coarsened (the used width is ``params["bin_s"]``).
    * Targets: ``volume`` (everything), ``hashtag`` (see :func:`_hashtags`), ``domain`` (outbound
      link hosts minus platform hosts) and ``near_dup`` (exact/minhash members of the WS2
      ``clusters``; skipped when ``clusters`` is not given). Per kind, at most
      ``max_targets_per_kind`` targets are scored, most concentrated (largest single bin) first.
    * A burst is a maximal run of bins with :func:`rolling_robust_z` ``>= z_threshold`` and
      ``count >= min_events``; for a target with ``>= kleinberg_min_events`` events it must also
      overlap a :func:`kleinberg_bursts` interval (base rate taken over the whole window). A
      run starting in the first bin is left-censored (its onset was not observed) and is
      dropped unless ``include_left_censored``.
    * ``value`` = peak bin count, ``baseline`` = rolling median count per bin at the peak,
      ``z`` = peak-bin z. ``doc_refs`` = up to 15 target documents in the window, most engaged
      first. Ids ``sig:burst#1..`` follow ``z`` descending, then start time; the output does
      not depend on input order.

    Fewer than ``min_events`` dated documents give ``[]``.
    """
    kinds = tuple(targets or ())
    bad = [k for k in kinds if k not in TARGET_KINDS]
    if bad:
        raise ValueError(f"unknown burst target kind(s) {bad}; expected a subset of {TARGET_KINDS}")
    min_events = max(1, int(min_events))
    z_threshold = float(z_threshold)
    dated = _dated(docs)
    if len(dated) < min_events or not kinds:
        return []
    t_first, t_last = dated[0][0], dated[-1][0]
    b, origin, n_bins = _grid(t_first, t_last, bin_s)
    window_s = max(t_last - t_first, float(b))           # Kleinberg base-rate window
    bix = [int((t - origin) // b) for t, _, _ in dated]

    # Score only targets that could burst, most concentrated first within each kind.
    candidates: dict[str, list[tuple[int, int, str, list[int], Counter]]] = defaultdict(list)
    for (kind, target), idxs in _target_members(dated, kinds, clusters).items():
        if len(idxs) < min_events:
            continue
        per_bin = Counter(bix[i] for i in idxs)
        peak = max(per_bin.values())
        if peak >= min_events:
            candidates[kind].append((-peak, -len(idxs), target, idxs, per_bin))

    params_common = {"bin_s": b, "z_window": int(z_window), "z_threshold": z_threshold, "min_events": min_events,
                     "min_mad": float(min_mad), "kleinberg_s": float(kleinberg_s),
                     "kleinberg_gamma": float(kleinberg_gamma), "kleinberg_min_events": int(kleinberg_min_events)}
    found: list[tuple[tuple, dict]] = []
    for kind in kinds:
        for _, _, target, idxs, per_bin in sorted(candidates.get(kind, []))[: max(0, int(max_targets_per_kind))]:
            counts = [per_bin.get(t, 0) for t in range(n_bins)]
            zs, meds = _robust_baseline(counts, z_window, min_mad)
            runs = [r for r in _runs(counts, zs, z_threshold, min_events) if include_left_censored or r[0] > 0]
            if not runs:
                continue
            kb: list[tuple[float, float]] | None = None
            if len(idxs) >= int(kleinberg_min_events):
                times = [dated[i][0] for i in idxs]
                kb = [(times[iv["start_idx"]], times[iv["end_idx"]])
                      for iv in kleinberg_bursts(times, s=kleinberg_s, gamma=kleinberg_gamma, span_s=window_s)]
            for a, z_end in runs:
                w_start, w_end = origin + a * b, origin + (z_end + 1) * b
                confirmed = None if kb is None else any(k0 < w_end and k1 >= w_start for k0, k1 in kb)
                if confirmed is False:
                    continue
                inside = [i for i in idxs if a <= bix[i] <= z_end]
                peak = max(range(a, z_end + 1), key=lambda t: (counts[t], zs[t], -t))
                first, last = dated[inside[0]][0], dated[inside[-1]][0]
                in_docs = [dated[i][2] for i in inside]
                plats = Counter(d.platform for d in in_docs)
                platforms = [p for p, _ in sorted(plats.items(), key=lambda kv: (-kv[1], kv[0]))]
                n_acc = _accounts(in_docs)
                edge = a == 0 or z_end == n_bins - 1
                ranked = sorted(inside, key=lambda i: (-dated[i][2].engagement(), dated[i][0], dated[i][1]))
                where = ", ".join(platforms[:3]) + (f" +{len(platforms) - 3} more" if len(platforms) > 3 else "")
                summary = (f"{_label(kind, target)}: {len(inside)} posts in {_duration(last - first)} "
                           f"({_when(first, last)}) vs median {_rate(meds[peak], b)} "
                           f"(z={zs[peak]:.1f}) on {where}; {n_acc} account(s)"
                           + ("; touches the edge of the collection window" if edge else ""))
                params = {
                    "target_kind": kind, "target": target, "start": _iso(first), "end": _iso(last),
                    "events_in_burst": len(inside), "share_of_target_total": round(len(inside) / len(idxs), 4),
                    "platforms": platforms, "kleinberg_confirmed": confirmed,
                    "window_start": _iso(w_start), "window_end": _iso(w_end), "peak_bin_start": _iso(origin + peak * b),
                    "target_total": len(idxs), "pre_burst_events": sum(1 for i in idxs if bix[i] < a),
                    "n_accounts": n_acc, "at_window_edge": edge, **params_common,
                }
                record = {"summary": summary, "value": float(counts[peak]), "baseline": float(meds[peak]),
                          "z": float(zs[peak]), "doc_refs": [dated[i][1] for i in ranked[:_MAX_REFS]],
                          "params": params}
                found.append(((-zs[peak], first, _KIND_RANK[kind], target), record))

    found.sort(key=lambda x: x[0])
    return [make_signal("burst", n, r["summary"], value=r["value"], doc_refs=r["doc_refs"], baseline=r["baseline"],
                        z=r["z"], params=r["params"], detector=DETECTOR)
            for n, (_, r) in enumerate(found[: max(0, int(max_signals))], 1)]


# --------------------------------------------------------------------------- profile
def sparkline(series: list[int], width: int = 48) -> str:
    """Unicode block sparkline (``▁``..``█``), max-pooled to at most ``width`` characters.

    Zero (and negative) bins are ``▁``; any positive bin is at least ``▂`` and the maximum is ``█``.
    """
    vals = [max(0.0, float(v or 0)) for v in series or []]
    w = int(width)
    if not vals or w <= 0:
        return ""
    n = len(vals)
    if n > w:
        vals = [max(vals[i * n // w:(i + 1) * n // w]) for i in range(w)]
    top = max(vals)
    if top <= 0:
        return _BLOCKS[0] * len(vals)
    return "".join(_BLOCKS[min(7, max(1, math.ceil(v / top * 7)))] if v > 0 else _BLOCKS[0] for v in vals)


def temporal_profile(docs: list[Document], bin_s: int | None = None) -> dict:
    """Activity curve of the dated documents.

    ``{"bin_s", "start", "end", "counts", "sparkline", "total", "undated", "peak", "version"}``
    where ``start`` / ``end`` bound the UTC-aligned grid (ISO 8601) and ``peak`` is
    ``{"start", "count"}`` of the first fullest bin (``None`` without dated documents).
    """
    docs = list(docs or [])
    dated = _dated(docs)
    undated = len({doc_ref(d) for d in docs if d is not None and d.created_at is None})
    if not dated:
        return {"bin_s": int(bin_s) if bin_s else None, "start": None, "end": None, "counts": [], "sparkline": "",
                "total": 0, "undated": undated, "peak": None, "version": TEMPORAL_VERSION}
    t_first, t_last = dated[0][0], dated[-1][0]
    b, origin, n_bins = _grid(t_first, t_last, bin_s)
    counts = [0] * n_bins
    for t, _, _ in dated:
        counts[int((t - origin) // b)] += 1
    p = counts.index(max(counts))
    return {"bin_s": b, "start": _iso(origin), "end": _iso(origin + len(counts) * b), "counts": counts,
            "sparkline": sparkline(counts), "total": len(dated), "undated": undated,
            "peak": {"start": _iso(origin + p * b), "count": counts[p]}, "version": TEMPORAL_VERSION}


def describe_profile(profile: dict) -> str:
    """One line for stats / prompts, e.g. ``▁▁█▁ 2026-11-01 00:00Z to 2026-11-08 00:00Z, 3 h bins, 208 dated posts, peak 43 at 2026-11-05 12:00Z``."""
    if not profile or not profile.get("counts"):
        return "no dated documents"
    fmt = lambda iso: datetime.fromisoformat(iso).strftime("%Y-%m-%d %H:%MZ")  # noqa: E731
    peak = profile.get("peak") or {}
    return (f"{profile['sparkline']} {fmt(profile['start'])} to {fmt(profile['end'])}, {_unit(int(profile['bin_s']))} "
            f"bins, {profile['total']} dated posts, peak {peak.get('count', 0)} at {fmt(peak['start'])}"
            + (f" ({profile['undated']} undated)" if profile.get("undated") else ""))


__all__ = [
    "TEMPORAL_VERSION", "DETECTOR", "BIN_CHOICES", "TARGET_KINDS", "MAX_BINS", "auto_bin_seconds", "bin_counts",
    "rolling_robust_z", "kleinberg_bursts", "detect_bursts", "temporal_profile", "sparkline", "describe_profile",
]
