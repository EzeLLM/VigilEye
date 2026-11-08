"""Authoritative vulnerability grounding: CISA KEV and FIRST EPSS.

The model must not be the source of truth for whether a CVE is exploited. This
module looks the CVEs found in the collection up in two keyless, read-only
reference sources and turns the answers into facts the prompts must respect:

* **CISA Known Exploited Vulnerabilities** (KEV): a bulk JSON download of every
  CVE CISA has confirmed as exploited in the wild. Nothing about the case is
  sent; the whole catalog is fetched and matched locally.
* **FIRST EPSS**: a modelled probability that a CVE is exploited in the next 30
  days. Only public CVE identifiers (validated against ``CVE-YYYY-NNNN``) are
  sent.

Fields follow OpenCTI's vulnerability model (``x_opencti_cisa_kev``,
``x_opencti_epss_score`` / ``_percentile``): :func:`ground_cves` writes
``rec['kev']`` and ``rec['epss']`` onto each CVE indicator, which the STIX export
reads. :func:`reference_facts_block` renders the prompt block, and
:func:`exploitation_claim_check` corroborates or flags "actively exploited"
claims in the finished report.

Everything goes through :class:`RefCache`: a file cache with a TTL, atomic
writes and a stale-cache fallback. ``offline=True`` or ``$VIGILEYE_OFFLINE``
disables the network entirely, and tests inject ``fetch``.

This module deliberately imports nothing from :mod:`vigileye.analysis`, so the
engine can import it without a cycle.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import tempfile
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable

from .. import __version__
from ..models import Report

log = logging.getLogger("vigileye.intel")

KEV_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
EPSS_URL = "https://api.first.org/data/v1/epss"
DEFAULT_USER_AGENT = f"VigilEye/{__version__} (+https://github.com/EzeLLM/VigilEye; reference-data)"
DEFAULT_CACHE_DIR = ".vigileye/refdata"

KEV_REF = "ref:cisa-kev"
EPSS_REF = "ref:first-epss"
REF_IDS = (KEV_REF, EPSS_REF)
SECTION_ID = "reference_facts"
#: Fields copied from a KEV catalog entry onto ``rec['kev']``.
KEV_FIELDS = ("dateAdded", "dueDate", "knownRansomwareCampaignUse", "vendorProject", "product",
              "vulnerabilityName", "cwes")
#: EPSS below this is treated as "exploitation unlikely" when KEV is also silent.
EPSS_LOW = 0.01

FACTS_HEADER = "AUTHORITATIVE REFERENCE FACTS (machine-retrieved; do not contradict{cite}):"
FACTS_NOTE = ("Note: absence from CISA KEV means no CISA-confirmed exploitation as of that catalog, not proof "
              "that a CVE is unexploited; EPSS is a modelled 30-day exploitation probability, not an observation.")

EXPLOIT_RE = re.compile(
    r"\b(?:exploit(?:ed|ing|ation)|in[ -]the[ -]wild|0-?day|zero[ -]?day"
    r"|actively (?:exploited|used|abused|targeted))\b", re.I)
#: A negation cue in the same clause before an exploitation phrase ("has not been exploited", "no evidence ... exploited").
_NEG_RE = re.compile(r"\b(?:not|never|nor|without|no (?:evidence|indication|signs?|known|reports?|confirmed|public"
                     r"|active|credible))\b|n[\u2019']t\b", re.I)
_CLAUSE_RE = re.compile(r"[,;:()]|\s(?:but|however|although|though|whereas|while)\s", re.I)
_CVE_RE = re.compile(r"\bCVE-(?:19|20)\d{2}-\d{4,7}\b", re.I)
_CVE_FULL = re.compile(r"CVE-(?:19|20)\d{2}-\d{4,7}")
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[(\"'])")

CHECK_STATUSES = ("corroborated", "unconfirmed", "uncorroborated", "contradicted", "consistent", "unchecked")


def _env_offline() -> bool:
    return os.environ.get("VIGILEYE_OFFLINE", "").strip().lower() not in ("", "0", "false", "no", "off")


def _section(config: Any) -> dict:
    """The ``intel`` section from a :class:`~vigileye.config.Config`, a full config dict, or the section itself."""
    if config is None:
        return {}
    if hasattr(config, "section"):
        return config.section("intel") or {}
    if isinstance(config, dict):
        inner = config.get("intel")
        return inner if isinstance(inner, dict) else config
    return {}


def _clean(value: Any, limit: int = 80) -> str:
    """One line, no control characters, capped: safe to place inside a prompt."""
    s = re.sub(r"[\x00-\x1f\x7f]+", " ", str(value or ""))
    s = " ".join(s.split())
    return s if len(s) <= limit else s[: limit - 1].rstrip() + "…"


def _norm_cve(value: Any) -> str | None:
    s = str(value or "").strip().upper()
    return s if _CVE_FULL.fullmatch(s) else None


# --------------------------------------------------------------------------- cache
class RefCache:
    """File cache for reference datasets: TTL by file age, atomic writes, stale fallback.

    ``fetch(url, params) -> parsed JSON`` is injectable (tests pass a stub). The
    default uses ``httpx.get(timeout=30, trust_env=True)``. Fetch failures are
    appended to :attr:`errors` and never raised.
    """

    def __init__(self, root: str | Path = DEFAULT_CACHE_DIR, *, ttl_hours: float = 24, offline: bool = False,
                 fetch: Callable[[str, dict | None], Any] | None = None, user_agent: str = DEFAULT_USER_AGENT):
        self.root = Path(root)
        self.ttl_hours = float(ttl_hours)
        self._offline = bool(offline)
        self._fetch = fetch or self._http_get_json
        self.user_agent = user_agent
        self.errors: list[str] = []
        self._memo: dict[Path, tuple[tuple[int, int], Any]] = {}
        self._lock = threading.Lock()

    @classmethod
    def from_config(cls, config: Any = None, **overrides: Any) -> "RefCache":
        """Build from the ``intel`` config section (``cache_dir``, ``ttl_hours``, ``offline``, ``user_agent``)."""
        sec = _section(config)
        kw: dict[str, Any] = {
            "ttl_hours": float(sec.get("ttl_hours") or 24),
            "offline": bool(sec.get("offline", False)),
            "user_agent": sec.get("user_agent") or DEFAULT_USER_AGENT,
        }
        kw.update(overrides)
        return cls(sec.get("cache_dir") or DEFAULT_CACHE_DIR, **kw)

    @property
    def offline(self) -> bool:
        return self._offline or _env_offline()

    def path(self, name: str, params: dict | None = None) -> Path:
        """Cache file for ``name`` (plus a short hash of ``params``), e.g. ``<root>/kev.json``."""
        stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._") or "data"
        if params:
            stem += "-" + hashlib.sha1(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()[:12]
        return self.root / f"{stem}.json"

    def _read(self, p: Path) -> tuple[Any, float] | None:
        try:
            st = p.stat()
        except OSError:
            return None
        key = (st.st_mtime_ns, st.st_size)
        memo = self._memo.get(p)
        if memo and memo[0] == key:
            data = memo[1]
        else:
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError) as e:
                log.warning("ignoring unreadable reference cache %s: %s", p, e)
                return None
            self._memo[p] = (key, data)
        return data, max(0.0, (time.time() - st.st_mtime) / 3600.0)

    def _write(self, p: Path, data: Any) -> None:
        tmp = None
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=f".{p.name}.", suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, separators=(",", ":"), ensure_ascii=False)
            os.replace(tmp, p)
            tmp = None
        except OSError as e:
            log.warning("could not write reference cache %s: %s", p, e)
        finally:
            if tmp:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass

    def _http_get_json(self, url: str, params: dict | None) -> Any:
        import httpx

        resp = httpx.get(url, params=params, timeout=30, trust_env=True, follow_redirects=True,
                         headers={"User-Agent": self.user_agent, "Accept": "application/json"})
        resp.raise_for_status()
        return resp.json()

    def get_json(self, name: str, url: str, params: dict | None = None, *, ttl_hours: float | None = None,
                 validate: Callable[[Any], bool] | None = None) -> Any | None:
        """Return the dataset ``name``: fresh cache, else a fetch, else stale cache, else ``None``.

        Offline (constructor flag or ``$VIGILEYE_OFFLINE``) never fetches and serves any cached copy
        regardless of age. ``ttl_hours=0`` forces a fetch. ``validate`` rejects a malformed payload,
        which then counts as a fetch error.
        """
        p = self.path(name, params)
        ttl = self.ttl_hours if ttl_hours is None else float(ttl_hours)
        with self._lock:
            cached = self._read(p)
            if cached is not None and (self.offline or cached[1] < ttl):
                return cached[0]
            if self.offline:
                return None
            try:
                data = self._fetch(url, params)
                if data is None or (validate is not None and not validate(data)):
                    raise ValueError("empty or malformed payload")
            except Exception as e:  # noqa: BLE001 - degrade to the stale copy
                msg = f"{name}: fetch failed ({type(e).__name__}: {_clean(e, 160)})"
                log.warning("reference data %s", msg)
                self.errors.append(msg + ("; using stale cache" if cached is not None else ""))
                return cached[0] if cached is not None else None
            self._write(p, data)
            self._memo.pop(p, None)
            return data


# --------------------------------------------------------------------------- KEV / EPSS
def _valid_kev(raw: Any) -> bool:
    return isinstance(raw, dict) and isinstance(raw.get("vulnerabilities"), list)


def _parse_kev(raw: Any) -> dict:
    if not _valid_kev(raw):
        return {}
    by_cve: dict[str, dict] = {}
    for entry in raw["vulnerabilities"]:
        cid = _norm_cve(entry.get("cveID")) if isinstance(entry, dict) else None
        if cid:
            by_cve[cid] = entry
    count = raw.get("count")
    return {"catalogVersion": str(raw.get("catalogVersion") or "") or None, "dateReleased": raw.get("dateReleased"),
            "count": int(count) if isinstance(count, (int, float)) else len(by_cve), "by_cve": by_cve}


def load_kev(cache: RefCache) -> dict:
    """The KEV catalog: ``{'catalogVersion', 'dateReleased', 'count', 'by_cve': {cveID: entry}}``; ``{}`` on miss."""
    return _parse_kev(cache.get_json("kev", KEV_URL, validate=_valid_kev))


def _prob(value: Any) -> float | None:
    try:
        v = float(value)  # the API returns numbers as strings
    except (TypeError, ValueError):
        return None
    return v if 0.0 <= v <= 1.0 and not math.isnan(v) else None


def _epss_lookup(cves: Iterable[str], cache: RefCache, batch: int = 100) -> tuple[dict[str, dict], set[str]]:
    """EPSS scores plus the set of CVEs whose batch was actually answered (to tell "no score" from "no data")."""
    ids = sorted({c for c in (_norm_cve(x) for x in cves) if c})
    scores: dict[str, dict] = {}
    answered: set[str] = set()
    step = max(1, int(batch))
    for i in range(0, len(ids), step):
        part = ids[i: i + step]
        payload = cache.get_json("epss", EPSS_URL, {"cve": ",".join(part)},
                                 validate=lambda d: isinstance(d, dict) and isinstance(d.get("data"), list))
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            continue
        answered.update(part)
        for row in payload["data"]:
            cid = _norm_cve(row.get("cve")) if isinstance(row, dict) else None
            epss, pct = (_prob(row.get("epss")), _prob(row.get("percentile"))) if cid else (None, None)
            if cid and epss is not None:
                scores[cid] = {"epss": epss, "percentile": pct, "date": str(row.get("date") or "") or None}
    return scores, answered


def epss_scores(cves: Iterable[str], cache: RefCache, *, batch: int = 100) -> dict[str, dict]:
    """``{cve: {'epss': float, 'percentile': float, 'date': str}}`` from ``GET ?cve=a,b,c`` (``batch`` per call)."""
    return _epss_lookup(cves, cache, batch)[0]


def _fmt_prob(v: float) -> str:
    if v >= 0.01:
        return f"{math.floor(v * 100) / 100:.2f}"  # floor: never round 0.9999 up to 1.00
    return f"{v:.2g}"


def _fact_line(cid: str, info: dict, kev_version: str | None) -> str:
    kev = info.get("kev")
    if info.get("in_kev") is True and kev:
        product = _clean(" ".join(str(x) for x in (kev.get("vendorProject"), kev.get("product")) if x), 60)
        line = (f"{cid}: in CISA KEV since {_clean(kev.get('dateAdded') or 'unknown date', 20)} ("
                + (f"{product}; " if product else "")
                + f"ransomware use: {_clean(kev.get('knownRansomwareCampaignUse') or 'Unknown', 20)})")
    elif info.get("in_kev") is False:
        line = f"{cid}: NOT in CISA KEV (catalog {kev_version or 'unknown version'})"
    else:
        line = f"{cid}: CISA KEV status unknown (catalog unavailable)"
    ep = info.get("epss")
    if ep:
        line += f"; EPSS {_fmt_prob(ep['epss'])}"
        if ep.get("percentile") is not None:
            line += f" (pctl {_fmt_prob(ep['percentile'])})"
    elif info.get("epss_checked"):
        line += "; no EPSS score"
    return line


def ground_cves(indicators: list[dict], *, cache: RefCache | None = None, config: Any = None) -> dict:
    """Ground every ``{'type': 'cve'}`` indicator in KEV and EPSS (records are mutated in place).

    Sets ``rec['kev']`` to the KEV fields (:data:`KEV_FIELDS`) or ``None`` when the catalog was checked
    and the CVE is not listed (the key is left absent when the catalog was unavailable), and
    ``rec['epss']`` to ``{'epss', 'percentile', 'date'}`` or ``None`` when EPSS has no score.

    ``config`` is the ``intel`` section (or a Config / full config dict holding one): ``cache_dir``,
    ``ttl_hours``, ``offline``, ``kev`` (default true) and ``epss`` (default true).

    Returns ``{'kev_version', 'kev_date_released', 'checked', 'in_kev', 'epss_scored', 'cves': {cve: {...}},
    'facts': [str], 'sources': [...], 'errors': [str]}``. Nothing is fetched when there are no CVEs.
    """
    sec = _section(config)
    cache = cache or RefCache.from_config(sec)
    recs = [r for r in indicators or [] if isinstance(r, dict) and r.get("type") == "cve" and _norm_cve(r.get("value"))]
    counts: Counter = Counter()
    for r in recs:
        counts[_norm_cve(r["value"])] += int(r.get("count") or 1)
    order = sorted(counts, key=lambda c: (-counts[c], c))
    out: dict[str, Any] = {"kev_version": None, "kev_date_released": None, "checked": 0, "in_kev": 0,
                           "epss_scored": 0, "cves": {}, "facts": [], "sources": [], "errors": []}
    if not order:
        return out
    n_err = len(cache.errors)
    kev = load_kev(cache) if sec.get("kev", True) else {}
    if sec.get("kev", True) and not kev:
        out["errors"].append("CISA KEV catalog unavailable (offline with no cached copy, or the download failed)")
    scores, answered = _epss_lookup(order, cache) if sec.get("epss", True) else ({}, set())
    out["kev_version"], out["kev_date_released"] = kev.get("catalogVersion"), kev.get("dateReleased")

    for cid in order:
        entry = kev["by_cve"].get(cid) if kev else None
        info: dict[str, Any] = {
            "count": counts[cid],
            "in_kev": (entry is not None) if kev else None,
            "kev": {k: entry.get(k) for k in KEV_FIELDS} if entry else None,
            "epss": scores.get(cid),
            "epss_checked": cid in answered,
        }
        out["cves"][cid] = info
    # Most-mentioned first; KEV-listed CVEs win ties so they survive the prompt block's limit.
    for cid in sorted(order, key=lambda c: (-counts[c], not out["cves"][c]["in_kev"], c)):
        out["facts"].append(_fact_line(cid, out["cves"][cid], out["kev_version"]))
    for r in recs:
        info = out["cves"][_norm_cve(r["value"])]
        if info["in_kev"] is not None:
            r["kev"] = dict(info["kev"]) if info["kev"] else None
        if info["epss_checked"]:
            r["epss"] = dict(info["epss"]) if info["epss"] else None

    out["checked"] = len(order) if kev else 0
    out["in_kev"] = sum(1 for i in out["cves"].values() if i["in_kev"])
    out["epss_scored"] = sum(1 for i in out["cves"].values() if i["epss"])
    if kev:
        out["sources"].append({
            "id": KEV_REF, "name": "CISA KEV", "platform": "reference", "source": "CISA KEV",
            "title": f"CISA Known Exploited Vulnerabilities catalog {out['kev_version'] or ''}".strip(),
            "version": out["kev_version"], "url": KEV_URL, "author": None,
            "created_at": out["kev_date_released"], "grade": "B1", "cited": 0})
    if out["epss_scored"]:
        dates = sorted({i["epss"]["date"] for i in out["cves"].values() if i["epss"] and i["epss"].get("date")})
        version = dates[-1] if dates else None
        out["sources"].append({
            "id": EPSS_REF, "name": "FIRST EPSS", "platform": "reference", "source": "FIRST EPSS",
            "title": f"FIRST Exploit Prediction Scoring System scores {version or ''}".strip(),
            "version": version, "url": EPSS_URL, "author": None, "created_at": version, "grade": "B2", "cited": 0})
    out["errors"] = cache.errors[n_err:] + out["errors"]
    return out


def reference_facts_block(grounding: dict, limit: int = 25) -> str:
    """The prompt block of grounded facts (``""`` when there is nothing to say)."""
    facts = list((grounding or {}).get("facts") or [])
    if not facts:
        return ""
    ids = [s["id"] for s in grounding.get("sources") or [] if s.get("id") in REF_IDS]
    cite = ("; cite " + " / ".join(f"[{i}]" for i in ids)) if ids else ""  # only refs that resolve
    lines = [FACTS_HEADER.format(cite=cite)] + [f"- {f}" for f in facts[: max(0, limit)]]
    if len(facts) > limit:
        lines.append(f"- ({len(facts) - limit} more CVE(s) omitted)")
    lines.append(FACTS_NOTE)
    return "\n".join(lines)


# --------------------------------------------------------------------------- report checks
def _negated(sentence: str) -> bool:
    """Whether an exploitation phrase in ``sentence`` is negated within its own clause."""
    for m in EXPLOIT_RE.finditer(sentence):
        start = max((c.end() for c in _CLAUSE_RE.finditer(sentence, 0, m.start())), default=0)
        if _NEG_RE.search(sentence, start, m.start()):
            return True
    return False


def _claims(text: str) -> dict[str, tuple[str, bool]]:
    """``{cve: (sentence, negated)}`` for each CVE the text says is (or is not) exploited.

    A sentence counts when it names the CVE and an exploitation phrase; when the text names exactly one
    CVE, an exploitation phrase in any sentence is attributed to it.
    """
    mentioned = list(dict.fromkeys(m.group(0).upper() for m in _CVE_RE.finditer(text or "")))
    out: dict[str, tuple[str, bool]] = {}
    if not mentioned:
        return out
    for sent in _SENT_RE.split(text):
        if not EXPLOIT_RE.search(sent):
            continue
        ids = list(dict.fromkeys(m.group(0).upper() for m in _CVE_RE.finditer(sent)))
        if not ids and len(mentioned) == 1:
            ids = mentioned
        for cid in ids:
            out.setdefault(cid, (" ".join(sent.split())[:240], _negated(sent)))
    return out


def _status(info: dict | None, negated: bool) -> str:
    if not info or info.get("in_kev") is None:
        return "unchecked"
    if info["in_kev"]:
        return "contradicted" if negated else "corroborated"
    if negated:
        return "consistent"
    ep = info.get("epss")
    return "unconfirmed" if ep and ep["epss"] < EPSS_LOW else "uncorroborated"


def _gap_line(status: str, cid: str, targets: list[str], info: dict | None, kev_version: str | None) -> str | None:
    who = "/".join(targets)
    if status == "unconfirmed":
        return (f"{who}: the exploitation claim for {cid} is not corroborated: it is not in CISA KEV (catalog "
                f"{kev_version or 'unknown'}) and its EPSS score is {_fmt_prob(info['epss']['epss'])}; treat "
                f"in-the-wild exploitation as unconfirmed pending independent reporting [{KEV_REF}] [{EPSS_REF}].")
    if status == "contradicted":
        since = (info.get("kev") or {}).get("dateAdded") or "an unknown date"
        return (f"{who}: describes {cid} as not exploited, but CISA KEV has listed it as known exploited since "
                f"{since} [{KEV_REF}]; reconcile before dissemination.")
    if status == "unchecked" and info is None:
        return (f"{who}: claims exploitation of {cid}, which appears in none of the collected documents and was "
                "not checked against CISA KEV or EPSS.")
    return None


def exploitation_claim_check(report: Report, grounding: dict) -> list[dict]:
    """Check exploitation claims in the BLUF and key judgments against the grounding.

    A claim is a sentence that names a CVE and matches :data:`EXPLOIT_RE`:

    * listed in KEV -> ``corroborated``; ``ref:cisa-kev`` is added to that judgment's evidence;
    * listed in KEV but the sentence negates exploitation -> ``contradicted``; a gap line is added;
    * not listed and EPSS < 0.01 -> ``unconfirmed``; a gap line is added;
    * not listed otherwise -> ``uncorroborated`` (recorded only); negated -> ``consistent``;
    * KEV unavailable -> ``unchecked``; a CVE absent from the collection also adds a gap line.

    Writes ``report.meta['reference_checks']`` and returns the checks. Safe to re-run (no duplicates).
    """
    g = grounding or {}
    cves: dict = g.get("cves") or {}
    targets: list[tuple[str, Any, str]] = [("BLUF", None, report.bluf or "")]
    targets += [(f"KJ{i}", kj, f"{kj.title}. {kj.summary}") for i, kj in enumerate(report.key_judgments, 1)]
    checks: list[dict] = []
    grouped: dict[tuple[str, str], list[str]] = {}
    for target, kj, text in targets:
        for cid, (sentence, negated) in _claims(text).items():
            info = cves.get(cid)
            status = _status(info, negated)
            ep = (info or {}).get("epss") or {}
            checks.append({"target": target, "cve": cid, "status": status, "negated": negated,
                           "in_kev": (info or {}).get("in_kev"),
                           "kev_date_added": ((info or {}).get("kev") or {}).get("dateAdded"),
                           "epss": ep.get("epss"), "kev_version": g.get("kev_version"), "excerpt": sentence})
            if status == "corroborated" and kj is not None and KEV_REF not in kj.evidence:
                kj.evidence.append(KEV_REF)
            grouped.setdefault((cid, status), []).append(target)
    new_gaps = [line for (cid, status), who in grouped.items()
                if (line := _gap_line(status, cid, who, cves.get(cid), g.get("kev_version")))]
    report.gaps = list(dict.fromkeys([*(report.gaps or []), *new_gaps]))
    if not isinstance(report.meta, dict):
        report.meta = {}
    report.meta["reference_checks"] = checks
    return checks


def grounding_summary(grounding: dict) -> dict:
    """Compact run record for ``meta['reference_grounding']``."""
    g = grounding or {}
    return {k: g.get(k) for k in ("kev_version", "kev_date_released", "checked", "in_kev", "epss_scored")} | {
        "errors": list(g.get("errors") or [])}


def reference_section(grounding: dict, checks: list[dict] | None = None) -> dict:
    """The ``reference_facts`` entry for ``meta['extra_sections']``."""
    g = grounding or {}
    ids = {s.get("id") for s in g.get("sources") or []}
    kev = (f"the CISA KEV catalog {g.get('kev_version')} [{KEV_REF}]" if KEV_REF in ids
           else "the CISA KEV catalog (unavailable for this run)")
    epss = f"FIRST EPSS [{EPSS_REF}]" if EPSS_REF in ids else "FIRST EPSS (no scores for this run)"
    L = [f"_CVEs found in the collection, checked against {kev} and {epss}. Absence from KEV is not proof that "
         "a CVE is unexploited; EPSS is a modelled 30-day probability._", ""]
    L += [f"- {f}" for f in g.get("facts") or []] or ["- _No CVEs to ground._"]
    if checks:
        L += ["", "**Exploitation claims**", ""]
        L += [f"- {c['target']}, {c['cve']}: {c['status']}" for c in checks]
    if g.get("errors"):
        L += ["", "**Lookup errors**", ""] + [f"- {_clean(e, 200)}" for e in g["errors"]]
    return {"id": SECTION_ID, "title": "Reference facts", "markdown": "\n".join(L)}


def apply_grounding(report: Report, grounding: dict, *, cited: Counter | dict | None = None) -> list[dict]:
    """Do every report-side step for a grounded run (mutates ``report``; idempotent).

    Runs :func:`exploitation_claim_check`, (re)appends the reference sources with ``cited`` counts
    (``cited`` is the engine's citation counter; when omitted, key-judgment evidence is counted),
    sets ``meta['reference_grounding']`` and replaces the ``reference_facts`` extra section.
    """
    before = Counter(e for kj in report.key_judgments for e in kj.evidence)
    checks = exploitation_claim_check(report, grounding)
    added = Counter(e for kj in report.key_judgments for e in kj.evidence) - before
    base = Counter(cited) if cited is not None else before
    report.sources = [s for s in report.sources or [] if not (isinstance(s, dict) and s.get("id") in REF_IDS)]
    report.sources += [{**s, "cited": base.get(s["id"], 0) + added.get(s["id"], 0)}
                       for s in (grounding or {}).get("sources") or []]
    report.meta["reference_grounding"] = grounding_summary(grounding)
    section = reference_section(grounding, checks)
    sections = [s for s in (report.meta.get("extra_sections") or []) if isinstance(s, dict)]
    idx = next((j for j, s in enumerate(sections) if s.get("id") == SECTION_ID), None)
    if idx is None:
        sections.append(section)
    else:
        sections[idx] = section
    report.meta["extra_sections"] = sections
    return checks


def refresh(cache: RefCache) -> dict:
    """Force a KEV download -> ``{'catalogVersion', 'dateReleased', 'count', 'refreshed', 'errors'}``.

    On failure (or offline) the cached copy is reported with ``refreshed=False``.
    """
    n_err = len(cache.errors)
    kev = _parse_kev(cache.get_json("kev", KEV_URL, ttl_hours=0, validate=_valid_kev))
    errors = cache.errors[n_err:]
    if cache.offline:
        errors.append("offline: served the cached catalog, if any, without downloading")
    return {"catalogVersion": kev.get("catalogVersion"), "dateReleased": kev.get("dateReleased"),
            "count": kev.get("count", 0), "refreshed": bool(kev) and not errors, "errors": errors}


def lookup_cve(cve: str, cache: RefCache) -> dict:
    """One CVE's full KEV entry and EPSS score (for ``vigileye refdata cve``). Raises ValueError on a bad id."""
    cid = _norm_cve(cve)
    if not cid:
        raise ValueError(f"not a CVE identifier: {cve!r}")
    n_err = len(cache.errors)
    kev = load_kev(cache)
    scores, answered = _epss_lookup([cid], cache)
    return {"cve": cid, "kev_version": kev.get("catalogVersion"),
            "in_kev": (cid in kev["by_cve"]) if kev else None, "kev": (kev.get("by_cve") or {}).get(cid),
            "epss": scores.get(cid) if cid in answered else None, "errors": cache.errors[n_err:]}


__all__ = [
    "KEV_URL", "EPSS_URL", "DEFAULT_USER_AGENT", "KEV_REF", "EPSS_REF", "REF_IDS", "KEV_FIELDS", "EXPLOIT_RE",
    "CHECK_STATUSES", "RefCache", "load_kev", "epss_scores", "ground_cves", "reference_facts_block",
    "exploitation_claim_check", "grounding_summary", "reference_section", "apply_grounding", "refresh",
    "lookup_cve",
]
