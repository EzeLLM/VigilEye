"""Evidence preservation and chain of custody for collected public content.

Follows the Berkeley Protocol on Digital Open Source Investigations (hash every
item at collection, log every handling event, keep a verifiable inventory) and
the hash enricher of Bellingcat's Auto Archiver:

* :func:`content_hash` / :func:`raw_hash`: SHA-256 over canonical JSON of a
  document's evidentiary content (what the source published) and of the
  platform's raw record.
* :func:`stamp`: writes ``doc.meta["evidence"]`` at collection time. An
  existing stamp is never overwritten.
* :class:`EvidenceLedger`: an append-only, hash-chained custody log (table
  ``custody``) in the same SQLite file as :class:`~vigileye.store.CaseStore`,
  with chain verification, case verification (stored documents against their
  stamps and against the hashes logged at collection) and a shareable
  manifest with a ``sha256sum``-style sidecar.
* :func:`wayback_lookup`: optional, read-only Internet Archive availability
  check. It is never called by default.

Canonical JSON (``vigileye-cjson/1``), reproducible with the stdlib alone:
every string (keys included) is NFC-normalised; values are converted the way
``json.dumps(default=str)`` converts them (which is how CaseStore stores
documents, so hashes survive the store round trip); non-finite floats become
the strings ``"NaN"``, ``"Infinity"`` and ``"-Infinity"``; the result is
``json.dumps(sort_keys=True, separators=(",", ":"), ensure_ascii=False)``
encoded as UTF-8. It resembles RFC 8785 but does not implement its number
formatting.

* content record: ``{"platform", "id", "source", "url", "title", "text",
  "created_at" (UTC ISO-8601 or null), "media" (list of media URLs)}``.
  Author, metrics, tags and meta are excluded.
* custody entry: ``entry_hash = sha256_hex(prev_hash + cjson({"case", "at",
  "action", "actor", "subject", "details"}))``. Each case starts its chain
  from :data:`GENESIS` (64 zeros).

Threat model: the chain is tamper-evident, not tamper-proof. Anyone who can
write the database can rebuild every hash, and deleting the newest entries
leaves a valid shorter chain. Anchor the head outside the database (hand over
the manifest and its sidecar, or timestamp it with RFC 3161) and pass it back
as ``anchor`` to :meth:`EvidenceLedger.verify_chain`.

Privacy: ledger details, verification results and manifests carry document
refs, fingerprints, hashes, timestamps and counts, never author names.
Profile-like URL segments and account handles are masked in manifests and in
:func:`format_entry`. The exact URLs stay in the case store, and each one is
bound by the document's ``content_sha256``.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from urllib.parse import quote

from . import __version__
from .models import Document, Report, _parse_dt, utcnow

EVIDENCE_VERSION = "1"
TOOL_VERSION = f"vigileye/{__version__}"
CANONICALIZATION = "vigileye-cjson/1"
MANIFEST_FORMAT = "vigileye-evidence-manifest/1"
GENESIS = "0" * 64
EVIDENCE_KEY = "evidence"
CONTENT_FIELDS = ("platform", "id", "source", "url", "title", "text", "created_at", "media")
STAMP_FIELDS = ("content_sha256", "raw_sha256", "collected_at", "collector", "target_spec", "tool_version")
WAYBACK_API = "https://archive.org/wayback/available?url="

_ACTION = re.compile(r"^[a-z][a-z0-9_.-]{0,39}$")
REPORT_ACTIONS = ("analyze", "amend")  # custody actions whose details bind a stored report's hash
_RETRIES = 3

_SCHEMA = """
CREATE TABLE IF NOT EXISTS custody(
    seq INTEGER PRIMARY KEY AUTOINCREMENT, case_name TEXT NOT NULL, at TEXT NOT NULL, action TEXT NOT NULL,
    actor TEXT NOT NULL, subject TEXT, details_json TEXT NOT NULL, prev_hash TEXT NOT NULL,
    entry_hash TEXT NOT NULL UNIQUE);
CREATE UNIQUE INDEX IF NOT EXISTS ux_custody_case_prev ON custody(case_name, prev_hash);
CREATE INDEX IF NOT EXISTS ix_custody_case_seq ON custody(case_name, seq);
CREATE TRIGGER IF NOT EXISTS custody_no_update BEFORE UPDATE ON custody
BEGIN SELECT RAISE(ABORT, 'custody log is append-only'); END;
CREATE TRIGGER IF NOT EXISTS custody_no_delete BEFORE DELETE ON custody
BEGIN SELECT RAISE(ABORT, 'custody log is append-only'); END;
"""


# --------------------------------------------------------------------------- canonical JSON
def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def _key(k: Any) -> str:
    """A dict key as ``json.dumps`` writes it, NFC-normalised."""
    if isinstance(k, str):
        s = str.__str__(k)
    elif k is True or k is False or k is None:
        s = json.dumps(k)
    elif isinstance(k, int):
        s = int.__repr__(k)
    elif isinstance(k, float):
        s = json.dumps(float(k))
    else:
        s = str(k)
    return _nfc(s)


def _plain(obj: Any) -> Any:
    """JSON-ready copy of ``obj``, converted like ``json.dumps(default=str)``, with NFC strings."""
    if isinstance(obj, str):
        return _nfc(str.__str__(obj))
    if obj is None or obj is True or obj is False:
        return obj
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        if math.isfinite(obj):
            return float(obj)
        return "NaN" if math.isnan(obj) else ("Infinity" if obj > 0 else "-Infinity")
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, dict):
        return {_key(k): _plain(v) for k, v in obj.items()}
    return _nfc(str(obj))


def canonical_json(obj: Any) -> str:
    """``vigileye-cjson/1`` text of ``obj`` (see the module docstring)."""
    return json.dumps(_plain(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


def _iso_utc(value: Any) -> str | None:
    dt = _parse_dt(value)
    return dt.astimezone(timezone.utc).isoformat() if dt else None


def _as_document(doc: Document | Mapping) -> Document:
    if isinstance(doc, Document):
        return doc
    if isinstance(doc, Mapping):
        return Document.from_dict(dict(doc))
    raise TypeError(f"expected a Document or a mapping, got {type(doc).__name__}")


# --------------------------------------------------------------------------- document hashes
def content_record(doc: Document | Mapping) -> dict:
    """The evidentiary content of ``doc`` that :func:`content_hash` covers (:data:`CONTENT_FIELDS`)."""
    d = _as_document(doc)
    return {"platform": d.platform, "id": d.id, "source": d.source, "url": d.url, "title": d.title,
            "text": d.text, "created_at": _iso_utc(d.created_at), "media": [m.url for m in d.media]}


def content_hash(doc: Document | Mapping) -> str:
    """SHA-256 hex of the canonical JSON of the document's evidentiary content."""
    return _sha256(canonical_json(content_record(doc)))


def raw_hash(doc: Document | Mapping) -> str:
    """SHA-256 hex of the canonical JSON of ``doc.raw``, the platform's own record (``{}`` when absent)."""
    raw = _as_document(doc).raw
    return _sha256(canonical_json(raw if raw is not None else {}))


def _is_stamp(ev: Any) -> bool:
    return isinstance(ev, dict) and isinstance(ev.get("content_sha256"), str)


def _stamp_of(doc: Document) -> dict | None:
    ev = doc.meta.get(EVIDENCE_KEY) if isinstance(doc.meta, dict) else None
    return ev if _is_stamp(ev) else None


def _collector_name(spec: str) -> str:
    try:
        from .collectors.base import resolve_target

        return resolve_target(spec)[0]
    except Exception:  # noqa: BLE001 - an unresolvable spec still gets stamped
        return "unknown"


def stamp(docs: Iterable[Document], *, spec: str, collector: str | None = None, tool_version: str | None = None,
          collected_at: Any = None, existing: Mapping[str, dict] | None = None) -> list[Document]:
    """Write a collection stamp into ``doc.meta["evidence"]`` of every document and return them as a list.

    Stamp: ``{content_sha256, raw_sha256, collected_at (UTC ISO), collector, target_spec, tool_version}``.
    ``collector`` defaults to the collector that ``spec`` resolves to, ``tool_version`` to
    :data:`TOOL_VERSION`, and ``collected_at`` to now (one timestamp for the whole call).

    A document that already carries a stamp keeps it. ``existing`` maps fingerprints to stamps
    already in the store (:meth:`EvidenceLedger.stamps`). When content and raw hashes are both
    unchanged, that earlier stamp is carried forward, so re-collecting an unchanged item keeps
    its first ``collected_at``. Documents are modified in place.
    """
    when = _iso_utc(utcnow() if collected_at is None else collected_at)
    if when is None:
        raise ValueError(f"unparseable collected_at {collected_at!r}")
    name = collector or _collector_name(spec)
    out: list[Document] = []
    for d in docs:
        if not isinstance(d.meta, dict):
            d.meta = {}
        if _stamp_of(d) is None:
            c, r = content_hash(d), raw_hash(d)
            prior = (existing or {}).get(d.fingerprint())
            if _is_stamp(prior) and prior.get("content_sha256") == c and prior.get("raw_sha256") == r:
                d.meta[EVIDENCE_KEY] = dict(prior)
            else:
                d.meta[EVIDENCE_KEY] = {"content_sha256": c, "raw_sha256": r, "collected_at": when,
                                        "collector": name, "target_spec": spec,
                                        "tool_version": tool_version or TOOL_VERSION}
        out.append(d)
    return out


# --------------------------------------------------------------------------- custody helpers
def collect_details(docs: Iterable[Document], *, new: int | None = None, **extra: Any) -> dict:
    """``details`` for a ``collect`` custody entry. It binds every collected item's hash into the chain.

    Returns ``{collected, new, batch_sha256, items: [[fingerprint, content_sha256], ...]}``, with items
    sorted and hashes recomputed from the content. :meth:`EvidenceLedger.verify_case` checks stored
    documents against these items. ``extra`` keys (e.g. ``error``) are added but cannot replace
    these four.
    """
    docs = list(docs)
    items = sorted([d.fingerprint(), content_hash(d)] for d in docs)
    core = {"collected": len(docs), "batch_sha256": _sha256(canonical_json(items)), "items": items}
    if new is not None:
        core["new"] = int(new)
    return {**extra, **core}


def report_sha256(report: Report | Mapping) -> str:
    """SHA-256 hex of the canonical JSON of ``report.to_dict()``. It is the same before and after CaseStore."""
    return _sha256(canonical_json(report.to_dict() if isinstance(report, Report) else dict(report)))


def file_sha256(path) -> str:
    """SHA-256 hex of a file's bytes, read in 1 MiB blocks."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def file_hashes(paths: Iterable) -> list[dict]:
    """``[{name, bytes, sha256}]`` for written files. Only base names are kept, never local directories."""
    return [{"name": Path(p).name, "bytes": Path(p).stat().st_size, "sha256": file_sha256(p)} for p in paths]


# Same profile-path pattern as analysis.coordination; plus @handles, u/ and user/ or user: target forms.
_PROFILE_PATH = re.compile(r"(/@)[^/?#]+|(/(?:u|user|users|profile|people|channel|c|author)/)[^/?#]+"
                           r"|(profile\.php\?)[^#]*", re.I)
_HANDLE = re.compile(r"(?<![\w#\]])@[\w.-]+")
_USER_PATH = re.compile(r"(?i)(?<![\w\]])(u|user)([/:])[\w.-]+")


def mask_accounts(text: str | None) -> str | None:
    """Mask account handles and profile-like URL segments as ``[account]`` (for display and manifests)."""
    if not text:
        return text
    from .policy import mask_profile_urls  # lazy: policy imports the collectors package

    s = _PROFILE_PATH.sub(lambda m: (m.group(1) or m.group(2) or m.group(3)) + "[account]", str(text))
    s = mask_profile_urls(s)
    s = _HANDLE.sub("@[account]", s)
    return _USER_PATH.sub(lambda m: m.group(1) + m.group(2) + "[account]", s)


def _short(v: Any) -> str:
    s = mask_accounts(str(v)) or ""
    return s if len(s) <= 40 else s[:16] + "..."


def format_entry(entry: Mapping) -> str:
    """One display line for a custody entry. Accounts are masked, lists are left out and long values are cut."""
    details = entry.get("details")
    details = details if isinstance(details, dict) else {}
    scalars = " ".join(f"{k}={_short(v)}" for k, v in sorted(details.items())
                       if not isinstance(v, (list, dict)) and k != "batch_sha256")
    subject = mask_accounts(entry.get("subject")) or "-"
    return (f"#{entry.get('seq')} {entry.get('at')} {entry.get('action', ''):<9} {entry.get('actor')} {subject}"
            + (f" {scalars}" if scalars else "") + f" [{str(entry.get('entry_hash', ''))[:16]}]")


def _entry_hash(prev_hash: str, case: str, at: str, action: str, actor: str, subject: str | None,
                details: Any) -> str:
    entry = {"case": case, "at": at, "action": action, "actor": actor, "subject": subject, "details": details}
    return _sha256(prev_hash + canonical_json(entry))


# --------------------------------------------------------------------------- ledger
class EvidenceLedger:
    """Append-only, hash-chained custody log stored in the case store's SQLite database.

    ``store`` is a :class:`~vigileye.store.CaseStore`, whose connection and lock are shared, or
    a database path, which opens its own CaseStore that :meth:`close` closes. ``clock`` (returns
    a datetime) can be injected for deterministic tests. Appends take an ``IMMEDIATE`` write lock
    and a unique ``(case_name, prev_hash)`` index, so writers in other processes cannot fork a
    chain. Triggers reject ``UPDATE`` and ``DELETE`` on the table.
    """

    def __init__(self, store: Any, *, clock: Callable[[], datetime] | None = None,
                 tool_version: str | None = None) -> None:
        self._owns = isinstance(store, (str, Path))
        if self._owns:
            from .store import CaseStore

            store = CaseStore(store)
        self.store = store
        self._db: sqlite3.Connection = store._db
        self._lock = store._lock
        self._clock = clock or utcnow
        self.tool_version = tool_version or TOOL_VERSION
        with self._lock:
            self._db.executescript(_SCHEMA)

    # -- writing
    def record(self, case: str, action: str, subject: str | None = None, details: Mapping | None = None,
               actor: str = "vigileye") -> dict:
        """Append a custody entry for ``case`` and return it (with ``seq``, ``prev_hash`` and ``entry_hash``).

        ``action`` is a short lowercase token (collect, analyze, export, manifest, verify, delete,
        note, ...). ``details`` is a JSON-able mapping stored in canonical form. Keep it free of
        author names: use refs, fingerprints, hashes and counts.
        """
        if not case or not isinstance(case, str):
            raise ValueError("case must be a non-empty string")
        if not isinstance(action, str) or not _ACTION.match(action):
            raise ValueError(f"invalid custody action {action!r}; expected a short lowercase token")
        if details is not None and not isinstance(details, Mapping):
            raise TypeError("details must be a mapping")
        if not actor or not isinstance(actor, str):
            raise ValueError("actor must be a non-empty string")
        subject = None if subject is None else str(subject)
        details_json = canonical_json(dict(details or {}))
        detail_obj = json.loads(details_json)
        db = self._db
        with self._lock:
            for attempt in range(_RETRIES):
                try:
                    if not db.in_transaction:
                        db.execute("BEGIN IMMEDIATE")
                    prev = self._head(case) or GENESIS
                    at = _iso_utc(self._clock())
                    if at is None:
                        raise ValueError("ledger clock returned an unusable timestamp")
                    h = _entry_hash(prev, case, at, action, actor, subject, detail_obj)
                    cur = db.execute(
                        "INSERT INTO custody(case_name, at, action, actor, subject, details_json, prev_hash, "
                        "entry_hash) VALUES(?,?,?,?,?,?,?,?)", (case, at, action, actor, subject, details_json, prev, h))
                    db.commit()
                except sqlite3.IntegrityError:  # another writer appended to this case's head first
                    db.rollback()
                    if attempt == _RETRIES - 1:
                        raise
                    continue
                except BaseException:
                    db.rollback()
                    raise
                return {"seq": int(cur.lastrowid), "case": case, "at": at, "action": action, "actor": actor,
                        "subject": subject, "details": detail_obj, "prev_hash": prev, "entry_hash": h}
        raise RuntimeError("unreachable")  # pragma: no cover

    # -- reading
    def _head(self, case: str) -> str | None:
        row = self._db.execute("SELECT entry_hash FROM custody WHERE case_name=? ORDER BY seq DESC LIMIT 1",
                               (case,)).fetchone()
        return row[0] if row else None

    def head(self, case: str) -> str | None:
        """Hash of the newest entry for ``case``, or ``None`` if the case has no custody entries."""
        with self._lock:
            return self._head(case)

    def _rows(self, case: str) -> list[tuple]:
        with self._lock:
            return self._db.execute(
                "SELECT seq, case_name, at, action, actor, subject, details_json, prev_hash, entry_hash "
                "FROM custody WHERE case_name=? ORDER BY seq", (case,)).fetchall()

    def entries(self, case: str) -> list[dict]:
        """All custody entries for ``case`` in append order (details that fail to parse come back as ``None``)."""
        out = []
        for seq, c, at, action, actor, subject, dj, prev, h in self._rows(case):
            try:
                details = json.loads(dj)
            except ValueError:
                details = None
            out.append({"seq": seq, "case": c, "at": at, "action": action, "actor": actor, "subject": subject,
                        "details": details, "prev_hash": prev, "entry_hash": h})
        return out

    def stamps(self, case: str, fingerprints: Iterable[str] | None = None) -> dict[str, dict]:
        """``{fingerprint: stamp}`` for stored documents of ``case``, optionally only the given fingerprints."""
        base = 'SELECT fingerprint, json FROM documents WHERE "case"=?'
        rows: list = []
        with self._lock:
            if fingerprints is None:
                rows = self._db.execute(base, (case,)).fetchall()
            else:
                fps = list(dict.fromkeys(fingerprints))
                for i in range(0, len(fps), 500):
                    part = fps[i:i + 500]
                    rows += self._db.execute(f"{base} AND fingerprint IN ({','.join('?' * len(part))})",
                                             (case, *part)).fetchall()
        out: dict[str, dict] = {}
        for fp, js in rows:
            try:
                ev = (json.loads(js).get("meta") or {}).get(EVIDENCE_KEY)
            except (ValueError, AttributeError):
                continue
            if _is_stamp(ev):
                out[fp] = ev
        return out

    # -- verification
    def verify_chain(self, case: str, *, anchor: str | None = None) -> dict:
        """Recompute the chain for ``case``.

        Returns ``{ok, entries, broken_at, head, reason}``. ``broken_at`` is the ``seq`` of the
        first entry that does not link to its predecessor or whose hash does not match its
        content. When ``anchor`` (a head recorded earlier, e.g. from a manifest) is given,
        ``anchor_found`` is added, and ``ok`` also requires the anchor to be one of the verified
        entries. That detects truncation or a rebuilt chain.
        """
        rows = self._rows(case)
        prev, broken_at, reason, seen = GENESIS, None, None, set()
        for seq, c, at, action, actor, subject, dj, prev_hash, h in rows:
            if prev_hash != prev:
                broken_at, reason = seq, "prev_hash does not link to the previous entry"
                break
            try:
                details = json.loads(dj)
            except ValueError:
                broken_at, reason = seq, "details_json is not valid JSON"
                break
            if _entry_hash(prev, c, at, action, actor, subject, details) != h:
                broken_at, reason = seq, "entry_hash does not match the entry content"
                break
            prev = h
            seen.add(h)
        out = {"ok": broken_at is None, "entries": len(rows), "broken_at": broken_at,
               "head": rows[-1][8] if rows else None, "reason": reason}
        if anchor is not None:
            out["anchor_found"] = anchor in seen
            if not out["anchor_found"]:
                out["ok"] = False
                out["reason"] = reason or "anchor not found among verified entries (log truncated or rewritten)"
        return out

    def _logged_items(self, case: str) -> set[tuple[str, str]] | None:
        logged: set[tuple[str, str]] | None = None
        for e in self.entries(case):
            items = (e["details"] or {}).get("items") if isinstance(e["details"], dict) else None
            if isinstance(items, list):
                if logged is None:
                    logged = set()
                logged.update((str(i[0]), str(i[1])) for i in items if isinstance(i, list) and len(i) == 2)
        return logged

    def _check_reports(self, case: str) -> dict:
        # The newest "analyze" or "amend" entry per report id holds the hash the stored report must match:
        # an amendment (e.g. the investigation provenance section) is itself a logged custody event.
        latest: dict[Any, str] = {}
        for e in self.entries(case):
            d = e["details"] if isinstance(e["details"], dict) else {}
            if e["action"] in REPORT_ACTIONS and "report_id" in d and "report_sha256" in d:
                latest.pop(d["report_id"], None)
                latest[d["report_id"]] = d["report_sha256"]
        res = {"checked": 0, "mismatched": [], "missing": []}
        for rid, digest in latest.items():
            try:
                rep = self.store.get_report(int(rid))
            except (TypeError, ValueError):
                rep = None
            if rep is None:
                res["missing"].append(rid)
                continue
            res["checked"] += 1
            if report_sha256(rep) != digest:
                res["mismatched"].append(rid)
        return res

    def verify_case(self, case: str, *, anchor: str | None = None) -> dict:
        """Verify the chain and every stored document of ``case``.

        Returns ``{ok, case, chain, documents_checked, mismatched, raw_mismatched, unstamped,
        unlogged, ledger_bound, reports}``. The document lists hold fingerprints.

        * ``mismatched``: stamped documents whose content no longer matches their ``content_sha256``.
        * ``raw_mismatched``: stamped documents whose raw record no longer matches ``raw_sha256``.
        * ``unstamped``: documents without a stamp (collected before stamping). These are not proven.
        * ``unlogged``: stamped documents whose current content hash was never logged by a
          ``collect`` entry (:func:`collect_details`). This catches an edit followed by a re-stamp.
          It is only checked once the chain carries item hashes (``ledger_bound``).
        * ``reports``: stored reports checked against the ``report_sha256`` of their newest ``analyze``
          or ``amend`` entry.

        ``ok`` requires a valid chain and empty ``mismatched``, ``raw_mismatched``, ``unlogged``
        and ``reports.mismatched`` lists. Unstamped documents and missing reports are reported
        but do not fail ``ok``.
        """
        chain = self.verify_chain(case, anchor=anchor)
        logged = self._logged_items(case)
        checked, mismatched, raw_bad, unstamped, unlogged = 0, [], [], [], []
        for d in self.store.get_documents(case):
            fp, ev = d.fingerprint(), _stamp_of(d)
            if ev is None:
                unstamped.append(fp)
                continue
            checked += 1
            c = content_hash(d)
            if c != ev["content_sha256"]:
                mismatched.append(fp)
            if ev.get("raw_sha256") is not None and raw_hash(d) != ev["raw_sha256"]:
                raw_bad.append(fp)
            if logged is not None and (fp, c) not in logged:
                unlogged.append(fp)
        reports = self._check_reports(case)
        ok = chain["ok"] and not mismatched and not raw_bad and not unlogged and not reports["mismatched"]
        return {"ok": ok, "case": case, "chain": chain, "documents_checked": checked,
                "mismatched": sorted(mismatched), "raw_mismatched": sorted(raw_bad), "unstamped": sorted(unstamped),
                "unlogged": sorted(unlogged), "ledger_bound": logged is not None, "reports": reports}

    def custody_summary(self, case: str, docs: Iterable[Document] | None = None) -> dict:
        """Compact custody state for a report's Methodology (``report.stats["custody"]``).

        Returns ``{chain_head, chain_entries, chain_ok}``. When ``docs`` (the analysed documents)
        is given, ``documents`` and ``documents_stamped`` are added.
        """
        chain = self.verify_chain(case)
        out: dict[str, Any] = {"chain_head": chain["head"], "chain_entries": chain["entries"], "chain_ok": chain["ok"]}
        if docs is not None:
            docs = list(docs)
            out["documents"] = len(docs)
            out["documents_stamped"] = sum(1 for d in docs if _stamp_of(d) is not None)
        return out

    # -- manifest
    def manifest(self, case: str) -> dict:
        """Evidence inventory for ``case`` (in the spirit of Berkeley Protocol Annex IV).

        Per document: ``{ref, fingerprint, url (accounts masked), collected_at, content_sha256,
        raw_sha256, status}``, where status is ``verified``, ``mismatch`` or ``unstamped``. For
        unstamped documents the hashes are taken now. Documents are sorted by ref.
        """
        chain = self.verify_chain(case)
        docs, counts = [], {"verified": 0, "mismatch": 0, "unstamped": 0}
        for d in sorted(self.store.get_documents(case), key=lambda x: (x.ref, x.fingerprint())):
            ev, c = _stamp_of(d), content_hash(d)
            status = "unstamped" if ev is None else ("verified" if ev["content_sha256"] == c else "mismatch")
            counts[status] += 1
            docs.append({"ref": d.ref, "fingerprint": d.fingerprint(), "url": mask_accounts(d.url),
                         "collected_at": ev.get("collected_at") if ev else None,
                         "content_sha256": ev["content_sha256"] if ev else c,
                         "raw_sha256": ev.get("raw_sha256") if ev else raw_hash(d), "status": status})
        return {"format": MANIFEST_FORMAT, "case": case, "generated_at": _iso_utc(self._clock()),
                "tool_version": self.tool_version,
                "hashing": {"algorithm": "sha256", "canonicalization": CANONICALIZATION,
                            "content_fields": list(CONTENT_FIELDS)},
                "chain": {"head": chain["head"], "entries": chain["entries"], "ok": chain["ok"]},
                "summary": {"documents": len(docs), **counts},
                "documents": docs}

    def write_manifest(self, case: str, path, *, record: bool = True) -> dict:
        """Write :meth:`manifest` as JSON to ``path`` and its SHA-256 to ``<path>.sha256`` (``sha256sum -c``).

        With ``record`` (the default), a ``manifest`` custody entry logs the file hash and the
        chain head the manifest attests to. Returns ``{path, sidecar, sha256, chain_head,
        documents, manifest}``.
        """
        m = self.manifest(case)
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = (json.dumps(m, indent=2) + "\n").encode("utf-8")
        p.write_bytes(data)
        digest = hashlib.sha256(data).hexdigest()
        side = p.with_name(p.name + ".sha256")
        side.write_text(f"{digest}  {p.name}\n", encoding="utf-8")
        if record:
            self.record(case, "manifest", subject=p.name,
                        details={"file": p.name, "sha256": digest, "chain_head": m["chain"]["head"],
                                 "documents": len(m["documents"])})
        return {"path": str(p), "sidecar": str(side), "sha256": digest, "chain_head": m["chain"]["head"],
                "documents": len(m["documents"]), "manifest": m}

    def close(self) -> None:
        if self._owns:
            self.store.close()

    def __enter__(self) -> "EvidenceLedger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# --------------------------------------------------------------------------- archive check
def _http_get_json(url: str, timeout: float = 20.0) -> Any:
    import httpx

    from .collectors.base import DEFAULT_USER_AGENT

    resp = httpx.get(url, timeout=timeout, follow_redirects=True,
                     headers={"User-Agent": DEFAULT_USER_AGENT, "Accept": "application/json"})
    resp.raise_for_status()
    return resp.json()


def _wayback_ts(ts: Any) -> str | None:
    try:
        return datetime.strptime(str(ts)[:14], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc).isoformat()
    except (TypeError, ValueError):
        return None


def wayback_lookup(url: str, fetch: Callable[[str], Any] | None = None) -> dict:
    """Ask the Internet Archive whether ``url`` has a snapshot (keyless, read-only, never called by default).

    ``fetch(api_url)`` returns the parsed JSON (or its text or bytes). The default is a single
    ``httpx`` GET. Calling this tells archive.org which URL you are looking at. Returns
    ``{url, archived, snapshot_url, timestamp (UTC ISO), status, error}``. It never raises.
    """
    out = {"url": url, "archived": False, "snapshot_url": None, "timestamp": None, "status": None, "error": None}
    u = str(url or "").strip()
    if not re.match(r"(?i)^https?://[^\s/]+", u):
        out["error"] = "not an http(s) URL"
        return out
    try:
        payload = (fetch or _http_get_json)(WAYBACK_API + quote(u, safe=""))
        if isinstance(payload, (bytes, str)):
            payload = json.loads(payload)
        closest = ((payload or {}).get("archived_snapshots") or {}).get("closest") or {}
    except Exception as e:  # noqa: BLE001 - an archive check must never break the caller
        out["error"] = f"{type(e).__name__}: {e}"
        return out
    if isinstance(closest, dict) and closest.get("available") and closest.get("url"):
        out.update(archived=True, snapshot_url=str(closest["url"]), timestamp=_wayback_ts(closest.get("timestamp")),
                   status=None if closest.get("status") is None else str(closest["status"]))
    return out


__all__ = [
    "EVIDENCE_VERSION", "TOOL_VERSION", "CANONICALIZATION", "MANIFEST_FORMAT", "GENESIS", "EVIDENCE_KEY",
    "CONTENT_FIELDS", "STAMP_FIELDS", "WAYBACK_API", "canonical_json", "content_record", "content_hash", "raw_hash",
    "stamp", "collect_details", "report_sha256", "file_sha256", "file_hashes", "mask_accounts", "format_entry",
    "EvidenceLedger", "wayback_lookup",
]
