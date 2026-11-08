"""STIX 2.1 bundle export: OpenCTI-compatible, idempotent, people-free.

:func:`report_to_stix` turns a finished :class:`~vigileye.models.Report` into a STIX 2.1 bundle that
OpenCTI imports natively, MISP reads through ``misp-stix`` and ``stix2`` parses:

* the report SDO carries the BLUF and one external reference per cited source (platform, URL,
  citation ref, Admiralty grade);
* tradecraft travels as ``note`` objects: one per key judgment (STIX Appendix A confidence
  15/50/85, OpenCTI ``likelihood`` from the WEP scale 5..95), plus the body, assumptions, gaps
  and the ACH matrix (``x_vigileye_ach``);
* observables become SCOs with spec-conformant deterministic ids; CVEs become vulnerabilities
  carrying the KEV/EPSS fields OpenCTI understands; ATT&CK T-ids become attack-patterns;
* SDO ids follow pycti's UUIDv5 recipes (e.g. report = name + published, note = content +
  created + abstract, attack-pattern = lowercased x_mitre_id), so exporting the same report
  twice yields the same ids and OpenCTI re-imports merge instead of duplicating.

Scope guard: no ``identity_class: individual``, no person/account entities, no post authors.
Author strings found in ``report.sources`` are replaced by a case-salted pseudonym in every
free-text field, account/profile URLs are never exported, and every object carries a statement
marking that the analysis is machine-assisted. Pure stdlib; ``stix2`` is only a test oracle.
"""
from __future__ import annotations

import hashlib
import ipaddress
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..models import Report

#: Namespace for deterministic STIX ids (STIX 2.1 section 2.9; also used by pycti for SDOs).
STIX_NS = uuid.UUID("00abedb4-aa42-466c-9c01-fed23315a9b7")

_TLP_WHITE = "marking-definition--613f2e26-407d-48c7-9eca-b8e91df99dc9"
#: TLP 1.0 marking ids from the STIX 2.1 spec. CLEAR reuses the WHITE id, as OpenCTI does.
TLP = {"clear": _TLP_WHITE,
       "white": _TLP_WHITE,
       "green": "marking-definition--34098fce-860f-48ae-8e50-ebd3cc5e41da",
       "amber": "marking-definition--f88d31f6-486f-44da-b317-01333bde0b82",
       "red": "marking-definition--5e57c739-391a-4eb3-b6be-7d15ca92d5ed"}
#: Full marking-definition objects, byte-for-byte as the spec and ``stix2`` define them.
TLP_OBJECTS: dict[str, dict] = {
    key: {"type": "marking-definition", "spec_version": "2.1", "id": mid, "created": "2017-01-20T00:00:00.000Z",
          "definition_type": "tlp", "name": f"TLP:{'WHITE' if key == 'clear' else key.upper()}",
          "definition": {"tlp": "white" if key == "clear" else key}}
    for key, mid in TLP.items()
}

#: ICD 203 words of estimative probability -> STIX Appendix A / OpenCTI note likelihood.
WEP_TO_STIX = {"almost no chance": 5, "very unlikely": 15, "unlikely": 30, "roughly even chance": 50,
               "likely": 70, "very likely": 85, "almost certain": 95}
#: Analytic confidence -> STIX Appendix A Low/Med/High scale.
CONFIDENCE_TO_STIX = {"low": 15, "moderate": 50, "high": 85}

#: VigilEye entity type -> (STIX type, fixed properties). Anything not listed is not exported.
ENTITY_MAP = {"organization": ("identity", {"identity_class": "organization"}),
              "media_outlet": ("identity", {"identity_class": "organization"}),
              "community": ("identity", {"identity_class": "group"}),
              "threat_actor": ("intrusion-set", {}),
              "malware": ("malware", {"is_family": True}),
              "tool": ("tool", {}),
              "campaign": ("campaign", {}),
              "vulnerability": ("vulnerability", {})}
#: Scope guard. 'location' is also skipped: a STIX location needs region/country/lat-long.
EXCLUDED_ENTITY_TYPES = {"person", "account", "individual", "user", "author"}

STATEMENT = "Machine-assisted analysis of public sources (VigilEye); not human-reviewed unless stated"
#: Fixed creation time of the objects shared by every bundle (producer identity, statement marking),
#: so their (id, created) pair never changes between exports.
SHARED_CREATED = "2020-01-01T00:00:00.000Z"

_ENTITY_ALIASES = {"org": "organization", "organisation": "organization", "company": "organization",
                   "agency": "organization", "media": "media_outlet", "outlet": "media_outlet",
                   "news_outlet": "media_outlet", "actor": "threat_actor", "apt": "threat_actor",
                   "threat_group": "threat_actor", "intrusion_set": "threat_actor", "ransomware": "malware",
                   "cve": "vulnerability", "people": "person", "handle": "account", "username": "account"}
_SCO_TYPES = {"domain": "domain-name", "ipv4": "ipv4-addr", "ipv6": "ipv6-addr", "url": "url"}
_HASH_KEYS = {"md5": ("MD5", 32), "sha1": ("SHA-1", 40), "sha256": ("SHA-256", 64)}
_OPENCTI_OBSERVABLE = {"domain-name": "Domain-Name", "ipv4-addr": "IPv4-Addr", "ipv6-addr": "IPv6-Addr",
                       "url": "Url", "file": "StixFile"}
_ACCOUNT_URL = re.compile(r"(?i)^[a-z][a-z0-9+.-]*://[^/?#]+/+(?:@|~|(?:u|user|users|profile|channel|author)/)")
_HANDLE = re.compile(r"(?i)^\s*(?:@|/?u/|/?user/)")
_CVE = re.compile(r"(?i)^CVE-\d{4}-\d{4,}$")
_TECHNIQUE = re.compile(r"^T\d{4}(?:\.\d{3})?$")
_DOMAIN = re.compile(r"(?:[\w-]+\.)+[\w-]{2,}")
_AUTHOR_STOP = {"[deleted]", "[removed]", "deleted", "removed", "anonymous", "unknown", "admin", "none", "null"}


# ----------------------------------------------------------------------------- primitives
def stix_ts(dt: datetime | str) -> str:
    """STIX timestamp ``YYYY-MM-DDTHH:MM:SS.mmmZ`` (UTC, millisecond precision; naive = UTC)."""
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt.strip().replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def canonical_json(obj: Any) -> str:
    """Compact, key-sorted JSON; equivalent to RFC 8785 (JCS) for the string dicts hashed here."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def deterministic_id(stix_type: str, contributing: dict) -> str:
    """``<type>--<uuid5(STIX_NS, canonical_json(contributing))>``, the STIX 2.1 / pycti recipe."""
    return f"{stix_type}--{uuid.uuid5(STIX_NS, canonical_json(contributing))}"


def _live(value: str) -> str:
    """Minimal refang so hand-written defanged values still map to the real observable."""
    v = re.sub(r"(?i)^hxxp", "http", value.strip())
    for a, b in (("[://]", "://"), ("[.]", "."), ("(.)", "."), ("{.}", "."), ("[dot]", "."), ("[:]", ":")):
        v = v.replace(a, b)
    return v


def sco_for_indicator(rec: dict) -> dict | None:
    """Map one indicator record to a STIX Cyber-observable with its deterministic id, or ``None``.

    domain -> domain-name, ipv4 -> ipv4-addr, ipv6 -> ipv6-addr, url -> url (account/profile URLs
    are refused), md5/sha1/sha256 -> file with one ``hashes`` entry. Every other type -> ``None``.
    """
    if not isinstance(rec, dict):
        return None
    kind = str(rec.get("type") or "").lower()
    value = _live(str(rec.get("value") or ""))
    if not value:
        return None
    if kind in _HASH_KEYS:
        algo, n = _HASH_KEYS[kind]
        if not re.fullmatch(rf"[0-9a-f]{{{n}}}", value.lower()):
            return None
        stype, props = "file", {"hashes": {algo: value.lower()}}
    elif kind in ("ipv4", "ipv6"):
        try:
            ip = ipaddress.ip_address(value)
        except ValueError:
            return None
        if ip.version != int(kind[-1]):
            return None
        stype, props = _SCO_TYPES[kind], {"value": ip.compressed}
    elif kind == "domain":
        dom = value.lower().rstrip(".")
        if not _DOMAIN.fullmatch(dom):
            return None
        stype, props = "domain-name", {"value": dom}
    elif kind == "url":
        if not re.match(r"(?i)^[a-z][a-z0-9+.-]*://\S+$", value) or _ACCOUNT_URL.match(value):
            return None
        stype, props = "url", {"value": value}
    else:
        return None
    return {"type": stype, "spec_version": "2.1", "id": deterministic_id(stype, props), **props}


def _pattern(sco: dict) -> str:
    def lit(v: str) -> str:
        return "'" + v.replace("\\", "\\\\").replace("'", "\\'") + "'"

    if sco["type"] == "file":
        (algo, v), = sco["hashes"].items()
        return f"[file:hashes.'{algo}' = {lit(v)}]"
    return f"[{sco['type']}:value = {lit(sco['value'])}]"


def _clean(v: Any) -> Any:
    """Drop None / empty values recursively (STIX forbids empty properties); keeps False and 0."""
    if isinstance(v, dict):
        out = {k: _clean(x) for k, x in v.items()}
        return {k: x for k, x in out.items() if x is not None and x != "" and x != [] and x != {}}
    if isinstance(v, (list, tuple)):
        return [x for x in (_clean(i) for i in v) if x is not None and x != "" and x != [] and x != {}]
    return v


def _norm_entity_type(t: Any) -> str:
    t = re.sub(r"[\s-]+", "_", str(t or "").strip().lower())
    return _ENTITY_ALIASES.get(t, t)


def _norm_conf(v: Any) -> str:
    s = str(v or "").strip().lower()
    return {"medium": "moderate", "med": "moderate"}.get(s, s if s in CONFIDENCE_TO_STIX else "moderate")


def _norm_wep(v: Any) -> str | None:
    s = re.sub(r"\s+", " ", str(v or "").strip().lower())
    if s in WEP_TO_STIX:
        return s
    for term in sorted(WEP_TO_STIX, key=len, reverse=True):  # "very likely" before "likely"
        if re.search(rf"\b{re.escape(term)}\b", s):
            return term
    return None


def _slug(s: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-") or "x"


class _Pseudonymizer:
    """Replaces author strings from ``report.sources`` with ``author-<hash>`` (salted by case)."""

    def __init__(self, report: Report):
        mapping: dict[str, str] = {}
        for s in report.sources or []:
            author = str((s or {}).get("author") or "").strip() if isinstance(s, dict) else ""
            if not author:
                continue
            pseudo = "author-" + hashlib.sha256(f"{report.case}\x00{author.casefold()}".encode()).hexdigest()[:10]
            bare = author.lstrip("@")
            for variant in (author, bare, bare.split("@", 1)[0]):
                v = variant.strip().casefold()
                if len(v) >= 3 and v not in _AUTHOR_STOP:
                    mapping.setdefault(v, pseudo)
        self.mapping = mapping
        keys = sorted(mapping, key=lambda k: (-len(k), k))
        self.rx = (re.compile(r"(?<![\w-])(?:" + "|".join(map(re.escape, keys)) + r")(?![\w-])", re.I)
                   if keys else None)

    def hit(self, text: Any) -> bool:
        return bool(self.rx and isinstance(text, str) and self.rx.search(text))

    def __call__(self, v: Any) -> Any:
        if self.rx is None:
            return v
        if isinstance(v, str):
            return self.rx.sub(lambda m: self.mapping.get(m.group(0).casefold(), "author-redacted"), v)
        if isinstance(v, dict):
            return {k: self(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [self(x) for x in v]
        return v


def _exportable(rec: dict, pseudo: _Pseudonymizer) -> bool:
    if rec.get("benign") or rec.get("warninglists") or rec.get("scope") == "non-routable":
        return False
    return not (str(rec.get("type")).lower() == "url" and pseudo.hit(str(rec.get("value") or "")))


def _safe_url(url: Any, pseudo: _Pseudonymizer) -> str | None:
    u = str(url or "").strip()
    if not re.match(r"(?i)^https?://\S+$", u) or _ACCOUNT_URL.match(u) or pseudo.hit(u):
        return None
    return u


def _num(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ----------------------------------------------------------------------------- export
def report_to_stix(report: Report, *, tlp: str = "clear", producer: str = "VigilEye",
                   emit_indicator_patterns: bool = False, max_external_refs: int = 200) -> dict:
    """Export ``report`` as a STIX 2.1 bundle (a plain dict, ready for ``json.dumps``).

    Deterministic: the same report always yields the same bundle. ``tlp`` is one of
    :data:`TLP`; ``emit_indicator_patterns`` adds a STIX-pattern indicator plus a ``based-on``
    relationship for every exported observable.
    """
    tlp_key = str(tlp or "clear").strip().lower().removeprefix("tlp:")
    if tlp_key not in TLP:
        raise ValueError(f"unknown TLP level {tlp!r}; expected one of {sorted(TLP)}")
    ts = stix_ts(report.created_at)
    pseudo = _Pseudonymizer(report)
    statement = {"type": "marking-definition", "spec_version": "2.1",
                 "id": deterministic_id("marking-definition", {"definition_type": "statement", "definition": STATEMENT}),
                 "created": SHARED_CREATED, "name": "VigilEye machine-assisted analysis",
                 "definition_type": "statement", "definition": {"statement": STATEMENT}}
    markings = [TLP[tlp_key], statement["id"]]
    identity = {"type": "identity", "spec_version": "2.1",
                "id": deterministic_id("identity", {"name": producer.lower().strip(), "identity_class": "system"}),
                "created": SHARED_CREATED, "modified": SHARED_CREATED, "name": producer, "identity_class": "system",
                "description": "Automated open-source intelligence analysis framework.",
                "object_marking_refs": markings}

    def sdo(stype: str, contributing: dict, **props: Any) -> dict:
        return _clean({"type": stype, "spec_version": "2.1", "id": deterministic_id(stype, contributing),
                       "created": ts, "modified": ts, "created_by_ref": identity["id"],
                       "object_marking_refs": markings, **props})

    name = f"{report.case}: {report.lens} assessment"
    report_id = deterministic_id("report", {"name": name.lower().strip(), "published": ts})
    content: dict[str, dict] = {}
    doc_links: list[tuple[set[str], list[str]]] = []

    def add(obj: dict) -> str:
        content.setdefault(obj["id"], obj)  # first writer wins (CVE indicator before vulnerability entity)
        return obj["id"]

    # Observables, vulnerabilities and techniques -------------------------------------------
    for rec in report.indicators or []:
        if not isinstance(rec, dict) or not _exportable(rec, pseudo):
            continue
        kind, value = str(rec.get("type") or "").lower(), _live(str(rec.get("value") or ""))
        ids: list[str] = []
        if kind == "cve" and _CVE.match(value):
            cve = value.upper()
            kev = rec.get("kev")
            epss = rec["epss"] if isinstance(rec.get("epss"), dict) else {}
            desc = None
            if isinstance(kev, dict):
                desc = f"{kev.get('vulnerabilityName') or cve}. Listed in CISA KEV since {kev.get('dateAdded') or 'n/a'}."
            ids.append(add(sdo("vulnerability", {"name": cve.lower()}, name=cve, description=desc,
                               external_references=[{"source_name": "cve", "external_id": cve,
                                                     "url": f"https://nvd.nist.gov/vuln/detail/{cve}"}],
                               x_opencti_cisa_kev=bool(kev) if "kev" in rec else None,
                               x_opencti_epss_score=_num(epss.get("epss")),
                               x_opencti_epss_percentile=_num(epss.get("percentile")))))
        elif kind == "attack_technique" and _TECHNIQUE.match(value.upper()):
            tid = value.upper()
            ids.append(add(sdo("attack-pattern", {"x_mitre_id": tid.lower()}, name=tid, x_mitre_id=tid,
                               external_references=[{"source_name": "mitre-attack", "external_id": tid,
                                                     "url": "https://attack.mitre.org/techniques/"
                                                            + tid.replace(".", "/") + "/"}])))
        else:
            sco = sco_for_indicator(rec)
            if sco is None:
                continue
            count = rec.get("count") if isinstance(rec.get("count"), int) else None
            sco.update(_clean({"object_marking_refs": markings, "x_vigileye_count": count,
                               "x_vigileye_docs": [str(d) for d in rec.get("docs") or []][:20]}))
            ids.append(add(sco))
            if emit_indicator_patterns:
                pattern = _pattern(sco)
                label = sco.get("value") or next(iter(sco["hashes"].values()))
                ind = sdo("indicator", {"pattern": pattern}, name=label, pattern=pattern, pattern_type="stix",
                          pattern_version="2.1", valid_from=ts, indicator_types=["unknown"],
                          description=f"Mentioned in {count or 1} public document(s); extracted by VigilEye, "
                                      "not validated as malicious.",
                          x_opencti_main_observable_type=_OPENCTI_OBSERVABLE[sco["type"]])
                rel = sdo("relationship", {"relationship_type": "based-on", "source_ref": ind["id"],
                                           "target_ref": sco["id"]},
                          relationship_type="based-on", source_ref=ind["id"], target_ref=sco["id"])
                ids += [add(ind), add(rel)]
        doc_links.append(({str(d) for d in rec.get("docs") or []}, ids))

    # Entities (scope-guarded) -------------------------------------------------------------
    for ent in report.entities or []:
        etype = _norm_entity_type(ent.type)
        ename = (ent.name or "").strip()
        if etype in EXCLUDED_ENTITY_TYPES or etype not in ENTITY_MAP or not ename:
            continue
        if pseudo.hit(ename) or _HANDLE.match(ename):
            continue
        stype, fixed = ENTITY_MAP[etype]
        if stype == "vulnerability" and _CVE.match(ename):
            ename = ename.upper()
        contributing = {"name": ename.lower()}
        if stype == "identity":
            contributing["identity_class"] = fixed["identity_class"]
        aliases: list[str] = []
        for a in ent.aliases or []:
            a = str(a).strip()
            if a and a.casefold() != ename.casefold() and not pseudo.hit(a) and not _HANDLE.match(a) \
                    and a.casefold() not in {x.casefold() for x in aliases}:
                aliases.append(a)
        alias_key = "x_opencti_aliases" if stype in ("identity", "vulnerability") else "aliases"
        add(sdo(stype, contributing, name=ename, **fixed, **{alias_key: aliases[:20]},
                x_vigileye_entity_type=etype, x_vigileye_mentions=ent.mentions))

    # Report --------------------------------------------------------------------------------
    ext_refs = []
    for s in report.sources or []:
        if not isinstance(s, dict) or len(ext_refs) >= max_external_refs:
            continue
        ref = str(s.get("id") or "").strip()
        ref = None if pseudo.hit(ref) else ref
        url = _safe_url(s.get("url"), pseudo)
        if ref or url:
            ext_refs.append(_clean({"source_name": str(s.get("platform") or "web"), "url": url, "external_id": ref,
                                    "description": f"Admiralty {s['grade']}" if s.get("grade") else None}))
    lens = str(report.lens or "general")
    rep = sdo("report", {"name": name.lower().strip(), "published": ts}, name=name,
              description=pseudo(report.bluf), published=ts, labels=[lens],
              report_types=["threat-report"] if lens in ("threat", "cyber") else [f"{_slug(lens)}-analysis"],
              object_refs=list(content) or [identity["id"]], external_references=ext_refs,
              x_vigileye_case=report.case, x_vigileye_lens=lens, x_vigileye_query=pseudo(report.query),
              x_vigileye_engine_version=str((report.meta or {}).get("engine_version") or "") or None)

    # Notes: body, key judgments, assumptions, gaps, ACH -------------------------------------
    notes: list[dict] = []

    def note(abstract: str, text: str, refs: list[str] | None = None, **props: Any) -> None:
        text, abstract = pseudo(text).strip(), pseudo(abstract).strip()
        notes.append(sdo("note", {"content": text, "created": ts, "attribute_abstract": abstract},
                         abstract=abstract, content=text, object_refs=refs or [report_id], **props))

    if (report.body_markdown or "").strip():
        note(f"Analysis: {name}", report.body_markdown.strip(), note_types=["analysis"], x_vigileye_section="body")
    for i, kj in enumerate(report.key_judgments or [], 1):
        conf, wep = _norm_conf(kj.confidence), _norm_wep(kj.likelihood)
        evidence = [str(e) for e in kj.evidence or []]
        refs = [report_id]
        for docs, ids in doc_links:
            if docs.intersection(evidence):
                refs += [x for x in ids if x not in refs]
        text = kj.summary or kj.title
        if evidence:
            text += "\n\nEvidence: " + ", ".join(f"[{e}]" for e in evidence)
        note(kj.title or f"Key judgment {i}", text, refs[:100], note_types=["assessment"],
             confidence=CONFIDENCE_TO_STIX[conf], likelihood=WEP_TO_STIX.get(wep or ""),
             labels=pseudo([str(t) for t in kj.tags or []]), x_vigileye_section="key_judgment", x_vigileye_index=i,
             x_vigileye_estimative_term=wep, x_vigileye_confidence_term=conf, x_vigileye_evidence=evidence)
    for section, title, items in (("assumptions", "Key assumptions", report.assumptions),
                                  ("gaps", "Intelligence gaps", report.gaps)):
        lines = [f"- {x}" for x in (str(i).strip() for i in items or []) if x]
        if lines:
            note(f"{title}: {name}", "\n".join(lines), note_types=["analysis"], x_vigileye_section=section)
    hyps = [h if isinstance(h, dict) else {"hypothesis": str(h)} for h in report.hypotheses or []]
    if hyps:
        rows = []
        for i, h in enumerate(hyps, 1):
            rows.append(f"- {h.get('id') or f'H{i}'}: {h.get('hypothesis', '')} (likelihood: "
                        f"{h.get('likelihood') or 'n/a'}; consistent: {', '.join(map(str, h.get('consistent') or [])) or '-'};"
                        f" inconsistent: {', '.join(map(str, h.get('inconsistent') or [])) or '-'})")
        conclusion = str(((report.meta or {}).get("ach") or {}).get("conclusion") or "").strip()
        keep = ("id", "hypothesis", "likelihood", "confidence", "consistent", "inconsistent",
                "inconsistency_score", "notes")
        note(f"Analysis of competing hypotheses: {name}",
             "\n".join(rows) + (f"\n\nConclusion: {conclusion}" if conclusion else ""),
             note_types=["assessment"], x_vigileye_section="ach",
             x_vigileye_ach=pseudo([{k: h[k] for k in keep if k in h} for h in hyps]))

    objects: dict[str, dict] = {}
    for obj in [TLP_OBJECTS[tlp_key], statement, identity, rep, *content.values(), *notes]:
        objects.setdefault(obj["id"], obj)
    return {"type": "bundle",
            "id": deterministic_id("bundle", {"case": report.case, "lens": lens, "created": ts}),
            "objects": list(objects.values())}


# ----------------------------------------------------------------------------- validation
_ID_RE = re.compile(r"^([a-z0-9][a-z0-9-]{1,248}[a-z0-9])--[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")
_TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?Z$")
_TS_KEYS = {"created", "modified", "published", "valid_from", "valid_until", "first_seen", "last_seen",
            "first_observed", "last_observed", "start_time", "stop_time"}
_SCO = {"artifact", "autonomous-system", "directory", "domain-name", "email-addr", "email-message", "file",
        "ipv4-addr", "ipv6-addr", "mac-addr", "mutex", "network-traffic", "process", "software", "url",
        "user-account", "windows-registry-key", "x509-certificate"}
_PEOPLE_SCO = {"user-account", "email-addr", "email-message"}


def _walk_refs(value: Any, path: str):
    if isinstance(value, dict):
        for k, v in value.items():
            if k.endswith("_ref") and isinstance(v, str):
                yield f"{path}.{k}", v
            elif k.endswith("_refs") and isinstance(v, list):
                yield from ((f"{path}.{k}[{i}]", r) for i, r in enumerate(v))
            else:
                yield from _walk_refs(v, f"{path}.{k}")
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from _walk_refs(v, f"{path}[{i}]")


def validate_bundle(bundle: dict) -> list[str]:
    """Structural STIX 2.1 checks plus the VigilEye scope guard. Returns problems (``[]`` = valid).

    Checks id syntax, duplicate ids, required ``spec_version``/``created``/``modified``, RFC 3339
    UTC timestamps, that every ``*_ref``/``*_refs`` resolves inside the bundle, non-empty report
    ``object_refs``, no empty property values, and that nothing describes a person
    (``identity_class: individual``, user accounts, e-mail addresses).
    """
    if not isinstance(bundle, dict) or bundle.get("type") != "bundle":
        return ["not a STIX bundle (type != 'bundle')"]
    errs: list[str] = []
    if not _ID_RE.match(str(bundle.get("id", ""))) or not str(bundle.get("id")).startswith("bundle--"):
        errs.append(f"bundle: malformed id {bundle.get('id')!r}")
    objects = bundle.get("objects")
    if not isinstance(objects, list):
        return errs + ["bundle: 'objects' must be a list"]
    ids: set[str] = set()
    for i, obj in enumerate(objects):
        oid = obj.get("id") if isinstance(obj, dict) else None
        m = _ID_RE.match(str(oid or ""))
        if not m or m.group(1) != obj.get("type"):
            errs.append(f"objects[{i}]: id {oid!r} does not match type {obj.get('type') if isinstance(obj, dict) else '?'}")
        if oid in ids:
            errs.append(f"{oid}: duplicate id")
        ids.add(oid)
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        oid, otype = obj.get("id"), obj.get("type")
        need = ["spec_version"] + ([] if otype in _SCO else ["created"]) + \
               ([] if otype in _SCO or otype == "marking-definition" else ["modified"])
        errs += [f"{oid}: missing {k}" for k in need if k not in obj]
        if obj.get("spec_version", "2.1") != "2.1":
            errs.append(f"{oid}: spec_version {obj.get('spec_version')!r} != '2.1'")
        for k in _TS_KEYS & obj.keys():
            if not isinstance(obj[k], str) or not _TS_RE.match(obj[k]):
                errs.append(f"{oid}: {k} {obj[k]!r} is not an RFC 3339 UTC timestamp")
        if isinstance(obj.get("created"), str) and isinstance(obj.get("modified"), str) \
                and obj["modified"] < obj["created"]:
            errs.append(f"{oid}: modified precedes created")
        errs += [f"{oid}: empty value for {k}" for k, v in obj.items() if v is None or v == "" or v == [] or v == {}]
        for where, ref in _walk_refs(obj, str(oid)):
            if ref not in ids:
                errs.append(f"{where}: unresolved reference {ref!r}")
        if otype == "report" and not obj.get("object_refs"):
            errs.append(f"{oid}: report.object_refs is empty")
        if otype == "note" and not (obj.get("content") and obj.get("object_refs")):
            errs.append(f"{oid}: note needs content and object_refs")
        conf = obj.get("confidence")
        if conf is not None and (isinstance(conf, bool) or not isinstance(conf, int) or not 0 <= conf <= 100):
            errs.append(f"{oid}: confidence must be an integer in 0..100")
        if otype == "identity" and obj.get("identity_class") == "individual":
            errs.append(f"{oid}: scope guard: identity_class 'individual' is not allowed")
        if otype in _PEOPLE_SCO:
            errs.append(f"{oid}: scope guard: {otype} objects are not allowed")
    return errs


def write_stix(report: Report, path, **kw: Any) -> Path:
    """Write :func:`report_to_stix` output as JSON. A directory ``path`` gets ``<case>-<lens>-<ts>.stix.json``."""
    p = Path(path)
    if p.is_dir():
        p = p / f"{_slug(report.case)}-{_slug(report.lens)}-{report.created_at:%Y%m%d-%H%M%S}.stix.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report_to_stix(report, **kw), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return p


__all__ = [
    "STIX_NS", "TLP", "TLP_OBJECTS", "WEP_TO_STIX", "CONFIDENCE_TO_STIX", "ENTITY_MAP", "EXCLUDED_ENTITY_TYPES",
    "STATEMENT", "stix_ts", "canonical_json", "deterministic_id", "sco_for_indicator", "report_to_stix",
    "validate_bundle", "write_stix",
]
