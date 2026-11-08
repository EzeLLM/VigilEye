"""Enrichment: indicator (IOC) extraction, imagery description, entity extraction.

Indicator extraction is pure regex (no LLM), handles defanged notation
(``hxxp://``, ``example[.]com``, ``1.2.3[.]4``) and deduplicates with counts
and the document ids each indicator was seen in.

Deliberately NOT extracted: e-mail addresses and phone numbers. VigilEye
analyses public content and communities; harvesting personal contact data is
out of scope.
"""
from __future__ import annotations

import hashlib
import ipaddress
import logging
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Iterable

from ..models import Document, Entity
from .chunking import doc_ref, fence_untrusted, render_chunk

if TYPE_CHECKING:  # pragma: no cover
    from .lenses import Lens

log = logging.getLogger("vigileye.analysis")

# ---------------------------------------------------------------------------
# Defang / refang
# ---------------------------------------------------------------------------

_REFANG_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bhxxp(s?)(?=\[?:)", re.I), r"http\1"),
    (re.compile(r"\bhxxp(s?)\b", re.I), r"http\1"),
    (re.compile(r"\bh\*\*p(s?)\b", re.I), r"http\1"),
    (re.compile(r"\bfxp\b", re.I), "ftp"),
    (re.compile(r"\[\s*:\s*\]//|\[://\]"), "://"),
    (re.compile(r"\[\s*:\s*\]"), ":"),
    (re.compile(r"\[\s*\.\s*\]|\(\s*\.\s*\)|\{\s*\.\s*\}|\[dot\]|\(dot\)|\{dot\}", re.I), "."),
    (re.compile(r"\[/\]"), "/"),
]

_DEFANG_MARK = re.compile(r"hxxp|h\*\*p|fxp|\[\s*[.:]\s*\]|\(\s*\.\s*\)|\{\s*\.\s*\}|\[dot\]|\(dot\)|\{dot\}|\[://\]", re.I)


def refang(text: str) -> str:
    """Convert defanged notation back to live form (for matching only)."""
    if not text:
        return ""
    for pat, rep in _REFANG_RULES:
        text = pat.sub(rep, text)
    return text


def defang(value: str, kind: str | None = None) -> str:
    """Return a safe-to-display defanged form of an indicator."""
    if not value:
        return value
    if kind in (None, "url"):
        v = re.sub(r"^http", "hxxp", value, flags=re.I)
        v = re.sub(r"^ftp", "fxp", v, flags=re.I)
        m = re.match(r"^([a-z]+://)([^/]+)(.*)$", v, re.I)
        if m:
            return m.group(1).replace("://", "[://]") + m.group(2).replace(".", "[.]") + m.group(3)
        if kind == "url":
            return v
    if kind in (None, "domain", "ipv4"):
        return value.replace(".", "[.]")
    if kind == "ipv6":
        return value.replace(":", "[:]")
    return value


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

_URL_RE = re.compile(r"\b(?:https?|ftp)://[^\s<>\"'`\])}|]+", re.I)
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")  # used only to MASK e-mails out of the text
_IPV4_RE = re.compile(r"(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(?!\d|\.\d)")
_IPV6_RE = re.compile(r"(?<![\w:.])(?:[0-9a-f]{0,4}:){2,7}[0-9a-f]{0,4}(?![\w:])", re.I)
_DOMAIN_RE = re.compile(r"(?<![\w@.-])((?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+([a-z]{2,24}))(?![\w-]|\.[a-z0-9])", re.I)
_MD5_RE = re.compile(r"(?<![\w])[a-f0-9]{32}(?![\w])", re.I)
_SHA1_RE = re.compile(r"(?<![\w])[a-f0-9]{40}(?![\w])", re.I)
_SHA256_RE = re.compile(r"(?<![\w])[a-f0-9]{64}(?![\w])", re.I)
_CVE_RE = re.compile(r"\bCVE-(?:19|20)\d{2}-\d{4,7}\b", re.I)
_BTC_LEGACY_RE = re.compile(r"(?<![\w])[13][a-km-zA-HJ-NP-Z1-9]{25,34}(?![\w])")
_BTC_BECH32_RE = re.compile(r"(?<![\w])bc1[ac-hj-np-z02-9]{11,71}(?![\w])", re.I)
_ETH_RE = re.compile(r"(?<![\w])0x[a-f0-9]{40}(?![\w])", re.I)
_ATTACK_TECH_RE = re.compile(r"(?<![\w])T1\d{3}(?:\.\d{3})?(?![\w])")
_ATTACK_TACTIC_RE = re.compile(r"(?<![\w])TA00\d{2}(?![\w])")
_HASHTAG_RE = re.compile(r"(?<![\w&#/])#([^\W\d][\w]{1,138})")
_CASHTAG_RE = re.compile(r"(?<![\w$])\$([A-Za-z]{1,6}(?:\.[A-Za-z]{1,2})?)(?![\w])")

#: TLDs accepted for bare (non-defanged) domain matches. Defanged domains are
#: always accepted because their author clearly meant them as indicators.
_COMMON_TLDS = set("""
com net org io gov edu mil int info biz co us uk de ru cn fr jp br in it nl eu au ca es ch se no fi dk pl ua by kz tr ir iq sy
il sa ae qa eg pk af kp kr tw hk sg my id th vn ph nz za ng ke mx ar cl pe ve cu be at cz sk hu ro bg gr pt ie lt lv ee rs hr
si ba me tv cc ly su to ws pw tk ml ga cf gq xyz top site online club app dev ai live shop store tech cloud link icu buzz vip
win bid loan work click space website fun pro mobi name asia lol news media blog wiki social network digital world today
onion bit i2p email zip mov ru rf one gg sh am fm im is la nu cx ms st vc so ac
""".split())

_BENIGN_DOMAINS = (
    "reddit.com", "redd.it", "redditmedia.com", "redditstatic.com", "imgur.com", "youtube.com", "youtu.be",
    "twitter.com", "x.com", "t.co", "google.com", "wikipedia.org", "archive.org", "github.com",
)

_B58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def _valid_base58check(addr: str) -> bool:
    n = 0
    for ch in addr:
        idx = _B58_ALPHABET.find(ch)
        if idx < 0:
            return False
        n = n * 58 + idx
    raw = n.to_bytes((n.bit_length() + 7) // 8, "big")
    raw = b"\x00" * (len(addr) - len(addr.lstrip("1"))) + raw
    if len(raw) < 5:
        return False
    body, check = raw[:-4], raw[-4:]
    return hashlib.sha256(hashlib.sha256(body).digest()).digest()[:4] == check


def _clean_url(u: str) -> str:
    return u.rstrip(".,;:!?'\"*_~")


INDICATOR_TYPES = (
    "url", "domain", "ipv4", "ipv6", "md5", "sha1", "sha256", "cve", "btc", "eth",
    "attack_technique", "attack_tactic", "hashtag", "cashtag",
)


def _raw_indicators(text: str) -> list[tuple[str, str]]:
    """Return (type, value) occurrences found in ``text`` (refanged)."""
    out: list[tuple[str, str]] = []
    if not text:
        return out
    live = refang(text)
    # Mask e-mail addresses entirely: we never extract them, nor their domains.
    live = _EMAIL_RE.sub(" ", live)

    urls = [_clean_url(m.group(0)) for m in _URL_RE.finditer(live)]
    out += [("url", u) for u in urls if len(u) > 10]
    no_urls = _URL_RE.sub(" ", live)

    for m in _CVE_RE.finditer(live):
        out.append(("cve", m.group(0).upper()))
    for m in _ATTACK_TECH_RE.finditer(live):
        out.append(("attack_technique", m.group(0)))
    for m in _ATTACK_TACTIC_RE.finditer(live):
        out.append(("attack_tactic", m.group(0)))
    for m in _ETH_RE.finditer(live):
        out.append(("eth", m.group(0).lower()))
    no_eth = _ETH_RE.sub(" ", live)
    for rx, kind in ((_SHA256_RE, "sha256"), (_SHA1_RE, "sha1"), (_MD5_RE, "md5")):
        for m in rx.finditer(no_eth):
            v = m.group(0)
            if len(set(v.lower())) > 4:  # skip 000000... / ffff...
                out.append((kind, v.lower()))
    for m in _BTC_BECH32_RE.finditer(no_urls):
        out.append(("btc", m.group(0).lower()))
    for m in _BTC_LEGACY_RE.finditer(no_urls):
        if _valid_base58check(m.group(0)):
            out.append(("btc", m.group(0)))

    for m in _IPV4_RE.finditer(live):
        out.append(("ipv4", m.group(0)))
    for m in _IPV6_RE.finditer(live):
        cand = m.group(0)
        if cand.count(":") < 2 or not re.search(r"[0-9a-f]", cand, re.I):
            continue
        try:
            ip = ipaddress.IPv6Address(cand)
        except ValueError:
            continue
        out.append(("ipv6", ip.compressed))

    # Domains: from URL hosts and bare mentions.
    for u in urls:
        host = re.sub(r"^[a-z]+://", "", u, flags=re.I).split("/", 1)[0].split("?", 1)[0].split("#", 1)[0]
        host = host.rsplit("@", 1)[-1].split(":", 1)[0].lower().strip(".")
        if host and not _IPV4_RE.fullmatch(host) and "." in host:
            out.append(("domain", host))
    defanged_ctx = bool(_DEFANG_MARK.search(text))
    for m in _DOMAIN_RE.finditer(no_urls):
        dom, tld = m.group(1).lower().strip("."), m.group(2).lower()
        if tld in _COMMON_TLDS or (defanged_ctx and _is_defanged_in(dom, text)):
            out.append(("domain", dom))

    for m in _HASHTAG_RE.finditer(no_urls):
        out.append(("hashtag", "#" + m.group(1)))
    for m in _CASHTAG_RE.finditer(no_urls):
        out.append(("cashtag", "$" + m.group(1).upper()))
    return out


def _is_defanged_in(value: str, original: str) -> bool:
    if not original:
        return False
    low = original.lower()
    if value.lower() in low:
        return False
    return value.lower() in refang(low)


def _annotate(kind: str, value: str) -> dict[str, Any]:
    info: dict[str, Any] = {}
    if kind in ("ipv4", "ipv6"):
        try:
            ip = ipaddress.ip_address(value)
            if ip.is_private or ip.is_loopback or ip.is_reserved or ip.is_link_local or ip.is_multicast:
                info["scope"] = "non-routable"
            else:
                info["scope"] = "public"
        except ValueError:
            pass
    elif kind == "domain":
        if value.endswith(".onion"):
            info["note"] = "tor hidden service"
        elif any(value == b or value.endswith("." + b) for b in _BENIGN_DOMAINS):
            info["benign"] = True
    elif kind == "url":
        host = re.sub(r"^[a-z]+://", "", value, flags=re.I).split("/", 1)[0].lower()
        if any(host == b or host.endswith("." + b) for b in _BENIGN_DOMAINS):
            info["benign"] = True
    elif kind == "attack_technique":
        info["reference"] = "https://attack.mitre.org/techniques/" + value.replace(".", "/") + "/"
    elif kind == "cve":
        info["reference"] = "https://nvd.nist.gov/vuln/detail/" + value
    return info


def extract_indicators(
    text: "str | Iterable[Document]",
    doc_id: str | None = None,
    *,
    types: Iterable[str] | None = None,
    include_benign: bool = True,
) -> list[dict]:
    """Extract and deduplicate indicators.

    ``text`` may be a string (optionally tagged with ``doc_id``) or an iterable of
    :class:`Document` (each document's title + text + media urls are scanned).

    Returns a list of dicts sorted by type then count::

        {"type": "domain", "value": "evil.example.com", "defanged": "evil[.]example[.]com",
         "count": 3, "docs": ["reddit:t3_abc"], "was_defanged": True, ...annotations}
    """
    wanted = set(types) if types else None
    agg: dict[tuple[str, str], dict] = {}

    def add(src: str, ref: str | None) -> None:
        for kind, value in _raw_indicators(src):
            if wanted is not None and kind not in wanted:
                continue
            key = (kind, value.lower() if kind not in ("btc",) else value)
            rec = agg.get(key)
            if rec is None:
                rec = {"type": kind, "value": value, "defanged": defang(value, kind), "count": 0, "docs": [],
                       "was_defanged": False}
                rec.update(_annotate(kind, value))
                agg[key] = rec
            rec["count"] += 1
            if ref and ref not in rec["docs"]:
                rec["docs"].append(ref)
            if not rec["was_defanged"] and kind in ("url", "domain", "ipv4", "ipv6") and _is_defanged_in(value, src):
                rec["was_defanged"] = True

    if isinstance(text, str):
        add(text, doc_id)
    else:
        for doc in text:
            # The document's own permalink is provenance, not an indicator: scanning it would make
            # every outlet's article URLs (and its domain) top the IOC list.
            parts = [doc.title or "", doc.text or ""]
            parts += [m.url for m in doc.media if m.url]
            parts += [m.description for m in doc.media if m.description]
            add("\n".join(p for p in parts if p), doc_ref(doc))

    out = list(agg.values())
    if not include_benign:
        out = [r for r in out if not r.get("benign")]
    order = {t: i for i, t in enumerate(INDICATOR_TYPES)}
    out.sort(key=lambda r: (order.get(r["type"], 99), -r["count"], r["value"]))
    return out


def summarize_indicators(indicators: list[dict], top_per_type: int = 10) -> dict[str, Any]:
    """Compact view for prompts: counts by type and the top values (defanged)."""
    by_type: dict[str, list[dict]] = {}
    for rec in indicators:
        by_type.setdefault(rec["type"], []).append(rec)
    summary: dict[str, Any] = {"counts": {k: len(v) for k, v in by_type.items()}, "top": {}}
    for kind, recs in by_type.items():
        recs = sorted((r for r in recs if not r.get("benign")), key=lambda r: -r["count"])[:top_per_type]
        if recs:
            summary["top"][kind] = [
                {"value": r["defanged"], "count": r["count"], "docs": r["docs"][:5]} for r in recs
            ]
    return summary


# ---------------------------------------------------------------------------
# Imagery
# ---------------------------------------------------------------------------

_IMAGE_EXT = re.compile(r"\.(png|jpe?g|gif|webp|bmp)(\?|$)", re.I)


def _from_local_file(doc: Document) -> bool:
    ev = doc.meta.get("evidence") if isinstance(doc.meta, dict) else None
    return doc.platform == "file" or (isinstance(ev, dict) and ev.get("collector") == "files")


def describe_media(docs: Iterable[Document], router: Any, max_images: int = 8, *,
                   prompt: str | None = None, max_workers: int = 4) -> int:
    """Fill ``Media.description`` for image attachments using the vision role.

    Errors are caught per image and recorded in ``doc.meta["vision_errors"]``.
    Returns the number of images successfully described.
    """
    targets: list[tuple[Document, Any]] = []
    for doc in docs:
        for m in doc.media:
            if m.description or not m.url:
                continue
            if m.kind not in ("image", "photo", "img") and not _IMAGE_EXT.search(m.url):
                continue
            targets.append((doc, m))
            if len(targets) >= max_images:
                break
        if len(targets) >= max_images:
            break
    if not targets:
        return 0

    def work(item: tuple[Document, Any]) -> bool:
        doc, media = item
        try:
            # Local image paths are read only for documents the analyst loaded with file:; a path inside
            # collected web/social content must never make VigilEye read and send a local file.
            local = {"allow_local": True} if _from_local_file(doc) else {}
            desc = router.describe_image(media.url, prompt=prompt, **local)
            media.description = (desc or "").strip() or None
            return media.description is not None
        except Exception as e:  # noqa: BLE001 - degrade gracefully
            log.warning("vision failed for %s: %s", media.url, e)
            doc.meta.setdefault("vision_errors", []).append(f"{media.url}: {e}")
            return False

    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(targets)))) as pool:
        return sum(1 for ok in pool.map(work, targets) if ok)


# ---------------------------------------------------------------------------
# Entities
# ---------------------------------------------------------------------------

def _norm_name(name: str) -> str:
    n = (name or "").strip().strip("\"'`").casefold()
    n = re.sub(r"^(?:u/|/u/|@)", "", n)
    n = re.sub(r"^/?r/", "r/", n)
    n = re.sub(r"\s+", " ", n)
    n = re.sub(r"^the ", "", n)
    return n.strip(" .,:;")


def coerce_entity(item: Any) -> Entity | None:
    """Build an :class:`Entity` from loosely-structured LLM output."""
    if isinstance(item, Entity):
        return item
    if isinstance(item, str):
        return Entity(name=item.strip(), type="unknown") if item.strip() else None
    if not isinstance(item, dict):
        return None
    name = str(item.get("name") or item.get("entity") or item.get("value") or "").strip()
    if not name:
        return None
    etype = str(item.get("type") or item.get("category") or item.get("kind") or "unknown").strip().lower() or "unknown"
    try:
        mentions = max(1, int(item.get("mentions") or item.get("count") or 1))
    except (TypeError, ValueError):
        mentions = 1
    aliases = item.get("aliases") or []
    if isinstance(aliases, str):
        aliases = [a.strip() for a in aliases.split(",") if a.strip()]
    ctx = item.get("context") or item.get("description") or item.get("evidence") or []
    if isinstance(ctx, str):
        ctx = [ctx]
    return Entity(name=name, type=etype, mentions=mentions,
                  aliases=[str(a) for a in aliases if str(a).strip() and str(a) != name],
                  context=[str(c) for c in ctx if str(c).strip()][:5])


def coerce_entities(payload: Any) -> list[Entity]:
    if isinstance(payload, dict):
        payload = payload.get("entities") or payload.get("items") or []
    if not isinstance(payload, list):
        return []
    return [e for e in (coerce_entity(x) for x in payload) if e is not None]


def merge_entities(entities: Iterable[Entity], *, max_context: int = 6) -> list[Entity]:
    """Deduplicate entities case-insensitively, merging aliases, mentions and context."""
    merged: list[Entity] = []
    index: dict[str, int] = {}
    type_votes: list[Counter] = []
    for ent in entities:
        if ent is None or not (ent.name or "").strip():
            continue
        keys = [_norm_name(ent.name)] + [_norm_name(a) for a in ent.aliases]
        keys = [k for k in keys if k]
        hit = next((index[k] for k in keys if k in index), None)
        if hit is None:
            new = Entity(name=ent.name.strip(), type=ent.type or "unknown", mentions=ent.mentions or 1,
                         aliases=[], context=[])
            merged.append(new)
            type_votes.append(Counter())
            hit = len(merged) - 1
        else:
            merged[hit].mentions += ent.mentions or 1
        target = merged[hit]
        type_votes[hit][(ent.type or "unknown").lower()] += ent.mentions or 1
        for alias in [ent.name, *ent.aliases]:
            alias = (alias or "").strip()
            if alias and _norm_name(alias) != _norm_name(target.name) and \
                    all(_norm_name(alias) != _norm_name(a) for a in target.aliases):
                target.aliases.append(alias)
        for c in ent.context:
            if c and c not in target.context and len(target.context) < max_context:
                target.context.append(c)
        for k in keys:
            index.setdefault(k, hit)
    for ent, votes in zip(merged, type_votes):
        known = [(t, n) for t, n in votes.most_common() if t not in ("unknown", "other", "")]
        ent.type = known[0][0] if known else (votes.most_common(1)[0][0] if votes else "unknown")
    merged.sort(key=lambda e: (-e.mentions, e.name.casefold()))
    return merged


def extract_entities(docs_chunk: list[Document], router: Any, *, lens: "Lens | None" = None,
                     env: Any = None, max_tokens: int | None = None, author_key: bytes | None = None) -> list[Entity]:
    """Named-entity extraction for one chunk via the cheap ``extractor`` role.

    Raises on LLM failure; callers (the engine) catch and degrade.
    """
    from .lenses import get_lens, template_env

    lens = lens or get_lens("general")
    env = env or template_env()
    documents, nonce = fence_untrusted(render_chunk(docs_chunk, max_tokens, key=author_key))
    prompt = env.get_template("entities.j2").render(
        lens=lens,
        documents=documents, nonce=nonce,
        entity_types=lens.entity_types,
    )
    system = env.get_template("system.j2").render(lens=lens, task="entity extraction", **_tradecraft_ctx())
    payload = router.chat_json(
        [{"role": "system", "content": system}, {"role": "user", "content": prompt}], role="extractor"
    )
    return merge_entities(coerce_entities(payload))


def _tradecraft_ctx() -> dict[str, Any]:
    from . import tradecraft as tc

    return {"icd203": tc.ICD203_BLOCK, "wep_scale": tc.WEP_SCALE, "wep_terms": tc.WEP_TERMS}


__all__ = [
    "refang", "defang", "extract_indicators", "summarize_indicators", "INDICATOR_TYPES",
    "describe_media", "extract_entities", "merge_entities", "coerce_entity", "coerce_entities",
]
