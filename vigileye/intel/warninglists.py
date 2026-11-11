"""MISP-style warninglists: annotate known-benign infrastructure instead of dropping it.

A warninglist is a named set of values with a match type, following MISP's
``misp-warninglists`` format (as read by ``pymispwarninglists``):

* ``string``: case-insensitive exact match (IP addresses are compared in canonical form);
* ``substring``: the entry occurs anywhere in the value;
* ``hostname``: the host (taken from a URL or domain) equals the entry or is a subdomain of it;
* ``cidr``: the IP address (or a URL's IP host) lies in one of the networks;
* ``regex``: ``re.search``; PCRE-style ``/pattern/flags`` entries are accepted.

``category`` is ``false_positive`` (the value is noise, e.g. RFC 1918 space) or ``known`` (the value is
recognised infrastructure, e.g. a link shortener). :func:`annotate_indicators` writes
``rec['warninglists']`` and sets ``rec['benign']``, so prompt summaries hide the value while the report
still lists it. The lists describe infrastructure, never individuals.

Values are refanged before matching, so ``hxxps://bit[.]ly/x`` matches ``bit.ly``. This module imports
nothing from :mod:`vigileye.analysis` at import time, so the engine can import it without a cycle.
"""
from __future__ import annotations

import ipaddress
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit

log = logging.getLogger("vigileye.intel")

MATCH_TYPES = ("string", "substring", "hostname", "cidr", "regex")
CATEGORIES = ("false_positive", "known")
KIND_TO_MISP: dict[str, list[str]] = {
    "domain": ["domain", "hostname"], "ipv4": ["ip-src", "ip-dst"], "ipv6": ["ip-src", "ip-dst"], "url": ["url"],
    "md5": ["md5"], "sha1": ["sha1"], "sha256": ["sha256"],
    # Beyond the core mapping: other indicator kinds and their nearest MISP attribute types.
    "cve": ["vulnerability"], "btc": ["btc"], "eth": ["eth"],
}
_HOST_ATTRS = ["hostname", "domain", "url", "domain|ip"]
_IP_ATTRS = ["ip-src", "ip-dst", "domain|ip", "ip-src|port", "ip-dst|port"]
_PCRE = re.compile(r"/(.*)/([a-zA-Z]*)", re.S)
_PCRE_FLAGS = {"i": re.I, "m": re.M, "s": re.S, "x": re.X}


def _refang(text: str) -> str:
    from ..analysis.enrich import refang  # lazy: vigileye.analysis may import this module

    return refang(text)


def _ip(value: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        return ipaddress.ip_address(value.strip().strip("[]"))
    except ValueError:
        return None


def _norm_str(value: str) -> str:
    v = value.strip().lower()
    ip = _ip(v)
    return ip.compressed if ip is not None else v


def _norm_host(value: str) -> str:
    v = value.strip().lower()
    if "://" in v:
        v = urlsplit(v).hostname or ""
    return re.sub(r"^\*?\.+", "", v).rstrip(".")


def _host_of(value: str, kind: str | None) -> str | None:
    """Host part of a URL or domain value (``None`` for other indicator kinds)."""
    v = value.strip()
    if kind == "url" or "://" in v:
        try:
            host = urlsplit(v if "://" in v else "http://" + v).hostname
        except ValueError:
            return None
    elif kind in (None, "", "domain", "hostname"):
        host = v
    else:
        return None
    host = (host or "").lower().rstrip(".")
    return host or None


def _compile(entry: str, list_name: str) -> re.Pattern | None:
    m = _PCRE.fullmatch(entry)
    pattern, flags = (m.group(1), m.group(2)) if m else (entry, "")
    fl = 0
    for ch in flags:
        fl |= _PCRE_FLAGS.get(ch.lower(), 0)
    try:
        return re.compile(pattern, fl)
    except re.error as e:
        log.warning("warninglist %s: skipping invalid regex %r: %s", list_name, entry, e)
        return None


@dataclass
class Warninglist:
    """One warninglist. The match index is built at construction; rebuild the object after editing ``values``."""

    name: str
    values: list[str]
    type: str = "string"
    matching_attributes: list[str] = field(default_factory=list)
    category: str = "false_positive"
    description: str = ""
    _strings: set[str] = field(default_factory=set, init=False, repr=False, compare=False)
    _subs: list[str] = field(default_factory=list, init=False, repr=False, compare=False)
    _regexes: list[re.Pattern] = field(default_factory=list, init=False, repr=False, compare=False)
    _nets: dict[int, dict[int, set[int]]] = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.type = str(self.type or "string").strip().lower()
        if self.type not in MATCH_TYPES:
            raise ValueError(f"warninglist {self.name!r}: unsupported type {self.type!r} (expected {MATCH_TYPES})")
        cat = str(self.category or "false_positive").strip().lower().replace("-", "_")
        self.category = cat if cat in CATEGORIES else "false_positive"
        self.values = [str(v) for v in self.values or [] if str(v).strip()]
        self.matching_attributes = [str(a).strip().lower() for a in self.matching_attributes or [] if str(a).strip()]
        if self.type == "string":
            self._strings = {_norm_str(v) for v in self.values}
        elif self.type == "hostname":
            self._strings = {h for h in (_norm_host(v) for v in self.values) if h}
        elif self.type == "substring":
            self._subs = [v.strip().lower() for v in self.values if v.strip()]
        elif self.type == "regex":
            self._regexes = [rx for rx in (_compile(v, self.name) for v in self.values) if rx is not None]
        else:  # cidr: index network prefixes by (version, prefix length) for O(#lengths) lookups
            for v in self.values:
                try:
                    net = ipaddress.ip_network(v.strip(), strict=False)
                except ValueError:
                    log.warning("warninglist %s: skipping invalid network %r", self.name, v)
                    continue
                shift = net.max_prefixlen - net.prefixlen
                self._nets.setdefault(net.version, {}).setdefault(net.prefixlen, set()).add(
                    int(net.network_address) >> shift)

    def applies_to(self, kind: str | None) -> bool:
        """Whether this list should be consulted for an indicator of ``kind`` (via ``matching_attributes``)."""
        if not self.matching_attributes or not kind:
            return True
        attrs = set(KIND_TO_MISP.get(kind, [kind]))
        if kind == "url" and self.type in ("hostname", "cidr"):
            attrs |= {"hostname", "domain", "ip-src", "ip-dst"}  # these types match on the URL's host
        return bool(attrs.intersection(self.matching_attributes))

    def matches(self, value: str, kind: str) -> bool:
        """Whether the (already refanged) ``value`` of indicator ``kind`` is on this list."""
        if not value or not self.applies_to(kind):
            return False
        if self.type == "string":
            return _norm_str(value) in self._strings
        if self.type == "substring":
            low = value.lower()
            return any(s in low for s in self._subs)
        if self.type == "regex":
            return any(rx.search(value) for rx in self._regexes)
        if self.type == "hostname":
            host = _host_of(value, kind)
            if not host:
                return False
            parts = host.split(".")
            return any(".".join(parts[i:]) in self._strings for i in range(len(parts)))
        ip = _ip(value) if kind in ("ipv4", "ipv6", None, "") else None
        if ip is None and (kind == "url" or "://" in value):
            ip = _ip(_host_of(value, "url") or "")
        if ip is None:
            return False
        n = int(ip)
        return any((n >> (ip.max_prefixlen - plen)) in keys for plen, keys in self._nets.get(ip.version, {}).items())


def _wl(name: str, kind: str, values: str, attrs: list[str], category: str, description: str) -> Warninglist:
    return Warninglist(name=name, values=values.split(), type=kind, matching_attributes=list(attrs),
                       category=category, description=description)


BUILTIN_LISTS: list[Warninglist] = [
    _wl("rfc1918-and-bogons", "cidr", """
        10.0.0.0/8 172.16.0.0/12 192.168.0.0/16 100.64.0.0/10 127.0.0.0/8 169.254.0.0/16 0.0.0.0/8
        192.0.0.0/24 192.0.2.0/24 198.18.0.0/15 198.51.100.0/24 203.0.113.0/24 224.0.0.0/4 240.0.0.0/4
        ::/128 ::1/128 fc00::/7 fe80::/10 ff00::/8 2001:db8::/32
    """, _IP_ATTRS + ["url"], "false_positive",
        "Private (RFC 1918), shared (RFC 6598), loopback, link-local, documentation (RFC 5737/3849), "
        "benchmarking, multicast and reserved address space: not routable public infrastructure."),
    _wl("public-dns-resolvers", "string", """
        8.8.8.8 8.8.4.4 1.1.1.1 1.0.0.1 9.9.9.9 149.112.112.112 208.67.222.222 208.67.220.220 4.2.2.1 4.2.2.2
        94.140.14.14 94.140.15.15 76.76.2.0 76.76.10.0 185.228.168.9 185.228.169.9
        2001:4860:4860::8888 2001:4860:4860::8844 2606:4700:4700::1111 2606:4700:4700::1001 2620:fe::fe 2620:fe::9
        2620:119:35::35 2620:119:53::53 2a10:50c0::ad1:ff 2a10:50c0::ad2:ff
        dns.google one.one.one.one cloudflare-dns.com dns.quad9.net
    """, _IP_ATTRS + ["domain", "hostname"], "false_positive",
        "Well-known public recursive DNS resolvers (Google, Cloudflare, Quad9, OpenDNS, Level3, AdGuard, "
        "Control D, CleanBrowsing)."),
    _wl("reserved-test-domains", "hostname", """
        example.com example.net example.org test example invalid localhost local home.arpa
    """, _HOST_ATTRS, "false_positive",
        "Reserved and special-use names (RFC 2606, RFC 6761, RFC 6762, RFC 8375): documentation, test and "
        "local-only domains."),
    _wl("link-shorteners", "hostname", """
        bit.ly bitly.com j.mp t.co tinyurl.com goo.gl ow.ly buff.ly is.gd v.gd rb.gy cutt.ly shorturl.at t.ly
        tiny.cc rebrand.ly bl.ink lnkd.in dlvr.it s.id trib.al amzn.to fb.me wp.me aka.ms
    """, _HOST_ATTRS, "known",
        "Public URL-shortening services. The shortener is benign infrastructure; the destination is not checked."),
    _wl("major-platforms-cdn", "hostname", """
        reddit.com redd.it redditmedia.com redditstatic.com imgur.com youtube.com youtu.be ytimg.com
        twitter.com x.com twimg.com google.com gstatic.com wikipedia.org wikimedia.org archive.org github.com
        facebook.com fbcdn.net instagram.com cdninstagram.com linkedin.com licdn.com tiktok.com bsky.app
        bsky.social mastodon.social ycombinator.com microsoft.com apple.com telegram.org substack.com medium.com
    """, _HOST_ATTRS, "known",
        "Front ends and first-party static CDNs of major platforms that collected content links to. "
        "User-content hosts that are often abused (raw.githubusercontent.com, cloud storage buckets) are "
        "deliberately not listed."),
    _wl("security-references", "hostname", """
        mitre.org cve.org nist.gov cisa.gov first.org cert.org ncsc.gov.uk cert.europa.eu enisa.europa.eu
        virustotal.com abuse.ch exploit-db.com otx.alienvault.com malpedia.caad.fkie.fraunhofer.de
        hybrid-analysis.com any.run attack.mitre.org msrc.microsoft.com
    """, _HOST_ATTRS, "known",
        "Vulnerability databases, CERTs and threat-intelligence reference sites that reports cite as sources."),
]


def _from_misp(data: Any, fallback_name: str) -> Warninglist:
    if not isinstance(data, dict) or not isinstance(data.get("list"), list):
        raise ValueError("not a warninglist (missing 'list' array)")
    return Warninglist(name=str(data.get("name") or fallback_name), values=[str(v) for v in data["list"]],
                       type=str(data.get("type") or "string"),
                       matching_attributes=list(data.get("matching_attributes") or []),
                       category=str(data.get("category") or "false_positive"),
                       description=str(data.get("description") or ""))


def load_misp_warninglists(directory: str | Path) -> list[Warninglist]:
    """Load warninglists from a MISP/misp-warninglists clone (``lists/*/list.json``) or a flat folder of ``*.json``.

    Files that cannot be read or parsed, or that use an unknown type, are logged and skipped.
    """
    root = Path(directory).expanduser()
    if not root.is_dir():
        log.warning("warninglists directory %s does not exist", root)
        return []
    files = sorted(root.glob("lists/*/list.json")) or sorted(root.glob("*/list.json")) or sorted(root.glob("*.json"))
    out: list[Warninglist] = []
    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            fallback = path.parent.name if path.name == "list.json" else path.stem
            out.append(_from_misp(data, fallback))
        except (OSError, ValueError, TypeError) as e:
            log.warning("skipping warninglist %s: %s", path, e)
    return out


def match(value: str, kind: str, lists: Iterable[Warninglist] | None = None) -> list[str]:
    """Names of the lists that ``value`` (an indicator of ``kind``, possibly defanged) is on."""
    if not value:
        return []
    live = _refang(str(value)).strip()
    return [wl.name for wl in (BUILTIN_LISTS if lists is None else lists) if wl.matches(live, kind)]


def annotate_indicators(indicators: list[dict], lists: Iterable[Warninglist] | None = None, *,
                        benign_categories: Iterable[str] = CATEGORIES) -> int:
    """Annotate indicator records in place; returns how many matched at least one list.

    Each match sets ``rec['warninglists']`` (sorted list names) and, when a matching list's category is
    in ``benign_categories``, ``rec['benign'] = True``. Nothing is removed: the report keeps every value,
    while ``summarize_indicators`` hides benign ones from prompts. Re-running is safe.
    """
    wls = list(BUILTIN_LISTS if lists is None else lists)
    benign_cats = set(benign_categories)
    hits = 0
    for rec in indicators or []:
        if not isinstance(rec, dict) or not rec.get("value"):
            continue
        live = _refang(str(rec["value"])).strip()
        matched = [wl for wl in wls if wl.matches(live, rec.get("type"))]
        if not matched:
            continue
        hits += 1
        rec["warninglists"] = sorted({*(rec.get("warninglists") or []), *(wl.name for wl in matched)})
        if any(wl.category in benign_cats for wl in matched):
            rec["benign"] = True
    return hits


__all__ = [
    "Warninglist", "BUILTIN_LISTS", "KIND_TO_MISP", "MATCH_TYPES", "CATEGORIES", "load_misp_warninglists",
    "match", "annotate_indicators",
]
