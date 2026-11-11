"""Target policy: which collection targets may be fetched, and SSRF-safe URL checks.

Three callers ask the same question, "may this target be collected?", with different trust:

* ``auto``: an LLM planner proposed the target (``vigileye investigate``). Only TOPIC-level
  targets pass: communities, searches, threads, feeds and pages. Account- or person-level
  targets (``reddit:u/``, ``hn:user:``, ``mastodon:@user``, ``bluesky:@handle``), search
  queries that name an account (``from:``, ``author:``, ``@handle``), profile-page URLs and
  ``file:`` targets are refused.
* ``manual``: the analyst typed the target (CLI). Any collector form is allowed.
* ``api``: an HTTP API client sent the target. Same rules as ``manual``; the server passes
  ``allow_files=False``.

In every mode ``file:`` needs ``allow_files=True``, and every URL-bearing target (``rss:``,
``web:``, a Mastodon instance, a URL inside a third-party collector's target) must be a public
http(s) URL: no other scheme, no credentials, and no host that is, or resolves to, a loopback,
private, link-local, multicast, reserved, CGNAT or cloud-metadata address (IPv4, IPv6, and
IPv4-mapped/6to4/NAT64 IPv6). Name resolution goes through an injectable ``resolver``
(default :func:`socket.getaddrinfo`) and fails closed.

:func:`is_public_url` and :func:`guard_request_hook` are exported for fetchers (the web
collector, image fetching) that must re-check every redirect hop: a check made here at
planning time does not stop DNS rebinding at fetch time.
"""
from __future__ import annotations

import ipaddress
import re
import socket
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable
from urllib.parse import urlsplit, urlunsplit

from .collectors.base import ALIASES, CollectorError, _PREFIX_ALIASES, resolve_target

POLICY_VERSION = "1"
MODES = ("auto", "manual", "api")
MAX_SPEC_LEN = 512
MAX_QUERY_LEN = 200

Resolver = Callable[[str, int], Iterable[Any]]

#: Topic-level target forms an LLM planner may propose (``mode="auto"``), per collector.
AUTO_TARGET_FORMS: tuple[dict, ...] = (
    {"collector": "reddit", "description": "subreddit listings and Reddit post search",
     "forms": ["reddit:r/<subreddit>", "reddit:search:<query>", "reddit:search:<query>@r/<subreddit>"],
     "examples": ["reddit:r/cybersecurity", "reddit:search:ransomware leak site"]},
    {"collector": "hackernews", "description": "Hacker News story search and single discussion threads",
     "forms": ["hn:search:<query>", "hn:item:<numeric story id>"],
     "examples": ["hn:search:open source intelligence", "hn:item:8863"]},
    {"collector": "rss", "description": "RSS/Atom feeds of news sites, blogs, government and CERT bulletins",
     "forms": ["rss:<http(s) feed URL>"],
     "examples": ["rss:https://www.cisa.gov/cybersecurity-advisories/all.xml"]},
    {"collector": "web", "description": "one public web page (robots.txt respected)",
     "forms": ["web:<http(s) page URL>"],
     "examples": ["web:https://www.example.org/report.html"]},
    {"collector": "mastodon", "description": "Mastodon hashtag timelines and status search on one instance",
     "forms": ["mastodon:#<tag>@<instance>", "mastodon:search:<query>@<instance>"],
     "examples": ["mastodon:#osint@infosec.exchange", "mastodon:search:disinformation@mastodon.social"]},
    {"collector": "bluesky", "description": "Bluesky public post search",
     "forms": ["bluesky:search:<query>"],
     "examples": ["bluesky:search:election misinformation"]},
)

_SHORT_PREFIX = {"hackernews": "hn", "files": "file"}
_CTRL = re.compile(r"[\x00-\x1f\x7f]")
_WS = re.compile(r"\s+")
_SUBREDDIT = re.compile(r"[A-Za-z0-9_]{2,50}")
_TAG = re.compile(r"[^\W_][\w]{0,99}")
_HOSTNAME = re.compile(r"(?=.{1,253}$)(?:[a-z0-9_](?:[a-z0-9_-]{0,61}[a-z0-9_])?)(?:\.[a-z0-9_](?:[a-z0-9_-]{0,61}"
                       r"[a-z0-9_])?)*")
_SCHEME = re.compile(r"^([a-z][a-z0-9+.\-]*):(?!\d)", re.I)
_EMBEDDED_URL = re.compile(r"[a-z][a-z0-9+.\-]*://[^\s,]+", re.I)
#: Search syntax that scopes a query to one account (Reddit, Bluesky, Mastodon operators, handles, DIDs).
_PERSON_QUERY = re.compile(r"(?:^|[\s(\"'])(?:(?:author|from|to|user|mentions?):(?=\S)|@[\w.\-]|u/[\w\-]"
                           r"|user/[\w\-]|did:(?:plc|web):)", re.I)
#: URL paths of account/profile pages (refused in auto mode for web: and rss: targets).
_PROFILE_PATH = re.compile(r"/@[^/?#]|^/(?:u|user|users|profile|profiles|people|author|authors|member|members|"
                           r"channel)/[^/?#]|profile\.php", re.I)
_PROFILE_HOST_PATHS = {  # host -> first path segments that are NOT a profile page on that host
    "news.ycombinator.com": {"", "item", "news", "newest", "front", "ask", "show", "best", "active"},
    "hnrss.org": {"", "frontpage", "newest", "polls", "ask", "show", "jobs", "best", "active", "bestcomments",
                  "newcomments", "item"},
    **{h: {"", "hashtag", "search", "explore", "tag", "tags", "topics", "trending", "company"}
       for h in ("twitter.com", "x.com", "instagram.com", "tiktok.com", "facebook.com", "threads.net",
                 "linkedin.com")},
    # every t.me/<name> (and t.me/s/<name>) is a channel or a person
    **{h: {"", "share", "addstickers", "iv"} for h in ("t.me", "telegram.me", "telegram.dog")},
    "vk.com": {"", "search", "feed", "wall", "topic", "video", "press"},
}
#: Hosts where a single-segment path (``/torvalds``) is a user or org profile, while deeper paths
#: (``/torvalds/linux``) are repositories; the listed first segments are site pages, not accounts.
_PROFILE_SINGLE_SEGMENT = {
    h: {"", "advisories", "topics", "search", "trending", "explore", "collections", "features", "security",
        "marketplace", "about", "pricing", "events", "sponsors", "orgs", "users", "help"}
    for h in ("github.com", "gitlab.com", "codeberg.org")
}

_BLOCKED_NAMES = {"localhost", "localhost.localdomain", "ip6-localhost", "ip6-loopback", "metadata",
                  "metadata.google.internal", "instance-data", "instance-data.ec2.internal"}
_BLOCKED_SUFFIXES = (".localhost", ".local", ".localdomain", ".internal", ".intranet", ".lan", ".home.arpa",
                     ".corp", ".private", ".in-addr.arpa", ".ip6.arpa", ".onion")
_METADATA_IPS = frozenset(ipaddress.ip_address(a) for a in (
    "169.254.169.254",  # AWS/GCP/Azure/OCI/DigitalOcean instance metadata
    "169.254.170.2",    # AWS ECS task metadata
    "169.254.169.123",  # AWS time sync
    "168.63.129.16",    # Azure WireServer (publicly routed, VM-only)
    "100.100.100.200",  # Alibaba Cloud metadata
    "192.0.0.192",      # Oracle Cloud metadata
    "fd00:ec2::254",    # AWS IMDS over IPv6
))
_CGNAT = ipaddress.ip_network("100.64.0.0/10")
_NAT64 = ipaddress.ip_network("64:ff9b::/96")


class PolicyError(CollectorError):
    """Raised by :func:`guard_request_hook` when a fetch would leave the public internet."""


@dataclass(frozen=True)
class PolicyDecision:
    """Outcome of :func:`check_target`.

    ``normalized`` is the canonical spec to collect (it resolves to the same collector and
    target); ``kind`` is ``topic``, ``account``, ``file`` or ``other`` (``""`` if unresolvable).
    """

    allowed: bool
    reason: str
    normalized: str
    kind: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------- addresses and URLs
def _parse_ip(value: Any) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    if isinstance(value, (ipaddress.IPv4Address, ipaddress.IPv6Address)):
        return value
    s = str(value or "").strip().strip("[]").split("%", 1)[0]
    try:
        return ipaddress.ip_address(s)
    except ValueError:
        return None


def ip_reason(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> str | None:
    """Why ``ip`` is not a public unicast address, or ``None`` when it is."""
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None:  # judged only by the embedded IPv4, the same on every Python version
            inner = ip_reason(ip.ipv4_mapped)
            return f"IPv4-mapped {inner}" if inner else None
        embedded = ip.sixtofour or (ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF) if ip in _NAT64 else None)
        if embedded is not None and ip_reason(embedded):
            return f"IPv6-embedded {ip_reason(embedded)}"
    if ip in _METADATA_IPS:
        return "cloud metadata address"
    if ip.version == 4 and ip in _CGNAT:
        return "carrier-grade NAT address"
    for label, flag in (("loopback", ip.is_loopback), ("link-local", ip.is_link_local),
                        ("multicast", ip.is_multicast), ("unspecified", ip.is_unspecified),
                        ("reserved", ip.is_reserved), ("private", ip.is_private),
                        ("site-local", getattr(ip, "is_site_local", False))):
        if flag:
            return f"{label} address"
    return None if ip.is_global else "non-global address"


def is_public_ip(address: Any) -> bool:
    """True for a public unicast IPv4/IPv6 address (string or :mod:`ipaddress` object)."""
    ip = _parse_ip(address)
    return ip is not None and ip_reason(ip) is None


def _resolve(host: str, port: int, resolver: Resolver | None) -> list:
    """Addresses for ``host``; ``[]`` on any failure or unparseable answer (fail closed)."""
    try:
        answers = list((resolver or socket.getaddrinfo)(host, port) or [])
    except (OSError, UnicodeError, ValueError):
        return []
    out = []
    for a in answers:
        raw = a[4][0] if isinstance(a, (tuple, list)) and len(a) >= 5 and a[4] else a
        ip = _parse_ip(raw)
        if ip is None:
            return []
        out.append(ip)
    return out


def host_reason(host: str, port: int = 443, *, resolver: Resolver | None = None) -> str | None:
    """Why ``host`` is not a public internet host, or ``None``. Resolves names with ``resolver``."""
    h = str(host or "").strip().strip("[]").rstrip(".").lower()
    if not h:
        return "missing host"
    ip = _parse_ip(h)
    if ip is None and re.fullmatch(r"[0-9a-fx.]+", h):  # 127.1, 0x7f.0.0.1, 2130706433, 0177.0.0.1 ...
        try:
            ip = ipaddress.IPv4Address(socket.inet_aton(h))
        except OSError:
            ip = None
    if ip is not None:
        r = ip_reason(ip)
        return f"host is a non-public address ({r})" if r else None
    try:
        name = h.encode("idna").decode("ascii")
    except UnicodeError:
        return "invalid hostname"
    if not _HOSTNAME.fullmatch(name):
        return "invalid hostname"
    if name in _BLOCKED_NAMES or name.endswith(_BLOCKED_SUFFIXES):
        return "internal hostname"
    if "." not in name:
        return "single-label (intranet) hostname"
    addrs = _resolve(name, port, resolver)
    if not addrs:
        return "hostname did not resolve"
    for a in addrs:
        r = ip_reason(a)
        if r:
            return f"hostname resolves to a non-public address ({r})"
    return None


def check_url(url: str, *, resolver: Resolver | None = None) -> str | None:
    """Why ``url`` may not be fetched, or ``None`` for a public http(s) URL."""
    if not isinstance(url, str) or not url.strip():
        return "empty URL"
    u = url.strip()
    if _CTRL.search(u) or "\\" in u:
        return "control character or backslash in URL"
    try:
        parts = urlsplit(u)
        port = parts.port
    except ValueError:
        return "malformed URL or port"
    scheme = parts.scheme.lower()
    if scheme not in ("http", "https"):
        return f"non-http(s) scheme {scheme + ':' if scheme else '(none)'}"
    if not parts.netloc:
        return "missing host"
    if "@" in parts.netloc:
        return "credentials in URL"
    return host_reason(parts.hostname or "", port or (443 if scheme == "https" else 80), resolver=resolver)


def is_public_url(url: str, resolver: Resolver | None = None) -> bool:
    """True when ``url`` is http(s), carries no credentials and its host is (and resolves to) public addresses."""
    return check_url(url, resolver=resolver) is None


def guard_request_hook(resolver: Resolver | None = None) -> Callable[[Any], None]:
    """An ``httpx`` request event hook that refuses non-public URLs, including every redirect hop.

    Usage: ``httpx.Client(event_hooks={"request": [guard_request_hook()]}, follow_redirects=True)``.
    Raises :class:`PolicyError` (a ``CollectorError``) before the request is sent.
    """

    def hook(request: Any) -> None:
        url = str(getattr(request, "url", request))
        reason = check_url(url, resolver=resolver)
        if reason:
            raise PolicyError(f"refusing to fetch {url}: {reason}")

    return hook


def _canon_url(url: str) -> str:
    """Lower-case scheme and host, drop the fragment; the path and query are kept as given."""
    try:
        p = urlsplit(url.strip())
    except ValueError:
        return url.strip()
    return urlunsplit((p.scheme.lower(), p.netloc.lower(), p.path or "/", p.query, ""))


# ---------------------------------------------------------------------------- target grammar
def _q(text: str) -> str:
    return _WS.sub(" ", text or "").strip()


def _rss_urls(target: str) -> list[str]:
    # Mirrors RSSCollector.collect: one or more feed URLs separated by commas/whitespace.
    return [u.strip() for u in re.split(r"[\s,]+(?=https?://)", target.strip()) if u.strip()]


def _web_url(target: str) -> tuple[str | None, str]:
    """Mirrors WebCollector.collect (a bare host gets ``https://``); returns ``(reason, url)``."""
    t = target.strip()
    if not re.match(r"^https?://", t, re.I):
        m = _SCHEME.match(t)
        if m:
            return f"non-http(s) scheme {m.group(1).lower()}:", t
        t = "https://" + t
    return None, t


def _parse(name: str, target: str) -> dict:
    """Canonical view of ``(collector, target)``: ``{kind, form, normalized, urls, query}``.

    ``kind`` is topic/account/file/other; ``form`` names the collector-specific shape (sub,
    search, item, tag, ...); ``urls`` are the URLs the collector will fetch from the target.
    """
    t = target.strip()
    out = {"kind": "topic", "form": "", "urls": [], "query": None,
           "normalized": f"{_SHORT_PREFIX.get(name, name)}:{t}"}
    if name == "reddit":
        from .collectors.reddit import RedditCollector

        form, value, scope = RedditCollector.parse_target(t)
        if form == "user":
            out.update(kind="account", form="user", normalized=f"reddit:u/{value}")
        elif form == "search":
            out.update(form="search", query=_q(value),
                       normalized=f"reddit:search:{_q(value)}" + (f"@r/{scope}" if scope else ""), scope=scope)
        else:
            out.update(form="sub" if t.lower().startswith("r/") else "bare-sub",
                       normalized=f"reddit:r/{value.lower()}", sub=value)
        # r/u_<name> is Reddit's per-user profile subreddit: account-level however it is spelled
        if str(out.get("sub") or out.get("scope") or "").lower().startswith("u_"):
            out["kind"] = "account"
    elif name == "hackernews":
        kind, _, value = t.partition(":")
        kind, value = kind.lower(), value.strip()
        if not value and kind not in {"front", "front_page", "frontpage"}:
            kind, value = "search", t
        canon = {"s": "search", "q": "search", "new": "recent", "date": "recent", "comment": "comments",
                 "author": "user", "u": "user", "front_page": "front", "frontpage": "front", "thread": "item",
                 "story": "item", "id": "item"}.get(kind, kind)
        if canon not in {"search", "recent", "comments", "user", "front", "item"}:
            canon, value = "search", t  # the collector searches the whole target for unknown kinds
        if canon == "item":
            value = value.split("id=")[-1].strip()
        if canon in {"search", "recent", "comments"}:
            value = _q(value)
            out["query"] = value
        out.update(form=canon, kind="account" if canon == "user" else "topic",
                   normalized="hn:front" if canon == "front" else f"hn:{canon}:{value}", value=value)
    elif name == "rss":
        urls = _rss_urls(t)
        out.update(form="feed", urls=urls, normalized="rss:" + ",".join(_canon_url(u) for u in urls))
    elif name == "web":
        reason, url = _web_url(t)
        out.update(form="page", urls=[url], url_reason=reason, explicit=bool(re.match(r"^https?://", t, re.I)),
                   normalized=f"web:{_canon_url(url) if not reason else url}")
    elif name == "mastodon":
        from .collectors.mastodon import DEFAULT_INSTANCE, MastodonCollector

        form, value, inst = MastodonCollector.parse_target(t, DEFAULT_INSTANCE)
        out.update(form=form, instance=inst, value=value, urls=[f"https://{inst}/"])
        if form == "account":
            out.update(kind="account", normalized=f"mastodon:@{value}@{inst}")
        elif form == "search":
            out.update(query=_q(value), normalized=f"mastodon:search:{_q(value)}@{inst}")
        elif form == "tag":
            out.update(normalized=f"mastodon:#{value.lower()}@{inst}")
        else:
            word = "local" if t.lower().startswith("local") else "public"
            out.update(normalized=f"mastodon:{word}@{inst}")
    elif name == "bluesky":
        if t.lower().startswith(("search:", "q:")):
            q = _q(t.split(":", 1)[1])
            out.update(form="search", query=q, normalized=f"bluesky:search:{q}")
        elif t.startswith("#"):
            out.update(form="tag", normalized=f"bluesky:{_q(t)}")
        else:
            actor = t.lstrip("@")
            if actor.startswith("https://bsky.app/profile/"):
                actor = actor.split("/profile/", 1)[1].split("/", 1)[0]
            out.update(kind="account", form="actor", normalized=f"bluesky:@{actor}")
    elif name == "files":
        out.update(kind="file", form="file", normalized=f"file:{t}")
    else:
        out.update(kind="other", form="other", urls=_EMBEDDED_URL.findall(t))
    if out["kind"] == "topic" and out["query"] and _PERSON_QUERY.search(out["query"]):
        out["kind"] = "account"  # a search scoped to an account is an account-level target
    if out["kind"] == "topic" and name in ("web", "rss") and any(_profile_url(u) for u in out["urls"]):
        out["kind"] = "account"
    return out


_URL_IN_TEXT = re.compile(r"https?://[^\s<>\"'`\])}|]+", re.I)
_TELEGRAM_HOSTS = ("t.me", "telegram.me", "telegram.dog")


def _bare_host(host: str) -> str:
    host = (host or "").lower().rstrip(".")
    for pre in ("www.", "m.", "mobile.", "old.", "new."):
        if host.startswith(pre):
            return host[len(pre):]
    return host


def mask_profile_url(url: str) -> str:
    """``url`` with the account-naming part masked as ``[account]`` on known social hosts.

    Covers the hosts of :data:`_PROFILE_HOST_PATHS` (``x.com/janedoe/status/1`` ->
    ``x.com/[account]/status/1``, ``t.me/s/channel/45``, ``linkedin.com/in/name``), repository
    owners on :data:`_PROFILE_SINGLE_SEGMENT` hosts (``github.com/[account]/repo``) and Hacker
    News ``?id=`` user links. Other URLs are returned unchanged.
    """
    try:
        p = urlsplit(url)
    except ValueError:
        return url
    host = _bare_host(p.hostname or "")
    parts = p.path.split("/")
    idx = [i for i, x in enumerate(parts) if x]
    query = p.query
    changed = False

    def mask(k: int) -> None:
        nonlocal changed
        if k < len(idx):
            parts[idx[k]] = "[account]"
            changed = True

    if host in ("news.ycombinator.com", "hnrss.org"):
        if not p.path.startswith("/item") and re.search(r"(?i)(?:^|&)id=", query):
            query, changed = re.sub(r"(?i)(^|&)(id=)[^&]*", r"\1\2[account]", query), True
    elif host in _PROFILE_HOST_PATHS and idx:
        first = parts[idx[0]].lower()
        if host in _TELEGRAM_HOSTS and first == "s":
            mask(1)
        elif host == "linkedin.com" and first in ("in", "pub"):
            mask(1)
        elif first not in _PROFILE_HOST_PATHS[host]:
            mask(0)
    elif host in _PROFILE_SINGLE_SEGMENT and idx:
        first = parts[idx[0]].lower()
        if first in ("orgs", "users"):
            mask(1)
        elif first not in _PROFILE_SINGLE_SEGMENT[host]:
            mask(0)
    if not changed:
        return url
    return urlunsplit(p._replace(path="/".join(parts), query=query))


def mask_profile_urls(text: str) -> str:
    """Apply :func:`mask_profile_url` to every http(s) URL inside ``text``."""
    return _URL_IN_TEXT.sub(lambda m: mask_profile_url(m.group(0)), text or "")


def _profile_url(url: str) -> bool:
    try:
        p = urlsplit(url)
    except ValueError:
        return False
    host = (p.hostname or "").lower().rstrip(".")  # "x.com." fetches x.com: compare without the root dot
    for pre in ("www.", "m.", "mobile.", "old.", "new."):
        if host.startswith(pre):
            host = host[len(pre):]
            break
    path = p.path or "/"
    if host == "reddit.com" and re.match(r"^/(?:u|user)/", path, re.I):
        return True
    allowed = _PROFILE_HOST_PATHS.get(host)
    if allowed is not None and path.strip("/").split("/", 1)[0].lower() not in allowed:
        return True
    single = _PROFILE_SINGLE_SEGMENT.get(host)
    segs = [x.lower() for x in path.split("/") if x]
    if single is not None and segs and (segs[0] in ("orgs", "users") or (len(segs) == 1 and segs[0] not in single)):
        return True
    if host in ("news.ycombinator.com", "hnrss.org") and re.search(r"(?:^|&)id=", p.query or "") \
            and not path.startswith("/item"):
        return True
    return bool(_PROFILE_PATH.search(path))


def _auto_reason(name: str, info: dict) -> str | None:
    """Why a parsed target is not one of the topic-level forms in :data:`AUTO_TARGET_FORMS`."""
    form = info["form"]
    if info["kind"] == "account":
        return "account/person-level target: auto mode allows only topic-level targets"
    if info.get("query") is not None and not 0 < len(info["query"]) <= MAX_QUERY_LEN:
        return f"search query must be 1-{MAX_QUERY_LEN} characters"
    if name == "reddit":
        if form in ("sub", "bare-sub") and _SUBREDDIT.fullmatch(info.get("sub", "")):
            return None
        if form == "search" and (not info.get("scope") or _SUBREDDIT.fullmatch(info["scope"])):
            return None
        return "allowed reddit forms are reddit:r/<subreddit> and reddit:search:<query>[@r/<subreddit>]"
    if name == "hackernews":
        if form == "search" or (form == "item" and info.get("value", "").isdigit()):
            return None
        return "allowed Hacker News forms are hn:search:<query> and hn:item:<numeric id>"
    if name in ("rss", "web"):
        if not info["urls"] or (name == "web" and not info.get("explicit")):
            return f"{name}: targets need an explicit http(s):// URL"
        return None
    if name == "mastodon":
        if form == "tag" and _TAG.fullmatch(info.get("value", "")):
            return None
        if form == "search" and "@" not in info.get("query", "@"):
            return None
        return "allowed Mastodon forms are mastodon:#<tag>@<instance> and mastodon:search:<query>@<instance>"
    if name == "bluesky":
        return None if form == "search" else "the allowed Bluesky form is bluesky:search:<query>"
    return f"collector {name!r} is not available to the planner"


def _network_reason(name: str, info: dict, resolver: Resolver | None) -> str | None:
    if info.get("url_reason"):
        return info["url_reason"]
    if name == "mastodon" and not _HOSTNAME.fullmatch(info.get("instance", "")):
        return "invalid Mastodon instance hostname"
    if name == "rss" and not info["urls"]:
        return "rss: target has no http(s) feed URL"
    for url in info["urls"]:
        reason = check_url(url, resolver=resolver)
        if reason:
            return f"{reason} ({url})"
    return None


def _collector_names(names: Iterable[str]) -> set[str]:
    out = set()
    for n in names:
        low = str(n).strip().lower().rstrip(":")
        out.add(ALIASES.get(low) or _PREFIX_ALIASES.get(low) or low)
    return out


def check_target(spec: str, *, mode: str = "auto", allow_files: bool = False,
                 allowed_collectors: Iterable[str] | None = None,
                 resolver: Resolver | None = None) -> PolicyDecision:
    """Decide whether ``spec`` may be collected in ``mode`` (``auto`` | ``manual`` | ``api``).

    ``allowed_collectors`` (names, aliases or prefixes such as ``hn``) restricts the collectors;
    ``resolver(host, port)`` replaces :func:`socket.getaddrinfo` (it may return getaddrinfo
    tuples or address strings). Never raises for a bad spec: the reason says what is wrong.
    """
    if mode not in MODES:
        raise ValueError(f"unknown policy mode {mode!r}; expected one of {MODES}")

    def deny(reason: str, normalized: str = "", kind: str = "") -> PolicyDecision:
        return PolicyDecision(False, reason, normalized, kind)

    if not isinstance(spec, str) or not spec.strip():
        return deny("empty target spec")
    s = spec.strip()
    if len(s) > MAX_SPEC_LEN:
        return deny(f"target spec longer than {MAX_SPEC_LEN} characters")
    if _CTRL.search(s):
        return deny("control characters in target spec")
    try:
        name, target = resolve_target(s)
    except CollectorError as e:
        return deny(f"unresolvable target: {e}")
    info = _parse(name, target)
    norm, kind = info["normalized"], info["kind"]
    if allowed_collectors is not None and name not in _collector_names(allowed_collectors):
        return deny(f"collector {name!r} is not allowed here", norm, kind)
    if name == "files":
        if mode == "auto":
            return deny("file: targets are never allowed in auto (planner) mode", norm, kind)
        if not allow_files:
            return deny("file: targets are disabled here (allow_files=False)", norm, kind)
        return PolicyDecision(True, "ok", norm, kind)
    if mode == "auto":
        reason = _auto_reason(name, info)
        if reason:
            return deny(reason, norm, kind)
    reason = _network_reason(name, info, resolver)
    if reason:
        return deny(f"non-public network target: {reason}", norm, kind)
    return PolicyDecision(True, "ok", norm, kind)


def check_targets(specs: Iterable[str], **kwargs: Any) -> tuple[list[str], list[dict]]:
    """Batch form for API handlers: ``(allowed normalized specs, [{spec, reason}] rejections)``."""
    ok, bad = [], []
    for spec in specs or []:
        d = check_target(spec, **kwargs)
        if d.allowed:
            ok.append(d.normalized)
        else:
            bad.append({"spec": spec if isinstance(spec, str) else repr(spec), "reason": d.reason})
    return ok, bad


__all__ = ["AUTO_TARGET_FORMS", "MODES", "POLICY_VERSION", "PolicyDecision", "PolicyError", "check_target",
           "check_targets", "check_url", "guard_request_hook", "host_reason", "ip_reason", "is_public_ip",
           "is_public_url"]
