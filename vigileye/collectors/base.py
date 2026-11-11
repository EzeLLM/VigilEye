"""Collector base class, registry and target resolution.

Collectors turn a *target spec* (``reddit:r/osint``, ``hn:search:ransomware``,
``rss:https://...``) into a stream of :class:`vigileye.models.Document`.

Scope: collectors only gather PUBLIC content (communities, topics, feeds and
public posts of public accounts). They intentionally provide no
de-anonymisation, identity-linking or location-finding capabilities.
"""

from __future__ import annotations

import html
import logging
import os
import random
import re
import threading
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import Any, Callable, ClassVar, Iterator

import httpx

log = logging.getLogger("vigileye.collectors")

DEFAULT_USER_AGENT = "VigilEye/2.0 (+https://github.com/EzeLLM/VigilEye)"
RETRY_STATUSES = {429, 500, 502, 503, 504, 520, 521, 522, 524}
MAX_RESPONSE_BYTES = 32 << 20   # largest response body any collector buffers (collectors.<name>.max_response_bytes)
FETCH_DEADLINE_S = 120.0        # overall time to read one response body (httpx timeouts are per read)

REGISTRY: dict[str, type["Collector"]] = {}
ALIASES: dict[str, str] = {}


class CollectorError(RuntimeError):
    """Raised when a collector cannot fulfil a request."""


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def register(cls: type["Collector"] | None = None, *, aliases: tuple[str, ...] = ()):
    """Class decorator adding a collector to :data:`REGISTRY`.

    Usable as ``@register`` or ``@register(aliases=("hn",))``.
    """

    def _wrap(klass: type["Collector"]) -> type["Collector"]:
        name = getattr(klass, "name", None)
        if not name:
            raise ValueError(f"{klass.__name__} must define a 'name' ClassVar")
        REGISTRY[name] = klass
        for alias in aliases:
            ALIASES[alias] = name
        return klass

    if cls is not None:
        return _wrap(cls)
    return _wrap


def _ensure_loaded() -> None:
    # Importing the package registers all built-in collectors.
    if len(REGISTRY) < 2:
        import importlib

        importlib.import_module("vigileye.collectors")


def get_collector(name: str, config: dict | None = None, http: httpx.Client | None = None) -> "Collector":
    """Instantiate a registered collector by name (or alias)."""
    _ensure_loaded()
    low = name.lower()
    key = ALIASES.get(low) or _PREFIX_ALIASES.get(low) or low
    if key not in REGISTRY:
        raise CollectorError(f"Unknown collector '{name}'. Available: {', '.join(sorted(REGISTRY))}")
    return REGISTRY[key](config=config, http=http)


def list_collectors() -> list[dict]:
    """Return ``[{name, description, examples}]`` for every registered collector."""
    _ensure_loaded()
    return [
        {
            "name": c.name,
            "description": c.description,
            "examples": list(c.examples),
        }
        for _, c in sorted(REGISTRY.items())
    ]


# --------------------------------------------------------------------------- #
# Target resolution
# --------------------------------------------------------------------------- #
_PREFIX_ALIASES = {
    "reddit": "reddit",
    "hn": "hackernews",
    "hackernews": "hackernews",
    "rss": "rss",
    "atom": "rss",
    "feed": "rss",
    "web": "web",
    "url": "web",
    "mastodon": "mastodon",
    "masto": "mastodon",
    "fedi": "mastodon",
    "bluesky": "bluesky",
    "bsky": "bluesky",
    "file": "files",
    "files": "files",
}
_FILE_EXTS = (".jsonl", ".ndjson", ".json", ".csv", ".txt", ".md", ".markdown")


def resolve_target(spec: str) -> tuple[str, str]:
    """Map a target spec to ``(collector_name, target)``.

    Examples::

        reddit:r/osint            -> ("reddit", "r/osint")
        r/osint                   -> ("reddit", "r/osint")      (legacy)
        u/someone                 -> ("reddit", "u/someone")    (legacy)
        hn:search:llm             -> ("hackernews", "search:llm")
        rss:https://x/feed.xml    -> ("rss", "https://x/feed.xml")
        https://example.com       -> ("web", "https://example.com")
        mastodon:#osint@infosec.exchange -> ("mastodon", "#osint@infosec.exchange")
        bluesky:@handle.bsky.social      -> ("bluesky", "@handle.bsky.social")
        file:data.jsonl / ./data.jsonl   -> ("files", "data.jsonl")
    """
    if spec is None:
        raise CollectorError("Empty target spec")
    s = spec.strip()
    if not s:
        raise CollectorError("Empty target spec")

    # Legacy reddit shorthands.
    if re.match(r"^/?(r|u|user)/[\w\-]+/?$", s, re.I):
        s2 = s.lstrip("/").rstrip("/")
        if s2.lower().startswith("user/"):
            s2 = "u/" + s2[5:]
        return "reddit", s2

    # Bare URLs.
    if re.match(r"^https?://", s, re.I):
        low = s.lower().split("?")[0]
        if low.endswith((".rss", ".atom", "/feed", "/rss", "/feed/", "/rss/", ".xml")):
            return "rss", s
        if "reddit.com/r/" in low or "reddit.com/user/" in low or "reddit.com/u/" in low:
            m = re.search(r"reddit\.com/(r|u|user)/([\w\-]+)", s, re.I)
            if m:
                kind = "r" if m.group(1).lower() == "r" else "u"
                return "reddit", f"{kind}/{m.group(2)}"
        return "web", s

    # prefix:rest
    if ":" in s:
        prefix, rest = s.split(":", 1)
        key = prefix.strip().lower()
        if key in _PREFIX_ALIASES:
            rest = rest.strip()
            if not rest:
                raise CollectorError(f"Target spec '{spec}' has no target after '{prefix}:'")
            name = _PREFIX_ALIASES[key]
            if name == "reddit":
                rest = rest.lstrip("/")
                if rest.lower().startswith("user/"):
                    rest = "u/" + rest[5:]
            return name, rest
        _ensure_loaded()
        if key in REGISTRY or key in ALIASES:
            return ALIASES.get(key, key), rest.strip()

    # Bare existing path or obvious file name.
    if os.path.exists(s) or s.lower().endswith(_FILE_EXTS):
        return "files", s

    raise CollectorError(
        f"Cannot resolve target '{spec}'. Use a prefix such as reddit:, hn:, rss:, web:, "
        "mastodon:, bluesky: or file:"
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
class _TextExtractor(HTMLParser):
    _BLOCK = {"p", "br", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "pre", "section", "article"}
    _SKIP = {"script", "style", "noscript", "template"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip += 1
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_startendtag(self, tag, attrs):
        if tag in self._BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._skip:
            self._skip -= 1
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def strip_html(text: str | None) -> str:
    """Convert an HTML fragment to plain text (keeps paragraph breaks)."""
    if not text:
        return ""
    if "<" not in text and "&" not in text:
        return text.strip()
    p = _TextExtractor()
    try:
        p.feed(text)
        p.close()
        out = "".join(p.parts)
    except Exception:  # pragma: no cover - extremely malformed input
        out = re.sub(r"<[^>]+>", " ", text)
        out = html.unescape(out)
    return normalize_ws(out)


def normalize_ws(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def to_utc(value: Any) -> datetime | None:
    """Parse epoch numbers / ISO-8601 / RFC-2822 strings / datetimes into tz-aware UTC."""
    if value is None or value == "":
        return None
    try:
        if isinstance(value, datetime):
            dt = value
        elif isinstance(value, (int, float)) or (isinstance(value, str) and re.fullmatch(r"-?\d+(\.\d+)?", value.strip())):
            num = float(value)
            if num > 1e12:  # milliseconds
                num /= 1000.0
            dt = datetime.fromtimestamp(num, tz=timezone.utc)
        elif isinstance(value, str):
            v = value.strip()
            try:
                iso = v.replace("Z", "+00:00") if v.endswith("Z") else v
                # Trim sub-microsecond precision (e.g. Bluesky nanoseconds).
                iso = re.sub(r"(\.\d{6})\d+", r"\1", iso)
                dt = datetime.fromisoformat(iso)
            except ValueError:
                from email.utils import parsedate_to_datetime

                try:
                    dt = parsedate_to_datetime(v)
                except (TypeError, ValueError):
                    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d"):
                        try:
                            dt = datetime.strptime(v, fmt)
                            break
                        except ValueError:
                            continue
                    else:
                        return None
        else:
            return None
    except (OverflowError, OSError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".avif")


def looks_like_image(url: str | None) -> bool:
    if not url:
        return False
    path = url.split("?", 1)[0].split("#", 1)[0].lower()
    return path.endswith(IMAGE_EXTS)


def _env_proxy_trust(config: dict) -> bool:
    return bool(config.get("trust_env", True))


# --------------------------------------------------------------------------- #
# Base class
# --------------------------------------------------------------------------- #
class Collector(ABC):
    """Abstract collector.

    Subclasses define ``name``, ``description``, ``examples`` and implement
    :meth:`collect`. Use :meth:`get_json` / :meth:`get` for HTTP: they apply
    polite rate limiting (``min_interval`` seconds between requests, default
    1.0) and retry with exponential backoff on 429/5xx and transport errors.
    """

    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    examples: ClassVar[list[str]] = []

    def __init__(self, config: dict | None = None, http: httpx.Client | None = None) -> None:
        self.config: dict = dict(config or {})
        self.log = logging.getLogger(f"vigileye.collectors.{self.name or 'base'}")
        self.user_agent: str = self.config.get("user_agent") or DEFAULT_USER_AGENT
        self.timeout: float = float(self.config.get("timeout", 20))
        self.max_retries: int = int(self.config.get("max_retries", 3))
        self.backoff: float = float(self.config.get("backoff", 1.5))
        self.max_response_bytes: int = int(self.config.get("max_response_bytes", MAX_RESPONSE_BYTES))
        self.fetch_deadline: float = float(self.config.get("fetch_deadline", FETCH_DEADLINE_S))
        self.min_interval: float = float(self.config.get("min_interval", self.default_min_interval()))
        self._last_request = 0.0
        self._lock = threading.Lock()
        self._owns_http = http is None
        self.http: httpx.Client = http or self._build_client()

    # -- construction ------------------------------------------------------ #
    def default_min_interval(self) -> float:
        return 1.0

    def _build_client(self) -> httpx.Client:
        """The collector's own client. Unless :attr:`allow_private`, a request hook refuses every request
        and redirect hop to a non-public host (:func:`vigileye.policy.guard_request_hook`). Clients
        passed in as ``http=`` are used as given."""
        headers = {"User-Agent": self.user_agent, "Accept": "*/*"}
        headers.update(self.config.get("headers") or {})
        kwargs: dict[str, Any] = dict(
            headers=headers,
            timeout=httpx.Timeout(self.timeout),
            follow_redirects=True,
            trust_env=_env_proxy_trust(self.config),  # honours HTTP(S)_PROXY / SSL_CERT_FILE
        )
        proxy = self.config.get("proxy")
        if proxy:
            kwargs["proxy"] = proxy
        verify = self.config.get("verify")
        if verify is not None:
            kwargs["verify"] = verify
        if not self.allow_private:
            from ..policy import guard_request_hook  # lazy: policy imports this module

            kwargs["event_hooks"] = {"request": [guard_request_hook()]}  # every request and redirect hop
        return httpx.Client(**kwargs)

    @property
    def allow_private(self) -> bool:
        """Lab opt-out of the SSRF guard: ``collectors.<name>.allow_private`` or ``collectors.http.allow_private``."""
        return bool(self.config.get("allow_private", False))

    def close(self) -> None:
        if self._owns_http:
            try:
                self.http.close()
            except Exception:  # pragma: no cover
                pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- HTTP helpers ------------------------------------------------------ #
    def throttle(self) -> None:
        """Block until ``min_interval`` seconds have passed since the last request."""
        with self._lock:
            wait = self.min_interval - (time.monotonic() - self._last_request)
            if wait > 0:
                time.sleep(wait)
            self._last_request = time.monotonic()

    def send_capped(self, method: str, url: str, *, max_bytes: int | None = None, **kwargs) -> httpx.Response:
        """One request whose body is streamed and refused past ``max_bytes`` (default
        :attr:`max_response_bytes`) or :attr:`fetch_deadline` seconds, so a hostile server cannot
        exhaust memory with a huge body or hang the collector by trickling bytes."""
        send_kw = {k: kwargs.pop(k) for k in ("follow_redirects", "auth") if k in kwargs}
        cap = int(max_bytes or self.max_response_bytes)
        req = self.http.build_request(method, url, **kwargs)
        deadline = time.monotonic() + self.fetch_deadline
        resp = self.http.send(req, stream=True, **send_kw)
        buf, size = [], 0
        try:
            for chunk in resp.iter_bytes():
                size += len(chunk)
                if size > cap:
                    raise CollectorError(f"{method} {url}: response body larger than {cap} bytes; aborted")
                if time.monotonic() > deadline:
                    raise CollectorError(f"{method} {url}: response body not received within {self.fetch_deadline:.0f}s")
                buf.append(chunk)
        finally:
            resp.close()
        headers = [(k, v) for k, v in resp.headers.multi_items()
                   if k.lower() not in ("content-encoding", "content-length", "transfer-encoding")]
        out = httpx.Response(resp.status_code, headers=headers, content=b"".join(buf), request=resp.request,
                             extensions=resp.extensions)
        out.history = list(resp.history)
        return out

    def request(self, method: str, url: str, *, max_bytes: int | None = None, **kwargs) -> httpx.Response:
        """HTTP request with throttling and retry/backoff on 429/5xx/transport errors (body capped,
        see :meth:`send_capped`)."""
        attempt = 0
        while True:
            self.throttle()
            try:
                resp = self.send_capped(method, url, max_bytes=max_bytes, **kwargs)
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                if attempt >= self.max_retries:
                    raise CollectorError(f"{method} {url} failed: {exc}") from exc
                delay = self._delay(attempt, None)
                self.log.warning("transport error on %s (%s); retry in %.1fs", url, exc, delay)
            else:
                if resp.status_code not in RETRY_STATUSES or attempt >= self.max_retries:
                    return resp
                delay = self._delay(attempt, resp.headers.get("Retry-After"))
                self.log.warning("HTTP %s on %s; retry in %.1fs", resp.status_code, url, delay)
            time.sleep(delay)
            attempt += 1

    def _delay(self, attempt: int, retry_after: str | None) -> float:
        if retry_after:
            try:
                return min(float(retry_after), 120.0)
            except ValueError:
                dt = to_utc(retry_after)
                if dt:
                    return max(0.0, min((dt - datetime.now(timezone.utc)).total_seconds(), 120.0))
        return min(self.backoff * (2**attempt) + random.uniform(0, 0.5), 60.0)

    def get(self, url: str, params: dict | None = None, **kwargs) -> httpx.Response:
        resp = self.request("GET", url, params=params, **kwargs)
        if resp.status_code >= 400:
            raise CollectorError(f"GET {resp.request.url} -> HTTP {resp.status_code}: {resp.text[:200]!r}")
        return resp

    def get_json(self, url: str, params: dict | None = None, **kwargs) -> Any:
        headers = {"Accept": "application/json"}
        headers.update(kwargs.pop("headers", {}) or {})
        resp = self.get(url, params=params, headers=headers, **kwargs)
        try:
            return resp.json()
        except ValueError as exc:
            raise CollectorError(f"GET {url}: response is not JSON") from exc

    # -- item safety ------------------------------------------------------- #
    def safe_map(self, fn: Callable[[Any], Any], item: Any) -> Any | None:
        """Apply a mapping function, logging (not raising) on malformed items."""
        try:
            return fn(item)
        except Exception as exc:  # noqa: BLE001 - one bad item must not kill the run
            self.log.warning("skipping malformed %s item: %s", self.name, exc, exc_info=self.log.isEnabledFor(logging.DEBUG))
            return None

    # -- API --------------------------------------------------------------- #
    @abstractmethod
    def collect(self, target: str, limit: int = 100, **opts) -> Iterator["Document"]:  # noqa: F821
        """Yield up to ``limit`` documents for ``target``."""

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"


UNTRUSTED_ORIGINS = ("api", "auto")


def collect(spec: str, limit: int = 100, config: dict | None = None, *, trust: str = "manual",
            **opts) -> Iterator["Document"]:  # noqa: F821
    """Convenience: resolve ``spec`` and stream documents from the right collector.

    ``config`` may be the whole ``collectors`` section (keyed by collector
    name, with a shared ``http`` sub-section) or a flat per-collector dict.
    ``trust`` is where the target came from (``manual`` CLI, ``api`` HTTP client, ``auto`` the
    investigation planner). For ``api`` and ``auto`` targets the fetch-time SSRF guard stays on
    whatever ``allow_private`` says, and collectors withhold credentials that are not scoped to the
    exact host being fetched (see ``config["trust"]``).
    """
    name, target = resolve_target(spec)
    cfg = collector_config(config, name)
    if trust in UNTRUSTED_ORIGINS:
        cfg = {**cfg, "allow_private": False, "trust": trust}
    col = get_collector(name, cfg)
    try:
        yield from col.collect(target, limit=limit, **opts)
    finally:
        col.close()


def collector_config(section: dict | None, name: str) -> dict:
    """Merge the shared ``http`` settings with the per-collector settings."""
    section = dict(section or {})
    known = set(REGISTRY) | {"http"}
    if not any(k in known for k in section):
        return section  # already flat
    merged: dict = {}
    merged.update(section.get("http") or {})
    merged.update(section.get(name) or {})
    return merged
