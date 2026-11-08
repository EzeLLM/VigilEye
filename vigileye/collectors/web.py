"""Generic public web page collector with optional same-domain crawl.

Respects robots.txt by default (``ignore_robots=False``).
"""

from __future__ import annotations

import hashlib
import re
from collections import deque
from html.parser import HTMLParser
from typing import Iterator
from urllib import robotparser
from urllib.parse import urldefrag, urljoin, urlparse

from ..models import Document, Media
from .base import Collector, CollectorError, normalize_ws, register, to_utc

_DROP = {"script", "style", "noscript", "template", "svg", "nav", "footer", "header", "aside", "form", "iframe", "button", "select"}
_BLOCK = {"p", "div", "br", "li", "ul", "ol", "tr", "table", "section", "article", "main", "blockquote", "pre",
          "h1", "h2", "h3", "h4", "h5", "h6", "figure", "figcaption", "dd", "dt", "hr"}
_VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}
_SKIP_EXT = (".pdf", ".zip", ".gz", ".tar", ".exe", ".dmg", ".mp4", ".mp3", ".avi", ".mov", ".jpg", ".jpeg", ".png",
             ".gif", ".webp", ".svg", ".css", ".js", ".ico", ".woff", ".woff2", ".ttf", ".xml", ".json")


class ReadableHTMLParser(HTMLParser):
    """Extract title, meta, readable text, images and links from HTML."""

    def __init__(self, base_url: str = "") -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.title = ""
        self.meta: dict[str, str] = {}
        self.images: list[tuple[str, str | None]] = []
        self.links: list[str] = []
        self._parts: list[str] = []
        self._main_parts: list[str] = []
        self._drop_depth = 0
        self._in_title = False
        self._main_depth = 0
        self._stack: list[str] = []

    # tag handling
    def handle_starttag(self, tag, attrs):
        a = {k.lower(): (v or "") for k, v in attrs}
        if tag == "base" and a.get("href"):
            self.base_url = urljoin(self.base_url, a["href"])
        if tag == "meta":
            key = (a.get("property") or a.get("name") or a.get("itemprop") or "").lower()
            if key and a.get("content"):
                self.meta.setdefault(key, a["content"].strip())
            return
        if tag == "title":
            self._in_title = True
        if tag in _VOID:
            if tag == "img" and not self._drop_depth:
                src = a.get("src") or a.get("data-src") or ""
                if src and not src.startswith("data:"):
                    self.images.append((urljoin(self.base_url, src), a.get("alt") or None))
            if tag in {"br", "hr"}:
                self._emit("\n")
            return
        self._stack.append(tag)
        if tag in _DROP:
            self._drop_depth += 1
        if tag in {"main", "article"}:
            self._main_depth += 1
        if tag == "a" and a.get("href") and not self._drop_depth:
            self.links.append(urljoin(self.base_url, a["href"]))
        elif tag == "a" and a.get("href"):
            self.links.append(urljoin(self.base_url, a["href"]))
        if tag in _BLOCK:
            self._emit("\n")

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in _VOID:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        if tag in _VOID or tag not in self._stack:
            return
        # pop until matching tag (tolerates unclosed children)
        while self._stack:
            t = self._stack.pop()
            if t in _DROP and self._drop_depth:
                self._drop_depth -= 1
            if t in {"main", "article"} and self._main_depth:
                self._main_depth -= 1
            if t == tag:
                break
        if tag in _BLOCK:
            self._emit("\n")

    def handle_data(self, data):
        if self._in_title:
            self.title += data
            return
        if self._drop_depth:
            return
        self._emit(data)

    def _emit(self, s: str) -> None:
        self._parts.append(s)
        if self._main_depth:
            self._main_parts.append(s)

    # results
    @property
    def text(self) -> str:
        main = normalize_ws("".join(self._main_parts))
        full = normalize_ws("".join(self._parts))
        # Prefer <main>/<article> when it carries most of the content.
        if main and len(main) >= 0.3 * len(full):
            return main
        return full


def extract(html_text: str, url: str = "") -> dict:
    """Return ``{title, text, description, author, published, images, links, lang}`` for an HTML page."""
    p = ReadableHTMLParser(url)
    try:
        p.feed(html_text)
        p.close()
    except Exception:  # pragma: no cover - html.parser is very tolerant
        pass
    m = p.meta
    images: list[tuple[str, str | None]] = []
    for key in ("og:image", "og:image:url", "twitter:image", "twitter:image:src"):
        if m.get(key):
            images.append((urljoin(p.base_url or url, m[key]), m.get("og:image:alt")))
            break
    images.extend(p.images)
    lang = None
    mm = re.search(r"<html[^>]*\blang=[\"']?([\w\-]+)", html_text[:5000], re.I)
    if mm:
        lang = mm.group(1)
    return {
        "title": normalize_ws(m.get("og:title") or p.title or ""),
        "text": p.text,
        "description": m.get("og:description") or m.get("description") or "",
        "author": m.get("author") or m.get("article:author") or None,
        "published": m.get("article:published_time") or m.get("datepublished") or m.get("date") or None,
        "site_name": m.get("og:site_name"),
        "images": images,
        "links": p.links,
        "lang": lang,
    }


def page_to_document(html_text: str, url: str, depth: int = 0, max_images: int = 10) -> Document:
    info = extract(html_text, url)
    seen: set[str] = set()
    media = []
    for src, alt in info["images"]:
        if src in seen or src.lower().endswith((".svg", ".ico")):
            continue
        seen.add(src)
        media.append(Media(url=src, kind="image", description=alt))
        if len(media) >= max_images:
            break
    text = info["text"]
    if info["description"] and info["description"] not in text[:2000]:
        text = info["description"] + "\n\n" + text
    host = urlparse(url).netloc
    return Document(
        id="web_" + hashlib.sha1(url.encode("utf-8")).hexdigest()[:16],
        platform="web",
        source=info.get("site_name") or host,
        title=info["title"],
        text=text,
        author=info["author"],
        url=url,
        created_at=to_utc(info["published"]),
        metrics={"chars": len(text), "links": len(info["links"])},
        media=media,
        tags=[],
        raw={"lang": info["lang"], "depth": depth, "host": host},
    )


@register
class WebCollector(Collector):
    name = "web"
    description = "Public web pages: readable text, title, og:image/img media; optional same-domain crawl (robots.txt respected)."
    examples = [
        "web:https://www.example.org/report.html",
        "https://www.cisa.gov/news-events/cybersecurity-advisories",
        "web:https://example.org/blog  (opts: crawl_depth=1)",
    ]

    def default_min_interval(self) -> float:
        return 1.0

    def __init__(self, config: dict | None = None, http=None) -> None:
        super().__init__(config, http)
        self._robots: dict[str, robotparser.RobotFileParser | None] = {}

    def allowed(self, url: str) -> bool:
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        if origin not in self._robots:
            rp = robotparser.RobotFileParser()
            try:
                resp = self.http.get(origin + "/robots.txt", timeout=10)
                if resp.status_code in (401, 403):
                    rp.disallow_all = True
                elif resp.status_code >= 400:
                    rp.allow_all = True
                else:
                    rp.parse(resp.text.splitlines())
            except Exception as exc:  # noqa: BLE001
                self.log.debug("robots.txt fetch failed for %s: %s", origin, exc)
                rp = None  # unreachable robots -> allow
            self._robots[origin] = rp
        rp = self._robots[origin]
        return True if rp is None else rp.can_fetch(self.user_agent, url)

    def collect(self, target: str, limit: int = 100, **opts) -> Iterator[Document]:
        start = target.strip()
        if not re.match(r"^https?://", start, re.I):
            start = "https://" + start
        depth_max = int(opts.get("crawl_depth", self.config.get("crawl_depth", 0)))
        ignore_robots = bool(opts.get("ignore_robots", self.config.get("ignore_robots", False)))
        same_path = opts.get("path_prefix")
        max_chars = int(opts.get("max_chars", self.config.get("max_chars", 200_000)))
        start_host = urlparse(start).netloc.lower()

        queue: deque[tuple[str, int]] = deque([(urldefrag(start)[0], 0)])
        seen: set[str] = {urldefrag(start)[0]}
        n = 0
        while queue and n < limit:
            url, depth = queue.popleft()
            if not ignore_robots and not self.allowed(url):
                self.log.info("robots.txt disallows %s; skipping (use ignore_robots=True to override)", url)
                if depth == 0 and n == 0 and not queue:
                    raise CollectorError(f"robots.txt disallows fetching {url}")
                continue
            try:
                resp = self.request("GET", url, headers={"Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.5"})
            except CollectorError as exc:
                self.log.warning("fetch failed %s: %s", url, exc)
                continue
            if resp.status_code >= 400:
                self.log.warning("HTTP %s for %s", resp.status_code, url)
                if depth == 0 and not queue and n == 0:
                    raise CollectorError(f"GET {url} -> HTTP {resp.status_code}")
                continue
            ctype = resp.headers.get("content-type", "").lower()
            final_url = str(resp.url)
            if "html" not in ctype and "xml" not in ctype and ctype:
                if ctype.startswith("text/"):
                    text = resp.text[:max_chars]
                    yield Document(
                        id="web_" + hashlib.sha1(final_url.encode()).hexdigest()[:16], platform="web",
                        source=urlparse(final_url).netloc, title=final_url.rsplit("/", 1)[-1], text=text,
                        url=final_url, metrics={"chars": len(text)}, raw={"content_type": ctype, "depth": depth},
                    )
                    n += 1
                else:
                    self.log.info("skipping non-HTML %s (%s)", url, ctype)
                continue
            html_text = resp.text
            doc = self.safe_map(lambda h: page_to_document(h, final_url, depth), html_text)
            if doc is None:
                continue
            if len(doc.text) > max_chars:
                doc.text = doc.text[:max_chars]
            yield doc
            n += 1
            if depth < depth_max:
                info_links = extract(html_text, final_url)["links"]
                for link in info_links:
                    link = urldefrag(link)[0]
                    pu = urlparse(link)
                    if pu.scheme not in ("http", "https") or pu.netloc.lower() != start_host:
                        continue
                    if pu.path.lower().endswith(_SKIP_EXT):
                        continue
                    if same_path and not pu.path.startswith(same_path):
                        continue
                    if link in seen:
                        continue
                    seen.add(link)
                    queue.append((link, depth + 1))
