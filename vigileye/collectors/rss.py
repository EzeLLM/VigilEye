"""RSS 2.0 / RSS 1.0 (RDF) / Atom feed collector (stdlib xml.etree)."""

from __future__ import annotations

import hashlib
import re
import xml.etree.ElementTree as ET
from typing import Iterator
from urllib.parse import urljoin

from ..models import Document, Media
from .base import Collector, CollectorError, looks_like_image, register, strip_html, to_utc


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower() if isinstance(tag, str) else ""


def _children(el: ET.Element, name: str) -> list[ET.Element]:
    return [c for c in el if _local(c.tag) == name]


def _child(el: ET.Element, *names: str) -> ET.Element | None:
    for name in names:
        for c in el:
            if _local(c.tag) == name:
                return c
    return None


def _text(el: ET.Element | None) -> str:
    if el is None:
        return ""
    # Atom content type="xhtml" holds child elements.
    if len(el):
        return (el.text or "") + "".join(ET.tostring(c, encoding="unicode", method="html") for c in el)
    return (el.text or "").strip()


def _ns_prefix_strip(xml_text: str) -> str:
    # Drop a leading BOM / junk before the XML declaration.
    xml_text = xml_text.lstrip("﻿ \t\r\n")
    # Replace HTML named entities that XML doesn't know.
    return re.sub(r"&(nbsp|ndash|mdash|hellip|rsquo|lsquo|ldquo|rdquo|copy|reg|trade);",
                  lambda m: {"nbsp": "&#160;", "ndash": "&#8211;", "mdash": "&#8212;", "hellip": "&#8230;",
                             "rsquo": "&#8217;", "lsquo": "&#8216;", "ldquo": "&#8220;", "rdquo": "&#8221;",
                             "copy": "&#169;", "reg": "&#174;", "trade": "&#8482;"}[m.group(1)], xml_text)


def parse_feed(xml_text: str | bytes, feed_url: str | None = None, limit: int | None = None) -> list[Document]:
    """Parse an RSS/Atom document into Documents. Malformed entries are skipped."""
    if isinstance(xml_text, bytes):
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            root = ET.fromstring(_ns_prefix_strip(xml_text.decode("utf-8", "replace")).encode("utf-8"))
    else:
        root = ET.fromstring(_ns_prefix_strip(xml_text).encode("utf-8"))

    rt = _local(root.tag)
    if rt == "feed":  # Atom
        channel = root
        entries = _children(root, "entry")
        feed_title = _text(_child(root, "title"))
        is_atom = True
    else:
        channel = _child(root, "channel") if rt in {"rss", "rdf"} else root
        if channel is None:
            channel = root
        entries = _children(channel, "item") or _children(root, "item")  # RSS 1.0 puts items under rdf:RDF
        feed_title = _text(_child(channel, "title"))
        is_atom = False

    source = feed_title or feed_url or "feed"
    docs: list[Document] = []
    for e in entries:
        try:
            doc = _map_atom(e, source, feed_url) if is_atom else _map_rss(e, source, feed_url)
        except Exception as exc:  # noqa: BLE001
            import logging

            logging.getLogger("vigileye.collectors.rss").warning("skipping bad feed entry: %s", exc)
            continue
        docs.append(doc)
        if limit and len(docs) >= limit:
            break
    return docs


def _stable_id(*parts: str | None) -> str:
    return "rss_" + hashlib.sha1("|".join(p or "" for p in parts).encode("utf-8")).hexdigest()[:16]


def _media(e: ET.Element, html_body: str, base: str | None) -> list[Media]:
    out: list[Media] = []
    seen: set[str] = set()

    def add(url: str | None, kind: str = "image", desc: str | None = None) -> None:
        if not url:
            return
        url = urljoin(base or "", url.strip())
        if url in seen:
            return
        seen.add(url)
        out.append(Media(url=url, kind=kind, description=desc or None))

    for el in e.iter():
        ln = _local(el.tag)
        if ln in {"enclosure", "content", "thumbnail"} and el.get("url"):
            mtype = (el.get("type") or el.get("medium") or "").lower()
            if mtype.startswith(("image", "video", "audio")) or ln == "thumbnail" or looks_like_image(el.get("url")):
                kind = "video" if "video" in mtype else "audio" if "audio" in mtype else "image"
                add(el.get("url"), kind)
        if ln == "link" and el.get("rel") == "enclosure" and el.get("href"):
            mtype = (el.get("type") or "").lower()
            add(el.get("href"), "video" if "video" in mtype else "image")
    for m in re.finditer(r"<img[^>]+src=[\"']([^\"']+)[\"'][^>]*>", html_body or "", re.I):
        alt = re.search(r"alt=[\"']([^\"']*)[\"']", m.group(0), re.I)
        add(m.group(1), "image", alt.group(1) if alt else None)
    return out


def _map_rss(e: ET.Element, source: str, feed_url: str | None) -> Document:
    title = strip_html(_text(_child(e, "title")))
    link = _text(_child(e, "link")) or None
    guid = _text(_child(e, "guid")) or None
    body_html = _text(_child(e, "encoded")) or _text(_child(e, "description"))
    date = _text(_child(e, "pubdate", "date", "published", "updated"))
    author = _text(_child(e, "creator", "author")) or None
    cats = [strip_html(_text(c)) for c in _children(e, "category") if _text(c)]
    if link:
        link = urljoin(feed_url or "", link)
    return Document(
        id=_stable_id(feed_url, guid or link or title),
        platform="rss",
        source=source,
        title=title,
        text=strip_html(body_html),
        author=author,
        url=link,
        created_at=to_utc(date),
        media=_media(e, body_html, link or feed_url),
        tags=cats,
        raw={"guid": guid, "feed_url": feed_url},
    )


def _map_atom(e: ET.Element, source: str, feed_url: str | None) -> Document:
    title = strip_html(_text(_child(e, "title")))
    link = None
    for l in _children(e, "link"):
        rel = l.get("rel", "alternate")
        if rel == "alternate" and l.get("href"):
            link = l.get("href")
            break
    if link is None:
        first = _child(e, "link")
        link = first.get("href") if first is not None else None
    eid = _text(_child(e, "id")) or None
    body_html = _text(_child(e, "content")) or _text(_child(e, "summary"))
    date = _text(_child(e, "published", "updated", "issued", "modified"))
    author_el = _child(e, "author")
    author = None
    if author_el is not None:
        author = _text(_child(author_el, "name")) or _text(author_el) or None
    cats = [c.get("term") or c.get("label") for c in _children(e, "category")]
    if link:
        link = urljoin(feed_url or "", link)
    return Document(
        id=_stable_id(feed_url, eid or link or title),
        platform="rss",
        source=source,
        title=title,
        text=strip_html(body_html),
        author=author,
        url=link,
        created_at=to_utc(date),
        media=_media(e, body_html, link or feed_url),
        tags=[c for c in cats if c],
        raw={"id": eid, "feed_url": feed_url},
    )


@register
class RSSCollector(Collector):
    name = "rss"
    description = "RSS 2.0 / RSS 1.0 / Atom feeds (news sites, blogs, government bulletins, CERT advisories)."
    examples = [
        "rss:https://feeds.bbci.co.uk/news/world/rss.xml",
        "rss:https://www.cisa.gov/cybersecurity-advisories/all.xml",
        "rss:https://hnrss.org/newest?q=osint",
        "rss:https://url1,https://url2",
    ]

    def default_min_interval(self) -> float:
        return 0.5

    def collect(self, target: str, limit: int = 100, **opts) -> Iterator[Document]:
        urls = [u.strip() for u in re.split(r"[\s,]+(?=https?://)", target.strip()) if u.strip()]
        if not urls:
            raise CollectorError("rss collector needs at least one feed URL")
        n = 0
        for url in urls:
            try:
                resp = self.get(url, headers={"Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml, */*"})
                docs = parse_feed(resp.content, feed_url=str(resp.url) or url)
            except (CollectorError, ET.ParseError) as exc:
                self.log.warning("feed %s failed: %s", url, exc)
                if len(urls) == 1:
                    raise CollectorError(f"feed {url} failed: {exc}") from exc
                continue
            for d in docs:
                yield d
                n += 1
                if n >= limit:
                    return
