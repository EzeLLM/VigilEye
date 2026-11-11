"""Hacker News collector via the public Algolia API."""

from __future__ import annotations

import html
from typing import Iterator

from ..models import Document
from .base import Collector, CollectorError, register, strip_html, to_utc

ALGOLIA = "https://hn.algolia.com/api/v1"
HN_ITEM = "https://news.ycombinator.com/item?id={id}"


def _map_hit(h: dict) -> Document:
    """Map an Algolia search hit (story or comment) to a Document."""
    oid = str(h.get("objectID") or h.get("id") or "")
    if not oid:
        raise ValueError("HN hit without objectID")
    tags = [t for t in (h.get("_tags") or []) if t in {"story", "comment", "ask_hn", "show_hn", "poll", "job"}]
    is_comment = "comment" in tags or (h.get("comment_text") and not h.get("title"))
    text = strip_html(h.get("story_text") or h.get("comment_text") or h.get("text") or "")
    link = h.get("url")
    if not text and link:
        text = f"[link] {link}"
    title = h.get("title") or h.get("story_title") or ""
    return Document(
        id=f"hn_{oid}",
        platform="hackernews",
        source="hn:comments" if is_comment else "hn:stories",
        title=html.unescape(title),
        text=text,
        author=h.get("author"),
        url=HN_ITEM.format(id=oid),
        created_at=to_utc(h.get("created_at_i") or h.get("created_at")),
        metrics={"points": h.get("points"), "num_comments": h.get("num_comments")},
        tags=tags,
        raw={"objectID": oid, "link_url": link, "story_id": h.get("story_id"), "parent_id": h.get("parent_id")},
    )


def _map_item(it: dict, root_title: str = "") -> Document:
    """Map an Algolia ``/items/{id}`` node to a Document (children ignored)."""
    iid = str(it.get("id") or "")
    if not iid:
        raise ValueError("HN item without id")
    kind = it.get("type") or "story"
    text = strip_html(it.get("text") or "")
    link = it.get("url")
    if not text and link:
        text = f"[link] {link}"
    return Document(
        id=f"hn_{iid}",
        platform="hackernews",
        source=f"hn:item:{it.get('story_id') or iid}",
        title=html.unescape(it.get("title") or root_title or ""),
        text=text,
        author=it.get("author"),
        url=HN_ITEM.format(id=iid),
        created_at=to_utc(it.get("created_at_i") or it.get("created_at")),
        metrics={"points": it.get("points"), "replies": len(it.get("children") or [])},
        tags=[kind],
        raw={"id": iid, "type": kind, "parent_id": it.get("parent_id"), "story_id": it.get("story_id"), "link_url": link},
    )


@register
class HackerNewsCollector(Collector):
    name = "hackernews"
    description = "Hacker News stories/comments via the Algolia API: search, recent, user submissions, item threads."
    examples = [
        "hn:search:open source intelligence",
        "hn:recent:llm",
        "hn:user:pg",
        "hn:item:8863",
        "hn:front",
    ]

    def default_min_interval(self) -> float:
        return 0.3

    def collect(self, target: str, limit: int = 100, **opts) -> Iterator[Document]:
        t = target.strip()
        kind, _, value = t.partition(":")
        kind = kind.lower()
        if not value and kind not in {"front", "front_page", "frontpage"}:
            kind, value = "search", t
        value = value.strip()
        tags = opts.get("tags")
        by_date = bool(opts.get("by_date", False))
        if kind in {"search", "s", "q"}:
            yield from self._search(value, tags or "story", limit, by_date)
        elif kind in {"recent", "new", "date"}:
            yield from self._search(value, tags or "story", limit, True)
        elif kind in {"comments", "comment"}:
            yield from self._search(value, "comment", limit, by_date)
        elif kind in {"user", "author", "u"}:
            utags = f"author_{value}" if not tags else f"author_{value},{tags}"
            yield from self._search("", utags, limit, True)
        elif kind in {"front", "front_page", "frontpage"}:
            yield from self._search("", "front_page", limit, False)
        elif kind in {"item", "thread", "story", "id"}:
            yield from self._item(value, limit)
        else:
            yield from self._search(t, tags or "story", limit, by_date)

    def _search(self, query: str, tags: str, limit: int, by_date: bool) -> Iterator[Document]:
        endpoint = f"{ALGOLIA}/{'search_by_date' if by_date else 'search'}"
        page = 0
        n = 0
        per = min(max(limit, 1), 100)
        while n < limit:
            params = {"query": query, "tags": tags, "hitsPerPage": per, "page": page}
            data = self.get_json(endpoint, params=params)
            hits = (data or {}).get("hits") or []
            if not hits:
                break
            for h in hits:
                doc = self.safe_map(_map_hit, h)
                if doc is None:
                    continue
                yield doc
                n += 1
                if n >= limit:
                    return
            page += 1
            if page >= int((data or {}).get("nbPages") or 0):
                break

    def _item(self, item_id: str, limit: int) -> Iterator[Document]:
        item_id = item_id.strip().split("id=")[-1]
        if not item_id.isdigit():
            raise CollectorError(f"Invalid HN item id '{item_id}'")
        root = self.get_json(f"{ALGOLIA}/items/{item_id}")
        root_title = (root or {}).get("title") or ""
        # Breadth-first flatten so the most prominent replies come first.
        queue = [root]
        n = 0
        while queue and n < limit:
            node = queue.pop(0)
            if not isinstance(node, dict):
                continue
            if node.get("text") or node.get("title"):
                doc = self.safe_map(lambda it: _map_item(it, root_title), node)
                if doc is not None:
                    yield doc
                    n += 1
            queue.extend(node.get("children") or [])


map_hit = _map_hit
map_item = _map_item
