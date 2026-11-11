"""Bluesky collector using the public AppView (no auth needed)."""

from __future__ import annotations

from typing import Iterator

from ..models import Document, Media
from .base import Collector, CollectorError, register, to_utc

APPVIEW = "https://public.api.bsky.app/xrpc"
# public.api.bsky.app answers searchPosts with 403 for anonymous clients; api.bsky.app serves it.
FALLBACK_APPVIEWS = ("https://api.bsky.app/xrpc",)


def post_url(uri: str, handle: str | None) -> str | None:
    """Build a bsky.app URL from an ``at://did/app.bsky.feed.post/rkey`` URI."""
    if not uri or not uri.startswith("at://"):
        return None
    parts = uri[5:].split("/")
    if len(parts) < 3:
        return None
    actor = handle or parts[0]
    return f"https://bsky.app/profile/{actor}/post/{parts[-1]}"


def _embed_media(embed: dict | None) -> tuple[list[Media], str]:
    media: list[Media] = []
    extra = ""
    if not embed:
        return media, extra
    t = embed.get("$type", "")
    if "images" in t or embed.get("images"):
        for img in embed.get("images") or []:
            url = img.get("fullsize") or img.get("thumb")
            if url:
                media.append(Media(url=url, kind="image", description=img.get("alt") or None))
    if "video" in t and (embed.get("playlist") or embed.get("thumbnail")):
        media.append(Media(url=embed.get("playlist") or embed.get("thumbnail"), kind="video", description=embed.get("alt")))
    if "external" in t or embed.get("external"):
        ext = embed.get("external") or {}
        if ext.get("uri"):
            extra += f"\n[link] {ext.get('title') or ''} {ext['uri']}".rstrip()
        if ext.get("thumb"):
            media.append(Media(url=ext["thumb"], kind="image", description=ext.get("title")))
    if "record" in t:
        rec = embed.get("record") or {}
        inner = rec.get("record") if isinstance(rec.get("record"), dict) and "value" not in rec else rec
        val = (inner or {}).get("value") or {}
        if val.get("text"):
            who = ((inner or {}).get("author") or {}).get("handle") or "?"
            extra += f"\n[quote @{who}] {val['text']}"
        sub_media = embed.get("media")
        if isinstance(sub_media, dict):
            m2, e2 = _embed_media(sub_media)
            media.extend(m2)
            extra += e2
    return media, extra


def _map_post(p: dict, source: str | None = None) -> Document:
    """Map an ``app.bsky.feed.defs#postView`` (or a feed item wrapping one) to a Document."""
    reason = None
    if "post" in p and isinstance(p["post"], dict):
        reason = (p.get("reason") or {}).get("$type")
        p = p["post"]
    uri = p.get("uri")
    if not uri:
        raise ValueError("bluesky post without uri")
    author = p.get("author") or {}
    handle = author.get("handle")
    record = p.get("record") or {}
    media, extra = _embed_media(p.get("embed"))
    text = (record.get("text") or "") + extra
    tags = [f"#{t}" for t in record.get("tags") or []]
    for facet in record.get("facets") or []:
        for feat in facet.get("features") or []:
            if feat.get("tag"):
                tags.append(f"#{feat['tag']}")
    return Document(
        id="bsky_" + uri.rsplit("/", 1)[-1] + "_" + (author.get("did") or "")[-8:],
        platform="bluesky",
        source=source or (f"@{handle}" if handle else "bluesky"),
        title="",
        text=text.strip(),
        author=handle,
        url=post_url(uri, handle),
        created_at=to_utc(record.get("createdAt") or p.get("indexedAt")),
        metrics={
            "likes": p.get("likeCount"),
            "reposts": p.get("repostCount"),
            "replies": p.get("replyCount"),
            "quotes": p.get("quoteCount"),
        },
        media=media,
        tags=sorted(set(tags)),
        raw={"uri": uri, "cid": p.get("cid"), "did": author.get("did"), "langs": record.get("langs"),
             "reply": bool(record.get("reply")), "reason": reason},
    )


@register
class BlueskyCollector(Collector):
    name = "bluesky"
    description = "Bluesky public post search and public author feeds via the public AppView."
    examples = [
        "bluesky:search:election misinformation",
        "bluesky:@bsky.app",
        "bluesky:#osint",
    ]

    def default_min_interval(self) -> float:
        return 0.3

    def collect(self, target: str, limit: int = 100, **opts) -> Iterator[Document]:
        t = target.strip()
        base = (self.config.get("appview") or APPVIEW).rstrip("/")
        if t.lower().startswith(("search:", "q:")):
            q = t.split(":", 1)[1].strip()
            params = {"q": q, "sort": opts.get("sort", "latest")}
            for k in ("since", "until", "lang", "author", "domain"):
                if opts.get(k):
                    params[k] = opts[k]
            yield from self._paged(f"{base}/app.bsky.feed.searchPosts", params, "posts", limit, f"search:{q}")
        elif t.startswith("#"):
            q = t
            yield from self._paged(f"{base}/app.bsky.feed.searchPosts", {"q": q, "sort": opts.get("sort", "latest")},
                                   "posts", limit, q)
        else:
            actor = t.lstrip("@")
            if actor.startswith("https://bsky.app/profile/"):
                actor = actor.split("/profile/", 1)[1].split("/", 1)[0]
            if not actor:
                raise CollectorError(f"Invalid bluesky target '{target}'")
            params = {"actor": actor, "filter": opts.get("filter", "posts_no_replies")}
            yield from self._paged(f"{base}/app.bsky.feed.getAuthorFeed", params, "feed", limit, f"@{actor}")

    def _paged(self, url: str, params: dict, key: str, limit: int, source: str) -> Iterator[Document]:
        n = 0
        cursor = None
        while n < limit:
            p = dict(params)
            p["limit"] = min(100, limit - n)
            if cursor:
                p["cursor"] = cursor
            data = self._get_with_fallback(url, p)
            items = (data or {}).get(key) or []
            if not items:
                break
            for it in items:
                doc = self.safe_map(lambda x: _map_post(x, source), it)
                if doc is None:
                    continue
                yield doc
                n += 1
                if n >= limit:
                    return
            cursor = (data or {}).get("cursor")
            if not cursor:
                break

    def _get_with_fallback(self, url: str, params: dict):
        try:
            return self.get_json(url, params=params)
        except CollectorError as exc:
            if "HTTP 403" not in str(exc) and "HTTP 401" not in str(exc):
                raise
            method = url.rsplit("/", 1)[-1]
            for alt in FALLBACK_APPVIEWS:
                alt_url = f"{alt}/{method}"
                if alt_url == url:
                    continue
                try:
                    data = self.get_json(alt_url, params=params)
                    self.log.info("bluesky: %s refused, using %s", url, alt)
                    return data
                except CollectorError:
                    continue
            raise


map_post = _map_post
