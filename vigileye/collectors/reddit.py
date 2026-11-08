"""Reddit collector: public subreddits, public user posts and search.

Uses PRAW when installed and credentials are configured (REDDIT_CLIENT_ID /
REDDIT_CLIENT_SECRET via env or config); otherwise falls back to the public
``.json`` endpoints, which need no credentials.
"""

from __future__ import annotations

import html
import os
from typing import Any, Iterator

from ..models import Document, Media
from .base import Collector, CollectorError, looks_like_image, register, to_utc

BASE = "https://www.reddit.com"
_SORTS = {"hot", "new", "top", "rising", "controversial", "relevance", "comments"}


def _full_url(permalink: str | None) -> str | None:
    if not permalink:
        return None
    if permalink.startswith("http"):
        return permalink
    return BASE + permalink


def _media_from_post(d: dict) -> list[Media]:
    media: list[Media] = []
    seen: set[str] = set()

    def add(url: str | None, kind: str = "image", desc: str | None = None) -> None:
        if not url:
            return
        url = html.unescape(url)
        if url in seen:
            return
        seen.add(url)
        media.append(Media(url=url, kind=kind, description=desc))

    url = d.get("url_overridden_by_dest") or d.get("url")
    if url and ("i.redd.it" in url or "i.imgur.com" in url or looks_like_image(url)):
        add(url)
    # Galleries
    meta = d.get("media_metadata") or {}
    items = (d.get("gallery_data") or {}).get("items") or []
    order = [it.get("media_id") for it in items if it.get("media_id")] or list(meta)
    captions = {it.get("media_id"): it.get("caption") for it in items}
    for mid in order:
        m = meta.get(mid) or {}
        src = (m.get("s") or {}).get("u") or (m.get("s") or {}).get("gif")
        if not src and m.get("m", "").startswith("image/"):
            ext = m["m"].split("/")[-1]
            src = f"https://i.redd.it/{mid}.{ext}"
        add(src, "image", captions.get(mid))
    # Hosted video
    rv = ((d.get("secure_media") or d.get("media") or {}) or {}).get("reddit_video") or {}
    if rv.get("fallback_url"):
        add(rv["fallback_url"], "video")
    # Preview image as last resort for link posts
    if not media:
        for img in ((d.get("preview") or {}).get("images") or [])[:1]:
            add((img.get("source") or {}).get("url"))
    return media


def _map_post(d: dict) -> Document:
    """Map a Reddit listing ``t3`` (submission) data dict to a Document."""
    if "data" in d and "kind" in d:
        d = d["data"]
    pid = d.get("name") or (f"t3_{d['id']}" if d.get("id") else None)
    if not pid:
        raise ValueError("reddit post without id")
    sub = d.get("subreddit") or ""
    url = _full_url(d.get("permalink"))
    link = d.get("url_overridden_by_dest") or d.get("url")
    text = d.get("selftext") or ""
    if not text and link and link != url and not d.get("is_self"):
        text = f"[link] {link}"
    tags = [t for t in (d.get("link_flair_text"),) if t]
    if d.get("over_18"):
        tags.append("nsfw")
    return Document(
        id=pid,
        platform="reddit",
        source=f"r/{sub}" if sub else "reddit",
        title=html.unescape(d.get("title") or ""),
        text=html.unescape(text),
        author=d.get("author"),
        url=url,
        created_at=to_utc(d.get("created_utc")),
        metrics={
            "score": d.get("score"),
            "num_comments": d.get("num_comments"),
            "upvote_ratio": d.get("upvote_ratio"),
        },
        media=_media_from_post(d),
        tags=tags,
        raw={
            "kind": "t3",
            "id": d.get("id"),
            "subreddit": sub,
            "domain": d.get("domain"),
            "link_url": link,
            "is_self": d.get("is_self"),
            "flair": d.get("link_flair_text"),
        },
    )


def _map_comment(d: dict, source: str | None = None) -> Document:
    """Map a Reddit ``t1`` (comment) data dict to a Document."""
    if "data" in d and "kind" in d:
        d = d["data"]
    cid = d.get("name") or (f"t1_{d['id']}" if d.get("id") else None)
    if not cid:
        raise ValueError("reddit comment without id")
    sub = d.get("subreddit") or ""
    return Document(
        id=cid,
        platform="reddit",
        source=source or (f"r/{sub}" if sub else "reddit"),
        title=html.unescape(d.get("link_title") or ""),
        text=html.unescape(d.get("body") or ""),
        author=d.get("author"),
        url=_full_url(d.get("permalink")),
        created_at=to_utc(d.get("created_utc")),
        metrics={"score": d.get("score")},
        tags=["comment"],
        raw={"kind": "t1", "id": d.get("id"), "parent_id": d.get("parent_id"), "link_id": d.get("link_id"), "subreddit": sub},
    )


@register
class RedditCollector(Collector):
    name = "reddit"
    description = "Reddit public subreddits, public user submissions/comments and search (PRAW or public JSON)."
    examples = [
        "reddit:r/osint",
        "reddit:r/worldnews",
        "reddit:u/spez",
        "reddit:search:ransomware leak site",
        "reddit:search:drone@r/ukraine",
        "r/osint",
    ]

    def default_min_interval(self) -> float:
        return 2.0  # unauthenticated JSON is rate limited fairly aggressively

    def __init__(self, config: dict | None = None, http=None) -> None:
        super().__init__(config, http)
        self._praw = None
        cid = self.config.get("client_id") or os.environ.get("REDDIT_CLIENT_ID")
        secret = self.config.get("client_secret") or os.environ.get("REDDIT_CLIENT_SECRET")
        if cid and secret and not self.config.get("force_json"):
            try:
                import praw  # type: ignore

                self._praw = praw.Reddit(
                    client_id=cid,
                    client_secret=secret,
                    user_agent=self.config.get("praw_user_agent") or self.user_agent,
                    check_for_async=False,
                )
                self._praw.read_only = True
            except ImportError:
                self.log.debug("praw not installed; using public JSON endpoints")
            except Exception as exc:  # pragma: no cover
                self.log.warning("praw init failed (%s); using public JSON endpoints", exc)
                self._praw = None

    # -- target parsing ---------------------------------------------------- #
    @staticmethod
    def parse_target(target: str) -> tuple[str, str, str | None]:
        """Return ``(kind, value, subreddit_scope)`` where kind in {sub, user, search}."""
        t = target.strip().lstrip("/")
        low = t.lower()
        if low.startswith("search:"):
            q = t[7:].strip()
            scope = None
            if "@r/" in q:
                q, scope = q.rsplit("@r/", 1)
                q, scope = q.strip(), scope.strip("/ ")
            return "search", q, scope
        if low.startswith(("u/", "user/")):
            return "user", t.split("/", 1)[1].strip("/"), None
        if low.startswith("r/"):
            return "sub", t[2:].strip("/"), None
        return "sub", t.strip("/"), None

    # -- collection -------------------------------------------------------- #
    def collect(self, target: str, limit: int = 100, **opts) -> Iterator[Document]:
        kind, value, scope = self.parse_target(target)
        if not value:
            raise CollectorError(f"Invalid reddit target '{target}'")
        include_comments = bool(opts.get("include_comments", self.config.get("include_comments", False)))
        per_post = int(opts.get("comments_per_post", self.config.get("comments_per_post", 10)))
        if self._praw is not None:
            try:
                yield from self._collect_praw(kind, value, scope, limit, include_comments, per_post, opts)
                return
            except Exception as exc:  # pragma: no cover - network path
                self.log.warning("praw collection failed (%s); falling back to JSON", exc)
        yield from self._collect_json(kind, value, scope, limit, include_comments, per_post, opts)

    def _listing(self, path: str, params: dict, limit: int) -> Iterator[dict]:
        after = None
        fetched = 0
        while fetched < limit:
            p = dict(params)
            p.update({"limit": min(100, limit - fetched), "raw_json": 1})
            if after:
                p["after"] = after
            data = self.get_json(f"{BASE}{path}", params=p)
            listing = (data or {}).get("data") or {}
            children = listing.get("children") or []
            if not children:
                break
            for child in children:
                yield child
                fetched += 1
                if fetched >= limit:
                    break
            after = listing.get("after")
            if not after:
                break

    def _collect_json(self, kind, value, scope, limit, include_comments, per_post, opts) -> Iterator[Document]:
        sort = (opts.get("sort") or self.config.get("sort") or ("relevance" if kind == "search" else "hot")).lower()
        if sort not in _SORTS:
            sort = "hot"
        time_filter = opts.get("time_filter") or self.config.get("time_filter")
        params: dict[str, Any] = {}
        if time_filter:
            params["t"] = time_filter

        if kind == "sub":
            path = f"/r/{value}/{sort if sort in {'hot', 'new', 'top', 'rising', 'controversial'} else 'hot'}.json"
            streams = [(path, params, "post")]
        elif kind == "search":
            params.update({"q": value, "sort": sort if sort in {"relevance", "hot", "top", "new", "comments"} else "relevance"})
            if scope:
                params["restrict_sr"] = 1
                streams = [(f"/r/{scope}/search.json", params, "post")]
            else:
                streams = [("/search.json", params, "post")]
        else:  # user: public submissions then public comments
            uparams = dict(params)
            if sort in {"hot", "new", "top", "controversial"}:
                uparams["sort"] = sort
            streams = [(f"/user/{value}/submitted.json", uparams, "post")]
            if opts.get("user_comments", True):
                streams.append((f"/user/{value}/comments.json", uparams, "comment"))

        emitted = 0
        for path, p, what in streams:
            remaining = limit - emitted
            if remaining <= 0:
                break
            for child in self._listing(path, p, remaining):
                data = child.get("data") or {}
                ck = child.get("kind")
                if ck == "t1" or what == "comment":
                    doc = self.safe_map(lambda d: _map_comment(d, f"u/{value}" if kind == "user" else None), data)
                else:
                    doc = self.safe_map(_map_post, data)
                    if doc is not None and kind == "user":
                        doc.meta["collected_from"] = f"u/{value}"
                if doc is None:
                    continue
                yield doc
                emitted += 1
                if emitted >= limit:
                    return
                if include_comments and ck == "t3" and data.get("num_comments"):
                    for c in self._comments_json(data, per_post):
                        yield c
                        emitted += 1
                        if emitted >= limit:
                            return

    def _comments_json(self, post: dict, per_post: int) -> Iterator[Document]:
        perma = post.get("permalink")
        if not perma or per_post <= 0:
            return
        try:
            data = self.get_json(f"{BASE}{perma.rstrip('/')}.json", params={"limit": per_post, "depth": 1, "raw_json": 1, "sort": "top"})
        except CollectorError as exc:
            self.log.warning("comments fetch failed for %s: %s", perma, exc)
            return
        if not isinstance(data, list) or len(data) < 2:
            return
        n = 0
        for child in (data[1].get("data") or {}).get("children") or []:
            if child.get("kind") != "t1":
                continue
            doc = self.safe_map(_map_comment, child.get("data") or {})
            if doc is None:
                continue
            doc.meta["parent_post"] = post.get("name")
            yield doc
            n += 1
            if n >= per_post:
                break

    # -- PRAW path --------------------------------------------------------- #
    def _collect_praw(self, kind, value, scope, limit, include_comments, per_post, opts) -> Iterator[Document]:  # pragma: no cover
        r = self._praw
        sort = (opts.get("sort") or self.config.get("sort") or ("relevance" if kind == "search" else "hot")).lower()
        tf = opts.get("time_filter") or self.config.get("time_filter") or "all"
        if kind == "sub":
            sr = r.subreddit(value)
            fn = {"new": sr.new, "top": lambda **k: sr.top(time_filter=tf, **k), "rising": sr.rising,
                  "controversial": lambda **k: sr.controversial(time_filter=tf, **k)}.get(sort, sr.hot)
            items = fn(limit=limit)
        elif kind == "search":
            sr = r.subreddit(scope or "all")
            items = sr.search(value, sort=sort if sort in {"relevance", "hot", "top", "new", "comments"} else "relevance",
                              time_filter=tf, limit=limit)
        else:
            red = r.redditor(value)
            items = red.submissions.new(limit=limit)
        n = 0
        for s in items:
            d = self._praw_submission_dict(s)
            doc = self.safe_map(_map_post, d)
            if doc is None:
                continue
            yield doc
            n += 1
            if n >= limit:
                return
            if include_comments and per_post > 0:
                try:
                    s.comment_sort = "top"
                    s.comments.replace_more(limit=0)
                    for c in list(s.comments)[:per_post]:
                        cd = {"name": c.fullname, "id": c.id, "body": c.body, "author": str(c.author) if c.author else None,
                              "permalink": c.permalink, "created_utc": c.created_utc, "score": c.score,
                              "subreddit": str(c.subreddit), "parent_id": c.parent_id, "link_id": c.link_id,
                              "link_title": s.title}
                        cdoc = self.safe_map(_map_comment, cd)
                        if cdoc is None:
                            continue
                        cdoc.meta["parent_post"] = s.fullname
                        yield cdoc
                        n += 1
                        if n >= limit:
                            return
                except Exception as exc:
                    self.log.warning("praw comments failed for %s: %s", s.id, exc)

    @staticmethod
    def _praw_submission_dict(s) -> dict:  # pragma: no cover
        fields = ["id", "name", "title", "selftext", "permalink", "url", "created_utc", "score", "num_comments",
                  "upvote_ratio", "is_self", "domain", "link_flair_text", "over_18", "media_metadata", "gallery_data",
                  "preview", "media", "secure_media"]
        d = {f: getattr(s, f, None) for f in fields}
        d["author"] = str(s.author) if getattr(s, "author", None) else None
        d["subreddit"] = str(s.subreddit)
        return d


# Public aliases for tests / external use.
map_post = _map_post
map_comment = _map_comment
