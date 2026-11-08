"""VigilEye collectors: public-source ingestion (Reddit, HN, RSS/Atom, web, Mastodon, Bluesky, local files).

Usage::

    from vigileye.collectors import resolve_target, get_collector, collect
    name, target = resolve_target("reddit:r/osint")
    docs = list(get_collector(name).collect(target, limit=50))
    # or simply
    docs = list(collect("hn:search:ransomware", limit=50))
"""

from __future__ import annotations

from .base import (
    ALIASES,
    REGISTRY,
    Collector,
    CollectorError,
    collect,
    collector_config,
    get_collector,
    list_collectors,
    looks_like_image,
    register,
    resolve_target,
    strip_html,
    to_utc,
)

# Import built-in collectors so they self-register.
from . import reddit as reddit  # noqa: E402,F401
from . import hackernews as hackernews  # noqa: E402,F401
from . import rss as rss  # noqa: E402,F401
from . import web as web  # noqa: E402,F401
from . import mastodon as mastodon  # noqa: E402,F401
from . import bluesky as bluesky  # noqa: E402,F401
from . import files as files  # noqa: E402,F401

from .bluesky import BlueskyCollector  # noqa: E402
from .files import FileCollector, load_documents, record_to_document  # noqa: E402
from .hackernews import HackerNewsCollector  # noqa: E402
from .mastodon import MastodonCollector  # noqa: E402
from .reddit import RedditCollector  # noqa: E402
from .rss import RSSCollector, parse_feed  # noqa: E402
from .web import WebCollector  # noqa: E402

__all__ = [
    "ALIASES",
    "REGISTRY",
    "Collector",
    "CollectorError",
    "collect",
    "collector_config",
    "get_collector",
    "list_collectors",
    "register",
    "resolve_target",
    "strip_html",
    "to_utc",
    "looks_like_image",
    "load_documents",
    "record_to_document",
    "parse_feed",
    "RedditCollector",
    "HackerNewsCollector",
    "RSSCollector",
    "WebCollector",
    "MastodonCollector",
    "BlueskyCollector",
    "FileCollector",
]
