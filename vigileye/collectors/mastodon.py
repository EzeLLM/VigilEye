"""Mastodon / Fediverse collector using public (often unauthenticated) API endpoints.

Targets::

    #tag@instance              public hashtag timeline
    @user@instance             a public account's public statuses
    search:query@instance      status search (many instances require a token)
    public@instance            local public timeline (``local=true``)

An access token may be supplied per instance with ``tokens: {instance: token}``, or as one
global token (config ``access_token`` or env ``MASTODON_TOKEN``) that is sent only to its home
instance: ``token_instance`` (default: the configured ``instance``, ``MASTODON_INSTANCE``, or
``mastodon.social``). A target naming any other instance is fetched without credentials.
"""

from __future__ import annotations

import os
import re
from typing import Iterator
from urllib.parse import quote

from ..models import Document, Media
from .base import Collector, CollectorError, register, strip_html, to_utc

DEFAULT_INSTANCE = "mastodon.social"


def _host(instance) -> str:
    return str(instance or "").strip().lower().rstrip(".")


def _map_status(s: dict, source: str | None = None) -> Document:
    """Map a Mastodon status JSON object to a Document (boosts are unwrapped)."""
    boosted_by = None
    if s.get("reblog"):
        boosted_by = (s.get("account") or {}).get("acct")
        s = s["reblog"]
    sid = s.get("id")
    if not sid:
        raise ValueError("status without id")
    acct = s.get("account") or {}
    url = s.get("url") or s.get("uri")
    host = re.sub(r"^https?://([^/]+).*$", r"\1", url or "") or "fediverse"
    text = strip_html(s.get("content") or "")
    if s.get("spoiler_text"):
        text = f"[CW: {s['spoiler_text']}] {text}"
    poll = s.get("poll")
    if poll:
        opts = "; ".join(f"{o.get('title')} ({o.get('votes_count')})" for o in poll.get("options") or [])
        text += f"\n[poll] {opts}"
    card = s.get("card") or {}
    if card.get("url"):
        text += f"\n[link] {card.get('title') or ''} {card['url']}".rstrip()
    media = []
    for m in s.get("media_attachments") or []:
        murl = m.get("url") or m.get("remote_url") or m.get("preview_url")
        if not murl:
            continue
        kind = m.get("type") or "image"
        media.append(Media(url=murl, kind="image" if kind in {"image", "unknown"} else kind, description=m.get("description")))
    acct_name = acct.get("acct") or acct.get("username")
    if acct_name and "@" not in acct_name and acct.get("url"):
        acct_host = re.sub(r"^https?://([^/]+).*$", r"\1", acct["url"])
        acct_name = f"{acct_name}@{acct_host}"
    return Document(
        id=f"masto_{host}_{sid}",
        platform="mastodon",
        source=source or host,
        title="",
        text=text,
        author=acct_name,
        url=url,
        created_at=to_utc(s.get("created_at")),
        metrics={
            "replies": s.get("replies_count"),
            "reblogs": s.get("reblogs_count"),
            "favourites": s.get("favourites_count"),
        },
        media=media,
        tags=[f"#{t.get('name')}" for t in s.get("tags") or [] if t.get("name")],
        raw={"id": sid, "uri": s.get("uri"), "language": s.get("language"), "visibility": s.get("visibility"),
             "in_reply_to_id": s.get("in_reply_to_id"), "boosted_by": boosted_by,
             "sensitive": s.get("sensitive")},
    )


@register
class MastodonCollector(Collector):
    name = "mastodon"
    description = "Mastodon/Fediverse public hashtag timelines, public account statuses and status search."
    examples = [
        "mastodon:#osint@infosec.exchange",
        "mastodon:@Gargron@mastodon.social",
        "mastodon:search:disinformation@mastodon.social",
        "mastodon:public@infosec.exchange",
    ]

    def default_min_interval(self) -> float:
        return 0.5

    @staticmethod
    def parse_target(target: str, default_instance: str = DEFAULT_INSTANCE) -> tuple[str, str, str]:
        """Return ``(kind, value, instance)`` with kind in {tag, account, search, public}."""
        t = target.strip()
        if t.lower().startswith("search:"):
            q = t[7:]
            q, inst = (q.rsplit("@", 1) if "@" in q else (q, default_instance))
            return "search", q.strip(), inst.strip().lower()
        if t.startswith("#"):
            tag, _, inst = t[1:].partition("@")
            return "tag", tag.strip(), (inst or default_instance).strip().lower()
        if t.lower().startswith(("tag:", "hashtag:")):
            rest = t.split(":", 1)[1].lstrip("#")
            tag, _, inst = rest.partition("@")
            return "tag", tag.strip(), (inst or default_instance).strip().lower()
        if re.match(r"^(public|local)(@|$)", t, re.I):
            _, _, inst = t.partition("@")
            return "public", "local", (inst or default_instance).strip().lower()
        if t.startswith("@") or t.count("@") >= 1:
            parts = t.lstrip("@").split("@")
            user = parts[0]
            inst = parts[1] if len(parts) > 1 and parts[1] else default_instance
            return "account", user.strip(), inst.strip().lower()
        if t.startswith("http"):
            m = re.match(r"https?://([^/]+)/@([\w.\-]+)", t)
            if m:
                return "account", m.group(2), m.group(1).lower()
            m = re.match(r"https?://([^/]+)/tags/([\w\-]+)", t)
            if m:
                return "tag", m.group(2), m.group(1).lower()
        return "tag", t.lstrip("#"), default_instance

    def token_instance(self) -> str:
        """The one instance the global ``access_token`` / ``MASTODON_TOKEN`` belongs to:
        ``token_instance``, else the configured ``instance``, else ``MASTODON_INSTANCE``, else
        :data:`DEFAULT_INSTANCE`."""
        inst = (self.config.get("token_instance") or self.config.get("instance")
                or os.environ.get("MASTODON_INSTANCE") or DEFAULT_INSTANCE)
        return _host(inst)

    def _headers(self, instance: str) -> dict:
        """Authorization for ``instance`` only. A token is sent solely to the instance it is scoped
        to (``tokens.<instance>``, or the global token's :meth:`token_instance`), so a target that
        names another instance never receives it. Planner-proposed (``trust: auto``) targets are
        fetched without credentials."""
        if self.config.get("trust") == "auto":
            return {}  # planner-proposed targets never carry the analyst's credentials
        instance = _host(instance)
        tokens = {_host(k): v for k, v in (self.config.get("tokens") or {}).items() if v}
        tok = tokens.get(instance)
        if not tok and instance == self.token_instance():
            tok = self.config.get("access_token") or os.environ.get("MASTODON_TOKEN")
        return {"Authorization": f"Bearer {tok}"} if tok else {}

    def collect(self, target: str, limit: int = 100, **opts) -> Iterator[Document]:
        kind, value, instance = self.parse_target(target, self.config.get("instance") or DEFAULT_INSTANCE)
        if not value:
            raise CollectorError(f"Invalid mastodon target '{target}'")
        base = f"https://{instance}"
        headers = self._headers(instance)
        if kind == "tag":
            src = f"#{value}@{instance}"
            yield from self._paged(f"{base}/api/v1/timelines/tag/{quote(value)}", {}, limit, headers, src)
        elif kind == "public":
            src = f"public@{instance}"
            yield from self._paged(f"{base}/api/v1/timelines/public", {"local": "true"}, limit, headers, src)
        elif kind == "account":
            acct = self.get_json(f"{base}/api/v1/accounts/lookup", params={"acct": value}, headers=headers)
            aid = (acct or {}).get("id")
            if not aid:
                raise CollectorError(f"Mastodon account '{value}@{instance}' not found")
            params = {"exclude_replies": str(bool(opts.get("exclude_replies", False))).lower(),
                      "exclude_reblogs": str(bool(opts.get("exclude_reblogs", False))).lower()}
            src = f"@{value}@{instance}"
            yield from self._paged(f"{base}/api/v1/accounts/{aid}/statuses", params, limit, headers, src)
        else:
            yield from self._search(base, value, limit, headers, f"search:{value}@{instance}")

    def _paged(self, url: str, params: dict, limit: int, headers: dict, source: str) -> Iterator[Document]:
        n = 0
        max_id = None
        while n < limit:
            p = dict(params)
            p["limit"] = min(40, limit - n)
            if max_id:
                p["max_id"] = max_id
            try:
                data = self.get_json(url, params=p, headers=headers)
            except CollectorError as exc:
                if not headers and any(c in str(exc) for c in ("HTTP 401", "HTTP 403", "HTTP 422")):
                    raise CollectorError(f"{exc} (this instance requires auth for this endpoint: set MASTODON_TOKEN "
                                         "or collectors.mastodon.tokens.<instance>)") from exc
                raise
            if not isinstance(data, list) or not data:
                break
            for s in data:
                doc = self.safe_map(lambda x: _map_status(x, source), s)
                if doc is None:
                    continue
                yield doc
                n += 1
                if n >= limit:
                    return
            max_id = data[-1].get("id")
            if not max_id:
                break

    def _search(self, base: str, q: str, limit: int, headers: dict, source: str) -> Iterator[Document]:
        n = 0
        offset = 0
        while n < limit:
            params = {"q": q, "type": "statuses", "limit": min(40, limit - n), "offset": offset}
            try:
                data = self.get_json(f"{base}/api/v2/search", params=params, headers=headers)
            except CollectorError as exc:
                if offset == 0 and not headers:
                    raise CollectorError(f"{exc} (status search usually needs an access token: set MASTODON_TOKEN)") from exc
                raise
            statuses = (data or {}).get("statuses") or []
            if not statuses:
                break
            for s in statuses:
                doc = self.safe_map(lambda x: _map_status(x, source), s)
                if doc is None:
                    continue
                yield doc
                n += 1
                if n >= limit:
                    return
            offset += len(statuses)


map_status = _map_status
