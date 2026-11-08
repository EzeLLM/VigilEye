"""Core data model shared by collectors, the analysis engine, the store and renderers."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from typing import Any


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_dt(value: Any) -> datetime | None:
    """Accept datetime, ISO-8601 string, or unix epoch (s or ms). Always returns tz-aware UTC."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        ts = float(value)
        if ts > 1e12:  # milliseconds
            ts /= 1000.0
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    if isinstance(value, str):
        s = value.strip()
        try:
            return _parse_dt(float(s))
        except ValueError:
            pass
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(s)
        except ValueError:
            return None
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    return None


def _dt_out(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _filter_kwargs(cls, data: dict) -> dict:
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in names}


@dataclass
class Media:
    url: str
    kind: str = "image"
    description: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | str) -> "Media":
        if isinstance(d, str):
            return cls(url=d)
        return cls(**_filter_kwargs(cls, d))


@dataclass
class Document:
    """One unit of collected content (post, comment, article, page, file)."""

    id: str
    platform: str
    source: str
    text: str
    title: str = ""
    author: str | None = None
    url: str | None = None
    created_at: datetime | None = None
    metrics: dict = field(default_factory=dict)
    media: list[Media] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    raw: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.id = str(self.id)
        self.text = self.text or ""
        self.title = self.title or ""
        self.created_at = _parse_dt(self.created_at)
        self.media = [m if isinstance(m, Media) else Media.from_dict(m) for m in (self.media or [])]

    @property
    def ref(self) -> str:
        """Citation handle used in prompts and reports, e.g. ``reddit:t3_abc``."""
        return self.id if self.id.startswith(f"{self.platform}:") else f"{self.platform}:{self.id}"

    def fingerprint(self) -> str:
        return hashlib.sha1(f"{self.platform}\x00{self.id}".encode("utf-8")).hexdigest()

    def engagement(self) -> float:
        total = 0.0
        for k in ("score", "likes", "num_comments", "replies", "reposts", "points", "favourites", "reblogs"):
            v = self.metrics.get(k)
            if isinstance(v, (int, float)):
                total += float(v)
        return total

    def render(self, max_chars: int | None = None) -> str:
        head = [f"[{self.ref}]", f"{self.platform}/{self.source}"]
        if self.author:
            head.append(f"by {self.author}")
        if self.created_at:
            head.append(self.created_at.strftime("%Y-%m-%d %H:%MZ"))
        if self.metrics:
            head.append(" ".join(f"{k}={v}" for k, v in self.metrics.items() if v is not None))
        lines = [" | ".join(head)]
        if self.title:
            lines.append(f"Title: {self.title}")
        if self.text:
            lines.append(self.text.strip())
        for m in self.media:
            if m.description:
                lines.append(f"[{m.kind}: {m.description}]")
            else:
                lines.append(f"[{m.kind}: {m.url}]")
        out = "\n".join(lines)
        if max_chars is not None and len(out) > max_chars:
            out = out[: max(0, max_chars - 15)].rstrip() + " …[truncated]"
        return out

    def to_dict(self) -> dict:
        d = asdict(self)
        d["created_at"] = _dt_out(self.created_at)
        d["media"] = [m.to_dict() for m in self.media]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Document":
        return cls(**_filter_kwargs(cls, dict(d)))


@dataclass
class Entity:
    name: str
    type: str
    mentions: int = 1
    aliases: list[str] = field(default_factory=list)
    context: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Entity":
        return cls(**_filter_kwargs(cls, d))


@dataclass
class Finding:
    title: str
    summary: str
    confidence: str = "moderate"
    likelihood: str | None = None
    evidence: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Finding":
        d = dict(d)
        ev = d.get("evidence")
        if isinstance(ev, str):
            d["evidence"] = [ev]
        return cls(**_filter_kwargs(cls, d))


@dataclass
class Report:
    case: str
    lens: str
    query: str
    bluf: str
    key_judgments: list[Finding]
    body_markdown: str
    entities: list[Entity] = field(default_factory=list)
    indicators: list[dict] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    hypotheses: list[dict] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=utcnow)

    def __post_init__(self) -> None:
        self.created_at = _parse_dt(self.created_at) or utcnow()
        self.key_judgments = [k if isinstance(k, Finding) else Finding.from_dict(k) for k in self.key_judgments or []]
        self.entities = [e if isinstance(e, Entity) else Entity.from_dict(e) for e in self.entities or []]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["created_at"] = _dt_out(self.created_at)
        d["key_judgments"] = [k.to_dict() for k in self.key_judgments]
        d["entities"] = [e.to_dict() for e in self.entities]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Report":
        return cls(**_filter_kwargs(cls, dict(d)))


__all__ = ["Media", "Document", "Entity", "Finding", "Report", "utcnow"]
