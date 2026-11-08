"""Local file collector: JSONL / JSON / CSV / TXT / Markdown, files or directories.

Lets analysts bring their own exports (scraped datasets, previous VigilEye
case exports, notes, transcripts) into the analysis pipeline.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Iterator

from ..models import Document, Media
from .base import Collector, CollectorError, register, to_utc

SUPPORTED = {".jsonl", ".ndjson", ".json", ".csv", ".tsv", ".txt", ".md", ".markdown"}

TEXT_KEYS = ("text", "body", "content", "selftext", "message", "full_text", "description", "comment", "post", "caption")
TITLE_KEYS = ("title", "headline", "subject", "name")
AUTHOR_KEYS = ("author", "user", "username", "screen_name", "handle", "from", "by", "creator")
URL_KEYS = ("url", "link", "permalink", "href", "source_url")
DATE_KEYS = ("created_at", "created_utc", "date", "timestamp", "time", "published", "published_at", "datetime", "pubDate")
ID_KEYS = ("id", "post_id", "guid", "uuid", "name", "fullname")
PLATFORM_KEYS = ("platform", "site", "network")
SOURCE_KEYS = ("source", "subreddit", "channel", "feed", "community", "group")
METRIC_KEYS = ("score", "likes", "upvotes", "num_comments", "comments", "replies", "shares", "retweets", "reposts",
               "views", "points", "favourites", "reblogs")
MEDIA_KEYS = ("media", "images", "image", "image_url", "attachments", "photos")
TAG_KEYS = ("tags", "hashtags", "labels", "categories")

_DOC_FIELDS = {"id", "platform", "source", "text"}
_SEQUENCE_ID = re.compile(r"^-?\d{1,12}(?:\.0)?$")


def _first(d: dict, keys: Iterable[str]) -> Any:
    lower = {str(k).lower(): v for k, v in d.items()}
    for k in keys:
        v = lower.get(k.lower())
        if v not in (None, ""):
            return v
    return None


def _as_str(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, dict):
        return v.get("name") or v.get("username") or v.get("screen_name") or json.dumps(v, ensure_ascii=False)
    return str(v)


def _media(v: Any) -> list[Media]:
    if not v:
        return []
    if isinstance(v, str):
        v = [u for u in re.split(r"[\s,;|]+", v) if u.startswith("http")]
    if isinstance(v, dict):
        v = [v]
    out = []
    for m in v if isinstance(v, list) else []:
        if isinstance(m, str):
            out.append(Media(url=m))
        elif isinstance(m, dict) and (m.get("url") or m.get("src")):
            out.append(Media(url=m.get("url") or m.get("src"), kind=m.get("kind") or m.get("type") or "image",
                             description=m.get("description") or m.get("alt")))
    return out


def _tags(v: Any) -> list[str]:
    if not v:
        return []
    if isinstance(v, str):
        return [t.strip() for t in re.split(r"[,;|]", v) if t.strip()]
    if isinstance(v, list):
        return [str(t.get("name") if isinstance(t, dict) else t) for t in v if t]
    return [str(v)]


def _num(v: Any) -> Any:
    if isinstance(v, str):
        s = v.strip().replace(",", "")
        try:
            return int(s)
        except ValueError:
            try:
                return float(s)
            except ValueError:
                return v
    return v


def record_to_document(rec: dict, origin: str, index: int = 0, default_platform: str = "file") -> Document:
    """Map an arbitrary record (dict) to a Document.

    Records that already look like serialised VigilEye Documents (``id``,
    ``platform``, ``source``, ``text``) round-trip via ``Document.from_dict``.
    """
    if not isinstance(rec, dict):
        rec = {"text": str(rec)}
    if _DOC_FIELDS.issubset(rec.keys()):
        return Document.from_dict(rec)
    # Some exports nest the payload ("data": {...}) e.g. Reddit listings.
    if "data" in rec and isinstance(rec["data"], dict) and len(rec) <= 3:
        rec = {**rec["data"], **{k: v for k, v in rec.items() if k != "data"}}
    text = _first(rec, TEXT_KEYS)
    title = _first(rec, TITLE_KEYS)
    if text is None and title is None:
        raise ValueError(f"record {index} has no text-like field (keys: {', '.join(list(rec)[:8])})")
    if isinstance(text, (list, dict)):
        text = json.dumps(text, ensure_ascii=False)
    rid = _first(rec, ID_KEYS)
    platform = _as_str(_first(rec, PLATFORM_KEYS)) or default_platform
    source = _as_str(_first(rec, SOURCE_KEYS)) or Path(origin).name
    if "subreddit" in {k.lower() for k in rec} and not source.startswith("r/"):
        source = f"r/{source}"
        if platform == default_platform:
            platform = "reddit"
    if rid is None:
        rid = hashlib.sha1(f"{origin}|{index}|{text}|{title}".encode("utf-8")).hexdigest()[:16]
    elif _SEQUENCE_ID.match(str(rid).strip()):
        # Row numbers and per-channel message ids (CSV ``id`` 1,2,..., Telegram/Discord exports) are
        # only unique inside their source: namespace them so two files or channels never merge.
        rid = f"{source}:{str(rid).strip()}"
    metrics = {}
    lower = {str(k).lower(): v for k, v in rec.items()}
    for k in METRIC_KEYS:
        if k in lower and lower[k] not in (None, "") and not isinstance(lower[k], (list, dict)):
            metrics[k] = _num(lower[k])
    known = set(TEXT_KEYS + TITLE_KEYS + AUTHOR_KEYS + URL_KEYS + DATE_KEYS + ID_KEYS + PLATFORM_KEYS + SOURCE_KEYS
                + METRIC_KEYS + MEDIA_KEYS + TAG_KEYS)
    raw = {k: v for k, v in rec.items() if str(k).lower() not in {x.lower() for x in known}
           and isinstance(v, (str, int, float, bool)) and len(str(v)) < 500}
    return Document(
        id=str(rid),
        platform=platform,
        source=source,
        title=str(title or "") if text is not None else "",
        text=str(text if text is not None else title),
        author=_as_str(_first(rec, AUTHOR_KEYS)),
        url=_as_str(_first(rec, URL_KEYS)),
        created_at=to_utc(_first(rec, DATE_KEYS)),
        metrics=metrics,
        media=_media(_first(rec, MEDIA_KEYS)),
        tags=_tags(_first(rec, TAG_KEYS)),
        raw=raw,
        meta={"origin": origin, "index": index},
    )


def _iter_json(path: Path) -> Iterator[Any]:
    with path.open("r", encoding="utf-8-sig") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        for key in ("documents", "docs", "items", "posts", "data", "results", "records", "entries", "hits", "statuses", "feed"):
            if isinstance(data.get(key), list):
                yield from data[key]
                return
            if isinstance(data.get(key), dict) and isinstance(data[key].get("children"), list):  # reddit listing
                yield from data[key]["children"]
                return
        yield data
    elif isinstance(data, list):
        yield from data


def _iter_jsonl(path: Path, log) -> Iterator[Any]:
    with path.open("r", encoding="utf-8-sig") as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                log.warning("%s:%d invalid JSON line skipped: %s", path, i, exc)


def _iter_csv(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        sample = fh.read(4096)
        fh.seek(0)
        delim = "\t" if path.suffix.lower() == ".tsv" else ","
        if path.suffix.lower() != ".tsv":
            try:
                delim = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
            except csv.Error:
                pass
        for row in csv.DictReader(fh, delimiter=delim):
            yield {k.strip() if isinstance(k, str) else k: v for k, v in row.items() if k is not None}


def _text_docs(path: Path, split: bool) -> Iterator[Document]:
    content = path.read_text(encoding="utf-8", errors="replace")
    stat = path.stat()
    mtime = to_utc(stat.st_mtime)
    title = path.stem
    if path.suffix.lower() in {".md", ".markdown"}:
        m = re.search(r"^#\s+(.+)$", content, re.M)
        if m:
            title = m.group(1).strip()
    if not split:
        yield Document(id=f"file_{hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:16]}", platform="file",
                       source=path.name, title=title, text=content.strip(), url=path.resolve().as_uri(),
                       created_at=mtime, metrics={"chars": len(content)}, meta={"origin": str(path)})
        return
    blocks = [b.strip() for b in re.split(r"\n\s*\n", content) if b.strip()]
    for i, block in enumerate(blocks):
        yield Document(id=f"file_{hashlib.sha1(f'{path.resolve()}|{i}'.encode()).hexdigest()[:16]}", platform="file",
                       source=path.name, title=f"{title} #{i + 1}", text=block, url=path.resolve().as_uri(),
                       created_at=mtime, meta={"origin": str(path), "block": i})


@register
class FileCollector(Collector):
    name = "files"
    description = "Local JSONL/JSON/CSV/TXT/Markdown files or directories (bring-your-own datasets and exports)."
    examples = [
        "file:data/posts.jsonl",
        "file:export/case.json",
        "file:dataset.csv",
        "file:notes/  (directory, recursive)",
        "./transcript.txt",
    ]

    def default_min_interval(self) -> float:
        return 0.0

    def __init__(self, config: dict | None = None, http=None) -> None:
        # No network needed: avoid building an HTTP client unless one is passed.
        self.config = dict(config or {})
        import logging

        self.log = logging.getLogger("vigileye.collectors.files")
        self.http = http
        self._owns_http = False
        self.min_interval = 0.0

    def close(self) -> None:  # nothing to close
        return None

    def collect(self, target: str, limit: int = 100, **opts) -> Iterator[Document]:
        path = Path(os.path.expanduser(target.strip()))
        if not path.exists():
            raise CollectorError(f"No such file or directory: {target}")
        split = bool(opts.get("split", self.config.get("split", False)))
        platform = opts.get("platform") or self.config.get("platform") or "file"
        n = 0
        owners: dict[tuple[str, str], tuple[str, str]] = {}  # (platform, id) -> (origin, source)
        for f in self._files(path):
            for doc in self.load_file(f, split=split, platform=platform):
                self._disambiguate(doc, owners, platform)
                yield doc
                n += 1
                if n >= limit:
                    return

    def _disambiguate(self, doc: Document, owners: dict, default_platform: str) -> None:
        """Give a mapped file record whose id another file or source already used its own id.

        Records with no platform identity carry ids from a file-local namespace; two such files
        (``a.csv`` and ``b.csv`` both with ``id`` x1) would otherwise share a fingerprint, and the
        later one would silently overwrite the earlier one in the case store.
        """
        origin = str((doc.meta or {}).get("origin") or "")
        if doc.platform != default_platform or not origin:
            return
        owner = (origin, doc.source)
        key = (doc.platform, doc.id)
        prev = owners.setdefault(key, owner)
        if prev == owner:
            return
        new_id = f"{doc.source}:{doc.id}"
        if owners.get((doc.platform, new_id), owner) != owner:
            new_id = f"{hashlib.sha1(origin.encode('utf-8')).hexdigest()[:8]}:{new_id}"
        self.log.warning("record id %s in %s is already used by %s; stored as %s", doc.id, origin, prev[0], new_id)
        doc.id = new_id
        owners[(doc.platform, new_id)] = owner

    @staticmethod
    def _files(path: Path) -> Iterator[Path]:
        if path.is_file():
            yield path
            return
        for p in sorted(path.rglob("*")):
            if p.is_file() and p.suffix.lower() in SUPPORTED and not any(part.startswith(".") for part in p.relative_to(path).parts):
                yield p

    def load_file(self, path: Path, split: bool = False, platform: str = "file") -> Iterator[Document]:
        """Yield Documents parsed from a single file (bad records are logged and skipped)."""
        ext = path.suffix.lower()
        origin = str(path)
        try:
            if ext in {".txt", ".md", ".markdown"} or ext not in SUPPORTED:
                yield from _text_docs(path, split)
                return
            if ext in {".jsonl", ".ndjson"}:
                records: Iterable[Any] = _iter_jsonl(path, self.log)
            elif ext == ".json":
                try:
                    records = list(_iter_json(path))
                except json.JSONDecodeError:
                    records = list(_iter_jsonl(path, self.log))  # JSONL saved as .json
            else:
                records = _iter_csv(path)
            for i, rec in enumerate(records):
                doc = self.safe_map(lambda r: record_to_document(r, origin, i, platform), rec)
                if doc is not None:
                    yield doc
        except (OSError, UnicodeDecodeError, csv.Error) as exc:
            self.log.warning("failed to read %s: %s", path, exc)


def load_documents(path: str, limit: int = 1_000_000, **opts) -> list[Document]:
    """Convenience: load all documents from a file or directory."""
    return list(FileCollector().collect(path, limit=limit, **opts))
