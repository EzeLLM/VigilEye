"""Hybrid retrieval over a case: SQLite FTS5 (BM25) + OpenAI-compatible embeddings, fused with RRF.

The pattern follows PaperQA2 and GPT Researcher (sparse plus dense retrieval), STORM's VectorRM
and the single-file "hybrid RAG" recipe (SQLite FTS5 + vectors + reciprocal rank fusion):

* **Lexical** (always on): an FTS5 virtual table (``porter unicode61``) over each document's title
  and text, in a separate SQLite file next to the case store (``cases.db`` -> ``cases.index.db``)
  or ``:memory:``. One table per case, so BM25 statistics never mix cases. User text only reaches
  ``MATCH`` as double-quoted phrases (:func:`fts_query`), so it can never break or inject into the
  query syntax. Where SQLite lacks FTS5, a pure-Python BM25 (same ranking idea) takes over.
* **Dense** (optional ``embedder`` role on any ``/v1/embeddings`` endpoint): document vectors are
  cached in an ``embeddings`` table keyed by ``(fingerprint, model)`` with a content hash, so
  re-indexing only embeds new or changed documents. No embedder, or any embedding failure, means
  lexical-only search (``use_embeddings="auto"``).
* **Fusion**: reciprocal rank fusion, ``score = sum(1 / (60 + rank))`` over both rankings.

Authors are never indexed, embedded or returned; :meth:`Hit.to_dict` carries refs and ranks only.
"""
from __future__ import annotations

import hashlib
import logging
import math
import operator
import re
import sqlite3
import struct
import threading
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .llm.router import LLMError
from .models import Document

log = logging.getLogger("vigileye.retrieval")

RETRIEVAL_VERSION = "1"
RRF_K = 60
EMBED_CHUNK = 256          # documents embedded (and committed to the cache) per router call
DEFAULT_EMBED_CHARS = 2000  # title + text characters sent to the embedder per document
MAX_QUERY_TERMS = 32

_EMB_SCHEMA = """
CREATE TABLE IF NOT EXISTS embeddings(
    fingerprint TEXT NOT NULL, model TEXT NOT NULL, content_sha TEXT NOT NULL, dim INTEGER NOT NULL,
    vec BLOB NOT NULL, PRIMARY KEY(fingerprint, model));
"""
_STOPWORDS = frozenset("""
a about above after again against all also am an and any are as at be because been before being below
between both but by can could did do does doing down during each few for from further had has have having
he her here hers him his how i if in into is it its itself just me more most my no nor not now of off on
once only or other our ours out over own same she should so some such than that the their theirs them then
there these they this those through to too under until up very was we were what when where which while who
whom why will with would you your yours
""".split())
_TERM_RE = re.compile(r"[^\W_]+(?:[-.:/'’][^\W_]+)*")
_WORD_RE = re.compile(r"[^\W_]+")


# ---------------------------------------------------------------------------- text helpers
def _fold(text: str) -> str:
    """NFKC, casefold and strip diacritics (what ``unicode61 remove_diacritics 2`` does)."""
    s = unicodedata.normalize("NFKD", unicodedata.normalize("NFKC", text or "").casefold())
    return unicodedata.normalize("NFC", "".join(c for c in s if not unicodedata.combining(c)))


def _stem(tok: str) -> str:
    """Light suffix stripper for the pure-Python fallback (FTS5 itself uses Porter)."""
    if not tok.isalpha():
        return tok
    if len(tok) > 4 and tok.endswith("ies"):
        tok = tok[:-3] + "y"
    elif len(tok) > 3 and tok.endswith("s") and not tok.endswith(("ss", "us", "is")):
        tok = tok[:-1]
    if len(tok) > 5 and tok.endswith("ing"):
        tok = tok[:-3]
    elif len(tok) > 4 and tok.endswith("ed"):
        tok = tok[:-2]
    elif len(tok) > 6 and tok.endswith("ation"):
        tok = tok[:-5]
    return tok


def _words(text: str) -> list[str]:
    return [_stem(w) for w in _WORD_RE.findall(_fold(text))]


def query_terms(text: str, max_terms: int = MAX_QUERY_TERMS) -> list[str]:
    """Search terms of free text: word runs, keeping joined forms (``cve-2024-3400``, ``pan-os``)
    whole, stopwords dropped (unless nothing else is left), deduplicated, at most ``max_terms``."""
    terms = list(dict.fromkeys(_TERM_RE.findall(_fold(text))))
    content = [t for t in terms if not all(w in _STOPWORDS or len(w) < 2 for w in _WORD_RE.findall(t))]
    return (content or terms)[: max(1, int(max_terms))]


def fts_query(text: str, max_terms: int = MAX_QUERY_TERMS) -> str:
    """An FTS5 ``MATCH`` expression for untrusted text: every term is a double-quoted phrase
    (inner quotes doubled) joined with ``OR``, so operators, column filters, ``NEAR``, ``*``, ``^``
    or parentheses in the input are plain tokens. Empty string when the text has no terms."""
    return " OR ".join('"' + t.replace('"', '""') + '"' for t in query_terms(text, max_terms))


def _content_sha(doc: Document) -> str:
    return hashlib.sha1(f"{doc.title}\x00{doc.text}".encode("utf-8")).hexdigest()


def _embed_text(doc: Document, max_chars: int) -> str:
    """What the embedder sees: title and text only (never the author), never empty."""
    body = "\n\n".join(p.strip() for p in (doc.title, doc.text) if p and p.strip())
    return (body[: max(1, int(max_chars))] if body else f"{doc.platform} {doc.source}".strip()) or doc.platform


def _unit(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm > 0 else list(vec)


def _pack(vec: list[float]) -> bytes:
    return struct.pack(f"<{len(vec)}f", *vec)


def _unpack(blob: bytes, dim: int) -> list[float]:
    return list(struct.unpack(f"<{dim}f", blob))


# ---------------------------------------------------------------------------- fusion
def rrf_fuse(rankings: Iterable[list[str]], k: int = RRF_K) -> dict[str, float]:
    """Reciprocal rank fusion: ``sum(1 / (k + rank))`` per key over the rankings (rank 1-based)."""
    out: dict[str, float] = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking, 1):
            out[key] = out.get(key, 0.0) + 1.0 / (k + rank)
    return out


@dataclass
class Hit:
    """One retrieved document: fused RRF ``score`` and its 1-based rank in each channel
    (``None`` when that channel did not return it or did not run)."""

    doc: Document
    score: float
    lexical_rank: int | None = None
    vector_rank: int | None = None

    @property
    def ref(self) -> str:
        return self.doc.ref

    def to_dict(self) -> dict:
        return {"ref": self.ref, "platform": self.doc.platform, "title": self.doc.title[:200],
                "score": round(self.score, 6), "lexical_rank": self.lexical_rank, "vector_rank": self.vector_rank}


# ---------------------------------------------------------------------------- BM25 fallback
class _PyBM25:
    """Okapi BM25 (k1=1.2, b=0.75, title counted twice) used when SQLite has no FTS5."""

    def __init__(self, docs: dict[str, Document], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf: dict[str, Counter] = {}
        self.dl: dict[str, int] = {}
        self.df: Counter = Counter()
        for fp, d in docs.items():
            c = Counter(_words(d.text))
            for w in _words(d.title):
                c[w] += 2
            self.tf[fp], self.dl[fp] = c, sum(c.values())
            self.df.update(c.keys())
        self.avgdl = (sum(self.dl.values()) / len(self.dl)) if self.dl else 0.0

    def search(self, query: str, depth: int, keep: set[str] | None = None) -> list[str]:
        terms = list(dict.fromkeys(_stem(w) for t in query_terms(query) for w in _WORD_RE.findall(t)))
        n = len(self.tf)
        if not terms or not n:
            return []
        idf = {t: math.log(1.0 + (n - self.df[t] + 0.5) / (self.df[t] + 0.5)) for t in terms}
        scored = []
        for fp, c in self.tf.items():
            if keep is not None and fp not in keep:
                continue
            norm = self.k1 * (1.0 - self.b + self.b * self.dl[fp] / (self.avgdl or 1.0))
            s = sum(idf[t] * c[t] * (self.k1 + 1.0) / (c[t] + norm) for t in terms if c[t])
            if s > 0:
                scored.append((-round(s, 12), fp))
        return [fp for _, fp in sorted(scored)[: max(0, depth)]]


def _fts5_available(db: sqlite3.Connection) -> bool:
    try:
        db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS temp._vigileye_fts5_probe USING fts5(x)")
        db.execute("DROP TABLE IF EXISTS temp._vigileye_fts5_probe")
        return True
    except sqlite3.OperationalError:
        return False


def default_index_path(store: Any) -> str:
    """Index file next to the case store (``.vigileye/cases.db`` -> ``.vigileye/cases.index.db``);
    ``:memory:`` for an in-memory store. Accepts a ``CaseStore`` or a path."""
    path = str(getattr(store, "path", store) or ":memory:")
    if path == ":memory:" or path.startswith("file::memory:"):
        return ":memory:"
    p = Path(path)
    return str(p.with_name(f"{p.stem}.index.db"))


# ---------------------------------------------------------------------------- index
class CaseIndex:
    """Hybrid lexical + dense index over one case's documents.

    ``use_embeddings``: ``"auto"`` (dense ranking when ``router`` has an ``embedder`` role; any
    embedding failure falls back to lexical and is noted in :attr:`errors`), ``True`` (required:
    failures raise :class:`LLMError`) or ``False`` (lexical only). ``backend``: ``"auto"``,
    ``"fts5"`` or ``"bm25"`` (pure Python). Pass ``documents`` instead of a store to index an
    ad-hoc list; its lexical tables then live in the connection's TEMP schema and only the
    embedding cache uses ``index_path``. The index syncs with the case lazily on the first
    :meth:`search`; call :meth:`refresh` after collecting more documents.
    """

    def __init__(self, store: Any, case: str, router: Any = None, *, use_embeddings: bool | str = "auto",
                 index_path: str | None = None, backend: str = "auto", role: str = "embedder",
                 embed_max_chars: int = DEFAULT_EMBED_CHARS, documents: Iterable[Document] | None = None):
        if use_embeddings not in ("auto", True, False):
            raise ValueError("use_embeddings must be 'auto', True or False")
        if backend not in ("auto", "fts5", "bm25"):
            raise ValueError("backend must be 'auto', 'fts5' or 'bm25'")
        if store is None and documents is None:
            raise ValueError("CaseIndex needs a store or a documents list")
        self.store, self.case, self.router, self.role = store, str(case), router, role
        self.use_embeddings = use_embeddings
        self.embed_max_chars = int(embed_max_chars)
        self._fixed = list(documents) if documents is not None else None
        self.path = str(index_path or (":memory:" if store is None else default_index_path(store)))
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._db.execute("PRAGMA busy_timeout=10000")
        if self.path != ":memory:":
            self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(_EMB_SCHEMA)
        tag = "adhoc" if self._fixed is not None else hashlib.sha1(self.case.encode("utf-8")).hexdigest()[:16]
        self._schema = "temp" if self._fixed is not None else "main"
        self._fts = f"fts{RETRIEVAL_VERSION}_{tag}"   # hex/fixed names only: never user text in SQL
        self._meta = f"docs{RETRIEVAL_VERSION}_{tag}"
        if backend == "bm25" or not _fts5_available(self._db):
            if backend == "fts5":
                self._db.close()
                raise RuntimeError("this SQLite build has no FTS5; use backend='bm25'")
            self.backend = "bm25"
        else:
            self.backend = "fts5"
            s = self._schema
            self._db.execute(f"CREATE TABLE IF NOT EXISTS {s}.{self._meta}(rid INTEGER PRIMARY KEY, "
                             "fingerprint TEXT UNIQUE NOT NULL, content_sha TEXT NOT NULL, platform TEXT)")
            self._db.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS {s}.{self._fts} USING fts5(title, text, "
                             "tokenize='porter unicode61 remove_diacritics 2')")
        self._docs: dict[str, Document] = {}
        self._bm25: _PyBM25 | None = None
        self._vectors: dict[str, dict[str, tuple[str, list[float]]]] = {}  # label -> fp -> (sha, unit vec)
        self._ready = False
        self.mode = "lexical"            # channels used by the last search: "lexical" or "hybrid"
        self.errors: list[str] = []      # embedding problems that forced lexical-only search

    @classmethod
    def from_documents(cls, docs: Iterable[Document], router: Any = None, **kw: Any) -> "CaseIndex":
        """Index an in-memory document list (no store); see the class docstring."""
        return cls(None, "", router, documents=docs, **kw)

    # ------------------------------------------------------------------ sync
    def _load(self) -> list[Document]:
        return list(self._fixed) if self._fixed is not None else self.store.get_documents(self.case)

    def refresh(self) -> dict:
        """Sync the lexical index with the case (incremental: only new/changed documents are
        written, vanished ones removed). Returns ``{documents, added, updated, removed, backend}``."""
        current: dict[str, Document] = {}
        for d in self._load():
            current.setdefault(d.fingerprint(), d)
        added = updated = removed = 0
        with self._lock:
            if self.backend == "fts5":
                s, db = self._schema, self._db
                db.execute("BEGIN IMMEDIATE")
                try:
                    rows = {r[1]: (r[0], r[2], r[3]) for r in
                            db.execute(f"SELECT rid, fingerprint, content_sha, platform FROM {s}.{self._meta}")}
                    for fp, (rid, _, _) in rows.items():
                        if fp not in current:
                            db.execute(f"DELETE FROM {s}.{self._fts} WHERE rowid=?", (rid,))
                            db.execute(f"DELETE FROM {s}.{self._meta} WHERE rid=?", (rid,))
                            removed += 1
                    for fp, d in current.items():
                        sha, old = _content_sha(d), rows.get(fp)
                        if old and old[1] == sha and old[2] == d.platform:
                            continue
                        if old:
                            rid = old[0]
                            db.execute(f"DELETE FROM {s}.{self._fts} WHERE rowid=?", (rid,))
                            db.execute(f"UPDATE {s}.{self._meta} SET content_sha=?, platform=? WHERE rid=?",
                                       (sha, d.platform, rid))
                            updated += 1
                        else:
                            rid = db.execute(f"INSERT INTO {s}.{self._meta}(fingerprint, content_sha, platform) "
                                             "VALUES(?,?,?)", (fp, sha, d.platform)).lastrowid
                            added += 1
                        db.execute(f"INSERT INTO {s}.{self._fts}(rowid, title, text) VALUES(?,?,?)",
                                   (rid, d.title, d.text))
                    db.execute("COMMIT")
                except BaseException:
                    db.execute("ROLLBACK")
                    raise
            else:
                added = len(set(current) - set(self._docs))
                removed = len(set(self._docs) - set(current))
                updated = sum(1 for fp in set(current) & set(self._docs)
                              if _content_sha(current[fp]) != _content_sha(self._docs[fp]))
            self._docs = current
            self._bm25 = _PyBM25(current) if self.backend == "bm25" else None
            self._ready = True
        return {"documents": len(current), "added": added, "updated": updated, "removed": removed,
                "backend": self.backend}

    # ------------------------------------------------------------------ channels
    def _lexical(self, query: str, depth: int, platform: str | None) -> list[str]:
        keep = None if platform is None else {fp for fp, d in self._docs.items() if d.platform == platform}
        if self.backend == "fts5":
            match = fts_query(query)
            if not match:
                return []
            s, t = self._schema, self._fts  # FTS5 needs the bare table name for MATCH and bm25(), not an alias
            sql = (f"SELECT m.fingerprint FROM {s}.{t} JOIN {s}.{self._meta} AS m ON m.rid = {t}.rowid "
                   f"WHERE {t} MATCH ?" + (" AND m.platform = ?" if platform is not None else "")
                   + f" ORDER BY bm25({t}, 2.0, 1.0), m.fingerprint LIMIT ?")
            args = [match] + ([platform] if platform is not None else []) + [int(depth)]
            try:
                with self._lock:
                    return [r[0] for r in self._db.execute(sql, args) if r[0] in self._docs]
            except sqlite3.OperationalError as e:  # never expected (terms are quoted); degrade, don't fail
                log.warning("FTS5 query failed (%s); using the Python BM25 fallback", e)
        if self._bm25 is None:
            self._bm25 = _PyBM25(self._docs)
        return self._bm25.search(query, depth, keep)

    def _embeddings_on(self) -> bool:
        if self.use_embeddings is False:
            return False
        has = self.router is not None and hasattr(self.router, "embeddings") and (
            self.router.has_role(self.role) if hasattr(self.router, "has_role") else True)
        if not has and self.use_embeddings is True:
            raise LLMError(f"use_embeddings=True but no '{self.role}' role is configured on the router")
        return has

    def _doc_vectors(self, label: str, dim: int) -> dict[str, list[float]]:
        """Unit vectors under ``label`` for every indexed document: memory, then the SQLite cache,
        then the embedder (in :data:`EMBED_CHUNK` batches, each committed as soon as it returns)."""
        want = {fp: _embed_text(d, self.embed_max_chars) for fp, d in self._docs.items()}
        sha = {fp: hashlib.sha1(t.encode("utf-8")).hexdigest() for fp, t in want.items()}
        mem = self._vectors.setdefault(label, {})

        def missing() -> list[str]:
            return [fp for fp in want if fp not in mem or mem[fp][0] != sha[fp] or len(mem[fp][1]) != dim]

        todo = missing()
        with self._lock:
            for i in range(0, len(todo), 500):
                part = todo[i : i + 500]
                q = (f"SELECT fingerprint, content_sha, dim, vec FROM main.embeddings WHERE model=? "
                     f"AND fingerprint IN ({','.join('?' * len(part))})")
                for fp, csha, d, blob in self._db.execute(q, [label, *part]):
                    if csha == sha[fp] and d == dim:
                        mem[fp] = (csha, _unpack(blob, d))
        todo = missing()
        for i in range(0, len(todo), EMBED_CHUNK):
            part = todo[i : i + EMBED_CHUNK]
            res = self.router.embeddings([want[fp] for fp in part], role=self.role)
            vecs = [_unit(v) for v in res.vectors]
            with self._lock:
                self._db.executemany(
                    "INSERT OR REPLACE INTO main.embeddings(fingerprint, model, content_sha, dim, vec) "
                    "VALUES(?,?,?,?,?)", [(fp, res.label, sha[fp], len(v), _pack(v)) for fp, v in zip(part, vecs)])
            if res.label != label or any(len(v) != dim for v in vecs):
                raise LLMError(f"embedder served documents with {res.label} but the query with {label} "
                               "(fallback mid-search or dimension mismatch); vector ranking skipped")
            for fp, v in zip(part, vecs):
                mem[fp] = (sha[fp], _unpack(_pack(v), len(v)))  # same float32 rounding as the cache
        return {fp: mem[fp][1] for fp in want}

    def _vector(self, query: str, depth: int, platform: str | None) -> list[str] | None:
        if not self._embeddings_on():
            return None
        try:
            q = self.router.embeddings([query], role=self.role)
            qv = _unit(_unpack(_pack(q.vectors[0]), len(q.vectors[0])))
            vecs = self._doc_vectors(q.label, len(qv))
        except Exception as e:  # noqa: BLE001 - lexical search must survive any embedder problem
            if self.use_embeddings is True:
                if isinstance(e, LLMError):
                    raise
                raise LLMError(f"embeddings failed: {e}") from e
            log.warning("embedding search unavailable, using lexical only: %s", e)
            self.errors.append(f"embeddings: {e}")
            return None
        scored = []
        for fp, v in vecs.items():
            if platform is not None and self._docs[fp].platform != platform:
                continue
            sim = sum(map(operator.mul, qv, v))
            if sim > 0:
                scored.append((-round(sim, 9), fp))
        return [fp for _, fp in sorted(scored)[: max(0, depth)]]

    # ------------------------------------------------------------------ search
    def _search(self, query: str, depth: int, platform: str | None) -> list[Hit]:
        if not self._ready:
            self.refresh()
        lex = self._lexical(query, depth, platform)
        vec = self._vector(query, depth, platform)
        self.mode = "lexical" if vec is None else "hybrid"
        lr = {fp: i for i, fp in enumerate(lex, 1)}
        vr = {fp: i for i, fp in enumerate(vec or [], 1)}
        fused = rrf_fuse([lex] + ([vec] if vec is not None else []))
        inf = float("inf")
        order = sorted(fused, key=lambda fp: (-round(fused[fp], 12), min(lr.get(fp, inf), vr.get(fp, inf)), fp))
        return [Hit(self._docs[fp], fused[fp], lr.get(fp), vr.get(fp)) for fp in order]

    def search(self, query: str, k: int = 10, *, platform: str | None = None) -> list[Hit]:
        """Top ``k`` documents for ``query`` (fused RRF over lexical and, when enabled, dense
        rankings; each channel contributes its top ``max(3k, 30)``). Ties break on the best single
        rank, then fingerprint, so results are deterministic. Empty query -> ``[]``."""
        q = " ".join(str(query or "").split())
        if not q or int(k) <= 0:
            return []
        return self._search(q, max(int(k) * 3, 30), platform)[: int(k)]

    @property
    def document_count(self) -> int:
        """Documents in the index (syncs with the case first if it has not been synced yet)."""
        if not self._ready:
            self.refresh()
        return len(self._docs)

    # ------------------------------------------------------------------ lifecycle
    def drop(self) -> None:
        """Delete this case's lexical tables and, with FTS5, the cached vectors that no other case's
        index references (call after ``CaseStore.delete_case``)."""
        with self._lock:
            self._docs, self._bm25, self._vectors, self._ready = {}, None, {}, False
            if self.backend != "fts5":
                return
            self._db.execute(f"DROP TABLE IF EXISTS {self._schema}.{self._fts}")
            self._db.execute(f"DROP TABLE IF EXISTS {self._schema}.{self._meta}")
            others = [r[0] for r in self._db.execute(
                "SELECT name FROM main.sqlite_master WHERE type='table' AND name LIKE ?",
                (f"docs{RETRIEVAL_VERSION}_%",)) if re.fullmatch(r"docs\d+_[0-9a-f]{16}", r[0])]
            keep = " UNION ".join(f"SELECT fingerprint FROM main.{t}" for t in others)
            self._db.execute("DELETE FROM main.embeddings" + (f" WHERE fingerprint NOT IN ({keep})" if keep else ""))

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def __enter__(self) -> "CaseIndex":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------- engine hook
def priority_scores(docs: list[Document]) -> list[float]:
    """Engagement (0.6) plus recency (0.4), rank-normalised to [0, 1]; mirrors
    ``AnalysisEngine._priority`` so a relevance blend keeps the engine's tie-breaking."""
    if len(docs) < 2:
        return [1.0] * len(docs)
    n = len(docs) - 1
    epoch = datetime.min.replace(tzinfo=timezone.utc)
    score = [0.0] * len(docs)
    for rank, i in enumerate(sorted(range(len(docs)), key=lambda i: docs[i].engagement())):
        score[i] += 0.6 * rank / n
    for rank, i in enumerate(sorted(range(len(docs)), key=lambda i: docs[i].created_at or epoch)):
        score[i] += 0.4 * rank / n
    return score


def rank_for_query(docs: Iterable[Document], query: str, router: Any = None, *,
                   use_embeddings: bool | str = "auto", relevance_weight: float = 0.6,
                   index_path: str = ":memory:") -> list[Document]:
    """Order ``docs`` for the engine's selection: ``relevance_weight`` x query relevance (hybrid RRF
    rank, rank-normalised to [0, 1]; documents the query does not match score 0) plus the rest x
    the engagement/recency priority. An empty query returns the engine's priority order exactly.
    ``index_path`` may point at a persistent index file to reuse its embedding cache."""
    docs = list(docs)
    prio = priority_scores(docs)
    q = " ".join(str(query or "").split())
    if len(docs) < 2 or not q:
        return [docs[i] for i in sorted(range(len(docs)), key=lambda i: -prio[i])]
    w = min(max(float(relevance_weight), 0.0), 1.0)
    with CaseIndex.from_documents(docs, router, use_embeddings=use_embeddings, index_path=index_path) as idx:
        hits = idx._search(q, depth=len(docs), platform=None)
    rel = {h.doc.fingerprint(): 1.0 - pos / len(hits) for pos, h in enumerate(hits)}
    combined = [w * rel.get(d.fingerprint(), 0.0) + (1.0 - w) * prio[i] for i, d in enumerate(docs)]
    return [docs[i] for i in sorted(range(len(docs)), key=lambda i: (-round(combined[i], 12), -prio[i], i))]


__all__ = [
    "RETRIEVAL_VERSION", "RRF_K", "Hit", "CaseIndex", "default_index_path", "fts_query", "query_terms",
    "rrf_fuse", "priority_scores", "rank_for_query",
]
