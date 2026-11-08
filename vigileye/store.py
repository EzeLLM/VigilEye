"""SQLite-backed case store for documents, reports and the LLM response cache. Stdlib only."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from .models import Document, Report, utcnow

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cases(name TEXT PRIMARY KEY, created_at TEXT, description TEXT);
CREATE TABLE IF NOT EXISTS documents(
    "case" TEXT NOT NULL, fingerprint TEXT NOT NULL, platform TEXT, source TEXT, created_at TEXT,
    json TEXT NOT NULL, collected_at TEXT, PRIMARY KEY("case", fingerprint));
CREATE INDEX IF NOT EXISTS ix_docs_case_created ON documents("case", created_at);
CREATE TABLE IF NOT EXISTS reports(
    id INTEGER PRIMARY KEY AUTOINCREMENT, "case" TEXT NOT NULL, lens TEXT, created_at TEXT, json TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS ix_reports_case ON reports("case");
CREATE TABLE IF NOT EXISTS llm_cache(key TEXT PRIMARY KEY, created_at TEXT, json TEXT NOT NULL);
"""


def _iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


class CaseStore:
    def __init__(self, path=".vigileye/cases.db") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA synchronous=NORMAL")
            self._db.executescript(_SCHEMA)
            self._db.commit()

    # -- cases
    def ensure_case(self, name: str, description: str = "") -> None:
        with self._lock, self._db:
            self._db.execute("INSERT OR IGNORE INTO cases(name, created_at, description) VALUES(?,?,?)",
                             (name, _iso(utcnow()), description or ""))
            if description:
                self._db.execute("UPDATE cases SET description=? WHERE name=?", (description, name))

    def list_cases(self) -> list[dict]:
        sql = """
        SELECT c.name, c.created_at, c.description,
          (SELECT COUNT(*) FROM documents d WHERE d."case"=c.name) AS documents,
          (SELECT COUNT(*) FROM reports r WHERE r."case"=c.name) AS reports,
          (SELECT MAX(t) FROM (SELECT MAX(collected_at) t FROM documents WHERE "case"=c.name
                               UNION ALL SELECT MAX(created_at) FROM reports WHERE "case"=c.name
                               UNION ALL SELECT c.created_at)) AS last_activity
        FROM cases c ORDER BY last_activity DESC, c.name"""
        with self._lock:
            return [dict(r) for r in self._db.execute(sql).fetchall()]

    def delete_case(self, name: str) -> None:
        with self._lock, self._db:
            for table in ("documents", "reports"):
                self._db.execute(f'DELETE FROM {table} WHERE "case"=?', (name,))
            self._db.execute("DELETE FROM cases WHERE name=?", (name,))

    # -- documents
    def add_documents(self, case: str, docs) -> int:
        self.ensure_case(case)
        now = _iso(utcnow())
        new = 0
        with self._lock, self._db:
            for d in docs:
                fp = d.fingerprint()
                row = (d.platform, d.source, _iso(d.created_at),
                       json.dumps(d.to_dict(), default=str, ensure_ascii=False))
                cur = self._db.execute(
                    'INSERT OR IGNORE INTO documents("case", fingerprint, platform, source, created_at, json, '
                    "collected_at) VALUES(?,?,?,?,?,?,?)", (case, fp, *row, now))
                if cur.rowcount:
                    new += 1
                else:
                    self._db.execute('UPDATE documents SET platform=?, source=?, created_at=?, json=? '
                                     'WHERE "case"=? AND fingerprint=?', (*row, case, fp))
        return new

    def get_documents(self, case: str, platform: str | None = None, since: datetime | None = None,
                      limit: int | None = None) -> list[Document]:
        sql, args = 'SELECT json FROM documents WHERE "case"=?', [case]
        if platform:
            sql += " AND platform=?"
            args.append(platform)
        if since is not None:
            sql += " AND created_at >= ?"
            args.append(_iso(since))
        sql += " ORDER BY created_at IS NULL, created_at DESC, collected_at DESC"
        if limit is not None:
            sql += " LIMIT ?"
            args.append(int(limit))
        with self._lock:
            rows = self._db.execute(sql, args).fetchall()
        return [Document.from_dict(json.loads(r["json"])) for r in rows]

    def count_documents(self, case: str) -> int:
        with self._lock:
            return self._db.execute('SELECT COUNT(*) FROM documents WHERE "case"=?', (case,)).fetchone()[0]

    def export_jsonl(self, case: str, path) -> int:
        docs = self.get_documents(case)
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as fh:
            for d in docs:
                fh.write(json.dumps(d.to_dict(), default=str, ensure_ascii=False) + "\n")
        return len(docs)

    # -- reports
    def save_report(self, report: Report) -> int:
        self.ensure_case(report.case)
        with self._lock, self._db:
            cur = self._db.execute('INSERT INTO reports("case", lens, created_at, json) VALUES(?,?,?,?)',
                                   (report.case, report.lens, _iso(report.created_at),
                                    json.dumps(report.to_dict(), default=str, ensure_ascii=False)))
            return int(cur.lastrowid)

    def list_reports(self, case: str) -> list[dict]:
        with self._lock:
            rows = self._db.execute('SELECT id, "case", lens, created_at, json FROM reports WHERE "case"=? '
                                    "ORDER BY created_at DESC, id DESC", (case,)).fetchall()
        out = []
        for r in rows:
            try:
                bluf = json.loads(r["json"]).get("bluf", "")
            except (ValueError, AttributeError):
                bluf = ""
            out.append({"id": r["id"], "case": r["case"], "lens": r["lens"], "created_at": r["created_at"],
                        "bluf": bluf})
        return out

    def get_report(self, id: int) -> Report | None:
        with self._lock:
            row = self._db.execute("SELECT json FROM reports WHERE id=?", (int(id),)).fetchone()
        return Report.from_dict(json.loads(row["json"])) if row else None

    # -- llm cache
    def cache_get(self, key: str) -> dict | None:
        with self._lock:
            row = self._db.execute("SELECT json FROM llm_cache WHERE key=?", (key,)).fetchone()
        if not row:
            return None
        try:
            return json.loads(row["json"])
        except ValueError:
            return None

    def cache_put(self, key: str, value: dict) -> None:
        with self._lock, self._db:
            self._db.execute("INSERT OR REPLACE INTO llm_cache(key, created_at, json) VALUES(?,?,?)",
                             (key, _iso(utcnow()), json.dumps(value, default=str, ensure_ascii=False)))

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def __enter__(self) -> "CaseStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


__all__ = ["CaseStore"]
