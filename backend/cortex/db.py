import re
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .chunking import Chunk

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    source_type TEXT NOT NULL,
    source TEXT,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ord INTEGER NOT NULL,
    text TEXT NOT NULL,
    heading TEXT,
    page INTEGER,
    embedding BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_doc ON chunks(doc_id, ord);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, title, tokenize = 'porter unicode61'
);
"""

_STOPWORDS = frozenset(
    "a an and are as at be but by can do does did for from had has have how i if in into is it "
    "its me my of on or so than that the their them then there these they this to was we were "
    "what when where which who why will with would you your about should could".split()
)


def fts_query(text: str) -> str | None:
    """Turn free text into a safe FTS5 query: quoted terms joined with OR."""
    terms = [t for t in re.findall(r"\w+", text.lower()) if t not in _STOPWORDS and len(t) > 1]
    if not terms:
        terms = re.findall(r"\w+", text.lower())
    if not terms:
        return None
    return " OR ".join(f'"{t}"' for t in dict.fromkeys(terms))


class Store:
    """SQLite storage for documents, chunks, embeddings and the keyword index."""

    def __init__(self, path: Path | str):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.executescript(_SCHEMA)
        self._matrix: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # ---- writes -------------------------------------------------------------

    def add_document(
        self,
        title: str,
        source_type: str,
        source: str | None,
        content: str,
        content_hash: str,
        chunks: list[Chunk],
        embeddings: np.ndarray,
    ) -> int:
        with self._lock, self._db:
            cur = self._db.execute(
                "INSERT INTO documents "
                "(title, source_type, source, content, content_hash, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (title, source_type, source, content, content_hash, datetime.now(timezone.utc).isoformat()),
            )
            doc_id = cur.lastrowid
            for ord_, (chunk, vector) in enumerate(zip(chunks, embeddings)):
                cur = self._db.execute(
                    "INSERT INTO chunks (doc_id, ord, text, heading, page, embedding) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        doc_id,
                        ord_,
                        chunk.text,
                        chunk.heading,
                        chunk.page,
                        np.asarray(vector, dtype=np.float32).tobytes(),
                    ),
                )
                self._db.execute(
                    "INSERT INTO chunks_fts (rowid, text, title) VALUES (?, ?, ?)",
                    (cur.lastrowid, chunk.text, f"{title} {chunk.heading or ''}"),
                )
            self._matrix = None
            return doc_id

    def delete_document(self, doc_id: int) -> bool:
        with self._lock, self._db:
            self._db.execute(
                "DELETE FROM chunks_fts WHERE rowid IN (SELECT id FROM chunks WHERE doc_id = ?)",
                (doc_id,),
            )
            cur = self._db.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            self._matrix = None
            return cur.rowcount > 0

    # ---- reads --------------------------------------------------------------

    def find_by_hash(self, content_hash: str) -> int | None:
        with self._lock:
            row = self._db.execute(
                "SELECT id FROM documents WHERE content_hash = ?", (content_hash,)
            ).fetchone()
        return row["id"] if row else None

    def list_documents(self) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT d.id, d.title, d.source_type, d.source, d.created_at, "
                "COUNT(c.id) AS chunk_count, LENGTH(d.content) AS chars "
                "FROM documents d LEFT JOIN chunks c ON c.doc_id = d.id "
                "GROUP BY d.id ORDER BY d.created_at DESC, d.id DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_document(self, doc_id: int) -> dict | None:
        with self._lock:
            doc = self._db.execute(
                "SELECT id, title, source_type, source, content, created_at "
                "FROM documents WHERE id = ?",
                (doc_id,),
            ).fetchone()
            if doc is None:
                return None
            chunks = self._db.execute(
                "SELECT id, ord, text, heading, page FROM chunks WHERE doc_id = ? ORDER BY ord",
                (doc_id,),
            ).fetchall()
        return {**dict(doc), "chunks": [dict(c) for c in chunks]}

    def counts(self) -> tuple[int, int]:
        with self._lock:
            docs = self._db.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            chunks = self._db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return docs, chunks

    def embedding_matrix(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (chunk_ids, doc_ids, embeddings) for every chunk, cached until a write."""
        with self._lock:
            if self._matrix is None:
                rows = self._db.execute(
                    "SELECT id, doc_id, embedding FROM chunks ORDER BY id"
                ).fetchall()
                chunk_ids = np.array([r["id"] for r in rows], dtype=np.int64)
                doc_ids = np.array([r["doc_id"] for r in rows], dtype=np.int64)
                if rows:
                    vectors = np.vstack(
                        [np.frombuffer(r["embedding"], dtype=np.float32) for r in rows]
                    )
                else:
                    vectors = np.zeros((0, 0), dtype=np.float32)
                self._matrix = (chunk_ids, doc_ids, vectors)
            return self._matrix

    def keyword_search(self, query: str, limit: int) -> list[int]:
        """Chunk ids ranked by BM25 (best first)."""
        match = fts_query(query)
        if match is None:
            return []
        with self._lock:
            rows = self._db.execute(
                "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? "
                "ORDER BY bm25(chunks_fts, 1.0, 0.5) LIMIT ?",
                (match, limit),
            ).fetchall()
        return [r["rowid"] for r in rows]

    def get_chunks(self, chunk_ids: list[int]) -> dict[int, dict]:
        if not chunk_ids:
            return {}
        marks = ",".join("?" * len(chunk_ids))
        with self._lock:
            rows = self._db.execute(
                "SELECT c.id AS chunk_id, c.doc_id, c.ord, c.text, c.heading, c.page, "
                "d.title, d.source_type, d.source "
                f"FROM chunks c JOIN documents d ON d.id = c.doc_id WHERE c.id IN ({marks})",
                chunk_ids,
            ).fetchall()
        return {r["chunk_id"]: dict(r) for r in rows}
