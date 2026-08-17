"""SQLite cache backend."""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

from pyantra_memory.cache.base import CacheBackend, CacheRegistry


class SQLiteCache(CacheBackend):
    """A durable, thread-safe cache backed by a single SQLite file.

    Uses WAL mode for concurrent reads. The schema is created automatically
    on first use. Suitable for single-machine workflows that need cache
    persistence across restarts.

    Example::

        cache = SQLiteCache("cache.db")
        cache.set("key", b"value", ttl=3600)
        data = cache.get("key")
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        self._ensure_table()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self._path))
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
        return self._conn

    def _ensure_table(self) -> None:
        conn = self._get_conn()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache (
                key       TEXT PRIMARY KEY,
                value     BLOB NOT NULL,
                expires_at REAL
            )
            """
        )
        conn.commit()

    def get(self, key: str) -> bytes | None:
        with self._lock:
            conn = self._get_conn()
            row = conn.execute(
                "SELECT value, expires_at FROM cache WHERE key = ?",
                (key,),
            ).fetchone()
            if row is None:
                return None
            value, expires_at = row
            if expires_at is not None and time.monotonic() > expires_at:
                conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                conn.commit()
                return None
            return bytes(value)

    def set(self, key: str, value: bytes, ttl: float | None = None) -> None:
        expires_at = (time.monotonic() + ttl) if ttl is not None else None
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "INSERT OR REPLACE INTO cache (key, value, expires_at) VALUES (?, ?, ?)",
                (key, value, expires_at),
            )
            conn.commit()

    def delete(self, key: str) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.execute("DELETE FROM cache WHERE key = ?", (key,))
            conn.commit()

    def clear(self) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.execute("DELETE FROM cache")
            conn.commit()

    def close(self) -> None:
        """Close the underlying database connection."""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None


CacheRegistry.register("sqlite", SQLiteCache)

__all__ = ["SQLiteCache"]
