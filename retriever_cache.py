import hashlib
import json
import sqlite3
from pathlib import Path

from config import BASE_DIR

CACHE_DB_PATH = BASE_DIR / "retriever_cache.db"

def _hash_query(query: str, plan_intent: str) -> str:
    """Hash the query and intent to create a unique cache key."""
    raw = f"{query.lower().strip()}|{plan_intent}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

class RetrieverCache:
    def __init__(self, db_path: Path = CACHE_DB_PATH):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        # WAL mode prevents OperationalError: database is locked under concurrent threadpool requests
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS query_cache (
                query_hash  TEXT PRIMARY KEY,
                query       TEXT NOT NULL,
                results     TEXT NOT NULL,
                created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    def get(self, query: str, plan_intent: str):
        qhash = _hash_query(query, plan_intent)
        row = self.conn.execute(
            "SELECT results FROM query_cache WHERE query_hash = ?",
            (qhash,)
        ).fetchone()
        if row:
            return json.loads(row[0])
        return None

    def set(self, query: str, plan_intent: str, results: dict):
        qhash = _hash_query(query, plan_intent)
        # Evict entries older than 7 days to prevent unbounded DB growth
        self.conn.execute(
            "DELETE FROM query_cache WHERE created_at < datetime('now', '-7 days')"
        )
        self.conn.execute(
            "INSERT OR REPLACE INTO query_cache (query_hash, query, results) VALUES (?, ?, ?)",
            (qhash, query.lower().strip(), json.dumps(results, ensure_ascii=False))
        )
        self.conn.commit()

    def close(self):
        self.conn.close()

# Singleton instance
_cache_instance = None

def get_retriever_cache() -> RetrieverCache:
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = RetrieverCache()
    return _cache_instance
