"""Simple SQLite-backed key-value storage for SatoshiRig."""

import os
import sqlite3
import logging
from contextlib import contextmanager
from typing import Optional

from .utils.logging_utils import _vlog

_logger = logging.getLogger("SatoshiRig.db")
_verbose_logging = True  # Always enable verbose logging for db

DB_PATH = os.environ.get("STATE_DB", os.path.join(os.getcwd(), "data", "state.db"))
_vlog(_logger, _verbose_logging, f"db: DB_PATH={DB_PATH}")


@contextmanager
def get_conn():
    _vlog(_logger, _verbose_logging, "db.get_conn: START")
    _vlog(_logger, _verbose_logging, f"db.get_conn: creating directory for DB_PATH={DB_PATH}")
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    _vlog(_logger, _verbose_logging, f"db.get_conn: connecting to database at {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    _vlog(_logger, _verbose_logging, f"db.get_conn: connection created")
    try:
        _vlog(_logger, _verbose_logging, "db.get_conn: creating table if not exists")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS kv_store (
                section TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT,
                PRIMARY KEY(section, key)
            )
            """
        )
        _vlog(_logger, _verbose_logging, "db.get_conn: table creation completed")
        _vlog(_logger, _verbose_logging, "db.get_conn: yielding connection")
        yield conn
        _vlog(_logger, _verbose_logging, "db.get_conn: committing transaction")
        conn.commit()
        _vlog(_logger, _verbose_logging, "db.get_conn: commit completed")
    finally:
        _vlog(_logger, _verbose_logging, "db.get_conn: closing connection")
        conn.close()
        _vlog(_logger, _verbose_logging, "db.get_conn: connection closed, END")


def get_value(section: str, key: str, default: Optional[str] = None) -> Optional[str]:
    _vlog(_logger, _verbose_logging, f"db.get_value: START section={section}, key={key}, default={default}")
    _vlog(_logger, _verbose_logging, "db.get_value: getting connection")
    with get_conn() as conn:
        _vlog(_logger, _verbose_logging, f"db.get_value: executing SELECT query")
        cur = conn.execute(
            "SELECT value FROM kv_store WHERE section = ? AND key = ?",
            (section, key),
        )
        _vlog(_logger, _verbose_logging, "db.get_value: fetching one row")
        row = cur.fetchone()
        _vlog(_logger, _verbose_logging, f"db.get_value: row={'present' if row else 'None'}")
        result = row[0] if row else default
        _vlog(_logger, _verbose_logging, f"db.get_value: result={'present' if result else 'None'}, length={len(result) if result else 0}, END")
        return result


def set_value(section: str, key: str, value: str) -> None:
    _vlog(_logger, _verbose_logging, f"db.set_value: START section={section}, key={key}, value length={len(value) if value else 0}")
    _vlog(_logger, _verbose_logging, "db.set_value: getting connection")
    with get_conn() as conn:
        _vlog(_logger, _verbose_logging, "db.set_value: executing INSERT/UPDATE query")
        conn.execute(
            """
            INSERT INTO kv_store(section, key, value)
            VALUES(?, ?, ?)
            ON CONFLICT(section, key)
            DO UPDATE SET value = excluded.value
            """,
            (section, key, value),
        )
        _vlog(_logger, _verbose_logging, "db.set_value: query executed, END")


def delete_value(section: str, key: str) -> None:
    _vlog(_logger, _verbose_logging, f"db.delete_value: START section={section}, key={key}")
    _vlog(_logger, _verbose_logging, "db.delete_value: getting connection")
    with get_conn() as conn:
        _vlog(_logger, _verbose_logging, "db.delete_value: executing DELETE query")
        conn.execute(
            "DELETE FROM kv_store WHERE section = ? AND key = ?",
            (section, key),
        )
        _vlog(_logger, _verbose_logging, "db.delete_value: query executed, END")


def get_section(section: str) -> dict:
    _vlog(_logger, _verbose_logging, f"db.get_section: START section={section}")
    _vlog(_logger, _verbose_logging, "db.get_section: getting connection")
    with get_conn() as conn:
        _vlog(_logger, _verbose_logging, "db.get_section: executing SELECT query")
        cur = conn.execute(
            "SELECT key, value FROM kv_store WHERE section = ?",
            (section,),
        )
        _vlog(_logger, _verbose_logging, "db.get_section: fetching all rows")
        rows = cur.fetchall()
        _vlog(_logger, _verbose_logging, f"db.get_section: rows count={len(rows)}")
        result = {row[0]: row[1] for row in rows}
        _vlog(_logger, _verbose_logging, f"db.get_section: result keys={list(result.keys())}, END")
        return result
