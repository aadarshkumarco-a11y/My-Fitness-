"""SQLite-backed time-series store for scraped market data and trade logs."""
from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Optional, Sequence

import pandas as pd

from ..utils import get_logger, settings

logger = get_logger(__name__)


_OPTION_CHAIN_DDL = """
CREATE TABLE IF NOT EXISTS option_chain (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    symbol TEXT NOT NULL,
    expiry TEXT,
    strike REAL NOT NULL,
    option_type TEXT NOT NULL CHECK(option_type IN ('CE','PE')),
    ltp REAL,
    oi INTEGER,
    iv REAL,
    spot REAL
);
"""

_EQUITY_DDL = """
CREATE TABLE IF NOT EXISTS equity_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    symbol TEXT NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume INTEGER,
    UNIQUE(symbol, timestamp)
);
"""

_TRADE_DDL = """
CREATE TABLE IF NOT EXISTS trade_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity REAL NOT NULL,
    price REAL NOT NULL,
    pnl REAL,
    notes TEXT
);
"""

_INDEX_DDL = [
    "CREATE INDEX IF NOT EXISTS idx_option_chain_lookup ON option_chain(symbol, timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_equity_lookup ON equity_history(symbol, timestamp);",
]


class Database:
    """Thin synchronous wrapper around a single SQLite file."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self._path = Path(path) if path else settings.storage.db_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            conn = sqlite3.connect(self._path)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            try:
                yield conn
                conn.commit()
            finally:
                conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(_OPTION_CHAIN_DDL)
            conn.execute(_EQUITY_DDL)
            conn.execute(_TRADE_DDL)
            for stmt in _INDEX_DDL:
                conn.execute(stmt)

    # ----------------------------------------------------------- option chain
    def insert_option_chain_rows(self, rows: Sequence[dict]) -> int:
        if not rows:
            return 0
        with self._connect() as conn:
            cur = conn.executemany(
                """
                INSERT INTO option_chain(timestamp, symbol, expiry, strike, option_type, ltp, oi, iv, spot)
                VALUES(:timestamp,:symbol,:expiry,:strike,:option_type,:ltp,:oi,:iv,:spot)
                """,
                rows,
            )
            return cur.rowcount or 0

    def fetch_option_chain(self, symbol: str, limit: int = 500) -> pd.DataFrame:
        with self._connect() as conn:
            return pd.read_sql_query(
                "SELECT * FROM option_chain WHERE symbol=? ORDER BY id DESC LIMIT ?",
                conn,
                params=(symbol.upper(), limit),
            )

    # ---------------------------------------------------------- equity history
    def insert_equity_rows(self, rows: Iterable[dict]) -> int:
        rows = list(rows)
        if not rows:
            return 0
        with self._connect() as conn:
            cur = conn.executemany(
                """
                INSERT OR IGNORE INTO equity_history(timestamp,symbol,open,high,low,close,volume)
                VALUES(:timestamp,:symbol,:open,:high,:low,:close,:volume)
                """,
                rows,
            )
            return cur.rowcount or 0

    def fetch_equity_history(self, symbol: str) -> pd.DataFrame:
        with self._connect() as conn:
            df = pd.read_sql_query(
                "SELECT timestamp, open, high, low, close, volume FROM equity_history "
                "WHERE symbol=? ORDER BY timestamp ASC",
                conn,
                params=(symbol.upper(),),
            )
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        return df

    # --------------------------------------------------------------- trades
    def insert_trades(self, run_id: str, trades: Iterable[dict]) -> int:
        rows = []
        for t in trades:
            rows.append(
                {
                    "run_id": run_id,
                    "timestamp": str(t.get("timestamp")),
                    "symbol": t.get("symbol", ""),
                    "side": t.get("side", ""),
                    "quantity": float(t.get("quantity", 0)),
                    "price": float(t.get("price", 0)),
                    "pnl": float(t.get("pnl") or 0.0),
                    "notes": t.get("notes", ""),
                }
            )
        if not rows:
            return 0
        with self._connect() as conn:
            cur = conn.executemany(
                """
                INSERT INTO trade_log(run_id,timestamp,symbol,side,quantity,price,pnl,notes)
                VALUES(:run_id,:timestamp,:symbol,:side,:quantity,:price,:pnl,:notes)
                """,
                rows,
            )
            return cur.rowcount or 0

    def fetch_trades(self, run_id: Optional[str] = None) -> pd.DataFrame:
        with self._connect() as conn:
            if run_id:
                return pd.read_sql_query(
                    "SELECT * FROM trade_log WHERE run_id=? ORDER BY id ASC",
                    conn,
                    params=(run_id,),
                )
            return pd.read_sql_query("SELECT * FROM trade_log ORDER BY id ASC", conn)

    # ---------------------------------------------------------------- helpers
    def execute(self, sql: str, params: Sequence[Any] = ()) -> List[Any]:
        with self._connect() as conn:
            cur = conn.execute(sql, params)
            return cur.fetchall()
