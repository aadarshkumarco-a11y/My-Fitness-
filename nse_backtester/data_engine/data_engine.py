"""Normalize NSE responses into clean DataFrames and persist them."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..database import Database
from ..scraper import NSEScraper
from ..utils import get_logger

logger = get_logger(__name__)


@dataclass
class OptionChainSnapshot:
    symbol: str
    timestamp: str
    spot: float
    atm_strike: Optional[float]
    expiries: List[str] = field(default_factory=list)
    rows: List[Dict[str, Any]] = field(default_factory=list)

    def as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


class DataEngine:
    """Bridges scraper output and our SQLite store."""

    def __init__(
        self,
        scraper: Optional[NSEScraper] = None,
        db: Optional[Database] = None,
    ) -> None:
        self._scraper = scraper or NSEScraper()
        self._db = db or Database()

    # -------------------------------------------------------------- option chain
    def fetch_option_chain(self, symbol: str = "NIFTY", persist: bool = True) -> Optional[OptionChainSnapshot]:
        result = self._scraper.fetch_option_chain(symbol=symbol)
        if not result.ok or result.data is None:
            logger.error("Failed to fetch option chain for %s: %s", symbol, result.error)
            return None
        snapshot = self._parse_option_chain(symbol, result.data)
        if persist and snapshot.rows:
            inserted = self._db.insert_option_chain_rows(snapshot.rows)
            logger.info("Inserted %d option-chain rows for %s", inserted, symbol)
        return snapshot

    @staticmethod
    def _safe_float(val: Any) -> Optional[float]:
        if val is None or val == "" or val == "-":
            return None
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _safe_int(val: Any) -> Optional[int]:
        f = DataEngine._safe_float(val)
        return None if f is None else int(f)

    def _parse_option_chain(self, symbol: str, payload: Dict[str, Any]) -> OptionChainSnapshot:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        records = payload.get("records", {}) or {}
        spot = self._safe_float(records.get("underlyingValue")) or 0.0
        expiries = list(records.get("expiryDates", []) or [])

        rows: List[Dict[str, Any]] = []
        strikes: List[float] = []

        for entry in records.get("data", []) or []:
            strike = self._safe_float(entry.get("strikePrice"))
            if strike is None:
                continue
            strikes.append(strike)
            expiry = entry.get("expiryDate")

            for opt_type_key, opt_type in (("CE", "CE"), ("PE", "PE")):
                node = entry.get(opt_type_key) or {}
                if not node:
                    continue
                rows.append(
                    {
                        "timestamp": timestamp,
                        "symbol": symbol.upper(),
                        "expiry": expiry,
                        "strike": strike,
                        "option_type": opt_type,
                        "ltp": self._safe_float(node.get("lastPrice")),
                        "oi": self._safe_int(node.get("openInterest")),
                        "iv": self._safe_float(node.get("impliedVolatility")),
                        "spot": spot,
                    }
                )

        atm_strike: Optional[float] = None
        if strikes and spot > 0:
            arr = np.array(sorted(set(strikes)))
            atm_strike = float(arr[np.abs(arr - spot).argmin()])

        return OptionChainSnapshot(
            symbol=symbol.upper(),
            timestamp=timestamp,
            spot=spot,
            atm_strike=atm_strike,
            expiries=expiries,
            rows=rows,
        )

    # ----------------------------------------------------------------- equity
    def fetch_equity_history(self, symbol: str, from_date: str, to_date: str) -> pd.DataFrame:
        result = self._scraper.fetch_historical(symbol, from_date=from_date, to_date=to_date)
        if not result.ok or result.data is None:
            logger.error("Historical fetch failed for %s: %s", symbol, result.error)
            return pd.DataFrame()
        records = result.data.get("data", []) or []
        rows: List[Dict[str, Any]] = []
        for r in records:
            rows.append(
                {
                    "timestamp": r.get("CH_TIMESTAMP") or r.get("mTIMESTAMP") or r.get("date"),
                    "symbol": symbol.upper(),
                    "open": self._safe_float(r.get("CH_OPENING_PRICE")),
                    "high": self._safe_float(r.get("CH_TRADE_HIGH_PRICE")),
                    "low": self._safe_float(r.get("CH_TRADE_LOW_PRICE")),
                    "close": self._safe_float(r.get("CH_CLOSING_PRICE")),
                    "volume": self._safe_int(r.get("CH_TOT_TRADED_QTY")),
                }
            )
        rows = [r for r in rows if r["timestamp"] and r["close"] is not None]
        if rows:
            self._db.insert_equity_rows(rows)
        return pd.DataFrame(rows)

    # ---------------------------------------------------- synthetic ohlcv (offline)
    @staticmethod
    def synthetic_ohlcv(
        days: int = 365,
        seed: int = 42,
        start_price: float = 18000.0,
        symbol: str = "NIFTY",
        freq: str = "B",
    ) -> pd.DataFrame:
        """Deterministic random-walk OHLCV generator for offline tests/demos."""
        rng = np.random.default_rng(seed)
        idx = pd.date_range(end=pd.Timestamp.today().normalize(), periods=days, freq=freq)
        if len(idx) == 0:
            idx = pd.date_range(end=pd.Timestamp.today().normalize(), periods=days)
        returns = rng.normal(loc=0.0004, scale=0.012, size=len(idx))
        prices = start_price * np.exp(np.cumsum(returns))
        opens = prices * (1 + rng.normal(0, 0.002, size=len(idx)))
        highs = np.maximum(opens, prices) * (1 + np.abs(rng.normal(0, 0.003, size=len(idx))))
        lows = np.minimum(opens, prices) * (1 - np.abs(rng.normal(0, 0.003, size=len(idx))))
        volumes = rng.integers(1_000_000, 5_000_000, size=len(idx))
        df = pd.DataFrame(
            {
                "timestamp": idx,
                "symbol": symbol.upper(),
                "open": opens,
                "high": highs,
                "low": lows,
                "close": prices,
                "volume": volumes,
            }
        )
        return df
