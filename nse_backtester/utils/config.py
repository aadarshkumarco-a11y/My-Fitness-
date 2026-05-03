"""Centralised, environment-overridable configuration.

All tunable parameters live here so we never hard-code values in business
logic. Override any value via environment variables prefixed with
``BACKTESTER_``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict


_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class ScraperSettings:
    base_url: str = "https://www.nseindia.com"
    option_chain_url: str = (
        "https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    )
    equity_quote_url: str = (
        "https://www.nseindia.com/api/quote-equity?symbol={symbol}"
    )
    historical_url: str = (
        "https://www.nseindia.com/api/historical/cm/equity"
        "?symbol={symbol}&series=[%22EQ%22]&from={from_date}&to={to_date}"
    )
    min_delay_seconds: float = field(default_factory=lambda: _env_float("BACKTESTER_SCRAPER_MIN_DELAY", 2.0))
    max_delay_seconds: float = field(default_factory=lambda: _env_float("BACKTESTER_SCRAPER_MAX_DELAY", 4.0))
    request_timeout: float = field(default_factory=lambda: _env_float("BACKTESTER_SCRAPER_TIMEOUT", 10.0))
    max_retries: int = field(default_factory=lambda: _env_int("BACKTESTER_SCRAPER_RETRIES", 3))
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )


@dataclass(frozen=True)
class BacktestSettings:
    initial_capital: float = field(default_factory=lambda: _env_float("BACKTESTER_CAPITAL", 1_000_000.0))
    brokerage_per_trade: float = field(default_factory=lambda: _env_float("BACKTESTER_BROKERAGE", 20.0))
    slippage_pct: float = field(default_factory=lambda: _env_float("BACKTESTER_SLIPPAGE_PCT", 0.005))
    risk_free_rate: float = field(default_factory=lambda: _env_float("BACKTESTER_RISK_FREE_RATE", 0.065))
    default_lot_size: int = field(default_factory=lambda: _env_int("BACKTESTER_LOT_SIZE", 50))


@dataclass(frozen=True)
class StorageSettings:
    db_path: Path = _PROJECT_ROOT / "data" / "market_data.db"
    cache_dir: Path = _PROJECT_ROOT / "data" / "cache"
    export_dir: Path = _PROJECT_ROOT / "data" / "exports"


@dataclass(frozen=True)
class Settings:
    scraper: ScraperSettings = field(default_factory=ScraperSettings)
    backtest: BacktestSettings = field(default_factory=BacktestSettings)
    storage: StorageSettings = field(default_factory=StorageSettings)
    nse_lot_sizes: Dict[str, int] = field(
        default_factory=lambda: {"NIFTY": 50, "BANKNIFTY": 15, "FINNIFTY": 40}
    )

    def ensure_dirs(self) -> None:
        self.storage.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.storage.cache_dir.mkdir(parents=True, exist_ok=True)
        self.storage.export_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
