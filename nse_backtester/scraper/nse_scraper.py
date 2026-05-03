"""Hardened NSE scraper.

NSE's public endpoints aggressively block clients that don't behave like a
browser. This module wraps ``requests.Session`` with:

* a warm-up GET against the homepage to acquire cookies
* realistic browser headers (User-Agent, Accept-Language, Referer)
* random 2-4s jitter between requests
* automatic session refresh when 401/403/Cloudflare-style responses appear
* exponential-with-jitter retry up to ``max_retries`` attempts
* a transparent on-disk JSON cache so we can replay scrapes offline

The scraper is intentionally side-effect-light: it never raises on a single
failed request — callers receive a typed result and can decide how to fall
back (e.g. to Black-Scholes pricing).
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from ..utils import get_logger, settings

logger = get_logger(__name__)


class ScraperError(RuntimeError):
    """Raised when the scraper fully exhausts retries."""


@dataclass
class ScrapeResult:
    ok: bool
    data: Optional[Dict[str, Any]]
    status_code: Optional[int]
    error: Optional[str] = None
    cached: bool = False

    @classmethod
    def success(cls, data: Dict[str, Any], status: int, cached: bool = False) -> "ScrapeResult":
        return cls(ok=True, data=data, status_code=status, cached=cached)

    @classmethod
    def failure(cls, error: str, status: Optional[int] = None) -> "ScrapeResult":
        return cls(ok=False, data=None, status_code=status, error=error)


class NSEScraper:
    """Threadsafe-ish wrapper around a single ``requests.Session``."""

    def __init__(self, cache_enabled: bool = True) -> None:
        self._cfg = settings.scraper
        self._cache_dir: Path = settings.storage.cache_dir
        self._cache_enabled = cache_enabled
        self._session: Optional[requests.Session] = None
        self._last_warmup_ts: float = 0.0
        self._warmup_ttl_seconds: float = 600.0

    # ------------------------------------------------------------------ session
    def _build_headers(self, referer: Optional[str] = None) -> Dict[str, str]:
        headers = {
            "User-Agent": self._cfg.user_agent,
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Referer": referer or self._cfg.base_url + "/",
            "X-Requested-With": "XMLHttpRequest",
        }
        return headers

    def _new_session(self) -> requests.Session:
        session = requests.Session()
        session.headers.update(self._build_headers())
        return session

    def _warmup(self, session: requests.Session, force: bool = False) -> bool:
        now = time.time()
        if not force and (now - self._last_warmup_ts) < self._warmup_ttl_seconds:
            return True
        try:
            logger.debug("Warming up NSE session via %s", self._cfg.base_url)
            resp = session.get(self._cfg.base_url + "/", timeout=self._cfg.request_timeout)
            if resp.status_code == 200:
                self._last_warmup_ts = now
                return True
            logger.warning("Warm-up returned status %s", resp.status_code)
            return False
        except requests.RequestException as exc:
            logger.warning("Warm-up failed: %s", exc)
            return False

    def _ensure_session(self, force_refresh: bool = False) -> requests.Session:
        if self._session is None or force_refresh:
            self._session = self._new_session()
            self._warmup(self._session, force=True)
        else:
            self._warmup(self._session)
        return self._session

    # ------------------------------------------------------------------- cache
    def _cache_path(self, key: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in key)
        return self._cache_dir / f"{safe}.json"

    def _read_cache(self, key: str, max_age_seconds: Optional[float]) -> Optional[Dict[str, Any]]:
        if not self._cache_enabled:
            return None
        path = self._cache_path(key)
        if not path.exists():
            return None
        if max_age_seconds is not None:
            age = time.time() - path.stat().st_mtime
            if age > max_age_seconds:
                return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed reading cache %s: %s", path, exc)
            return None

    def _write_cache(self, key: str, payload: Dict[str, Any]) -> None:
        if not self._cache_enabled:
            return
        path = self._cache_path(key)
        try:
            path.write_text(json.dumps(payload), encoding="utf-8")
        except OSError as exc:
            logger.warning("Failed writing cache %s: %s", path, exc)

    # ---------------------------------------------------------------- requests
    def _sleep_jitter(self) -> None:
        delay = random.uniform(self._cfg.min_delay_seconds, self._cfg.max_delay_seconds)
        time.sleep(delay)

    def _get(self, url: str, referer: Optional[str] = None) -> ScrapeResult:
        last_status: Optional[int] = None
        last_error: Optional[str] = None

        for attempt in range(1, self._cfg.max_retries + 1):
            session = self._ensure_session(force_refresh=(attempt > 1 and last_status in (401, 403)))
            try:
                resp = session.get(
                    url,
                    headers=self._build_headers(referer=referer),
                    timeout=self._cfg.request_timeout,
                )
                last_status = resp.status_code
                if resp.status_code == 200:
                    try:
                        return ScrapeResult.success(resp.json(), resp.status_code)
                    except ValueError as exc:
                        last_error = f"non-JSON response: {exc}"
                        logger.warning("Attempt %d: %s", attempt, last_error)
                elif resp.status_code in (401, 403):
                    last_error = f"blocked by NSE ({resp.status_code})"
                    logger.warning("Attempt %d: %s — refreshing session", attempt, last_error)
                    self._session = None
                else:
                    last_error = f"status={resp.status_code}"
                    logger.warning("Attempt %d for %s: %s", attempt, url, last_error)
            except requests.Timeout as exc:
                last_error = f"timeout: {exc}"
                logger.warning("Attempt %d timed out: %s", attempt, exc)
            except requests.RequestException as exc:
                last_error = f"request error: {exc}"
                logger.warning("Attempt %d failed: %s", attempt, exc)

            if attempt < self._cfg.max_retries:
                self._sleep_jitter()

        return ScrapeResult.failure(last_error or "unknown error", last_status)

    # ------------------------------------------------------------------ public
    def fetch_option_chain(
        self,
        symbol: str = "NIFTY",
        cache_max_age: Optional[float] = 60.0,
    ) -> ScrapeResult:
        """Fetch the option chain for an index symbol like ``NIFTY``."""
        symbol = symbol.upper().strip()
        cache_key = f"option_chain_{symbol}"
        cached = self._read_cache(cache_key, cache_max_age)
        if cached is not None:
            logger.info("Returning option chain for %s from cache", symbol)
            return ScrapeResult.success(cached, 200, cached=True)

        url = self._cfg.option_chain_url.format(symbol=symbol)
        referer = f"{self._cfg.base_url}/option-chain"
        result = self._get(url, referer=referer)
        if result.ok and result.data is not None:
            self._write_cache(cache_key, result.data)
        return result

    def fetch_equity_quote(
        self,
        symbol: str,
        cache_max_age: Optional[float] = 60.0,
    ) -> ScrapeResult:
        symbol = symbol.upper().strip()
        cache_key = f"equity_{symbol}"
        cached = self._read_cache(cache_key, cache_max_age)
        if cached is not None:
            logger.info("Returning equity quote for %s from cache", symbol)
            return ScrapeResult.success(cached, 200, cached=True)

        url = self._cfg.equity_quote_url.format(symbol=symbol)
        referer = f"{self._cfg.base_url}/get-quotes/equity?symbol={symbol}"
        result = self._get(url, referer=referer)
        if result.ok and result.data is not None:
            self._write_cache(cache_key, result.data)
        return result

    def fetch_historical(
        self,
        symbol: str,
        from_date: str,
        to_date: str,
        cache_max_age: Optional[float] = 24 * 3600.0,
    ) -> ScrapeResult:
        """Fetch historical EQ candles for a stock between two ``DD-MM-YYYY`` dates."""
        symbol = symbol.upper().strip()
        cache_key = f"hist_{symbol}_{from_date}_{to_date}"
        cached = self._read_cache(cache_key, cache_max_age)
        if cached is not None:
            return ScrapeResult.success(cached, 200, cached=True)

        url = self._cfg.historical_url.format(
            symbol=symbol, from_date=from_date, to_date=to_date
        )
        referer = f"{self._cfg.base_url}/get-quotes/equity?symbol={symbol}"
        result = self._get(url, referer=referer)
        if result.ok and result.data is not None:
            self._write_cache(cache_key, result.data)
        return result

    def close(self) -> None:
        if self._session is not None:
            try:
                self._session.close()
            except Exception:  # pragma: no cover
                pass
            self._session = None

    def __enter__(self) -> "NSEScraper":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
