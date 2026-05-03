"""Scraper resilience tests using mocked HTTP responses."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import requests

from nse_backtester.scraper import NSEScraper, ScraperError  # noqa: F401
from nse_backtester.scraper.nse_scraper import ScrapeResult


def _ok_response(payload):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = payload
    return resp


def _err_response(status):
    resp = MagicMock()
    resp.status_code = status
    resp.json.side_effect = ValueError("not json")
    return resp


def test_fetch_option_chain_success(tmp_path, monkeypatch):
    scraper = NSEScraper(cache_enabled=False)
    payload = {"records": {"underlyingValue": 18000, "expiryDates": ["29-May-2024"], "data": []}}

    with patch.object(requests.Session, "get") as mock_get:
        mock_get.side_effect = [_ok_response({}), _ok_response(payload)]
        result = scraper.fetch_option_chain("NIFTY")
    assert result.ok
    assert result.data["records"]["underlyingValue"] == 18000


def test_fetch_option_chain_blocked_then_recover():
    scraper = NSEScraper(cache_enabled=False)
    good = {"records": {"underlyingValue": 17000, "expiryDates": [], "data": []}}

    responses = [
        _ok_response({}),  # warmup
        _err_response(403),  # first attempt blocked
        _ok_response({}),  # warmup after refresh
        _ok_response(good),  # second attempt succeeds
    ]
    with patch.object(requests.Session, "get") as mock_get:
        mock_get.side_effect = responses
        scraper._cfg = scraper._cfg.__class__(min_delay_seconds=0, max_delay_seconds=0, max_retries=3)
        result = scraper.fetch_option_chain("NIFTY")
    assert result.ok
    assert result.data["records"]["underlyingValue"] == 17000


def test_fetch_option_chain_all_retries_fail():
    scraper = NSEScraper(cache_enabled=False)
    scraper._cfg = scraper._cfg.__class__(min_delay_seconds=0, max_delay_seconds=0, max_retries=2)
    with patch.object(requests.Session, "get") as mock_get:
        mock_get.return_value = _err_response(500)
        result = scraper.fetch_option_chain("NIFTY")
    assert not result.ok
    assert result.error is not None


def test_timeout_handling():
    scraper = NSEScraper(cache_enabled=False)
    scraper._cfg = scraper._cfg.__class__(min_delay_seconds=0, max_delay_seconds=0, max_retries=2)
    with patch.object(requests.Session, "get", side_effect=requests.Timeout("boom")) as _:
        result = scraper.fetch_option_chain("NIFTY")
    assert not result.ok
    assert "timeout" in (result.error or "").lower()
