"""Centralized logging configuration.

All modules should obtain a logger via ``get_logger(__name__)``. Logs are
written both to stdout and to a rotating file under ``logs/``.
"""
from __future__ import annotations

import logging
import logging.handlers
import os
from pathlib import Path

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_LOG_DIR = _PROJECT_ROOT / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = _LOG_DIR / "backtester.log"

_configured = False


def _configure_root() -> None:
    global _configured
    if _configured:
        return

    level_name = os.getenv("BACKTESTER_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    root = logging.getLogger("nse_backtester")
    root.setLevel(level)
    root.propagate = False

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)

    file_handler = logging.handlers.RotatingFileHandler(
        _LOG_FILE,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    if not root.handlers:
        root.addHandler(stream_handler)
        root.addHandler(file_handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a project logger, configuring the root logger on first call."""
    _configure_root()
    if not name.startswith("nse_backtester"):
        name = f"nse_backtester.{name}"
    return logging.getLogger(name)
