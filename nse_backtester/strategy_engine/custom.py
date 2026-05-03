"""Load a user-supplied Python ``Strategy`` subclass from a string of code.

⚠️ **Security note:** ``exec`` runs arbitrary user code. This is fine for
local research where the user supplies their own code, but DO NOT expose
this loader on a multi-tenant deployment without a sandbox.
"""
from __future__ import annotations

import inspect
import textwrap
from typing import Any, Dict, Type

from .base import Strategy


class CustomStrategyError(ValueError):
    pass


_DEFAULT_TEMPLATE = '''\
"""Edit this template — must define a class extending Strategy."""
from nse_backtester.strategy_engine.base import Strategy
from nse_backtester.strategy_engine.indicators import rsi_strategy_signals

class MyStrategy(Strategy):
    def generate_signals(self, data):
        return rsi_strategy_signals(data, period=14, oversold=30, overbought=70)
'''


def default_template() -> str:
    return _DEFAULT_TEMPLATE


def load_custom_strategy(code: str) -> Type[Strategy]:
    """Execute user code and return the first concrete ``Strategy`` subclass."""
    if not code or not code.strip():
        raise CustomStrategyError("Custom code is empty.")

    src = textwrap.dedent(code)
    namespace: Dict[str, Any] = {"__name__": "user_strategy"}

    try:
        compiled = compile(src, "<user_strategy>", "exec")
    except SyntaxError as exc:
        raise CustomStrategyError(f"Syntax error: {exc.msg} (line {exc.lineno})") from exc

    try:
        exec(compiled, namespace)  # noqa: S102 — intentional, user-controlled
    except Exception as exc:
        raise CustomStrategyError(f"Error while loading strategy: {exc}") from exc

    candidates = [
        obj for obj in namespace.values()
        if inspect.isclass(obj) and issubclass(obj, Strategy) and obj is not Strategy
    ]
    if not candidates:
        raise CustomStrategyError(
            "No Strategy subclass found. Define a class that extends "
            "`nse_backtester.strategy_engine.base.Strategy`."
        )
    if len(candidates) > 1:
        # Prefer the one explicitly defined in the user file (not imported).
        defined_here = [c for c in candidates if getattr(c, "__module__", "") == "user_strategy"]
        if defined_here:
            return defined_here[0]
    return candidates[0]
