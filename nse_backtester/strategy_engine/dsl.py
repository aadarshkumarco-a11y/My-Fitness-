"""Rule-based DSL strategy.

Grammar (one rule per line, ``#`` starts a comment):

    BUY  WHEN <expr>
    SELL WHEN <expr>
    EXIT WHEN <expr>
    STOP_LOSS  <pct>%
    TAKE_PROFIT <pct>%

Atoms supported in ``<expr>``:

* OHLCV columns: ``close``, ``open``, ``high``, ``low``, ``volume``
* Indicators:    ``RSI(period)``, ``EMA(period)``, ``SMA(period)``, ``VWAP()``
* Numbers:       ``30``, ``70.5``
* Comparisons:   ``<``, ``>``, ``<=``, ``>=``, ``==``, ``!=``
* Logical:       ``AND``, ``OR``, ``NOT``
* Cross helpers: ``A cross_above B``, ``A cross_below B``

Example::

    # Mean-reversion + trend filter
    BUY  WHEN RSI(14) < 30 AND close > EMA(200)
    SELL WHEN RSI(14) > 70
    STOP_LOSS  2%
    TAKE_PROFIT 5%
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..utils import get_logger
from .base import Strategy
from .indicators import _ema, _rsi, _vwap

logger = get_logger(__name__)


_INDICATOR_PATTERN = re.compile(r"\b(RSI|EMA|SMA|VWAP)\s*\(\s*([^)]*)\s*\)", re.IGNORECASE)


def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(int(period)).mean()


def _cross_above(a: pd.Series, b: pd.Series) -> pd.Series:
    a, b = pd.Series(a), pd.Series(b)
    a, b = a.align(b, join="inner")
    return (a.shift(1) <= b.shift(1)) & (a > b)


def _cross_below(a: pd.Series, b: pd.Series) -> pd.Series:
    a, b = pd.Series(a), pd.Series(b)
    a, b = a.align(b, join="inner")
    return (a.shift(1) >= b.shift(1)) & (a < b)


@dataclass
class _Rule:
    action: str  # BUY / SELL / EXIT
    raw: str


class DSLParseError(ValueError):
    pass


class DSLStrategy(Strategy):
    """Strategy compiled from a small declarative DSL."""

    name = "dsl"
    description = "User-defined strategy via the BUY/SELL WHEN ... DSL."

    def __init__(self, dsl_text: str, **kwargs):
        super().__init__(**kwargs)
        self.dsl_text = dsl_text or ""
        self.rules: List[_Rule] = []
        self._parse()

    # ----------------------------------------------------------------- parsing
    def _parse(self) -> None:
        for raw in self.dsl_text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            up = line.upper()
            if up.startswith("STOP_LOSS"):
                pct = self._parse_pct(line.split(None, 1)[1])
                if pct is not None:
                    self.params["stop_loss_pct"] = pct
                continue
            if up.startswith("TAKE_PROFIT"):
                pct = self._parse_pct(line.split(None, 1)[1])
                if pct is not None:
                    self.params["take_profit_pct"] = pct
                continue

            m = re.match(r"(BUY|SELL|EXIT)\s+WHEN\s+(.+)$", line, re.IGNORECASE)
            if not m:
                raise DSLParseError(f"Could not parse line: {raw!r}")
            self.rules.append(_Rule(action=m.group(1).upper(), raw=m.group(2).strip()))

        if not self.rules:
            raise DSLParseError("No rules parsed. Need at least one BUY/SELL/EXIT WHEN ... line.")

    @staticmethod
    def _parse_pct(text: str) -> Optional[float]:
        text = text.strip().rstrip("%")
        try:
            return float(text) / 100.0
        except ValueError:
            return None

    # --------------------------------------------------------------- evaluation
    def _resolve_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Find every indicator reference in the DSL and pre-compute it."""
        ns: Dict[str, pd.Series] = {
            "close": data["close"],
            "open": data.get("open", data["close"]),
            "high": data.get("high", data["close"]),
            "low": data.get("low", data["close"]),
            "volume": data.get("volume", pd.Series(np.zeros(len(data)), index=data.index)),
        }
        for rule in self.rules:
            for fn, args in _INDICATOR_PATTERN.findall(rule.raw):
                key = self._ind_key(fn, args)
                if key in ns:
                    continue
                ns[key] = self._compute_indicator(fn, args, data)
        return ns

    @staticmethod
    def _ind_key(fn: str, args: str) -> str:
        return f"_ind_{fn.lower()}_{re.sub(r'[^A-Za-z0-9]+', '_', args.strip())}"

    def _compute_indicator(self, fn: str, args: str, data: pd.DataFrame) -> pd.Series:
        fn_u = fn.upper()
        if fn_u == "VWAP":
            return _vwap(data)
        try:
            period = int(float(args.strip()))
        except (TypeError, ValueError):
            raise DSLParseError(f"Bad period for {fn}: {args!r}")
        if fn_u == "RSI":
            return _rsi(data["close"], period=period)
        if fn_u == "EMA":
            return _ema(data["close"], period=period)
        if fn_u == "SMA":
            return _sma(data["close"], period=period)
        raise DSLParseError(f"Unknown indicator: {fn}")

    def _expr_to_python(self, expr: str) -> str:
        """Translate a DSL expression into a pandas-compatible Python expression.

        Handles operator-precedence carefully: pandas overloads ``&``/``|``
        with higher precedence than the comparison operators, so we wrap each
        comparison sub-expression in explicit parentheses before swapping
        ``AND``/``OR`` for the bitwise equivalents.
        """
        out = expr

        # Replace indicator calls with pre-computed namespace keys.
        out = _INDICATOR_PATTERN.sub(
            lambda m: self._ind_key(m.group(1), m.group(2)), out
        )

        # cross_above / cross_below — split lhs/rhs around the keyword.
        for keyword, fn in (("cross_above", "_cross_above"), ("cross_below", "_cross_below")):
            pattern = re.compile(rf"(.+?)\s+{keyword}\s+(.+)", re.IGNORECASE)
            m = pattern.search(out)
            if m:
                lhs, rhs = m.group(1).strip(), m.group(2).strip()
                out = f"{fn}({lhs}, {rhs})"

        # Split around logical keywords so each comparison can be parenthesized.
        parts = re.split(r"\b(AND|OR|NOT)\b", out, flags=re.IGNORECASE)
        pieces: List[str] = []
        for piece in parts:
            stripped = piece.strip()
            if not stripped:
                continue
            upper = stripped.upper()
            if upper == "AND":
                pieces.append("&")
            elif upper == "OR":
                pieces.append("|")
            elif upper == "NOT":
                pieces.append("~")
            else:
                pieces.append(f"({stripped})")

        return " ".join(pieces)

    def _evaluate_rule(self, expr: str, ns: Dict[str, pd.Series], data: pd.DataFrame) -> pd.Series:
        py_expr = self._expr_to_python(expr)
        eval_globals = {
            "__builtins__": {},
            "_cross_above": _cross_above,
            "_cross_below": _cross_below,
        }
        eval_locals = dict(ns)
        try:
            result = eval(py_expr, eval_globals, eval_locals)  # noqa: S307
        except Exception as exc:
            raise DSLParseError(f"Failed to evaluate {expr!r}: {exc}") from exc
        if isinstance(result, bool):
            result = pd.Series([result] * len(data), index=data.index)
        if isinstance(result, np.ndarray):
            result = pd.Series(result, index=data.index)
        if not isinstance(result, pd.Series):
            raise DSLParseError(f"Expression {expr!r} did not produce a boolean Series")
        return result.fillna(False).astype(bool)

    # ----------------------------------------------------------------- public
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        ns = self._resolve_indicators(data)
        signal = pd.Series(["HOLD"] * len(data), index=data.index)
        # Process EXIT first (highest precedence), then SELL, then BUY so that
        # an EXIT on the same bar wins.
        order = {"EXIT": 0, "SELL": 1, "BUY": 2}
        for rule in sorted(self.rules, key=lambda r: order.get(r.action, 99)):
            mask = self._evaluate_rule(rule.raw, ns, data)
            signal = signal.where(~mask, rule.action)
        out = data.copy()
        out["signal"] = signal.values
        return out
