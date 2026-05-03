"""Translate a small subset of Pine Script v4/v5 to our DSL.

We do **not** attempt to support all of Pine Script. The supported subset is:

* ``//@version=N`` and ``strategy("...")``/``indicator("...")`` headers
* Variable assignments of indicator outputs:
    ``rsiVal = ta.rsi(close, 14)``
    ``fast   = ta.ema(close, 9)``
* Indicator helpers: ``ta.rsi``, ``ta.ema``, ``ta.sma``, ``ta.vwap``,
  ``ta.crossover``, ``ta.crossunder``
* Single-line ``if <cond>`` followed by an indented ``strategy.entry(...)``
  or ``strategy.close(...)``/``strategy.exit(...)``

Anything else is ignored with a warning. The output is the canonical DSL
string that ``DSLStrategy`` understands. This is enough to express the bulk
of beginner / intermediate Pine strategies.
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

from ..utils import get_logger

logger = get_logger(__name__)


_TA_RSI = re.compile(r"ta\.rsi\s*\(\s*close\s*,\s*([0-9]+)\s*\)", re.IGNORECASE)
_TA_EMA = re.compile(r"ta\.ema\s*\(\s*close\s*,\s*([0-9]+)\s*\)", re.IGNORECASE)
_TA_SMA = re.compile(r"ta\.sma\s*\(\s*close\s*,\s*([0-9]+)\s*\)", re.IGNORECASE)
_TA_VWAP = re.compile(r"ta\.vwap\b", re.IGNORECASE)
_CROSSOVER = re.compile(r"ta\.crossover\s*\(\s*([^,]+?)\s*,\s*([^)]+?)\s*\)", re.IGNORECASE)
_CROSSUNDER = re.compile(r"ta\.crossunder\s*\(\s*([^,]+?)\s*,\s*([^)]+?)\s*\)", re.IGNORECASE)

_ASSIGN = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)$")
_IF_LINE = re.compile(r"^\s*if\s+(.+?)\s*$", re.IGNORECASE)
_ENTRY = re.compile(r"strategy\.(entry|long)\s*\(", re.IGNORECASE)
_LONG_DIR = re.compile(r"strategy\.long\b", re.IGNORECASE)
_SHORT_DIR = re.compile(r"strategy\.short\b", re.IGNORECASE)
_CLOSE_OR_EXIT = re.compile(r"strategy\.(close|exit)\s*\(", re.IGNORECASE)


def _replace_ta(expr: str) -> str:
    expr = _TA_RSI.sub(lambda m: f"RSI({m.group(1)})", expr)
    expr = _TA_EMA.sub(lambda m: f"EMA({m.group(1)})", expr)
    expr = _TA_SMA.sub(lambda m: f"SMA({m.group(1)})", expr)
    expr = _TA_VWAP.sub("VWAP()", expr)
    expr = _CROSSOVER.sub(lambda m: f"({m.group(1).strip()}) cross_above ({m.group(2).strip()})", expr)
    expr = _CROSSUNDER.sub(lambda m: f"({m.group(1).strip()}) cross_below ({m.group(2).strip()})", expr)
    expr = re.sub(r"\band\b", "AND", expr, flags=re.IGNORECASE)
    expr = re.sub(r"\bor\b", "OR", expr, flags=re.IGNORECASE)
    expr = re.sub(r"\bnot\b", "NOT", expr, flags=re.IGNORECASE)
    return expr.strip()


def _substitute_vars(expr: str, env: Dict[str, str]) -> str:
    out = expr
    # Substitute longest names first so prefixes don't get clobbered.
    for var in sorted(env.keys(), key=len, reverse=True):
        out = re.sub(rf"\b{re.escape(var)}\b", f"({env[var]})", out)
    return out


class PineParseError(ValueError):
    pass


def pine_to_dsl(pine_text: str) -> str:
    """Translate Pine source to our DSL. Returns the DSL string."""
    lines = pine_text.replace("\r\n", "\n").split("\n")
    env: Dict[str, str] = {}  # variable name → DSL expression
    rules: List[Tuple[str, str]] = []  # (action, expr)

    i = 0
    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()
        i += 1

        if not stripped or stripped.startswith("//"):
            continue
        if stripped.lower().startswith(("strategy(", "indicator(", "//@version")):
            continue

        # Variable assignment
        if "=" in stripped and not stripped.startswith("if") and "==" not in stripped.split("//")[0]:
            m = _ASSIGN.match(stripped)
            if m and not m.group(2).strip().startswith("strategy."):
                name = m.group(1)
                expr = _substitute_vars(_replace_ta(m.group(2).strip()), env)
                env[name] = expr
                continue

        # if <cond>
        m = _IF_LINE.match(stripped)
        if m:
            cond = _substitute_vars(_replace_ta(m.group(1)), env)
            # Look ahead at indented body lines for entry/close calls.
            actions: List[str] = []
            while i < len(lines):
                body = lines[i]
                if not body.strip():
                    i += 1
                    continue
                if not (body.startswith(" ") or body.startswith("\t")):
                    break
                actions.append(body.strip())
                i += 1
            for act in actions:
                if _CLOSE_OR_EXIT.search(act):
                    rules.append(("EXIT", cond))
                elif _ENTRY.search(act) or "strategy.long" in act.lower():
                    if _SHORT_DIR.search(act):
                        rules.append(("SELL", cond))
                    else:
                        rules.append(("BUY", cond))
            continue

        logger.debug("pine_to_dsl ignored unsupported line: %s", stripped)

    if not rules:
        raise PineParseError(
            "No BUY/SELL/EXIT rules detected. Did your Pine script use "
            "`if <cond>` blocks with `strategy.entry/close/exit`?"
        )

    out_lines = [f"{action} WHEN {expr}" for action, expr in rules]
    return "\n".join(out_lines)
