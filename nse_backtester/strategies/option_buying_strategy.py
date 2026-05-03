"""Directional weekly ATM option buying driven by an EMA crossover."""
from __future__ import annotations

import pandas as pd

from ..strategy_engine.base import Strategy
from ..strategy_engine.indicators import ema_crossover_signals
from ..strategy_engine.registry import register_strategy


@register_strategy("option_buying")
class OptionBuyingStrategy(Strategy):
    description = (
        "Buy weekly ATM CE on bullish EMA crossover, ATM PE on bearish crossover. "
        "Use this with OptionsBacktestEngine."
    )

    def __init__(self, fast: int = 9, slow: int = 21, **kwargs):
        super().__init__(fast=fast, slow=slow, **kwargs)

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        return ema_crossover_signals(
            data,
            fast=int(self.params["fast"]),
            slow=int(self.params["slow"]),
        )
