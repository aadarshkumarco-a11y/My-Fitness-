"""Trend-following on VWAP regime."""
from __future__ import annotations

import pandas as pd

from ..strategy_engine.base import Strategy
from ..strategy_engine.indicators import vwap_strategy_signals
from ..strategy_engine.registry import register_strategy


@register_strategy("vwap")
class VWAPStrategy(Strategy):
    description = "Long when close > VWAP, exit when close drops below."

    def __init__(self, period: int = 20, **kwargs):
        super().__init__(period=period, **kwargs)

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        return vwap_strategy_signals(data, period=int(self.params["period"]))
