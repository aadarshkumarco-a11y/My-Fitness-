"""RSI-based mean-reversion strategy."""
from __future__ import annotations

import pandas as pd

from ..strategy_engine.base import Strategy
from ..strategy_engine.indicators import rsi_strategy_signals
from ..strategy_engine.registry import register_strategy


@register_strategy("rsi")
class RSIStrategy(Strategy):
    description = "Buy when RSI < oversold, sell when RSI > overbought."

    def __init__(self, period: int = 14, oversold: float = 30.0, overbought: float = 70.0, **kwargs):
        super().__init__(period=period, oversold=oversold, overbought=overbought, **kwargs)

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        return rsi_strategy_signals(
            data,
            period=int(self.params["period"]),
            oversold=float(self.params["oversold"]),
            overbought=float(self.params["overbought"]),
        )
