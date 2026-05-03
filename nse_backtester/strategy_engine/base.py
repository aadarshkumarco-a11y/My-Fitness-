"""Base strategy contract and signal types."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

import pandas as pd


class SignalType(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    EXIT = "EXIT"


@dataclass
class Signal:
    timestamp: pd.Timestamp
    signal: SignalType
    price: float
    confidence: float = 1.0
    metadata: Dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


class Strategy:
    """Subclass and implement ``generate_signals``.

    The contract is strict: ``generate_signals(data)`` must return a DataFrame
    indexed by timestamp with at minimum a ``signal`` column whose values are
    ``SignalType`` members (or their string equivalents).

    Strategies that need richer execution semantics than the standard equity
    or options engines provide (e.g. multi-leg short option spreads, custom
    rebalancing logic) can opt into self-contained backtesting by setting
    ``supports_self_backtest = True`` and overriding ``run_full_backtest``.
    The dashboard and CLI dispatch to that method instead of the generic
    engine when the flag is set.
    """

    name: str = "base"
    description: str = ""
    supports_self_backtest: bool = False

    def __init__(self, **params: Any) -> None:
        self.params: Dict[str, Any] = params

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("generate_signals must be implemented")

    def run_full_backtest(
        self,
        data: pd.DataFrame,
        initial_capital: float,
        symbol: str = "SYMBOL",
        **kwargs: Any,
    ):
        """Optional: run the entire backtest internally and return a
        :class:`nse_backtester.backtesting.engine.BacktestResult`.

        Override this when ``supports_self_backtest = True``. The default
        raises :class:`NotImplementedError`.
        """
        raise NotImplementedError(
            "run_full_backtest is only implemented by strategies that set "
            "supports_self_backtest = True"
        )

    # Optional helpers that subclasses can override --------------------------
    def position_size(
        self,
        capital: float,
        price: float,
        signal: SignalType,
    ) -> float:
        if price <= 0:
            return 0.0
        # default: invest 100% of available capital, integer quantity
        return float(int(capital // price))

    def stop_loss(self, entry_price: float, signal: SignalType) -> Optional[float]:
        sl_pct = self.params.get("stop_loss_pct")
        if sl_pct is None:
            return None
        if signal == SignalType.BUY:
            return entry_price * (1 - sl_pct)
        if signal == SignalType.SELL:
            return entry_price * (1 + sl_pct)
        return None

    def take_profit(self, entry_price: float, signal: SignalType) -> Optional[float]:
        tp_pct = self.params.get("take_profit_pct")
        if tp_pct is None:
            return None
        if signal == SignalType.BUY:
            return entry_price * (1 + tp_pct)
        if signal == SignalType.SELL:
            return entry_price * (1 - tp_pct)
        return None
