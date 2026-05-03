"""Event-driven backtesting engine for equities and options.

Design notes
------------
* The engine consumes a single OHLCV-style ``DataFrame`` and a ``Strategy``.
* Each row is treated as an event. Signals are evaluated against the *close*
  of that bar; fills assume the close ± slippage.
* For options we use ``OptionsBacktestEngine`` which simulates ATM weekly
  expiry buying with optional Black-Scholes time-decay handling when an
  external price series is unavailable.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..strategy_engine.base import Signal, SignalType, Strategy
from ..utils import get_logger, settings

logger = get_logger(__name__)


@dataclass
class Trade:
    timestamp: pd.Timestamp
    symbol: str
    side: str  # BUY / SELL
    quantity: float
    price: float
    pnl: float = 0.0
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat() if hasattr(self.timestamp, "isoformat") else str(self.timestamp),
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "price": self.price,
            "pnl": self.pnl,
            "notes": self.notes,
        }


@dataclass
class Position:
    symbol: str
    quantity: float = 0.0
    avg_price: float = 0.0
    direction: int = 0  # +1 long, -1 short, 0 flat
    entry_time: Optional[pd.Timestamp] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    def is_flat(self) -> bool:
        return self.quantity == 0 or self.direction == 0


@dataclass
class BacktestResult:
    run_id: str
    initial_capital: float
    final_equity: float
    trades: List[Trade]
    equity_curve: pd.DataFrame
    drawdown_curve: pd.DataFrame
    metrics: Dict[str, float] = field(default_factory=dict)
    signals: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def trade_log_df(self) -> pd.DataFrame:
        return pd.DataFrame([t.to_dict() for t in self.trades])


class BacktestEngine:
    """Vectorised single-symbol equity backtester.

    Parameters
    ----------
    strategy: Strategy
    initial_capital: float
    brokerage_per_trade: float
    slippage_pct: float
        Applied as a multiplicative cost on every fill.
    allow_short: bool
        When False, ``SELL`` only flattens an existing long.
    """

    def __init__(
        self,
        strategy: Strategy,
        initial_capital: Optional[float] = None,
        brokerage_per_trade: Optional[float] = None,
        slippage_pct: Optional[float] = None,
        allow_short: bool = False,
    ) -> None:
        cfg = settings.backtest
        self.strategy = strategy
        self.initial_capital = float(initial_capital if initial_capital is not None else cfg.initial_capital)
        self.brokerage_per_trade = float(
            brokerage_per_trade if brokerage_per_trade is not None else cfg.brokerage_per_trade
        )
        self.slippage_pct = float(slippage_pct if slippage_pct is not None else cfg.slippage_pct)
        self.allow_short = allow_short

    # ------------------------------------------------------------- helpers
    def _apply_slippage(self, price: float, side: str) -> float:
        if side.upper() == "BUY":
            return price * (1 + self.slippage_pct)
        return price * (1 - self.slippage_pct)

    def _coerce_signals(self, signals_df: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        if "signal" not in signals_df.columns:
            raise ValueError("Strategy must produce a 'signal' column")
        df = signals_df.copy()
        if "close" not in df.columns and "close" in data.columns:
            df["close"] = data["close"].values
        if "timestamp" not in df.columns:
            df = df.reset_index().rename(columns={"index": "timestamp"})
            if "timestamp" in data.columns and len(df) == len(data):
                df["timestamp"] = data["timestamp"].values
        df["signal"] = df["signal"].astype(str).str.upper()
        return df

    # ------------------------------------------------------------- main loop
    def run(self, data: pd.DataFrame, symbol: str = "SYMBOL") -> BacktestResult:
        if data is None or data.empty:
            raise ValueError("Backtest data is empty")
        required = {"close"}
        if not required.issubset(data.columns):
            raise ValueError(f"Data missing columns: {required - set(data.columns)}")

        signals_df = self._coerce_signals(self.strategy.generate_signals(data), data)

        run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"
        cash = self.initial_capital
        position = Position(symbol=symbol)
        trades: List[Trade] = []
        equity_history: List[Dict[str, Any]] = []

        for idx, row in signals_df.iterrows():
            ts = row.get("timestamp", idx)
            close = float(row["close"]) if not pd.isna(row["close"]) else None
            if close is None:
                continue
            sig = SignalType(row["signal"]) if row["signal"] in SignalType._value2member_map_ else SignalType.HOLD

            cash, position = self._maybe_apply_stops(cash, position, close, ts, trades, symbol)

            if sig == SignalType.BUY:
                cash, position = self._handle_buy(cash, position, close, ts, trades, symbol)
            elif sig == SignalType.SELL:
                cash, position = self._handle_sell(cash, position, close, ts, trades, symbol)
            elif sig == SignalType.EXIT and not position.is_flat():
                cash, position = self._close_position(cash, position, close, ts, trades, symbol, note="EXIT")

            equity = cash + position.quantity * close * position.direction
            if position.direction == 0:
                equity = cash
            equity_history.append({"timestamp": ts, "equity": equity, "cash": cash, "close": close})

        # Force-close any open position at the last close.
        if not position.is_flat():
            last_row = signals_df.iloc[-1]
            cash, position = self._close_position(
                cash, position, float(last_row["close"]), last_row.get("timestamp", signals_df.index[-1]), trades, symbol, note="EOD"
            )
            if equity_history:
                equity_history[-1]["equity"] = cash
                equity_history[-1]["cash"] = cash

        eq_df = pd.DataFrame(equity_history)
        if not eq_df.empty:
            eq_df["timestamp"] = pd.to_datetime(eq_df["timestamp"], errors="coerce")
            eq_df = eq_df.set_index("timestamp")
            running_max = eq_df["equity"].cummax()
            dd = (eq_df["equity"] - running_max) / running_max
            drawdown = dd.to_frame(name="drawdown")
        else:
            drawdown = pd.DataFrame(columns=["drawdown"])

        return BacktestResult(
            run_id=run_id,
            initial_capital=self.initial_capital,
            final_equity=float(cash),
            trades=trades,
            equity_curve=eq_df,
            drawdown_curve=drawdown,
            metrics={},
            signals=signals_df,
        )

    # ------------------------------------------------------------- order ops
    def _handle_buy(
        self, cash: float, position: Position, close: float,
        ts: pd.Timestamp, trades: List[Trade], symbol: str,
    ):
        if position.direction == -1:
            cash, position = self._close_position(cash, position, close, ts, trades, symbol, note="cover-short")
        if position.direction == 1:
            return cash, position  # already long, hold
        fill_price = self._apply_slippage(close, "BUY")
        qty = self.strategy.position_size(cash - self.brokerage_per_trade, fill_price, SignalType.BUY)
        if qty <= 0:
            return cash, position
        cost = qty * fill_price + self.brokerage_per_trade
        if cost > cash:
            qty = max(0.0, np.floor((cash - self.brokerage_per_trade) / fill_price))
            if qty <= 0:
                return cash, position
            cost = qty * fill_price + self.brokerage_per_trade
        cash -= cost
        position = Position(
            symbol=symbol, quantity=qty, avg_price=fill_price, direction=1, entry_time=ts,
            stop_loss=self.strategy.stop_loss(fill_price, SignalType.BUY),
            take_profit=self.strategy.take_profit(fill_price, SignalType.BUY),
        )
        trades.append(Trade(ts, symbol, "BUY", qty, fill_price, 0.0, "open-long"))
        return cash, position

    def _handle_sell(
        self, cash: float, position: Position, close: float,
        ts: pd.Timestamp, trades: List[Trade], symbol: str,
    ):
        if position.direction == 1:
            return self._close_position(cash, position, close, ts, trades, symbol, note="close-long")
        if position.direction == -1 or not self.allow_short:
            return cash, position
        fill_price = self._apply_slippage(close, "SELL")
        qty = self.strategy.position_size(cash - self.brokerage_per_trade, fill_price, SignalType.SELL)
        if qty <= 0:
            return cash, position
        cash += qty * fill_price - self.brokerage_per_trade  # short proceeds
        position = Position(
            symbol=symbol, quantity=qty, avg_price=fill_price, direction=-1, entry_time=ts,
            stop_loss=self.strategy.stop_loss(fill_price, SignalType.SELL),
            take_profit=self.strategy.take_profit(fill_price, SignalType.SELL),
        )
        trades.append(Trade(ts, symbol, "SELL", qty, fill_price, 0.0, "open-short"))
        return cash, position

    def _close_position(
        self, cash: float, position: Position, close: float,
        ts: pd.Timestamp, trades: List[Trade], symbol: str, note: str = "",
    ):
        if position.is_flat():
            return cash, position
        side = "SELL" if position.direction == 1 else "BUY"
        fill_price = self._apply_slippage(close, side)
        gross = position.quantity * fill_price
        if position.direction == 1:
            cash += gross - self.brokerage_per_trade
            pnl = (fill_price - position.avg_price) * position.quantity - self.brokerage_per_trade
        else:
            cash -= gross + self.brokerage_per_trade
            pnl = (position.avg_price - fill_price) * position.quantity - self.brokerage_per_trade
        trades.append(Trade(ts, symbol, side, position.quantity, fill_price, pnl, note or "close"))
        return cash, Position(symbol=symbol)

    def _maybe_apply_stops(self, cash, position, close, ts, trades, symbol):
        if position.is_flat():
            return cash, position
        if position.direction == 1:
            if position.stop_loss and close <= position.stop_loss:
                return self._close_position(cash, position, close, ts, trades, symbol, note="stop-loss")
            if position.take_profit and close >= position.take_profit:
                return self._close_position(cash, position, close, ts, trades, symbol, note="take-profit")
        else:
            if position.stop_loss and close >= position.stop_loss:
                return self._close_position(cash, position, close, ts, trades, symbol, note="stop-loss")
            if position.take_profit and close <= position.take_profit:
                return self._close_position(cash, position, close, ts, trades, symbol, note="take-profit")
        return cash, position
