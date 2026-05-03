"""Options-specific backtesting helpers.

Simulates buying weekly ATM options on a spot data series. Option prices are
re-derived each bar with Black-Scholes when no live chain is available, which
naturally captures time decay.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import numpy as np
import pandas as pd

from ..pricing import black_scholes
from ..strategy_engine.base import Signal, SignalType, Strategy
from ..utils import get_logger, settings
from .engine import BacktestResult, Position, Trade

logger = get_logger(__name__)


def next_thursday(d: pd.Timestamp) -> pd.Timestamp:
    """Return the upcoming weekly expiry (Thursday). If d is Thursday, returns d."""
    days_ahead = (3 - d.weekday()) % 7  # Mon=0 ... Sun=6, Thu=3
    return (d + pd.Timedelta(days=days_ahead)).normalize()


def round_to_strike(spot: float, step: float = 50.0) -> float:
    return round(spot / step) * step


@dataclass
class OptionLeg:
    timestamp: pd.Timestamp
    expiry: pd.Timestamp
    option_type: str  # CE / PE
    strike: float
    entry_premium: float
    quantity: int  # in lots * lot_size


class OptionsBacktestEngine:
    """Buy-only weekly ATM options simulator using BS for pricing fallback."""

    def __init__(
        self,
        strategy: Strategy,
        initial_capital: Optional[float] = None,
        brokerage_per_trade: Optional[float] = None,
        slippage_pct: Optional[float] = None,
        lot_size: Optional[int] = None,
        volatility: float = 0.18,
        risk_free_rate: Optional[float] = None,
        strike_step: float = 50.0,
    ) -> None:
        cfg = settings.backtest
        self.strategy = strategy
        self.initial_capital = float(initial_capital if initial_capital is not None else cfg.initial_capital)
        self.brokerage_per_trade = float(
            brokerage_per_trade if brokerage_per_trade is not None else cfg.brokerage_per_trade
        )
        self.slippage_pct = float(slippage_pct if slippage_pct is not None else cfg.slippage_pct)
        self.lot_size = int(lot_size if lot_size is not None else cfg.default_lot_size)
        self.volatility = volatility
        self.risk_free_rate = float(risk_free_rate if risk_free_rate is not None else cfg.risk_free_rate)
        self.strike_step = strike_step

    def _premium(self, spot: float, strike: float, days_to_expiry: float, option_type: str) -> float:
        T = max(days_to_expiry, 0.0001) / 365.0
        result = black_scholes(spot, strike, T, self.volatility, self.risk_free_rate)
        return result.call_price if option_type == "CE" else result.put_price

    def run(self, data: pd.DataFrame, symbol: str = "NIFTY") -> BacktestResult:
        if data is None or data.empty:
            raise ValueError("Options backtest data is empty")
        if "timestamp" not in data.columns:
            data = data.reset_index().rename(columns={"index": "timestamp"})
        data = data.copy()
        data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
        data = data.dropna(subset=["timestamp", "close"]).sort_values("timestamp").reset_index(drop=True)

        signals_df = self.strategy.generate_signals(data).copy()
        if "timestamp" not in signals_df.columns:
            signals_df["timestamp"] = data["timestamp"].values
        signals_df["signal"] = signals_df["signal"].astype(str).str.upper()
        if "close" not in signals_df.columns:
            signals_df["close"] = data["close"].values

        run_id = f"opt-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"
        cash = self.initial_capital
        leg: Optional[OptionLeg] = None
        trades: List[Trade] = []
        equity_history = []

        for _, row in signals_df.iterrows():
            ts = pd.Timestamp(row["timestamp"])
            spot = float(row["close"])
            sig = row["signal"]

            # Mark-to-market and force-close on expiry day
            if leg is not None:
                dte = max((leg.expiry - ts).days, 0)
                premium_now = self._premium(spot, leg.strike, dte, leg.option_type)
                if dte == 0:
                    pnl = (premium_now - leg.entry_premium) * leg.quantity - self.brokerage_per_trade
                    cash += premium_now * leg.quantity - self.brokerage_per_trade
                    trades.append(Trade(ts, f"{symbol}-{int(leg.strike)}{leg.option_type}", "SELL", leg.quantity, premium_now, pnl, "expiry"))
                    leg = None

            if leg is None and sig in ("BUY", "SELL"):
                option_type = "CE" if sig == "BUY" else "PE"
                strike = round_to_strike(spot, self.strike_step)
                expiry = next_thursday(ts)
                if expiry == ts.normalize():
                    expiry = expiry + pd.Timedelta(days=7)
                dte = max((expiry - ts).days, 1)
                premium = self._premium(spot, strike, dte, option_type)
                premium *= 1 + self.slippage_pct
                if premium <= 0:
                    continue
                budget = cash * 0.5  # do not deploy 100% of capital on a single weekly leg
                lots = int(budget // (premium * self.lot_size))
                qty = lots * self.lot_size
                if qty <= 0:
                    continue
                cost = qty * premium + self.brokerage_per_trade
                if cost > cash:
                    continue
                cash -= cost
                leg = OptionLeg(ts, expiry, option_type, strike, premium, qty)
                trades.append(Trade(ts, f"{symbol}-{int(strike)}{option_type}", "BUY", qty, premium, 0.0, f"open-{option_type}"))

            if leg is not None:
                dte_now = max((leg.expiry - ts).days, 0)
                mark = self._premium(spot, leg.strike, dte_now, leg.option_type)
                equity = cash + mark * leg.quantity
            else:
                equity = cash
            equity_history.append({"timestamp": ts, "equity": equity, "cash": cash, "close": spot})

        # Force close at end
        if leg is not None and not data.empty:
            ts = pd.Timestamp(data["timestamp"].iloc[-1])
            spot = float(data["close"].iloc[-1])
            dte = max((leg.expiry - ts).days, 0)
            premium = self._premium(spot, leg.strike, dte, leg.option_type)
            pnl = (premium - leg.entry_premium) * leg.quantity - self.brokerage_per_trade
            cash += premium * leg.quantity - self.brokerage_per_trade
            trades.append(Trade(ts, f"{symbol}-{int(leg.strike)}{leg.option_type}", "SELL", leg.quantity, premium, pnl, "force-close"))
            leg = None
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
