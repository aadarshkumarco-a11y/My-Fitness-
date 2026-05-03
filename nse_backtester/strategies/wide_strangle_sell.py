"""BankNifty / Nifty wide-strangle SELL strategy.

Paste-ready for the **Custom Python** strategy editor in the Streamlit UI, or
use directly via the CLI / Python import. This strategy owns its own backtest
simulator (``supports_self_backtest = True``) because the generic equity /
long-options engines cannot model short option premium collection + time-decay.

The strategy:
1. Every **Monday** (or the first trading day of the week), SELL:
   - 1 OTM Call = ATM strike + ``gap``  (rounded to nearest ``strike_step``)
   - 1 OTM Put  = ATM strike - ``gap``  (rounded to nearest ``strike_step``)
2. On **Thursday** (weekly expiry) CLOSE both legs at intrinsic value.
3. PnL per week = net premium received − intrinsic payout − brokerage − slippage.

Black-Scholes is used to price the premiums at entry with configurable IV and
risk-free rate. At expiry the option value collapses to max(0, S−K) / max(0, K−S).
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from nse_backtester.pricing.black_scholes import black_scholes
from nse_backtester.strategy_engine.base import Strategy
from nse_backtester.strategy_engine.registry import register_strategy


@dataclass
class _StrangleTrade:
    entry_date: pd.Timestamp
    expiry_date: pd.Timestamp
    spot_at_entry: float
    strike_ce: float
    strike_pe: float
    premium_ce: float
    premium_pe: float
    total_premium: float
    spot_at_expiry: float
    payout_ce: float
    payout_pe: float
    net_pnl: float
    lots: int
    lot_size: int


@register_strategy("wide_strangle_sell")
class WideStrangleSellStrategy(Strategy):
    """Short wide-strangle on weekly BankNifty / Nifty options.

    Parameters
    ----------
    gap : int
        Strike gap from spot in index points (default 1700 for BankNifty).
    strike_step : int
        Strike rounding step (100 for BankNifty, 50 for Nifty).
    lot_size : int
        Contracts per lot (35 for BankNifty, 50 for Nifty).
    lots : int
        Number of lots to sell per leg.
    volatility : float
        Annualised implied volatility assumption (default 0.18 = 18%).
    risk_free_rate : float
        Risk-free rate for BS (default 0.06 = 6%).
    brokerage_per_leg : float
        Brokerage per single-leg trade (default ₹20).
    slippage_pct : float
        Slippage as fraction of premium (default 0.005 = 0.5%).
    max_capital_pct : float
        Max fraction of capital to risk per week (margin block, default 0.5).
    """

    name = "wide_strangle_sell"
    description = "Short OTM weekly strangle on BankNifty / Nifty."
    supports_self_backtest = True

    def __init__(
        self,
        gap: int = 1700,
        strike_step: int = 100,
        lot_size: int = 35,
        lots: int = 1,
        volatility: float = 0.18,
        risk_free_rate: float = 0.06,
        brokerage_per_leg: float = 20.0,
        slippage_pct: float = 0.005,
        max_capital_pct: float = 0.50,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.gap = int(gap)
        self.strike_step = int(strike_step)
        self.lot_size = int(lot_size)
        self.lots = int(lots)
        self.volatility = float(volatility)
        self.risk_free_rate = float(risk_free_rate)
        self.brokerage_per_leg = float(brokerage_per_leg)
        self.slippage_pct = float(slippage_pct)
        self.max_capital_pct = float(max_capital_pct)

    # -- helpers ---------------------------------------------------------------

    def _round_strike(self, raw: float) -> float:
        return round(raw / self.strike_step) * self.strike_step

    @staticmethod
    def _bs_premium(spot, strike, dte_days, vol, rf, option_type: str) -> float:
        if dte_days <= 0:
            if option_type == "CE":
                return max(0.0, spot - strike)
            return max(0.0, strike - spot)
        t = dte_days / 365.0
        res = black_scholes(spot, strike, t, vol, rf)
        return res.call_price if option_type == "CE" else res.put_price

    def _find_expiry(self, entry_date: pd.Timestamp, ts_series: pd.Series) -> Optional[pd.Timestamp]:
        """Find Thursday (weekday 3) at or after entry_date within the same week."""
        for t in ts_series:
            if t > entry_date and pd.Timestamp(t).weekday() == 3:
                return pd.Timestamp(t)
        # If no Thursday found (data ends), use last available date.
        later = ts_series[ts_series > entry_date]
        return pd.Timestamp(later.iloc[-1]) if len(later) > 0 else None

    # -- generate_signals (fallback for UI display) ----------------------------

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        out = data.copy()
        out["signal"] = "HOLD"
        return out

    # -- full backtest ---------------------------------------------------------

    def run_full_backtest(
        self,
        data: pd.DataFrame,
        initial_capital: float,
        symbol: str = "BANKNIFTY",
        **kwargs,
    ):
        from nse_backtester.backtesting.engine import BacktestResult, Trade

        if data is None or data.empty:
            raise ValueError("Data is empty")

        df = data.copy().reset_index(drop=True)
        if "timestamp" not in df.columns:
            raise ValueError("Data must have a 'timestamp' column")

        df["_ts"] = pd.to_datetime(df["timestamp"])
        df["_weekday"] = df["_ts"].dt.weekday  # 0=Mon ... 4=Fri

        run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"
        capital = float(initial_capital)
        trades: List[Trade] = []
        strangle_trades: List[_StrangleTrade] = []
        equity_hist: List[Dict[str, Any]] = []

        total_qty = self.lot_size * self.lots

        # Choose entry days: Mondays preferred; if none in current week, use first
        # weekday available (Tue/Wed). Skip weeks where there are < 2 days to expiry.
        i = 0
        while i < len(df):
            row = df.iloc[i]
            entry_date = row["_ts"]
            weekday = int(row["_weekday"])

            # Only enter on Mon-Wed; otherwise mark equity and advance.
            if weekday > 2:
                equity_hist.append({"timestamp": entry_date, "equity": capital})
                i += 1
                continue

            spot = float(row["close"])
            strike_ce = self._round_strike(spot + self.gap)
            strike_pe = self._round_strike(spot - self.gap)

            expiry_date = self._find_expiry(entry_date, df["_ts"])
            if expiry_date is None:
                equity_hist.append({"timestamp": entry_date, "equity": capital})
                i += 1
                continue

            dte = max(1, (expiry_date - entry_date).days)
            if dte < 2:
                # Same-day or next-day expiry — premium too low; skip.
                equity_hist.append({"timestamp": entry_date, "equity": capital})
                i += 1
                continue

            prem_ce = self._bs_premium(spot, strike_ce, dte, self.volatility, self.risk_free_rate, "CE")
            prem_pe = self._bs_premium(spot, strike_pe, dte, self.volatility, self.risk_free_rate, "PE")
            prem_ce *= (1 - self.slippage_pct)
            prem_pe *= (1 - self.slippage_pct)

            total_premium = (prem_ce + prem_pe) * total_qty
            entry_brokerage = 2 * self.brokerage_per_leg

            # Skip weeks where premium is essentially zero (deep OTM, low IV)
            if total_premium <= entry_brokerage:
                equity_hist.append({"timestamp": entry_date, "equity": capital})
                i += 1
                continue

            capital += total_premium - entry_brokerage
            trades.append(Trade(
                timestamp=entry_date, symbol=symbol, side="SELL",
                quantity=total_qty, price=prem_ce,
                notes=f"Sell {int(strike_ce)}CE @ {prem_ce:.2f}",
            ))
            trades.append(Trade(
                timestamp=entry_date, symbol=symbol, side="SELL",
                quantity=total_qty, price=prem_pe,
                notes=f"Sell {int(strike_pe)}PE @ {prem_pe:.2f}",
            ))

            expiry_idx = None
            for j in range(i, len(df)):
                r = df.iloc[j]
                s = float(r["close"])
                t = r["_ts"]
                remaining_dte = max(0, (expiry_date - t).days)
                mtm_ce = self._bs_premium(s, strike_ce, remaining_dte, self.volatility, self.risk_free_rate, "CE")
                mtm_pe = self._bs_premium(s, strike_pe, remaining_dte, self.volatility, self.risk_free_rate, "PE")
                mtm_cost = (mtm_ce + mtm_pe) * total_qty
                equity_hist.append({"timestamp": t, "equity": capital - mtm_cost})
                if t >= expiry_date:
                    expiry_idx = j
                    break

            if expiry_idx is None:
                expiry_idx = len(df) - 1

            spot_exp = float(df.iloc[expiry_idx]["close"])
            payout_ce = max(0.0, spot_exp - strike_ce) * total_qty
            payout_pe = max(0.0, strike_pe - spot_exp) * total_qty
            exit_brokerage = 2 * self.brokerage_per_leg

            capital -= (payout_ce + payout_pe + exit_brokerage)
            net_pnl = total_premium - entry_brokerage - payout_ce - payout_pe - exit_brokerage

            trades.append(Trade(
                timestamp=df.iloc[expiry_idx]["_ts"], symbol=symbol, side="BUY",
                quantity=total_qty, price=payout_ce / max(total_qty, 1),
                pnl=-payout_ce,
                notes=f"Close {int(strike_ce)}CE @ intrinsic {payout_ce / total_qty:.2f}",
            ))
            trades.append(Trade(
                timestamp=df.iloc[expiry_idx]["_ts"], symbol=symbol, side="BUY",
                quantity=total_qty, price=payout_pe / max(total_qty, 1),
                pnl=-payout_pe,
                notes=f"Close {int(strike_pe)}PE @ intrinsic {payout_pe / total_qty:.2f}",
            ))

            strangle_trades.append(_StrangleTrade(
                entry_date=entry_date, expiry_date=df.iloc[expiry_idx]["_ts"],
                spot_at_entry=spot, strike_ce=strike_ce, strike_pe=strike_pe,
                premium_ce=prem_ce, premium_pe=prem_pe, total_premium=total_premium,
                spot_at_expiry=spot_exp, payout_ce=payout_ce, payout_pe=payout_pe,
                net_pnl=net_pnl, lots=self.lots, lot_size=self.lot_size,
            ))

            if equity_hist:
                equity_hist[-1]["equity"] = capital

            i = expiry_idx + 1

        while i < len(df):
            equity_hist.append({"timestamp": df.iloc[i]["_ts"], "equity": capital})
            i += 1

        eq_df = pd.DataFrame(equity_hist)
        eq_df["equity"] = eq_df["equity"].astype(float)
        eq_df["timestamp"] = pd.to_datetime(eq_df["timestamp"])

        peak = eq_df["equity"].expanding().max()
        dd = (eq_df["equity"] - peak) / peak
        dd_df = eq_df[["timestamp"]].copy()
        dd_df["drawdown"] = dd.values

        signals_df = self.generate_signals(data)

        return BacktestResult(
            run_id=run_id,
            initial_capital=initial_capital,
            final_equity=capital,
            trades=trades,
            equity_curve=eq_df,
            drawdown_curve=dd_df,
            signals=signals_df,
        )
