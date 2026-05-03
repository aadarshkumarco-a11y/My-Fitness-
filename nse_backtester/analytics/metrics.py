"""Performance metrics on equity curves and trade logs."""
from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import pandas as pd


def equity_curve_to_returns(equity: pd.Series) -> pd.Series:
    if equity is None or equity.empty:
        return pd.Series(dtype=float)
    returns = equity.pct_change().fillna(0.0)
    return returns


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    return float(drawdown.min())


def _sharpe(returns: pd.Series, risk_free_rate: float = 0.065, periods_per_year: int = 252) -> float:
    if returns.empty or returns.std(ddof=0) == 0:
        return 0.0
    excess = returns - risk_free_rate / periods_per_year
    return float(np.sqrt(periods_per_year) * excess.mean() / returns.std(ddof=0))


def _profit_factor(trades: pd.DataFrame) -> float:
    if trades is None or trades.empty or "pnl" not in trades.columns:
        return 0.0
    wins = trades.loc[trades["pnl"] > 0, "pnl"].sum()
    losses = -trades.loc[trades["pnl"] < 0, "pnl"].sum()
    if losses == 0:
        return float("inf") if wins > 0 else 0.0
    return float(wins / losses)


def _win_rate(trades: pd.DataFrame) -> float:
    if trades is None or trades.empty or "pnl" not in trades.columns:
        return 0.0
    closing = trades[trades["pnl"] != 0]
    if closing.empty:
        return 0.0
    wins = (closing["pnl"] > 0).sum()
    return float(wins / len(closing))


def compute_metrics(
    equity_curve: pd.DataFrame,
    trades: pd.DataFrame,
    initial_capital: float,
    risk_free_rate: float = 0.065,
    periods_per_year: int = 252,
) -> Dict[str, float]:
    if equity_curve is None or equity_curve.empty:
        return {
            "net_profit": 0.0, "roi_pct": 0.0, "win_rate_pct": 0.0,
            "max_drawdown_pct": 0.0, "sharpe": 0.0, "profit_factor": 0.0,
            "num_trades": 0, "final_equity": float(initial_capital),
        }

    equity = equity_curve["equity"] if "equity" in equity_curve.columns else equity_curve.iloc[:, 0]
    returns = equity_curve_to_returns(equity)
    final_equity = float(equity.iloc[-1])
    net_profit = final_equity - initial_capital
    roi_pct = (net_profit / initial_capital) * 100.0 if initial_capital > 0 else 0.0
    sharpe = _sharpe(returns, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year)
    max_dd = _max_drawdown(equity) * 100.0
    win_rate = _win_rate(trades) * 100.0
    profit_factor = _profit_factor(trades)

    return {
        "net_profit": float(net_profit),
        "roi_pct": float(roi_pct),
        "win_rate_pct": float(win_rate),
        "max_drawdown_pct": float(max_dd),
        "sharpe": float(sharpe),
        "profit_factor": float(profit_factor),
        "num_trades": int(0 if trades is None or trades.empty else len(trades)),
        "final_equity": final_equity,
    }
