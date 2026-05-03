"""Tests for the wide strangle sell strategy."""
import pytest

from nse_backtester.data_engine import DataEngine
from nse_backtester.strategies.wide_strangle_sell import WideStrangleSellStrategy


def _data(days=180, seed=7):
    return DataEngine.synthetic_ohlcv(days=days, seed=seed, symbol="BANKNIFTY")


def test_strangle_runs():
    # Synthetic data starts ~18000, so use gap ~3% (500) and higher IV.
    strat = WideStrangleSellStrategy(gap=500, lot_size=35, lots=1, volatility=0.25)
    result = strat.run_full_backtest(_data(), initial_capital=500_000, symbol="BANKNIFTY")
    assert result.run_id
    assert result.equity_curve.shape[0] > 0
    assert len(result.trades) > 0


def test_strangle_equity_not_constant():
    strat = WideStrangleSellStrategy(gap=400, lot_size=35, lots=1, volatility=0.30)
    result = strat.run_full_backtest(_data(days=120), initial_capital=1_000_000, symbol="BANKNIFTY")
    eq = result.equity_curve["equity"]
    assert eq.nunique() > 1, "Equity should vary week to week"


def test_strangle_params_propagate():
    strat = WideStrangleSellStrategy(gap=500, strike_step=50, lot_size=50, lots=2)
    assert strat.gap == 500
    assert strat.strike_step == 50
    assert strat.lot_size == 50
    assert strat.lots == 2
    assert strat.supports_self_backtest is True


def test_strangle_round_strike():
    strat = WideStrangleSellStrategy(strike_step=100)
    assert strat._round_strike(48323) == 48300
    assert strat._round_strike(48350) == 48400
    assert strat._round_strike(50000) == 50000


def test_strangle_small_data():
    small = _data(days=10, seed=99)
    strat = WideStrangleSellStrategy(gap=500, lot_size=35, lots=1)
    result = strat.run_full_backtest(small, initial_capital=500_000)
    assert result.equity_curve.shape[0] > 0
