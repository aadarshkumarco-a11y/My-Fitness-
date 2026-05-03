import numpy as np
import pandas as pd

from nse_backtester.data_engine import DataEngine
from nse_backtester.strategies import *  # noqa: F401,F403
from nse_backtester.strategy_engine.indicators import (
    ema_crossover_signals,
    rsi_strategy_signals,
    vwap_strategy_signals,
)
from nse_backtester.strategy_engine.registry import get_strategy, list_strategies


def _sample():
    return DataEngine.synthetic_ohlcv(days=200, seed=7)


def test_registry_has_all_strategies():
    names = set(list_strategies())
    assert {"rsi", "ema_crossover", "vwap", "option_buying"}.issubset(names)


def test_rsi_signals_within_bounds():
    df = _sample()
    out = rsi_strategy_signals(df, period=14, oversold=30, overbought=70)
    assert "signal" in out.columns
    assert set(out["signal"].dropna().unique()).issubset({"BUY", "SELL", "HOLD"})


def test_ema_signals_produce_buy_or_sell():
    df = _sample()
    out = ema_crossover_signals(df, fast=5, slow=20)
    assert "signal" in out.columns
    assert set(out["signal"]).issubset({"BUY", "SELL", "HOLD"})


def test_vwap_signals_complete():
    df = _sample()
    out = vwap_strategy_signals(df)
    assert "signal" in out.columns
    assert len(out) == len(df)


def test_strategy_class_callable():
    strat_cls = get_strategy("rsi")
    strat = strat_cls(period=10)
    out = strat.generate_signals(_sample())
    assert "signal" in out.columns
