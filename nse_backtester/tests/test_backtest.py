import pandas as pd
import pytest

from nse_backtester.analytics import compute_metrics
from nse_backtester.backtesting import BacktestEngine, OptionsBacktestEngine
from nse_backtester.data_engine import DataEngine
from nse_backtester.strategies import *  # noqa
from nse_backtester.strategy_engine.registry import get_strategy


def _data():
    return DataEngine.synthetic_ohlcv(days=180, seed=11)


def test_equity_backtest_runs_end_to_end():
    strategy = get_strategy("ema_crossover")()
    engine = BacktestEngine(strategy=strategy, initial_capital=500_000)
    result = engine.run(_data(), symbol="DEMO")
    assert result.run_id
    assert result.equity_curve.shape[0] > 0
    assert result.final_equity > 0
    metrics = compute_metrics(result.equity_curve, result.trade_log_df, result.initial_capital)
    for k in ("net_profit", "roi_pct", "sharpe", "max_drawdown_pct"):
        assert k in metrics


def test_empty_data_raises():
    strategy = get_strategy("rsi")()
    engine = BacktestEngine(strategy=strategy)
    with pytest.raises(ValueError):
        engine.run(pd.DataFrame())


def test_options_backtest_runs():
    strategy = get_strategy("option_buying")()
    engine = OptionsBacktestEngine(strategy=strategy, initial_capital=200_000)
    result = engine.run(_data(), symbol="NIFTY")
    assert result.run_id
    assert result.final_equity >= 0


def test_brokerage_and_slippage_applied():
    strategy = get_strategy("ema_crossover")()
    cheap = BacktestEngine(strategy=strategy, brokerage_per_trade=0, slippage_pct=0)
    expensive = BacktestEngine(strategy=strategy, brokerage_per_trade=200, slippage_pct=0.02)
    cheap_res = cheap.run(_data(), symbol="A")
    exp_res = expensive.run(_data(), symbol="A")
    # Higher costs should never produce strictly better final equity for the same data/strategy.
    assert exp_res.final_equity <= cheap_res.final_equity + 1e-6
