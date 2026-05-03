import pytest

from nse_backtester.data_engine import DataEngine
from nse_backtester.strategy_engine import DSLStrategy, pine_to_dsl, PineParseError
from nse_backtester.backtesting import BacktestEngine


PINE_RSI = """
//@version=5
strategy("RSI mean reversion")
rsiVal = ta.rsi(close, 14)
if rsiVal < 30
    strategy.entry("Long", strategy.long)
if rsiVal > 70
    strategy.close("Long")
"""


PINE_EMA_CROSS = """
//@version=5
strategy("EMA crossover")
fast = ta.ema(close, 9)
slow = ta.ema(close, 21)
if ta.crossover(fast, slow)
    strategy.entry("Long", strategy.long)
if ta.crossunder(fast, slow)
    strategy.close("Long")
"""


def test_pine_rsi_translates():
    dsl = pine_to_dsl(PINE_RSI)
    assert "RSI(14)" in dsl
    assert "BUY WHEN" in dsl
    assert "EXIT WHEN" in dsl


def test_pine_ema_translates_with_vars():
    dsl = pine_to_dsl(PINE_EMA_CROSS)
    assert "cross_above" in dsl
    assert "cross_below" in dsl


def test_pine_translation_runs_in_backtest():
    dsl = pine_to_dsl(PINE_RSI)
    strat = DSLStrategy(dsl)
    data = DataEngine.synthetic_ohlcv(days=180, seed=2)
    res = BacktestEngine(strategy=strat, initial_capital=100_000).run(data, "X")
    assert res.run_id


def test_pine_empty_raises():
    with pytest.raises(PineParseError):
        pine_to_dsl("// just a comment\nstrategy('X')\n")
