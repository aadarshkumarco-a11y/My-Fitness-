import pytest

from nse_backtester.data_engine import DataEngine
from nse_backtester.strategy_engine import DSLStrategy, DSLParseError
from nse_backtester.backtesting import BacktestEngine


def _data():
    return DataEngine.synthetic_ohlcv(days=200, seed=3)


def test_simple_rsi_dsl():
    dsl = """
    BUY  WHEN RSI(14) < 30
    SELL WHEN RSI(14) > 70
    """
    strat = DSLStrategy(dsl)
    out = strat.generate_signals(_data())
    assert "signal" in out.columns
    assert set(out["signal"].unique()).issubset({"BUY", "SELL", "HOLD"})


def test_logical_and_indicator_combo():
    dsl = """
    # buy oversold pullback only when above the 50 EMA
    BUY  WHEN RSI(14) < 35 AND close > EMA(50)
    SELL WHEN close < EMA(50)
    STOP_LOSS 2%
    TAKE_PROFIT 5%
    """
    strat = DSLStrategy(dsl)
    out = strat.generate_signals(_data())
    assert (out["signal"] == "BUY").any() or (out["signal"] == "SELL").any()
    assert strat.params["stop_loss_pct"] == pytest.approx(0.02)
    assert strat.params["take_profit_pct"] == pytest.approx(0.05)


def test_cross_above_below():
    dsl = """
    BUY  WHEN EMA(9) cross_above EMA(21)
    SELL WHEN EMA(9) cross_below EMA(21)
    """
    strat = DSLStrategy(dsl)
    out = strat.generate_signals(_data())
    assert (out["signal"] == "BUY").any()
    assert (out["signal"] == "SELL").any()


def test_dsl_runs_in_backtest():
    strat = DSLStrategy("BUY WHEN RSI(14) < 35\nSELL WHEN RSI(14) > 65")
    res = BacktestEngine(strategy=strat, initial_capital=200_000).run(_data(), symbol="X")
    assert res.run_id
    assert res.equity_curve.shape[0] > 0


def test_invalid_dsl_raises():
    with pytest.raises(DSLParseError):
        DSLStrategy("totally not valid")
    with pytest.raises(DSLParseError):
        DSLStrategy("")
