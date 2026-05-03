import pytest

from nse_backtester.data_engine import DataEngine
from nse_backtester.strategy_engine import (
    CustomStrategyError,
    default_template,
    load_custom_strategy,
)


def test_default_template_is_loadable():
    cls = load_custom_strategy(default_template())
    inst = cls()
    out = inst.generate_signals(DataEngine.synthetic_ohlcv(days=100, seed=4))
    assert "signal" in out.columns


def test_simple_custom_strategy():
    code = """
from nse_backtester.strategy_engine.base import Strategy
class GreenCandle(Strategy):
    def generate_signals(self, data):
        out = data.copy()
        out["signal"] = "HOLD"
        out.loc[out["close"] > out["open"], "signal"] = "BUY"
        out.loc[out["close"] < out["open"], "signal"] = "SELL"
        return out
"""
    cls = load_custom_strategy(code)
    out = cls().generate_signals(DataEngine.synthetic_ohlcv(days=50, seed=1))
    assert (out["signal"] == "BUY").any()
    assert (out["signal"] == "SELL").any()


def test_no_strategy_raises():
    with pytest.raises(CustomStrategyError):
        load_custom_strategy("x = 1\n")


def test_syntax_error_raises():
    with pytest.raises(CustomStrategyError):
        load_custom_strategy("def : pass")
