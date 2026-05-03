import pandas as pd
from nse_backtester.strategy_engine.base import Strategy
from nse_backtester.strategy_engine.indicators import rsi, ema, sma, vwap, atr, crossover, crossunder

class MyStrategy(Strategy):
    name = "starter_rsi"

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        out = data.copy()
        out["rsi14"] = rsi(out["close"], 14)

        out["signal"] = "HOLD"
        # Entry: oversold (RSI < 40)
        out.loc[out["rsi14"] < 40, "signal"] = "BUY"

        # Exit: overbought (RSI > 60) — guard with HOLD so it does NOT overwrite BUY
        is_hold = out["signal"] == "HOLD"
        out.loc[(out["rsi14"] > 60) & is_hold, "signal"] = "EXIT"

        return out
