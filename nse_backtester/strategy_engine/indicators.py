"""Re-usable indicator helpers built on top of ``ta`` with pandas fallbacks."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

try:  # pragma: no cover
    from ta.momentum import RSIIndicator
    from ta.trend import EMAIndicator
    from ta.volume import VolumeWeightedAveragePrice

    _HAS_TA = True
except ImportError:  # pragma: no cover
    _HAS_TA = False


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    if _HAS_TA:
        return RSIIndicator(close=series, window=period, fillna=False).rsi()
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _ema(series: pd.Series, period: int) -> pd.Series:
    if _HAS_TA:
        return EMAIndicator(close=series, window=period, fillna=False).ema_indicator()
    return series.ewm(span=period, adjust=False).mean()


def _vwap(df: pd.DataFrame, period: Optional[int] = None) -> pd.Series:
    if "volume" not in df or df["volume"].isna().all():
        return df["close"].rolling(period or 14).mean()
    if _HAS_TA and period is None:
        return VolumeWeightedAveragePrice(
            high=df["high"], low=df["low"], close=df["close"], volume=df["volume"], fillna=False
        ).volume_weighted_average_price()
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = typical * df["volume"]
    if period:
        return pv.rolling(period).sum() / df["volume"].rolling(period).sum().replace(0, np.nan)
    return pv.cumsum() / df["volume"].cumsum().replace(0, np.nan)


def _signal_from_state(prev_state: int, state: int) -> str:
    if prev_state <= 0 and state > 0:
        return "BUY"
    if prev_state >= 0 and state < 0:
        return "SELL"
    return "HOLD"


def rsi_strategy_signals(
    df: pd.DataFrame,
    period: int = 14,
    oversold: float = 30.0,
    overbought: float = 70.0,
) -> pd.DataFrame:
    out = df.copy()
    out["rsi"] = _rsi(out["close"], period=period)
    out["signal"] = "HOLD"
    out.loc[out["rsi"] < oversold, "signal"] = "BUY"
    out.loc[out["rsi"] > overbought, "signal"] = "SELL"
    return out


def ema_crossover_signals(
    df: pd.DataFrame,
    fast: int = 9,
    slow: int = 21,
) -> pd.DataFrame:
    out = df.copy()
    out["ema_fast"] = _ema(out["close"], period=fast)
    out["ema_slow"] = _ema(out["close"], period=slow)
    state = np.sign(out["ema_fast"] - out["ema_slow"]).fillna(0).astype(int)
    prev = state.shift(1).fillna(0).astype(int)
    signals = []
    for p, s in zip(prev, state):
        signals.append(_signal_from_state(p, s))
    out["signal"] = signals
    return out


def vwap_strategy_signals(df: pd.DataFrame, period: Optional[int] = None) -> pd.DataFrame:
    out = df.copy()
    out["vwap"] = _vwap(out, period=period)
    out["signal"] = "HOLD"
    out.loc[out["close"] > out["vwap"], "signal"] = "BUY"
    out.loc[out["close"] < out["vwap"], "signal"] = "SELL"
    return out
