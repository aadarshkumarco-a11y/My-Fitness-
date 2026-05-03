"""Streamlit dashboard for running backtests interactively.

Run:
    streamlit run nse_backtester/ui/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the parent package importable when this file is launched directly.
_HERE = Path(__file__).resolve()
_ROOT = _HERE.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import streamlit as st

from nse_backtester.analytics import (
    compute_metrics,
    export_results,
    plot_drawdown,
    plot_equity_curve,
)
from nse_backtester.backtesting import BacktestEngine, OptionsBacktestEngine
from nse_backtester.data_engine import DataEngine
from nse_backtester.database import Database
from nse_backtester.scraper import NSEScraper
from nse_backtester import strategies as _builtin_strategies  # noqa: F401
from nse_backtester.strategy_engine import (
    CustomStrategyError,
    DSLParseError,
    DSLStrategy,
    PineParseError,
    default_template,
    load_custom_strategy,
    pine_to_dsl,
)
from nse_backtester.strategy_engine.registry import get_strategy, list_strategies
from nse_backtester.utils import settings


st.set_page_config(page_title="NSE Backtester", page_icon="📈", layout="wide")
st.title("📈 NSE Backtesting Platform")
st.caption("Strategy backtesting for Indian markets — equities & options. Built-in, DSL, Pine Script, or custom Python.")


_DEFAULT_DSL = """# Mean-reversion with trend filter
BUY  WHEN RSI(14) < 30 AND close > EMA(50)
SELL WHEN RSI(14) > 70 OR close < EMA(50)
STOP_LOSS  2%
TAKE_PROFIT 5%
"""

_DEFAULT_PINE = """//@version=5
strategy("EMA crossover")
fast = ta.ema(close, 9)
slow = ta.ema(close, 21)
if ta.crossover(fast, slow)
    strategy.entry("Long", strategy.long)
if ta.crossunder(fast, slow)
    strategy.close("Long")
"""


with st.sidebar:
    st.header("Configuration")
    strategy_source = st.radio(
        "Strategy source",
        ["Built-in", "Rule DSL", "Pine Script (subset)", "Custom Python"],
        index=0,
        help="Choose how you want to define your strategy.",
    )

    if strategy_source == "Built-in":
        strategy_name = st.selectbox("Strategy", list_strategies(), index=0)
    else:
        strategy_name = None

    symbol = st.text_input("Symbol", value="NIFTY")
    capital = st.number_input(
        "Initial capital (₹)",
        min_value=10_000.0,
        max_value=1e9,
        value=float(settings.backtest.initial_capital),
        step=10_000.0,
    )
    days = st.slider("Days of history", min_value=60, max_value=1500, value=365, step=30)
    seed = st.number_input("RNG seed (for synthetic data)", value=42, step=1)
    use_options = st.checkbox("Run on options (weekly ATM, BS pricing)", value=False)
    data_source = st.radio("Data source", ["Synthetic", "Live (NSE scraper)"], index=0)
    brokerage = st.number_input("Brokerage per trade (₹)", value=20.0, step=1.0)
    slippage = st.number_input("Slippage (% as decimal)", value=0.005, format="%f")
    run_btn = st.button("▶ Run backtest", use_container_width=True, type="primary")


# Strategy editor area (main panel, above results)
strategy_payload = None
if strategy_source == "Rule DSL":
    with st.expander("📜 DSL strategy editor", expanded=True):
        st.markdown(
            "**Syntax:** `BUY/SELL/EXIT WHEN <expr>` — supports `RSI(p)`, `EMA(p)`, "
            "`SMA(p)`, `VWAP()`, `close/open/high/low/volume`, `<, >, <=, >=, ==, !=`, "
            "`AND/OR/NOT`, `cross_above`, `cross_below`. Add `STOP_LOSS X%` and "
            "`TAKE_PROFIT X%` lines for risk management."
        )
        strategy_payload = st.text_area("DSL", value=_DEFAULT_DSL, height=200, label_visibility="collapsed")

elif strategy_source == "Pine Script (subset)":
    with st.expander("🌲 Pine Script editor", expanded=True):
        st.markdown(
            "Supported subset: `ta.rsi`, `ta.ema`, `ta.sma`, `ta.vwap`, "
            "`ta.crossover`, `ta.crossunder`, simple variable assignments, "
            "`if <cond>` with `strategy.entry`/`strategy.close`/`strategy.exit`. "
            "Will be auto-translated to the DSL."
        )
        strategy_payload = st.text_area("Pine", value=_DEFAULT_PINE, height=240, label_visibility="collapsed")

elif strategy_source == "Custom Python":
    with st.expander("🐍 Custom Python strategy", expanded=True):
        st.warning(
            "⚠️ Custom code is executed in this Python process. Only paste code "
            "you trust — do not run code from strangers on a hosted instance.",
            icon="⚠️",
        )
        st.markdown(
            "Define a class extending "
            "`nse_backtester.strategy_engine.base.Strategy` with a "
            "`generate_signals(self, data)` method that returns a DataFrame "
            "with a `signal` column (`BUY` / `SELL` / `HOLD` / `EXIT`)."
        )
        strategy_payload = st.text_area("Python", value=default_template(), height=320, label_visibility="collapsed")


@st.cache_data(show_spinner=False)
def _synthetic_data(symbol: str, days: int, seed: int) -> pd.DataFrame:
    return DataEngine.synthetic_ohlcv(days=days, seed=int(seed), symbol=symbol)


def _build_strategy(source: str, name: str | None, payload: str | None):
    if source == "Built-in":
        return get_strategy(name)()
    if source == "Rule DSL":
        return DSLStrategy(payload or "")
    if source == "Pine Script (subset)":
        dsl = pine_to_dsl(payload or "")
        st.info(f"Translated Pine → DSL:\n```\n{dsl}\n```")
        return DSLStrategy(dsl)
    if source == "Custom Python":
        cls = load_custom_strategy(payload or "")
        return cls()
    raise ValueError(f"Unknown strategy source: {source}")


def _load_data(source: str, symbol: str, days: int, seed: int) -> pd.DataFrame:
    if source == "Synthetic":
        return _synthetic_data(symbol, days, int(seed))
    try:
        de = DataEngine(scraper=NSEScraper())
        snap = de.fetch_option_chain(symbol)
        if snap is not None:
            st.info(
                f"Fetched live spot for {symbol}: ₹{snap.spot:,.2f} (ATM strike: {snap.atm_strike})"
            )
        else:
            st.warning("Could not fetch live data — falling back to synthetic OHLCV.")
    except Exception as exc:  # pragma: no cover
        st.warning(f"Live fetch failed ({exc}). Falling back to synthetic OHLCV.")
    return _synthetic_data(symbol, days, int(seed))


if run_btn:
    try:
        strategy = _build_strategy(strategy_source, strategy_name, strategy_payload)
    except (DSLParseError, PineParseError, CustomStrategyError) as exc:
        st.error(f"Strategy error: {exc}")
        st.stop()
    except Exception as exc:
        st.error(f"Could not build strategy: {exc}")
        st.stop()

    with st.spinner("Loading data..."):
        try:
            data = _load_data(data_source, symbol, days, seed)
        except Exception as exc:
            st.error(f"Failed to load data: {exc}")
            st.stop()

    if data is None or data.empty:
        st.error("Data is empty.")
        st.stop()

    with st.spinner("Running backtest..."):
        try:
            if use_options:
                engine = OptionsBacktestEngine(
                    strategy=strategy,
                    initial_capital=capital,
                    brokerage_per_trade=brokerage,
                    slippage_pct=slippage,
                )
            else:
                engine = BacktestEngine(
                    strategy=strategy,
                    initial_capital=capital,
                    brokerage_per_trade=brokerage,
                    slippage_pct=slippage,
                )
            result = engine.run(data, symbol=symbol)
        except Exception as exc:
            st.error(f"Backtest failed: {exc}")
            st.stop()

    trades_df = result.trade_log_df
    metrics = compute_metrics(result.equity_curve, trades_df, result.initial_capital)
    result.metrics = metrics

    Database().insert_trades(result.run_id, [t.to_dict() for t in result.trades])
    export_results(result.run_id, trades_df, result.equity_curve, result.drawdown_curve, metrics)

    st.success(f"Run complete — id `{result.run_id}`")

    cols = st.columns(4)
    cols[0].metric("Net Profit (₹)", f"{metrics['net_profit']:,.0f}")
    cols[1].metric("ROI", f"{metrics['roi_pct']:.2f}%")
    cols[2].metric("Sharpe", f"{metrics['sharpe']:.2f}")
    cols[3].metric("Max Drawdown", f"{metrics['max_drawdown_pct']:.2f}%")

    cols2 = st.columns(4)
    cols2[0].metric("Win Rate", f"{metrics['win_rate_pct']:.2f}%")
    pf = metrics["profit_factor"]
    cols2[1].metric("Profit Factor", "∞" if pf == float("inf") else f"{pf:.2f}")
    cols2[2].metric("# Trades", f"{metrics['num_trades']}")
    cols2[3].metric("Final Equity", f"₹{metrics['final_equity']:,.0f}")

    fig_eq = plot_equity_curve(result.equity_curve)
    if fig_eq is not None:
        st.plotly_chart(fig_eq, use_container_width=True)

    fig_dd = plot_drawdown(result.drawdown_curve)
    if fig_dd is not None:
        st.plotly_chart(fig_dd, use_container_width=True)

    st.subheader("Trade log")
    if trades_df.empty:
        st.info("No trades were executed.")
    else:
        st.dataframe(trades_df, use_container_width=True)
        st.download_button(
            "⬇ Download trades CSV",
            trades_df.to_csv(index=False).encode("utf-8"),
            file_name=f"trades_{result.run_id}.csv",
            mime="text/csv",
        )

    st.subheader("Signals preview")
    st.dataframe(result.signals.tail(50), use_container_width=True)
else:
    st.info("Configure the run on the left and click **Run backtest**.")
