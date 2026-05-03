# NSE Backtesting Platform

A modular, production-grade backtesting platform for **Indian markets** (NSE
stocks and options). Designed to behave like a real quant research stack — not
a demo notebook.

## Highlights

* **Hardened NSE scraper** — session warm-up, rotating headers, 401/403 detection,
  randomised 2–4 s delays, exponential retries, transparent JSON cache.
* **Modular architecture** — `scraper`, `data_engine`, `strategy_engine`,
  `backtesting`, `pricing`, `analytics`, `database`, `ui`, `utils` —
  strict separation of concerns.
* **Strategy plug-in system** — add a strategy by subclassing `Strategy` and
  decorating with `@register_strategy("name")`.
* **Four ways to define strategies** — built-in registry, declarative Rule
  DSL (`BUY WHEN RSI(14) < 30`), Pine Script subset (auto-translated), or
  arbitrary Python uploaded via the UI.
* **Backtesting engine** — equities + weekly ATM options; brokerage, slippage,
  stop-loss/take-profit, position sizing, full trade log.
* **Black-Scholes pricing** — used as a fallback when scraped option data is
  missing or stale; also exposes greeks and implied-vol solver.
* **Analytics engine** — Net Profit, ROI, Win Rate, Max Drawdown, Sharpe,
  Profit Factor, equity & drawdown curves, CSV export.
* **Streamlit UI** — strategy/capital input, run button, KPI cards, Plotly
  equity & drawdown charts, trade table, CSV download.
* **Robust by design** — never crashes on missing data; uses synthetic OHLCV
  fallback when NSE is unreachable so research never blocks.

## Project layout

```
nse_backtester/
├── scraper/            # Hardened NSE HTTP scraper
├── data_engine/        # Normalize + persist scraped data
├── strategy_engine/    # Strategy contract + indicator helpers + registry
├── strategies/         # Built-in strategy plug-ins (RSI / EMA / VWAP / Options)
├── backtesting/        # Equity + Options backtest engines
├── pricing/            # Black-Scholes + implied volatility
├── analytics/          # Metrics + reporting + plotting
├── database/           # SQLite layer (option_chain, equity_history, trade_log)
├── ui/                 # Streamlit dashboard
├── utils/              # Logger + config (env-overridable)
├── tests/              # Pytest suite (scraper / strategy / backtest / pricing / DB)
├── main.py             # CLI entrypoint
└── requirements.txt
```

## Setup

```bash
# 1. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the test suite
pytest -q nse_backtester/tests
```

> Deploying to Streamlit Community Cloud? See [DEPLOY.md](DEPLOY.md) for a
> 3-minute walkthrough.

## CLI usage

The CLI lives in `nse_backtester/main.py` and is run as a module so relative
imports work correctly:

```bash
# List registered strategies
python -m nse_backtester.main strategies

# Equity backtest with synthetic data (offline-safe default)
python -m nse_backtester.main run --strategy ema_crossover --symbol NIFTY --capital 1000000

# Options backtest (weekly ATM, BS pricing)
python -m nse_backtester.main run --strategy option_buying --options

# Live scrape an NSE option chain
python -m nse_backtester.main scrape --symbol NIFTY
```

Outputs are written under `nse_backtester/data/exports/<run_id>/`:

* `trades.csv` — full trade log
* `equity_curve.csv` — bar-by-bar portfolio equity
* `drawdown.csv` — running drawdown
* `metrics.json` — KPI summary

## Streamlit UI

```bash
streamlit run nse_backtester/ui/app.py
```

The dashboard exposes strategy choice, capital, brokerage, slippage, options
mode, and renders KPI cards + Plotly charts + trade table with CSV download.

## Defining your own strategy

You have **four** options:

### 1. Rule DSL (no code, write rules in plain text)

In the UI sidebar pick **Strategy source → Rule DSL**, then write:

```
# Mean-reversion with trend filter
BUY  WHEN RSI(14) < 30 AND close > EMA(50)
SELL WHEN RSI(14) > 70 OR close < EMA(50)
STOP_LOSS  2%
TAKE_PROFIT 5%
```

Supported: `RSI(p)`, `EMA(p)`, `SMA(p)`, `VWAP()`, `close/open/high/low/volume`,
`<, >, <=, >=, ==, !=`, `AND/OR/NOT`, `cross_above`, `cross_below`,
`STOP_LOSS X%`, `TAKE_PROFIT X%`.

### 2. Pine Script (subset)

Pick **Strategy source → Pine Script (subset)** in the UI. A subset is
auto-translated to the DSL:

```pine
//@version=5
strategy("EMA crossover")
fast = ta.ema(close, 9)
slow = ta.ema(close, 21)
if ta.crossover(fast, slow)
    strategy.entry("Long", strategy.long)
if ta.crossunder(fast, slow)
    strategy.close("Long")
```

Supported: `ta.rsi`, `ta.ema`, `ta.sma`, `ta.vwap`, `ta.crossover`,
`ta.crossunder`, simple variable assignments, `if <cond>` blocks containing
`strategy.entry`/`strategy.close`/`strategy.exit`.

### 3. Custom Python in the UI

Pick **Strategy source → Custom Python**. Paste any class extending
`Strategy` — it's executed in-process so you get the full power of pandas,
numpy, and the `ta` library.

### 4. Plug-in registered strategy (programmatic)

```python
# nse_backtester/strategies/my_strategy.py
import pandas as pd
from nse_backtester.strategy_engine.base import Strategy
from nse_backtester.strategy_engine.registry import register_strategy

@register_strategy("my_strategy")
class MyStrategy(Strategy):
    description = "Buy on green candle, sell on red."

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        out = data.copy()
        out["signal"] = "HOLD"
        out.loc[out["close"] > out["open"], "signal"] = "BUY"
        out.loc[out["close"] < out["open"], "signal"] = "SELL"
        return out
```

Add `from . import my_strategy` to `nse_backtester/strategies/__init__.py`,
and it appears in the built-in dropdown automatically.

## Configuration

Every tunable knob lives in `nse_backtester/utils/config.py`. Override at
runtime via environment variables (all prefixed with `BACKTESTER_`):

| Env var | Default | Purpose |
|---|---|---|
| `BACKTESTER_LOG_LEVEL` | `INFO` | Logger level |
| `BACKTESTER_CAPITAL` | `1000000` | Initial capital |
| `BACKTESTER_BROKERAGE` | `20` | Brokerage per trade (₹) |
| `BACKTESTER_SLIPPAGE_PCT` | `0.005` | Slippage (decimal) |
| `BACKTESTER_RISK_FREE_RATE` | `0.065` | Used by Black-Scholes & Sharpe |
| `BACKTESTER_LOT_SIZE` | `50` | Default options lot size |
| `BACKTESTER_SCRAPER_TIMEOUT` | `10` | NSE request timeout (s) |
| `BACKTESTER_SCRAPER_RETRIES` | `3` | Retry attempts |
| `BACKTESTER_SCRAPER_MIN_DELAY` | `2` | Min jitter delay (s) |
| `BACKTESTER_SCRAPER_MAX_DELAY` | `4` | Max jitter delay (s) |

## NSE scraping notes

* Always start with a warm-up GET against `https://www.nseindia.com/` to seed
  cookies — the API endpoints reject requests without them.
* The scraper rotates the session and retries on `401/403` automatically.
* All responses are cached as JSON under `nse_backtester/data/cache/` so you
  can re-run backtests offline. TTLs are configurable per call.
* If NSE is unreachable, the data engine seamlessly falls back to deterministic
  synthetic OHLCV so research never gets blocked.

## Disclaimer

This tool is for **research only**. Backtests use simplified assumptions
(notably a constant volatility surface for Black-Scholes and synthetic OHLCV
when live data is unavailable). Validate independently before risking capital.
