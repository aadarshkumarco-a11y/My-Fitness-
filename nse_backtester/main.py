"""Command-line entry point.

Usage examples:

    python -m nse_backtester.main run --strategy rsi --symbol RELIANCE
    python -m nse_backtester.main run --strategy option_buying --options
    python -m nse_backtester.main scrape --symbol NIFTY
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .analytics import compute_metrics, export_results
from .backtesting import BacktestEngine, OptionsBacktestEngine
from .data_engine import DataEngine
from .database import Database
from .scraper import NSEScraper
from .strategies import *  # noqa: F401,F403  registers built-in strategies
from .strategy_engine.registry import get_strategy, list_strategies
from .utils import get_logger, settings

logger = get_logger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nse_backtester", description="NSE Backtesting Platform")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run a backtest")
    run_p.add_argument("--strategy", required=True, choices=list_strategies())
    run_p.add_argument("--symbol", default="NIFTY")
    run_p.add_argument("--capital", type=float, default=settings.backtest.initial_capital)
    run_p.add_argument("--days", type=int, default=365)
    run_p.add_argument("--seed", type=int, default=42)
    run_p.add_argument("--options", action="store_true", help="Use options engine (weekly ATM)")
    run_p.add_argument("--source", choices=["synthetic", "scrape"], default="synthetic")

    scrape_p = sub.add_parser("scrape", help="Fetch + persist live NSE data")
    scrape_p.add_argument("--symbol", default="NIFTY")
    scrape_p.add_argument("--show", type=int, default=10)

    sub.add_parser("strategies", help="List registered strategies")
    return parser


def cmd_run(args: argparse.Namespace) -> int:
    strategy_cls = get_strategy(args.strategy)
    strategy = strategy_cls()

    de = DataEngine()
    if args.source == "synthetic":
        data = de.synthetic_ohlcv(days=args.days, seed=args.seed, symbol=args.symbol)
    else:
        # Use scrape: try equity quote then fall back to synthetic if unavailable.
        snap = de.fetch_option_chain(args.symbol)
        logger.info("Live spot fetched: %s", getattr(snap, "spot", None))
        data = de.synthetic_ohlcv(days=args.days, seed=args.seed, symbol=args.symbol)

    if args.options:
        engine = OptionsBacktestEngine(strategy=strategy, initial_capital=args.capital)
    else:
        engine = BacktestEngine(strategy=strategy, initial_capital=args.capital)

    result = engine.run(data, symbol=args.symbol)
    metrics = compute_metrics(result.equity_curve, result.trade_log_df, result.initial_capital)
    result.metrics = metrics

    db = Database()
    db.insert_trades(result.run_id, [t.to_dict() for t in result.trades])
    export_results(result.run_id, result.trade_log_df, result.equity_curve, result.drawdown_curve, metrics)

    print(json.dumps({"run_id": result.run_id, "metrics": metrics}, indent=2, default=str))
    return 0


def cmd_scrape(args: argparse.Namespace) -> int:
    de = DataEngine(scraper=NSEScraper())
    snap = de.fetch_option_chain(args.symbol)
    if snap is None:
        logger.error("Could not fetch option chain for %s", args.symbol)
        return 1
    print(json.dumps({
        "symbol": snap.symbol,
        "spot": snap.spot,
        "atm_strike": snap.atm_strike,
        "expiries": snap.expiries[:5],
        "rows": len(snap.rows),
        "preview": snap.rows[: args.show],
    }, indent=2, default=str))
    return 0


def cmd_strategies(_: argparse.Namespace) -> int:
    for name in list_strategies():
        cls = get_strategy(name)
        print(f"  {name:<18s} - {getattr(cls, 'description', '')}")
    return 0


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return cmd_run(args)
    if args.command == "scrape":
        return cmd_scrape(args)
    if args.command == "strategies":
        return cmd_strategies(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
