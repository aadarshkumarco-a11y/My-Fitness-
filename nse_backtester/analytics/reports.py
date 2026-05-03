"""CSV exports + Plotly chart helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from ..utils import get_logger, settings

logger = get_logger(__name__)

try:  # pragma: no cover
    import plotly.graph_objects as go

    _HAS_PLOTLY = True
except ImportError:  # pragma: no cover
    _HAS_PLOTLY = False


def _ensure_export_dir() -> Path:
    settings.storage.export_dir.mkdir(parents=True, exist_ok=True)
    return settings.storage.export_dir


def export_results(
    run_id: str,
    trades: pd.DataFrame,
    equity_curve: pd.DataFrame,
    drawdown_curve: pd.DataFrame,
    metrics: Optional[dict] = None,
) -> Tuple[Path, Path, Path]:
    out_dir = _ensure_export_dir() / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    trades_path = out_dir / "trades.csv"
    equity_path = out_dir / "equity_curve.csv"
    drawdown_path = out_dir / "drawdown.csv"

    if trades is not None and not trades.empty:
        trades.to_csv(trades_path, index=False)
    else:
        pd.DataFrame(columns=["timestamp", "symbol", "side", "quantity", "price", "pnl", "notes"]).to_csv(trades_path, index=False)

    if equity_curve is not None and not equity_curve.empty:
        equity_curve.to_csv(equity_path)
    if drawdown_curve is not None and not drawdown_curve.empty:
        drawdown_curve.to_csv(drawdown_path)

    if metrics:
        (out_dir / "metrics.json").write_text(pd.Series(metrics).to_json(indent=2))

    logger.info("Exported run %s to %s", run_id, out_dir)
    return trades_path, equity_path, drawdown_path


def plot_equity_curve(equity_curve: pd.DataFrame):
    if not _HAS_PLOTLY or equity_curve is None or equity_curve.empty:
        return None
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=equity_curve.index,
            y=equity_curve["equity"],
            mode="lines",
            name="Equity",
            line=dict(color="#1f77b4", width=2),
        )
    )
    fig.update_layout(
        title="Portfolio Equity Curve",
        xaxis_title="Date",
        yaxis_title="Equity (₹)",
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


def plot_drawdown(drawdown_curve: pd.DataFrame):
    if not _HAS_PLOTLY or drawdown_curve is None or drawdown_curve.empty:
        return None
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=drawdown_curve.index,
            y=drawdown_curve["drawdown"] * 100,
            mode="lines",
            fill="tozeroy",
            name="Drawdown %",
            line=dict(color="#d62728", width=1.5),
        )
    )
    fig.update_layout(
        title="Drawdown",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        template="plotly_white",
    )
    return fig
