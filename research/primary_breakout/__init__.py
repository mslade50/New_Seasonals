"""Research-only primary-universe breakout portfolio backtest."""

from .engine import (
    BacktestConfig,
    BacktestResult,
    build_stock_universe,
    compute_features,
    load_price_panel,
    monthly_block_bootstrap_sharpe,
    period_metrics,
    run_backtest,
    slice_metrics,
)

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "build_stock_universe",
    "compute_features",
    "load_price_panel",
    "monthly_block_bootstrap_sharpe",
    "period_metrics",
    "run_backtest",
    "slice_metrics",
]
