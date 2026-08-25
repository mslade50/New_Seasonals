"""Research-only VIX-compression x fragility UVXY strategy."""

from .backtest import StrategyConfig, build_features, build_signals, run_research

__all__ = ["StrategyConfig", "build_features", "build_signals", "run_research"]
