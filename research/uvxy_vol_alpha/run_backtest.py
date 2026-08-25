"""CLI for the research-only UVXY compression × fragility backtest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.uvxy_vol_alpha.backtest import (
    StrategyConfig,
    run_research,
    write_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest the frozen VRC x ex-VRC fragility UVXY hypothesis."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data",
        help="Directory containing master_prices and fragility parquet files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "artifacts" / "uvxy-vol-alpha",
        help="Ignored directory for HTML/JSON/CSV outputs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = StrategyConfig()
    results = run_research(args.data_dir, config)
    paths = write_outputs(results, args.output_dir)
    primary = results["summaries"]["primary"]
    print(
        f"verdict={results['verdict']['status']} "
        f"trades={primary['n_trades']} "
        f"mean={primary['mean_return']:.6f} "
        f"win_rate={primary['win_rate']:.3f}"
    )
    for label, path in paths.items():
        print(f"{label}: {path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
