"""Orchestration and artifact writing for research-only Trend V2 runs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    FROZEN_BENCHMARK,
    PREREGISTERED_CROSS_SECTIONAL_SPEC,
    PREREGISTERED_MULTISPEED_SPECS,
    CrossSectionalSpec,
    MultiSpeedSpec,
)
from .engine import (
    BacktestResult,
    PriceData,
    TargetResult,
    backtest_next_period,
    cross_sectional_targets,
    frozen_benchmark_targets,
    multispeed_targets,
    performance_summary,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESERVED_OUTPUT_ROOTS = {
    "data",
    "dist",
    "site",
    "functions",
    ".github",
    "research",
    "scripts",
    "tests",
}


@dataclass(frozen=True)
class TrialRun:
    name: str
    family: str
    specification: dict[str, Any]
    targets: TargetResult
    backtest: BacktestResult
    summary: dict[str, Any]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(float(value)) else float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def _spec_parameter_fields(spec: object) -> list[str]:
    return [field.name for field in fields(spec) if field.name != "name"]


def _prepare_output_dir(output_dir: str | Path) -> Path:
    """Create a new artifact directory and refuse production-adjacent paths."""
    output = Path(output_dir).expanduser().resolve()
    try:
        relative = output.relative_to(PROJECT_ROOT)
    except ValueError:
        relative = None
    if relative is not None:
        if not relative.parts or relative.parts[0].lower() != "artifacts":
            raise ValueError(
                f"output must be under {PROJECT_ROOT / 'artifacts'}; got {output}"
            )
        if any(part.lower() in RESERVED_OUTPUT_ROOTS for part in relative.parts[1:]):
            raise ValueError(f"output path contains a reserved production/source directory: {output}")
    elif "artifacts" not in {part.lower() for part in output.parts}:
        raise ValueError("output outside the project must still be inside a directory named artifacts")
    if output.exists() and not output.is_dir():
        raise FileExistsError(f"artifact output exists and is not a directory: {output}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty artifact directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def infer_sector_tickers(sector_history: pd.DataFrame) -> list[str]:
    lower = {str(column).lower(): column for column in sector_history.columns}
    if "ticker" in lower:
        return sorted(
            sector_history[lower["ticker"]]
            .dropna()
            .astype(str)
            .str.upper()
            .str.strip()
            .unique()
        )
    if isinstance(sector_history.index, pd.DatetimeIndex):
        return sorted(str(column).upper().strip() for column in sector_history.columns)
    raise ValueError("cannot infer stock tickers from undated sector history")


def run_trend_v2_research(
    prices: PriceData,
    sector_history: pd.DataFrame | None = None,
    market_ticker: str = "SPY",
    stock_tickers: list[str] | None = None,
    multispeed_specs: tuple[MultiSpeedSpec, ...] = PREREGISTERED_MULTISPEED_SPECS,
    cross_sectional_spec: CrossSectionalSpec = PREREGISTERED_CROSS_SECTIONAL_SPEC,
) -> list[TrialRun]:
    """Run the frozen comparator, fixed ETF grid, and optional stock family."""
    market_ticker = market_ticker.upper().strip()
    cash_returns = None
    if "^IRX" in prices.close.columns:
        cash_returns = (
            prices.close["^IRX"]
            .dropna()
            .resample("ME")
            .last()
            .div(100.0 * 12.0)
            .shift(1)
        )
    missing_benchmark = sorted(set(FROZEN_BENCHMARK.universe) - set(prices.close.columns))
    if missing_benchmark:
        raise ValueError(f"frozen benchmark tickers missing: {missing_benchmark}")
    benchmark_close = prices.close.loc[:, list(FROZEN_BENCHMARK.universe)]
    benchmark_open = (
        prices.open.reindex(columns=benchmark_close.columns)
        if prices.open is not None
        else None
    )
    benchmark_target = frozen_benchmark_targets(benchmark_close)
    benchmark_backtest = backtest_next_period(
        targets=benchmark_target.targets,
        close=benchmark_close,
        open_prices=benchmark_open,
        cost_bps_per_side=FROZEN_BENCHMARK.cost_bps_per_side,
        cash_returns=cash_returns,
    )
    benchmark_summary = performance_summary(benchmark_backtest)
    benchmark_net = benchmark_backtest.monthly["net_return"]
    runs = [
        TrialRun(
            name=FROZEN_BENCHMARK.name,
            family="frozen_benchmark_not_a_candidate_trial",
            specification=asdict(FROZEN_BENCHMARK),
            targets=benchmark_target,
            backtest=benchmark_backtest,
            summary=benchmark_summary,
        )
    ]

    for spec in multispeed_specs:
        target = multispeed_targets(benchmark_close, spec)
        backtest = backtest_next_period(
            targets=target.targets,
            close=benchmark_close,
            open_prices=benchmark_open,
            cost_bps_per_side=spec.cost_bps_per_side,
            cash_returns=cash_returns,
        )
        runs.append(
            TrialRun(
                name=spec.name,
                family="multi_speed_time_series_etf",
                specification=asdict(spec),
                targets=target,
                backtest=backtest,
                summary=performance_summary(backtest, benchmark_net=benchmark_net),
            )
        )

    if sector_history is not None:
        if market_ticker not in prices.close.columns:
            raise ValueError(f"market ticker missing from prices: {market_ticker}")
        selected = stock_tickers or infer_sector_tickers(sector_history)
        selected = sorted(
            ticker
            for ticker in {ticker.upper().strip() for ticker in selected}
            if ticker in prices.close.columns and ticker != market_ticker
        )
        if not selected:
            raise ValueError("no stock tickers overlap sector history and price data")
        stock_close = prices.close.loc[:, selected]
        stock_open = prices.open.reindex(columns=selected) if prices.open is not None else None
        target = cross_sectional_targets(
            stock_close=stock_close,
            market_close=prices.close[market_ticker],
            sector_history=sector_history,
            spec=cross_sectional_spec,
        )
        backtest = backtest_next_period(
            targets=target.targets,
            close=stock_close,
            open_prices=stock_open,
            cost_bps_per_side=cross_sectional_spec.cost_bps_per_side,
            cash_returns=cash_returns,
        )
        runs.append(
            TrialRun(
                name=cross_sectional_spec.name,
                family="stock_cross_sectional_residual_trend_separate_family",
                specification=asdict(cross_sectional_spec),
                targets=target,
                backtest=backtest,
                summary=performance_summary(backtest, benchmark_net=benchmark_net),
            )
        )
    return runs


def _input_metadata(prices: PriceData) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "source": prices.source,
        "price_basis_declared": "adjusted",
        "rows": len(prices.close),
        "tickers": int(prices.close.shape[1]),
        "start": str(prices.close.index.min().date()),
        "end": str(prices.close.index.max().date()),
        "has_open": prices.open is not None,
    }
    if prices.source:
        source = Path(prices.source)
        if source.is_file():
            stat = source.stat()
            metadata.update(
                {
                    "size_bytes": int(stat.st_size),
                    "modified_utc": datetime.fromtimestamp(
                        stat.st_mtime, tz=timezone.utc
                    ).isoformat(),
                }
            )
    return metadata


def _benchmark_fingerprint() -> str:
    payload = json.dumps(asdict(FROZEN_BENCHMARK), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_research_artifacts(
    output_dir: str | Path,
    prices: PriceData,
    runs: list[TrialRun],
    stock_family_requested: bool,
) -> Path:
    """Write an append-free, self-contained research bundle under artifacts/."""
    output = _prepare_output_dir(output_dir)
    returns_dir = output / "monthly_returns"
    targets_dir = output / "targets"
    returns_dir.mkdir()
    targets_dir.mkdir()

    summary_rows: list[dict[str, Any]] = []
    trial_records: list[dict[str, Any]] = []
    for run in runs:
        run.backtest.monthly.to_csv(returns_dir / f"{run.name}.csv", index_label="month")
        run.targets.targets.to_parquet(targets_dir / f"{run.name}.parquet")
        row = {"name": run.name, "family": run.family, **run.summary}
        summary_rows.append(row)
        trial_records.append(
            {
                "name": run.name,
                "family": run.family,
                "specification": run.specification,
                "summary": run.summary,
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output / "summary.csv", index=False)

    candidate_runs = [run for run in runs if "not_a_candidate" not in run.family]
    multi_fields = _spec_parameter_fields(PREREGISTERED_MULTISPEED_SPECS[0])
    stock_fields = _spec_parameter_fields(PREREGISTERED_CROSS_SECTIONAL_SPEC)
    manifest = {
        "schema_version": "trend_v2_research_bundle.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "no_order": True,
        "production_writes": False,
        "automatic_promotion": False,
        "input": _input_metadata(prices),
        "execution_semantics": (
            "next_open_to_next_open" if prices.open is not None else "next_close_to_next_close"
        ),
        "frozen_benchmark": {
            "name": FROZEN_BENCHMARK.name,
            "fingerprint_sha256": _benchmark_fingerprint(),
            "candidate_trial": False,
            "book_state_overlay": (
                "production fragility gate held outside the price-only family comparison; "
                "replay unchanged at portfolio-increment gate"
            ),
        },
        "trial_accounting": {
            "preregistered_multispeed_trials": len(PREREGISTERED_MULTISPEED_SPECS),
            "preregistered_stock_family_trials": 1,
            "stock_family_requested": bool(stock_family_requested),
            "candidate_trials_executed": len(candidate_runs),
            "total_rows_including_benchmark": len(runs),
            "multispeed_parameter_field_count": len(multi_fields),
            "multispeed_parameter_fields": multi_fields,
            "stock_parameter_field_count": len(stock_fields),
            "stock_parameter_fields": stock_fields,
            "warning": "Every unreported rerun, threshold, universe, and cost variant also counts as a trial.",
        },
        "artifacts": {
            "summary": "summary.csv",
            "trial_details": "trial_details.json",
            "monthly_returns": "monthly_returns/",
            "targets": "targets/",
            "support_note": "support_note.md",
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, allow_nan=False), encoding="utf-8"
    )
    (output / "trial_details.json").write_text(
        json.dumps(_json_safe(trial_records), indent=2, allow_nan=False), encoding="utf-8"
    )

    benchmark_row = next(row for row in summary_rows if "not_a_candidate" in row["family"])
    best = max(
        (row for row in summary_rows if "not_a_candidate" not in row["family"]),
        key=lambda row: float(row["net_sharpe"])
        if row.get("net_sharpe") is not None and np.isfinite(row["net_sharpe"])
        else -np.inf,
        default=None,
    )
    note_lines = [
        "# Trend V2 research support note",
        "",
        "Research-only output. Nothing here changes the production sleeve, stages orders, or promotes a rule.",
        "",
        f"- Frozen benchmark net Sharpe: {benchmark_row['net_sharpe']:.3f}",
        f"- Candidate trials executed: {len(candidate_runs)}",
        f"- Execution: {manifest['execution_semantics']}",
    ]
    if best is not None:
        note_lines.append(
            f"- Highest full-sample candidate net Sharpe: {best['name']} ({best['net_sharpe']:.3f})"
        )
    note_lines.extend(
        [
            "",
            "The highest in-sample row is not a selection rule. Apply the preregistered holdout, stability, cost-stress, concentration, and portfolio-increment gates in research/trend_v2/PREREGISTRATION.md before any shadow discussion.",
            "",
        ]
    )
    (output / "support_note.md").write_text("\n".join(note_lines), encoding="utf-8")
    return output
