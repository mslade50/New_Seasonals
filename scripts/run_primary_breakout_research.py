"""Run the research-only primary-stock breakout portfolio study.

Example:
    python scripts/run_primary_breakout_research.py \
      --prices C:/path/to/data/master_prices.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.primary_breakout import (
    BacktestConfig,
    build_stock_universe,
    load_price_panel,
    monthly_block_bootstrap_sharpe,
    period_metrics,
    run_backtest,
)
from research.primary_breakout.report import render_report
from strategy_config import (
    ACCOUNT_VALUE,
    LIQUID_PLUS_COMMODITIES,
    OLV_CAP_EXEMPT_ETFS,
)

PREREG_PATH = REPO_ROOT / "research" / "primary_breakout" / "PREREGISTRATION.md"
IMPLEMENTATION_PATHS = (
    REPO_ROOT / "research" / "primary_breakout" / "__init__.py",
    REPO_ROOT / "research" / "primary_breakout" / "engine.py",
    REPO_ROOT / "research" / "primary_breakout" / "report.py",
    Path(__file__).resolve(),
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            [
                "git",
                "-c",
                f"safe.directory={REPO_ROOT.as_posix()}",
                "rev-parse",
                "HEAD",
            ],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _resolve_output_dir(requested: str | None, run_id: str) -> Path:
    artifacts_root = (REPO_ROOT / "artifacts").resolve()
    output = Path(requested).resolve() if requested else artifacts_root / "research" / "primary_breakout" / run_id
    try:
        output.relative_to(artifacts_root)
    except ValueError as exc:
        raise ValueError(f"output must stay under {artifacts_root}") from exc
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    return output


def _period_table(primary) -> pd.DataFrame:
    definitions = [
        ("2000s", "2000-01-03", "2009-12-31"),
        ("2010s", "2010-01-01", "2019-12-31"),
        ("2020+", "2020-01-01", None),
        ("2015+ holdout", "2015-01-01", None),
    ]
    rows = []
    for label, start, end in definitions:
        metrics = period_metrics(primary, start, end)
        rows.append({"Period": label, "Start": start, "End": end or primary.metrics["end"], **metrics})
    return pd.DataFrame(rows)


def _robustness_configs(primary: BacktestConfig) -> list[tuple[str, BacktestConfig, bool]]:
    return [
        ("Primary: 100d / CK 3.0 / 5 bps", primary, True),
        ("Cost: 0 bps", replace(primary, cost_bps=0.0), False),
        ("Cost: 10 bps", replace(primary, cost_bps=10.0), False),
        ("Cost: 20 bps", replace(primary, cost_bps=20.0), False),
        (
            "Breakout: 80d",
            replace(primary, breakout_window=80, roc_window=80),
            False,
        ),
        (
            "Breakout: 120d",
            replace(primary, breakout_window=120, roc_window=120),
            False,
        ),
        ("CK multiplier: 2.5", replace(primary, ck_atr_multiple=2.5), False),
        ("CK multiplier: 3.5", replace(primary, ck_atr_multiple=3.5), False),
    ]


def _robustness_table(panel, calendar, universe, primary_config, quality):
    rows = []
    primary_result = None
    for label, config, is_primary in _robustness_configs(primary_config):
        print(f"Running {label}...", flush=True)
        result = run_backtest(panel, calendar, universe, config, data_quality=quality)
        if is_primary:
            primary_result = result
        rows.append(
            {
                "Variant": label,
                "Primary": is_primary,
                "BreakoutWindow": config.breakout_window,
                "ROCWindow": config.roc_window,
                "CKATRPeriod": config.ck_atr_period,
                "CKMultiplier": config.ck_atr_multiple,
                "CKStopPeriod": config.ck_stop_period,
                "CostBps": config.cost_bps,
                "CAGR": result.metrics["cagr"],
                "Sharpe": result.metrics["sharpe"],
                "MaxDD": result.metrics["max_drawdown"],
                "Trades": result.metrics["trades"],
                "AverageGross": result.metrics["average_gross"],
                "AveragePositions": result.metrics["average_positions"],
                "AverageR": result.metrics["average_r"],
            }
        )
    if primary_result is None:
        raise RuntimeError("primary result was not produced")
    return primary_result, pd.DataFrame(rows)


def _decision_gates(primary, robustness: pd.DataFrame, periods: pd.DataFrame) -> dict[str, bool]:
    regime_rows = periods[periods["Period"].isin(["2000s", "2010s", "2020+"])]
    cost20 = robustness.loc[robustness["Variant"] == "Cost: 20 bps"].iloc[0]
    neighbor_names = [
        "Breakout: 80d",
        "Breakout: 120d",
        "CK multiplier: 2.5",
        "CK multiplier: 3.5",
    ]
    neighbors = robustness[robustness["Variant"].isin(neighbor_names)]
    concentration = primary.metrics.get("top5_pnl_concentration", np.nan)
    return {
        "positive_each_major_decade": bool(
            (regime_rows["cagr"] > 0).all() and (regime_rows["sharpe"] > 0).all()
        ),
        "positive_at_20_bps_per_side": bool(cost20["Sharpe"] > 0 and cost20["CAGR"] > 0),
        "all_parameter_neighbors_positive": bool(
            len(neighbors) == len(neighbor_names)
            and (neighbors["Sharpe"] > 0).all()
            and (neighbors["CAGR"] > 0).all()
        ),
        "drawdown_better_than_equal_weight": bool(
            abs(float(primary.metrics["max_drawdown"]))
            < abs(float(primary.metrics["ew_max_drawdown"]))
        ),
        "top5_contribution_below_50pct": bool(
            np.isfinite(concentration) and float(concentration) < 0.50
        ),
    }


def _support_note(primary, gates: dict[str, bool], bootstrap: dict[str, float]) -> str:
    verdict = "ADVANCE TO RAW / POINT-IN-TIME VALIDATION" if all(gates.values()) else "DO NOT ADVANCE YET"
    gate_lines = "\n".join(f"- {'PASS' if value else 'FAIL'} — {name}" for name, value in gates.items())
    return f"""# Primary Breakout Research Support Note

Verdict: **{verdict}**

This bundle is research-only and cannot promote, stage, or place orders.

## Primary result

- Period: {primary.metrics['start']} through {primary.metrics['end']}
- Universe: {len(primary.universe)} current primary stock-like tickers
- Net CAGR: {primary.metrics['cagr']:.4%}
- Sharpe: {primary.metrics['sharpe']:.3f}
- Maximum drawdown: {primary.metrics['max_drawdown']:.4%}
- Trades: {int(primary.metrics['trades']):,}
- Monthly-block Sharpe 90% interval: {bootstrap['sharpe_p05']:.3f} to {bootstrap['sharpe_p95']:.3f}

## Gates

{gate_lines}

## Reliance boundary

The adjusted-cache, static-current-universe result is exploratory. A pass means
only that raw/as-traded bars and point-in-time/dead-name universe evidence are
worth acquiring. It is not a production recommendation.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prices", required=True, help="Explicit local master_prices.parquet path")
    parser.add_argument("--output-dir", help="New directory under this worktree's artifacts root")
    parser.add_argument("--start-date", default="2000-01-03")
    parser.add_argument("--end-date", help="Optional end date; defaults to common latest stock session")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prices_path = Path(args.prices).resolve()
    if not prices_path.is_file():
        raise FileNotFoundError(prices_path)
    universe, excluded = build_stock_universe(LIQUID_PLUS_COMMODITIES, OLV_CAP_EXEMPT_ETFS)
    if len(universe) != 162 or len(excluded) != 35:
        raise ValueError(
            f"frozen universe expectation failed: {len(universe)} stocks / {len(excluded)} exclusions"
        )
    universe_payload = {"stocks": universe, "excluded_instruments": excluded}
    universe_hash = _sha256_json(universe_payload)
    data_hash = _sha256_file(prices_path)
    prereg_hash = _sha256_file(PREREG_PATH)
    implementation_hashes = {
        path.relative_to(REPO_ROOT).as_posix(): _sha256_file(path)
        for path in IMPLEMENTATION_PATHS
    }
    run_id = f"run-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{universe_hash[:8]}"
    output_dir = _resolve_output_dir(args.output_dir, run_id)

    print(f"Loading {len(universe)} primary stocks from {prices_path}...", flush=True)
    panel, calendar, quality = load_price_panel(prices_path, universe, benchmark_ticker="SPY")
    latest = {
        ticker: panel[ticker]["Close"].last_valid_index()
        for ticker in universe
        if ticker in panel and panel[ticker]["Close"].last_valid_index() is not None
    }
    if len(latest) != len(universe):
        missing = sorted(set(universe) - set(latest))
        raise ValueError(f"universe names without any close data: {missing}")
    common_end = min(latest.values())
    end_date = pd.Timestamp(args.end_date).normalize() if args.end_date else common_end
    if end_date > common_end:
        raise ValueError(
            f"requested end {end_date.date()} exceeds common stock coverage {common_end.date()}"
        )
    quality.update(
        {
            "common_stock_coverage_end": common_end.date().isoformat(),
            "last_observed_by_ticker": {
                ticker: value.date().isoformat() for ticker, value in latest.items()
            },
            "universe_hash": universe_hash,
            "price_sha256": data_hash,
            "preregistration_sha256": prereg_hash,
            "implementation_sha256": implementation_hashes,
        }
    )

    primary_config = BacktestConfig(
        start_date=args.start_date,
        end_date=end_date.date().isoformat(),
        initial_equity=float(ACCOUNT_VALUE),
    )
    config_hash = _sha256_json(asdict(primary_config))
    primary, robustness = _robustness_table(
        panel, calendar, universe, primary_config, quality
    )
    periods = _period_table(primary)
    bootstrap = monthly_block_bootstrap_sharpe(primary.equity)
    gates = _decision_gates(primary, robustness, periods)

    # Human artifact plus complete support outputs. The manifest is published
    # last so its presence means the bundle is complete.
    output_dir.mkdir(parents=True, exist_ok=False)
    primary.equity.reset_index().to_csv(output_dir / "equity_curve.csv", index=False)
    primary.trades.to_csv(output_dir / "trades.csv", index=False)
    primary.candidates.to_csv(output_dir / "candidates.csv", index=False)
    primary.yearly_returns.to_csv(output_dir / "yearly_returns.csv", index=False)
    periods.to_csv(output_dir / "period_metrics.csv", index=False)
    robustness.to_csv(output_dir / "robustness.csv", index=False)
    _write_json(output_dir / "universe.json", universe_payload)
    _write_json(output_dir / "metrics.json", primary.metrics)
    _write_json(output_dir / "data_quality.json", primary.data_quality)
    _write_json(output_dir / "bootstrap.json", bootstrap)
    _write_json(output_dir / "decision_gates.json", gates)
    (output_dir / "support_note.md").write_text(
        _support_note(primary, gates, bootstrap), encoding="utf-8"
    )
    render_report(
        primary,
        robustness,
        periods,
        bootstrap,
        gates,
        excluded,
        universe_hash,
        data_hash,
        output_dir / "report.html",
    )

    bundle_files = sorted(path for path in output_dir.iterdir() if path.is_file())
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "production_writes": False,
        "order_writes": False,
        "broker_calls": False,
        "network_calls": False,
        "automatic_promotion": False,
        "git_commit": _git_commit(),
        "prices_path": str(prices_path),
        "price_sha256": data_hash,
        "preregistration_path": str(PREREG_PATH),
        "preregistration_sha256": prereg_hash,
        "universe_sha256": universe_hash,
        "config_sha256": config_hash,
        "implementation_sha256": implementation_hashes,
        "config": asdict(primary_config),
        "gates": gates,
        "verdict": (
            "advance_to_raw_pit_validation" if all(gates.values()) else "do_not_advance"
        ),
        "files": {path.name: _sha256_file(path) for path in bundle_files},
    }
    _write_json(output_dir / "manifest.json", manifest)
    print(json.dumps({"output_dir": str(output_dir), "verdict": manifest["verdict"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
