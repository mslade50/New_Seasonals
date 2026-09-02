"""Exhaustively regenerate Legend futures candidates and compare a golden CSV.

This is an intentionally slow release check (~6 minutes on the purchased
2016-2026 archive), not part of the fast unit-test suite.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Capture the candidate source tree before importing any production candidate
# module.  The later attestation must match this bootstrap snapshot, which
# prevents a source change during module import or the expensive archive hash
# from binding evidence to code other than the code loaded for this run.
CANDIDATE_SOURCE_LABELS = frozenset(
    {
        "legend_etf/__init__.py",
        "legend_etf/calendar.py",
        "legend_etf/config.py",
        "legend_etf/core.py",
        "legend_etf/databento_source.py",
        "legend_etf/paths.py",
        "legend_etf/reservations.py",
        "legend_etf/storage.py",
        "scripts/prepare_legend_etf_signals.py",
        "scripts/verify_legend_futures_candidate_parity.py",
        "requirements-legend-etf.txt",
    }
)


def _bootstrap_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bootstrap_candidate_source_tree_sha256(root: Path) -> str:
    base = Path(root).resolve()
    rows: list[dict[str, str]] = []
    for label in sorted(CANDIDATE_SOURCE_LABELS):
        source = (base / Path(label)).resolve()
        try:
            source.relative_to(base)
        except ValueError as exc:  # pragma: no cover - constant labels are bounded
            raise RuntimeError(f"candidate source escapes Legend root: {label}") from exc
        if not source.is_file():
            raise FileNotFoundError(f"candidate pipeline source is missing: {source}")
        rows.append({"label": label, "sha256": _bootstrap_file_sha256(source)})
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


EARLY_CANDIDATE_SOURCE_TREE_SHA256 = _bootstrap_candidate_source_tree_sha256(ROOT)

import numpy as np
import pandas as pd

from legend_etf.calendar import _calendar
from legend_etf.config import NY_TZ
from legend_etf.core import evaluate_futures_setup
from legend_etf.reservations import (
    CANDIDATE_PARITY_PROTOCOL,
    CANDIDATE_PIPELINE_FILES,
    candidate_pipeline_attestation,
    file_sha256,
)
from legend_etf.storage import atomic_write_json, content_hash

KEYS = ["root", "setup_date", "entry_date"]
EXPECTED_CANDIDATE_COUNT = 342
RATIO_ATOL = 1e-12
ATR_ATOL = 1e-10


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--historical-engine", type=Path, required=True)
    result.add_argument("--archive-dir", type=Path, required=True)
    result.add_argument("--golden", type=Path, required=True)
    result.add_argument("--start", default="2016-01-01")
    result.add_argument("--end", default="2026-08-31")
    result.add_argument(
        "--evidence",
        type=Path,
        help="Optional JSON evidence artifact written after all assertions pass",
    )
    return result


def _duplicates(frame: pd.DataFrame) -> int:
    return int(frame.duplicated(KEYS, keep=False).sum())


def _archive_manifest(directory: Path) -> tuple[str, list[dict[str, object]]]:
    files = [
        {
            "path": str(path.resolve()),
            "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
            "sha256": file_sha256(path),
        }
        for path in sorted(Path(directory).glob("*.parquet"))
    ]
    return content_hash(files), files


def _file_manifest(path: Path) -> dict[str, object]:
    source = Path(path).resolve()
    stat = source.stat()
    return {
        "path": str(source),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": file_sha256(source),
    }


def _input_manifest(args: argparse.Namespace) -> dict[str, object]:
    archive_hash, archive_files = _archive_manifest(args.archive_dir)
    return {
        "historical_engine": _file_manifest(args.historical_engine),
        "golden": _file_manifest(args.golden),
        "archive": {
            "path": str(args.archive_dir.resolve()),
            "metadata_manifest_sha256": archive_hash,
            "files": archive_files,
        },
    }


def load_engine(path: Path):
    spec = importlib.util.spec_from_file_location(
        "legend_futures_engine_full_parity", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import historical engine: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def full_entry_dates(start: str, end: str) -> list[pd.Timestamp]:
    calendar = _calendar()
    result: list[pd.Timestamp] = []
    for session in calendar.sessions_in_range(pd.Timestamp(start), pd.Timestamp(end)):
        close = pd.Timestamp(calendar.session_close(session)).tz_convert(NY_TZ)
        if (close.hour, close.minute) != (16, 0):
            continue
        stamp = pd.Timestamp(session)
        if stamp.tz is not None:
            stamp = stamp.tz_convert("UTC").tz_localize(None)
        result.append(stamp.normalize())
    return result


def main() -> int:
    candidate_pipeline = candidate_pipeline_attestation(ROOT)
    if CANDIDATE_SOURCE_LABELS != CANDIDATE_PIPELINE_FILES:
        raise RuntimeError("bootstrap candidate source list does not match release policy")
    if (
        candidate_pipeline["source_tree_sha256"]
        != EARLY_CANDIDATE_SOURCE_TREE_SHA256
    ):
        raise RuntimeError("candidate pipeline changed while verifier modules loaded")
    args = parser().parse_args()
    input_manifest = _input_manifest(args)
    engine = load_engine(args.historical_engine)
    expected = pd.read_csv(
        args.golden, parse_dates=["setup_date", "entry_date"]
    )
    entries = full_entry_dates(args.start, args.end)
    all_research: list[dict[str, object]] = []
    all_production: list[dict[str, object]] = []
    started = time.perf_counter()

    for root, instrument in engine.INSTRUMENTS.items():
        root_started = time.perf_counter()
        minutes = engine.load_symbol_minutes(args.archive_dir, instrument.symbol)
        bars15 = engine.build_15_minute_bars(minutes)
        daily = engine.build_daily_sessions(minutes, bars15)
        trusted = pd.Timestamp(instrument.trusted_start)
        qualifying = daily.loc[
            daily.complete_rth
            & daily.setup_rth.fillna(False)
            & daily.trend_ratio.ge(0.75)
        ]
        research_rows: list[dict[str, object]] = []
        for setup_date, setup in qualifying.iterrows():
            entry_date = setup.next_session
            if pd.isna(entry_date) or entry_date not in daily.index:
                continue
            entry_date = pd.Timestamp(entry_date)
            if entry_date < trusted or not bool(daily.loc[entry_date, "complete_rth"]):
                continue
            if engine._spans_contract_change(minutes, setup_date, entry_date):
                continue
            research_rows.append(
                {
                    "root": root,
                    "setup_date": pd.Timestamp(setup_date),
                    "entry_date": entry_date,
                    "direction": int(setup.trend_direction),
                    "ratio": float(setup.trend_ratio),
                    "atr14": float(setup.atr14),
                    "instrument_id": int(setup.instrument_id),
                }
            )
        all_research.extend(research_rows)

        dates = pd.DatetimeIndex(daily.index)
        minutes_utc = minutes.copy()
        minutes_utc.index = minutes_utc.index.tz_convert("UTC")
        maximum_day = minutes.index.max().tz_localize(None).normalize()
        production_rows: list[dict[str, object]] = []
        reasons: dict[str, int] = {}
        prefilter_count = 0
        for entry_date in entries:
            if entry_date > maximum_day:
                break
            position = int(dates.searchsorted(entry_date, side="left")) - 1
            if position < 0:
                continue
            setup_date = pd.Timestamp(dates[position])
            setup = daily.iloc[position]
            if setup_date < trusted:
                continue
            if (
                not bool(setup.complete_rth)
                or not np.isfinite(setup.trend_ratio)
                or float(setup.trend_ratio) < 0.75
                or int(setup.trend_direction) == 0
            ):
                continue
            prefilter_count += 1
            as_of = entry_date.tz_localize(NY_TZ) + pd.Timedelta(
                hours=8, minutes=45
            )
            end = as_of.tz_convert("UTC")
            window = minutes_utc.loc[
                (minutes_utc.index >= end - pd.Timedelta(days=150))
                & (minutes_utc.index < end)
            ]
            setup_result = evaluate_futures_setup(
                window,
                setup_date=setup_date,
                entry_date=entry_date,
                as_of=as_of,
            )
            reasons[setup_result.reason] = reasons.get(setup_result.reason, 0) + 1
            if setup_result.qualifies:
                production_rows.append(
                    {
                        "root": root,
                        "setup_date": setup_date,
                        "entry_date": entry_date,
                        "direction": setup_result.trend_direction,
                        "ratio": setup_result.trend_ratio,
                        "instrument_id": setup_result.instrument_id,
                    }
                )
        all_production.extend(production_rows)
        frozen_root = expected.loc[expected.root.eq(root)]
        frozen_keys = set(zip(frozen_root.setup_date, frozen_root.entry_date))
        production_keys = {
            (row["setup_date"], row["entry_date"]) for row in production_rows
        }
        print(
            f"{root}: research={len(research_rows)} frozen={len(frozen_root)} "
            f"prefilter={prefilter_count} production={len(production_rows)} "
            f"missing={len(frozen_keys - production_keys)} "
            f"extra={len(production_keys - frozen_keys)} reasons={reasons} "
            f"seconds={time.perf_counter() - root_started:.2f}",
            flush=True,
        )
        del minutes, minutes_utc, bars15, daily, qualifying
        gc.collect()

    research = pd.DataFrame(all_research)
    production = pd.DataFrame(all_production)
    duplicate_counts = {
        "golden": _duplicates(expected),
        "research": _duplicates(research),
        "production": _duplicates(production),
    }
    if any(duplicate_counts.values()):
        raise RuntimeError(f"candidate parity contains duplicate keys: {duplicate_counts}")
    counts = {
        "golden": len(expected),
        "research": len(research),
        "production": len(production),
    }
    if set(counts.values()) != {EXPECTED_CANDIDATE_COUNT}:
        raise RuntimeError(
            f"expected exactly {EXPECTED_CANDIDATE_COUNT} candidates: {counts}"
        )
    frozen_keys = set(zip(expected.root, expected.setup_date, expected.entry_date))
    research_keys = set(zip(research.root, research.setup_date, research.entry_date))
    production_keys = set(
        zip(production.root, production.setup_date, production.entry_date)
    )
    if research_keys != frozen_keys or production_keys != frozen_keys:
        raise RuntimeError(
            "candidate parity failed: "
            f"research missing={len(frozen_keys - research_keys)} "
            f"extra={len(research_keys - frozen_keys)}; "
            f"production missing={len(frozen_keys - production_keys)} "
            f"extra={len(production_keys - frozen_keys)}"
        )
    research_joined = expected.merge(
        research, on=KEYS, how="inner", validate="one_to_one"
    )
    research_direction_matches = int(
        (
            research_joined.prior_futures_trend_direction
            == research_joined.direction
        ).sum()
    )
    research_ratio_delta = float(
        (
            research_joined.prior_futures_trend_ratio
            - research_joined.ratio
        ).abs().max()
    )
    research_atr_delta = float(
        (research_joined.futures_atr14 - research_joined.atr14).abs().max()
    )
    research_contract_matches = int(
        (
            research_joined.futures_contract_id
            == research_joined.instrument_id
        ).sum()
    )
    production_joined = expected.merge(
        production, on=KEYS, how="inner", validate="one_to_one"
    )
    production_direction_matches = int(
        (
            production_joined.prior_futures_trend_direction
            == production_joined.direction
        ).sum()
    )
    production_ratio_delta = float(
        (
            production_joined.prior_futures_trend_ratio
            - production_joined.ratio
        ).abs().max()
    )
    production_contract_matches = int(
        (
            production_joined.futures_contract_id
            == production_joined.instrument_id
        ).sum()
    )
    failures: list[str] = []
    for label, value in {
        "research direction": research_direction_matches,
        "research contract": research_contract_matches,
        "production direction": production_direction_matches,
        "production contract": production_contract_matches,
    }.items():
        if value != EXPECTED_CANDIDATE_COUNT:
            failures.append(f"{label}={value}/{EXPECTED_CANDIDATE_COUNT}")
    if research_ratio_delta > RATIO_ATOL:
        failures.append(f"research ratio delta={research_ratio_delta}")
    if production_ratio_delta > RATIO_ATOL:
        failures.append(f"production ratio delta={production_ratio_delta}")
    if research_atr_delta > ATR_ATOL:
        failures.append(f"research ATR delta={research_atr_delta}")
    if failures:
        raise RuntimeError("candidate attribute parity failed: " + "; ".join(failures))

    elapsed = time.perf_counter() - started
    if candidate_pipeline_attestation(ROOT) != candidate_pipeline:
        raise RuntimeError("candidate pipeline source/runtime changed during parity run")
    if _input_manifest(args) != input_manifest:
        raise RuntimeError("parity engine/golden/archive changed during parity run")
    evidence = {
        "protocol": CANDIDATE_PARITY_PROTOCOL,
        "status": "pass",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "inputs": input_manifest,
        "range": {"start": args.start, "end": args.end},
        "candidate_pipeline": candidate_pipeline,
        "full_session_count": len(entries),
        "counts": counts,
        "duplicate_key_rows": duplicate_counts,
        "matches": {
            "research_direction": research_direction_matches,
            "research_contract": research_contract_matches,
            "production_direction": production_direction_matches,
            "production_contract": production_contract_matches,
        },
        "max_deltas": {
            "research_ratio": research_ratio_delta,
            "production_ratio": production_ratio_delta,
            "research_atr14": research_atr_delta,
        },
        "tolerances": {"ratio_atol": RATIO_ATOL, "atr_atol": ATR_ATOL},
        "runtime_seconds": elapsed,
    }
    if args.evidence is not None:
        atomic_write_json(args.evidence, evidence)
    print(
        f"PASS: {len(expected)} candidates across {len(entries)} full sessions; "
        f"research direction={research_direction_matches}/{len(research_joined)} "
        f"contract={research_contract_matches}/{len(research_joined)}; "
        f"production direction={production_direction_matches}/{len(production_joined)} "
        f"contract={production_contract_matches}/{len(production_joined)}; "
        f"max research ratio delta={research_ratio_delta:.3g} "
        f"max production ratio delta={production_ratio_delta:.3g} "
        f"max ATR delta={research_atr_delta:.3g} seconds={elapsed:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
