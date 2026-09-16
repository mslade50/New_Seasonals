"""Replay SPY/QQQ-only setups against the original ETF research rule.

Both evaluators receive the same 20-calendar-day window. This is signal
parity, not a claim that 15-minute bars prove 09:31 broker fills.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
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
        "legend_etf/etf_source.py",
        "research/legend_ema_backtest.py",
        "scripts/verify_legend_etf_candidate_parity.py",
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


import pandas as pd

from legend_etf.calendar import (
    is_full_session,
    previous_session,
    rth_bar_starts,
    session_labels,
)
from legend_etf.etf_source import ETF_SYMBOLS, HISTORY_DAYS, evaluate_etf_setup
from legend_etf.reservations import candidate_pipeline_attestation, file_sha256
from legend_etf.storage import atomic_write_json
from research.legend_ema_backtest import (
    BacktestConfig,
    qualifies_setup,
    session_touch_flags,
)

PROTOCOL = "legend-etf-native-candidate-parity-v1"
START = "2012-01-01"
END = "2026-08-28"


def replay(frame: pd.DataFrame, symbol: str) -> tuple[dict, list]:
    counts = {
        "evaluated": 0,
        "blocked_history": 0,
        "reference": 0,
        "production": 0,
        "mismatches": 0,
        "max_ema_delta": 0.0,
        "max_ratio_delta": 0.0,
    }
    rows = []
    for entry in session_labels(START, END):
        if not is_full_session(entry) or not is_full_session(previous_session(entry)):
            continue
        day = entry.date().isoformat()
        setup = previous_session(entry).date().isoformat()
        start = (pd.Timestamp(day) - pd.Timedelta(days=HISTORY_DAYS)).tz_localize("America/New_York")
        end = pd.Timestamp(f"{setup} 16:00", tz="America/New_York")
        sample = frame.loc[(frame.index >= start) & (frame.index < end)]
        expected = rth_bar_starts(start.date(), setup)
        complete = sample.index.equals(expected) and len(sample) >= 226
        try:
            actual = evaluate_etf_setup(sample, entry_date=day)
        except ValueError:
            if complete:
                raise
            counts["blocked_history"] += 1
            continue
        if not complete:
            raise AssertionError(
                f"Production admitted incomplete history: {symbol} {day}"
            )
        # Invoke the retained independent research's actual qualification function.
        ref = sample.copy()
        ref["ema"] = ref["close"].ewm(span=20, adjust=False).mean()
        ref["ema_prev"] = ref["ema"].shift()
        prior = ref.loc[ref.index.date == pd.Timestamp(setup).date()]
        width = prior["high"].max() - prior["low"].min()
        ratio = (
            abs(prior.iloc[-1]["close"] - prior.iloc[0]["open"]) / width
            if width > 0
            else 0.0
        )
        reference = qualifies_setup(
            pd.Series(
                {
                    "full_session": len(prior) == 26,
                    "bars_seen": len(sample),
                    "trend_ratio": ratio,
                    **session_touch_flags(prior),
                }
            ),
            BacktestConfig(),
        )
        counts["evaluated"] += 1
        counts["reference"] += int(reference)
        counts["production"] += int(actual["qualifies"])
        counts["mismatches"] += int(reference != actual["qualifies"])
        counts["max_ema_delta"] = max(
            counts["max_ema_delta"],
            abs(float(ref.iloc[-1]["ema"]) - actual["initial_ema"]),
        )
        counts["max_ratio_delta"] = max(
            counts["max_ratio_delta"], abs(float(ratio) - actual["trend_ratio"])
        )
        if reference or actual["qualifies"]:
            rows.append(
                {"etf": symbol, **actual, "reference_qualifies": bool(reference)}
            )
    return counts, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    source_before = candidate_pipeline_attestation(ROOT)
    if source_before["source_tree_sha256"] != EARLY_CANDIDATE_SOURCE_TREE_SHA256:
        raise RuntimeError("Candidate source changed during imports")
    counts, inputs, candidates = {}, {}, []
    for symbol in ETF_SYMBOLS:
        path = (args.data_dir / f"{symbol}_15min.parquet").resolve()
        stat = path.stat()
        inputs[symbol] = {
            "path": str(path),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": file_sha256(path),
        }
        frame = pd.read_parquet(path)
        index = pd.DatetimeIndex(pd.to_datetime(frame.pop("ts")))
        frame.index = (
            index.tz_localize("America/New_York")
            if index.tz is None
            else index.tz_convert("America/New_York")
        )
        if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
            raise ValueError(f"{symbol} archive timestamps are ambiguous")
        counts[symbol], rows = replay(frame, symbol)
        candidates.extend(rows)
        print(json.dumps({symbol: counts[symbol]}), flush=True)
    passed = all(
        item["evaluated"] >= 2500
        and item["reference"] > 0
        and item["reference"] == item["production"]
        and item["mismatches"] == 0
        and item["max_ema_delta"] <= 1e-10
        and item["max_ratio_delta"] <= 1e-12
        for item in counts.values()
    )
    if candidate_pipeline_attestation(ROOT) != source_before:
        raise RuntimeError("Candidate source/runtime changed during replay; rerun on the final tree")
    for record in inputs.values():
        path = Path(record["path"])
        if path.stat().st_mtime_ns != record["mtime_ns"] or file_sha256(path) != record["sha256"]:
            raise RuntimeError("Historical inputs changed during replay")
    evidence = {
        "protocol": PROTOCOL,
        "status": "pass" if passed else "fail",
        "range": {"start": START, "end": END},
        "completed_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "counts": counts,
        "inputs": inputs,
        "candidate_pipeline": source_before,
        "runtime_seconds": time.monotonic() - started,
    }
    atomic_write_json(args.output, evidence)
    pd.DataFrame(candidates).to_csv(
        args.output.with_suffix(".candidates.csv"), index=False
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
