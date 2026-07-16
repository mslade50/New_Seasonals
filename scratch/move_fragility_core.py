"""One-shot refactor: move the fragility scoring core out of
pages/risk_dashboard_v2.py into a streamlit-free fragility_core.py, leaving
re-export imports behind so every existing `from pages.risk_dashboard_v2
import X` keeps working (RISK_DIALS_2026-07-16.md A3)."""
import io
import re

PAGE = "pages/risk_dashboard_v2.py"

src = io.open(PAGE, encoding="utf-8").read()
lines = src.splitlines(keepends=True)


def find_line(pred, start=0):
    for i in range(start, len(lines)):
        if pred(lines[i]):
            return i
    raise SystemExit(f"anchor not found from line {start}")


def next_toplevel_def(start):
    return find_line(lambda l: l.startswith("def ") or l.startswith("@"), start)


# Block A: load_horizon_stats .. end of compute_horizon_fragility
a_start = find_line(lambda l: l.startswith("def load_horizon_stats"))
chf = find_line(lambda l: l.startswith("def compute_horizon_fragility"))
a_end = next_toplevel_def(chf + 1)  # first top-level def after it

# Block B: compute_fragility_timeseries .. next top-level def
b_start = find_line(lambda l: l.startswith("def compute_fragility_timeseries"))
b_end = next_toplevel_def(b_start + 1)

block_a = "".join(lines[a_start:a_end])
block_b = "".join(lines[b_start:b_end])

# Sanity: every def we expect is inside the captured blocks
expected = ["load_horizon_stats", "_signal_edge", "_calm_mult_for_streak",
            "_compute_calm_multiplier_scalar", "_compute_calm_multiplier_series",
            "_days_since_last_fire", "_signal_decay_weight",
            "_compute_decay_metadata", "compute_horizon_fragility",
            "compute_fragility_timeseries"]
combined = block_a + block_b
for name in expected:
    assert f"def {name}" in combined, f"missing {name}"
assert "HORIZON_DAYS" in block_a and "CALM_STREAK_THRESHOLDS" in block_a

CORE_HEADER = '''"""fragility_core — the dial scoring core, streamlit-free.

Extracted verbatim from pages/risk_dashboard_v2.py on 2026-07-16
(RISK_DIALS_2026-07-16.md A3). The page re-imports everything here, so both
`from pages.risk_dashboard_v2 import compute_horizon_fragility` and
`from fragility_core import compute_horizon_fragility` resolve to the same
objects. The three consumers of the duplicated scoring pipeline (the page,
daily_risk_report, weekly_market_rundown) all call compute_fragility_bundle.

Behavior contract: functions are MOVED, not modified — the golden values in
tests/test_fragility_core.py lock the scoring math.
"""
from __future__ import annotations

import datetime
import json
import os

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_ROOT, "data")
HORIZON_STATS_PATH = os.path.join(DATA_DIR, "signal_horizon_stats.json")


'''

BUNDLE = '''

def compute_fragility_bundle(signals_ordered, regime_mult, price_ctx,
                             spy_close, ts_write_path=None):
    """The scoring pipeline shared by the page, daily_risk_report and
    weekly_market_rundown (previously three hand-kept copies).

    Returns dict: horizon_stats, h_scores (5d-smoothed latest), h_scores_10d,
    frag_df (raw timeseries). All None-safe when the stats JSON is missing.

    ts_write_path: optional explicit destination for the RAW timeseries
    (rd2_fragility_ts.parquet — research/ML only, NEVER a sizing input).
    Written with vintage metadata; the write is a caller decision now, not a
    side effect buried in a cached compute path.
    """
    horizon_stats = load_horizon_stats()
    h_scores = None
    h_scores_10d = None
    frag_df = None
    if horizon_stats is not None:
        h_scores = compute_horizon_fragility(
            signals_ordered, regime_mult, horizon_stats, price_ctx, spy_close)
        frag_df = compute_fragility_timeseries(
            signals_ordered, spy_close, horizon_stats)
        if frag_df is not None and len(frag_df) >= 1:
            # 5d moving average for dial display (smooths day-to-day noise)
            h_scores = frag_df.rolling(5, min_periods=1).mean().iloc[-1].to_dict()
            h_scores_10d = frag_df.rolling(10, min_periods=1).mean().iloc[-1].to_dict()
        if ts_write_path and frag_df is not None and not frag_df.empty:
            try:
                import pyarrow as pa
                import pyarrow.parquet as pq
                table = pa.Table.from_pandas(frag_df)
                md = dict(table.schema.metadata or {})
                md[b"fragility_basis"] = b"raw_recompute"
                md[b"fragility_generated"] = datetime.datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S").encode()
                md[b"fragility_note"] = (b"full recompute vintage - research/ML "
                                         b"only, never a sizing input")
                pq.write_table(table.replace_schema_metadata(md), ts_write_path)
            except Exception:
                frag_df.to_parquet(ts_write_path)
    return {
        "horizon_stats": horizon_stats,
        "h_scores": h_scores,
        "h_scores_10d": h_scores_10d,
        "frag_df": frag_df,
    }
'''

io.open("fragility_core.py.new", "w", encoding="utf-8", newline="").write(
    CORE_HEADER + block_a + "\n" + block_b + BUNDLE)

REEXPORT = '''# Fragility scoring core moved to fragility_core.py (2026-07-16, A3).
# Re-exported here so existing `from pages.risk_dashboard_v2 import X`
# consumers (daily_risk_report, weekly_market_rundown, scripts, tests)
# keep working unchanged.
from fragility_core import (  # noqa: F401
    HORIZON_STATS_PATH,
    HORIZON_DAYS,
    HORIZON_DECAY_DD,
    CALM_STREAK_THRESHOLDS,
    CALM_STREAK_MULTIPLIERS,
    load_horizon_stats,
    _signal_edge,
    _calm_mult_for_streak,
    _compute_calm_multiplier_scalar,
    _compute_calm_multiplier_series,
    _days_since_last_fire,
    _signal_decay_weight,
    _compute_decay_metadata,
    compute_horizon_fragility,
    compute_fragility_bundle,
)

'''
REEXPORT_B = '''# compute_fragility_timeseries also lives in fragility_core (A3 move).
from fragility_core import compute_fragility_timeseries  # noqa: F401

'''

new_lines = (lines[:a_start] + [REEXPORT] + lines[a_end:b_start]
             + [REEXPORT_B] + lines[b_end:])
io.open(PAGE + ".new", "w", encoding="utf-8", newline="").write("".join(new_lines))
print(f"block A: lines {a_start+1}-{a_end} ({a_end-a_start} lines)")
print(f"block B: lines {b_start+1}-{b_end} ({b_end-b_start} lines)")
print("wrote fragility_core.py.new and", PAGE + ".new — inspect then swap")
