"""Backward-compatibility check: with NO overflow_universe.parquet present, the
overflow tier daily_scan produces must equal the legacy static computation
(CSV_UNIVERSE - LIQUID_PLUS_COMMODITIES). Pure import-level check — does NOT run
the scan, hit the network, R2, or Sheets.
"""
import os
import sys
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

# Stub heavy/broken-in-this-env imports that daily_scan pulls at module load but
# that build_effective_strategy_book does not use. (The local yfinance install
# has a protobuf version mismatch unrelated to these changes.) This lets us
# exercise the pure universe-construction logic without the data/Sheets stack.
for _name in ("yfinance", "gspread"):
    if _name not in sys.modules:
        sys.modules[_name] = types.ModuleType(_name)

import overflow_universe as ou

# Guard: this check is only meaningful when the parquet is absent (fallback path).
assert not os.path.exists(ou.OVERFLOW_UNIVERSE_PATH), (
    f"overflow_universe.parquet present at {ou.OVERFLOW_UNIVERSE_PATH} — "
    "this check expects the fallback (no-parquet) path."
)

import daily_scan as ds
from strategy_config import CSV_UNIVERSE, LIQUID_PLUS_COMMODITIES

static_overflow = sorted(set(CSV_UNIVERSE) - set(LIQUID_PLUS_COMMODITIES))

book = ds.build_effective_strategy_book(scope="overflow")
assert book, "overflow book is empty"
for s in book:
    assert s["_scan_source"] == "Overflow", s["name"]
    got = s["universe_tickers"]
    assert got == static_overflow, (
        f"{s['name']}: overflow universe diverged from legacy static set "
        f"(len got={len(got)} vs static={len(static_overflow)})"
    )

# 'all' scope must still contain a Liquid pass untouched.
book_all = ds.build_effective_strategy_book(scope="all")
liquid = [s for s in book_all if s["_scan_source"] == "Liquid"]
assert liquid, "no Liquid pass in scope=all"

print(f"OK: {len(book)} overflow strategies, each universe == legacy static "
      f"({len(static_overflow)} tickers). Liquid pass intact ({len(liquid)} strats).")
print("BACKWARD-COMPAT OK")
