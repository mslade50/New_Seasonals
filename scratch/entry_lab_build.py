"""
entry_lab_build.py — RESEARCH-ONLY one-off: entry-rule what-if sweeps over the
full production book, emitted as a static payload for the private site's Entry
Lab page (site/research/entry_lab.json).

Method (mirrors scripts/build_trade_ledger.py):
  precompute_all_indicators + generate_candidates_fast run ONCE on the prod
  book (entry rules are consumed in the PROCESS stage, not candidate
  generation), then process_signals_fast is re-run per variant over deep-copied
  STRATEGY_BOOK execution/settings dicts. The engine is never patched.

Sweeps (9 process runs total: 1 baseline + 4 offsets + 3 fill windows + 1 spare
budgeted but unused):
  1. Persistent-limit offset — the engine parses the offset OUT OF THE
     entry_type STRING ('0.25' / '0.75' / '1 ATR' / default 0.5), so only
     {0.25, 0.5, 0.75, 1.0} ATR are representable. The requested 0.0 / 0.1 /
     0.4 / 0.6 points are dropped (see dropped_dimensions).
  2. OLV fill_window_days in {2, 3, 5, 10} (engine-read execution config;
     3 = prod, reused from the baseline run).

Run:  python scratch/entry_lab_build.py   (several minutes — full-book precompute)
"""
import copy
import datetime
import json
import os
import sys
import time

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("OVERFLOW_UNIVERSE_ACTIVE", "0")

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

import data_provider
from strategy_config import ACCOUNT_VALUE
from pages.strat_backtester import (
    load_seasonal_map,
    load_atr_seasonal_map,
    precompute_all_indicators,
    generate_candidates_fast,
    process_signals_fast,
)
from daily_portfolio_report import build_full_strategy_book

OUT_JSON = os.path.join(_ROOT, "site", "research", "entry_lab.json")
DATA_START = "2000-01-01"
BT_START = datetime.date(2003, 1, 1)

OFFSET_POINTS = [0.25, 0.5, 0.75, 1.0]
OFFSET_STRINGS = {
    0.25: "Limit Order -0.25 ATR (Persistent)",
    0.5: "Limit Order -0.5 ATR (Persistent)",
    0.75: "Limit Order -0.75 ATR (Persistent)",
    1.0: "Limit Order -1 ATR (Persistent)",
}
FILL_WINDOW_POINTS = [2, 3, 5, 10]
OLV_NAME = "Oversold Low Volume"


def _prod_offset(entry_type):
    """Mirror the engine's substring parse (strat_backtester ~line 1851)."""
    if "0.25" in entry_type:
        return 0.25
    if "1 ATR" in entry_type or "1.0 ATR" in entry_type:
        return 1.0
    if "0.75" in entry_type:
        return 0.75
    return 0.5


def _pf(r):
    pos = r[r > 0].sum()
    neg = r[r < 0].sum()
    if neg == 0:
        return None
    return float(pos / abs(neg))


def _round(v, d=4):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if np.isnan(f) or np.isinf(f):
        return None
    return round(f, d)


def _prep(sig):
    if sig.empty:
        return sig
    sig = sig.copy()
    sig["R"] = sig["PnL"] / sig["Risk $"].replace(0, np.nan)
    return sig


def _strategy_row(name, g, n_signals):
    r = g["R"].dropna() if len(g) else pd.Series(dtype=float)
    return {
        "strategy": name,
        "n_signals": int(n_signals),
        "n": int(len(g)),
        "fill_rate": _round(len(g) / n_signals, 3) if n_signals else None,
        "tot_r": _round(r.sum(), 2),
        "avg_r": _round(r.mean(), 3),
        "win_pct": _round((g["PnL"] > 0).mean() * 100, 1) if len(g) else None,
        "pf": _round(_pf(r), 2),
        "pnl_flat": _round(g["PnL"].sum(), 0) if len(g) else 0.0,
    }


def _book_row(sig):
    r = sig["R"].dropna() if len(sig) else pd.Series(dtype=float)
    return {
        "n": int(len(sig)),
        "tot_r": _round(r.sum(), 2),
        "avg_r": _round(r.mean(), 3),
        "win_pct": _round((sig["PnL"] > 0).mean() * 100, 1) if len(sig) else None,
        "pnl_flat": _round(sig["PnL"].sum(), 0) if len(sig) else 0.0,
    }


def summarize(sig, names, sig_counts):
    """Per-strategy rows (tiers merged by name) restricted to `names`, plus
    whole-book totals."""
    sig = _prep(sig)
    rows = []
    for name in names:
        g = sig[sig["Strategy"] == name] if not sig.empty else pd.DataFrame()
        rows.append(_strategy_row(name, g, sig_counts.get(name, 0)))
    return rows, _book_row(sig)


def main():
    t0 = time.time()
    print("=" * 70)
    print("ENTRY LAB — entry-rule what-if sweeps (research only)")
    print("=" * 70)

    base_book = copy.deepcopy(build_full_strategy_book())
    persistent = [s["name"] for s in base_book
                  if "PERSISTENT" in s["settings"]["entry_type"].upper()]
    persistent = sorted(set(persistent))
    prod_offsets = {}
    for s in base_book:
        if s["name"] in persistent:
            prod_offsets.setdefault(s["name"], _prod_offset(s["settings"]["entry_type"]))
    print(f"Persistent-limit strategies: {persistent}")
    print(f"Prod offsets: {prod_offsets}")

    all_names = sorted({s["name"] for s in base_book})

    sznl_map = load_seasonal_map()
    atr_sznl_map = load_atr_seasonal_map()
    if not atr_sznl_map:
        print("WARNING: atr_seasonal_ranks.parquet missing — ATR-seasonal strategies under-fire.")

    tickers = set()
    for s in base_book:
        tickers.update(s["universe_tickers"])
    tickers.update(["SPY", "^VIX"])
    print(f"Loading {len(tickers)} tickers from master_prices ...")
    md = data_provider.get_history(sorted(tickers), start=DATA_START)

    vix_df = md.get("^VIX")
    vix_series = None
    if vix_df is not None and not vix_df.empty:
        vd = vix_df.copy()
        if isinstance(vd.columns, pd.MultiIndex):
            vd.columns = vd.columns.get_level_values(0)
        vd.columns = [c.capitalize() for c in vd.columns]
        vix_series = vd["Close"]

    print("Precomputing indicators (the slow part, runs once) ...")
    processed = precompute_all_indicators(md, base_book, sznl_map, vix_series, atr_sznl_map)
    print(f"  done in {time.time() - t0:.0f}s")

    candidates, signal_data = generate_candidates_fast(processed, base_book, sznl_map, BT_START)
    print(f"  {len(candidates)} candidate signal-dates")

    sig_counts = {}
    for c in candidates:
        name = base_book[c[3]]["name"]
        sig_counts[name] = sig_counts.get(name, 0) + 1

    def run(book, tag):
        t = time.time()
        sig = process_signals_fast(
            candidates, signal_data, processed, book, ACCOUNT_VALUE,
            cap_bps=250, overflow_active=True, flat_sizing=True,
        )
        print(f"  [{tag}] {len(sig)} trades in {time.time() - t:.0f}s")
        return sig

    # --- baseline (prod config) ---
    print("\nRun 1/8: baseline (prod config)")
    sig_base = run(base_book, "baseline")
    base_all_rows, base_book_row = summarize(sig_base, all_names, sig_counts)

    # --- sweep 1: persistent-limit offset ---
    run_i = 1
    offset_points = []
    for v in OFFSET_POINTS:
        run_i += 1
        print(f"\nRun {run_i}/8: persistent limit offset = {v} ATR")
        book = copy.deepcopy(base_book)
        for s in book:
            if s["name"] in persistent:
                s["settings"]["entry_type"] = OFFSET_STRINGS[v]
        sig = run(book, f"offset {v}")
        rows, book_row = summarize(sig, persistent, sig_counts)
        offset_points.append({"value": v, "per_strategy": rows, "book": book_row})

    # --- sweep 2: OLV fill window ---
    fw_points = []
    for v in FILL_WINDOW_POINTS:
        if v == 3:  # prod — reuse baseline
            rows, book_row = summarize(sig_base, [OLV_NAME], sig_counts)
            fw_points.append({"value": v, "per_strategy": rows, "book": book_row,
                              "reused_baseline": True})
            continue
        run_i += 1
        print(f"\nRun {run_i}/8: OLV fill_window_days = {v}")
        book = copy.deepcopy(base_book)
        for s in book:
            if s["name"] == OLV_NAME:
                s["execution"]["fill_window_days"] = int(v)
        sig = run(book, f"olv fw {v}")
        rows, book_row = summarize(sig, [OLV_NAME], sig_counts)
        fw_points.append({"value": v, "per_strategy": rows, "book": book_row})

    out = {
        "computed_asof": str(datetime.date.today()),
        "bt_start": str(BT_START),
        "basis": "flat_750k",
        "engine": ("process_signals_fast over build_full_strategy_book() "
                   "(liquid + overflow passes, tiers merged by name), cap_bps=250, "
                   "overflow_active=True, flat_sizing=True; candidates generated once "
                   "with prod filters and reused across variants"),
        "baseline": {"per_strategy": base_all_rows, "book": base_book_row},
        "sweeps": [
            {
                "dimension": "persistent_limit_offset_atr",
                "label": "Persistent GTC limit offset (ATR below signal close)",
                "strategy_scope": persistent,
                "prod_values": prod_offsets,
                "points": offset_points,
                "notes": ("All 6 persistent-limit strategies moved to the same offset "
                          "simultaneously; per-strategy rows isolate each. Deeper limits "
                          "= fewer fills but better prices — read fill_rate with avg_r."),
            },
            {
                "dimension": "olv_fill_window_days",
                "label": "OLV entry-order live window (trading days)",
                "strategy_scope": [OLV_NAME],
                "prod_values": {OLV_NAME: 3},
                "points": fw_points,
                "notes": "Prod = 3 (2026-06-24 change). Point 3 reuses the baseline run.",
            },
        ],
        "dropped_dimensions": [
            "Offsets 0.0 / 0.1 / 0.4 / 0.6 ATR: the engine has NO numeric offset config — "
            "it parses the offset from the entry_type STRING and only recognizes "
            "0.25 / 0.75 / '1 ATR' / default-0.5 (pages/strat_backtester.py ~1851). "
            "Unrecognized values silently fall back to 0.5, so those points would be lies. "
            "Engine not patched per ground rules.",
            "T+1 Open vs persistent-limit entry-style flip: representable via entry_type "
            "but out of the 8-run budget; a candidate for a follow-up run.",
        ],
        "caveats": [
            "RESEARCH ONLY — one-off local computation, not rebuilt nightly, nothing here stages orders.",
            "Candidates were generated ONCE with prod filters; only fill/exit processing is re-run per variant. Valid because entry rules are process-stage config, but any variant that would have changed SIGNAL generation is out of scope.",
            "fill_rate = executed trades / candidate signals. The denominator includes signals dropped by non-entry gates (OVS gap-skip and earnings blackout, OLV sector gate, caps), so absolute levels understate pure limit-fill probability; COMPARISONS across points are clean because the denominator is fixed.",
            "Liquid + overflow tiers are merged per strategy name (the site ledger splits them).",
            "Flat $750k sizing throughout; R stats are sizing-invariant.",
            "In-sample parameter sweep — the prod values were themselves chosen on this history. Beating prod in a cell here is a candidate for study, not evidence.",
            "The offset sweep moves all 6 persistent strategies together; book totals at each point include cross-strategy interactions (caps, MTM equity, exposure).",
        ],
    }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, separators=(",", ":"), allow_nan=False)
    print(f"\nWrote {OUT_JSON} ({os.path.getsize(OUT_JSON) / 1024:.0f} KB)")
    print(f"Total runtime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
