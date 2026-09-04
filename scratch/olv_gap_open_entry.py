"""OLV entry what-if: limit at T+1 OPEN - 0.25 ATR, placed only on a gap-down
(T+1 open < signal close), vs prod's persistent limit at signal CLOSE - 0.25 ATR
(live T+1..T+3).

Method mirrors scratch/entry_lab_build.py / scripts/build_trade_ledger.py:
precompute_all_indicators runs ONCE; candidates + process are re-run per
variant over deep-copied OLV config dicts. The engine is never patched:
- the gap-down gate is the generic t1_open_filters candidate gate
  (NextOpen < Close + 0*ATR), same mechanism the 3x Leader Gap Fade uses live
- the open-anchored entry is the engine's single-day "Limit (Open +/- 0.25 ATR)"
  path (fills only if Low <= T+1 open - 0.25*ATR, no persistence)

Variants (whole variants only — no marginal-fill decompositions):
  baseline            prod OLV: persistent close-0.25 ATR, fill window T+3
  gate_only           prod order, but only placed when T+1 gaps down
  proposal            gap-down gate + day limit at T+1 open - 0.25 ATR
  open_anchor_nogate  day limit at T+1 open - 0.25 ATR, no gate

Book restricted to OLV (liquid + overflow tiers) for all variants, so the
cap environment is held constant; R stats are sizing-invariant, flat $750k
PnL is indicative.

Run:  python scratch/olv_gap_open_entry.py
"""
import copy
import datetime
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

OLV_NAME = "Oversold Low Volume"
DATA_START = "2000-01-01"
BT_START = datetime.date(2003, 1, 1)
GAP_DOWN_FILTER = [{"reference": "Close", "logic": "<", "atr_offset": 0.0}]
DAY_LIMIT_ENTRY = "Limit (Open +/- 0.25 ATR)"
ERAS = [
    ("2003-2009", "2003-01-01", "2009-12-31"),
    ("2010-2014", "2010-01-01", "2014-12-31"),
    ("2015-2019", "2015-01-01", "2019-12-31"),
    ("2020-2022", "2020-01-01", "2022-12-31"),
    ("2023-now", "2023-01-01", "2099-01-01"),
]


def pf(r):
    pos = r[r > 0].sum()
    neg = r[r < 0].sum()
    return float(pos / abs(neg)) if neg != 0 else float("inf")


def stats_line(sig, n_candidates):
    if sig.empty:
        return dict(n_sig=n_candidates, n=0)
    r = (sig["PnL"] / sig["Risk $"].replace(0, np.nan)).dropna()
    return dict(
        n_sig=n_candidates,
        n=len(sig),
        fill=len(sig) / n_candidates if n_candidates else np.nan,
        win=(sig["PnL"] > 0).mean(),
        avgR=r.mean(),
        medR=r.median(),
        totR=r.sum(),
        pf=pf(r),
        flat=sig["PnL"].sum(),
    )


def print_stats(tag, s):
    if s.get("n", 0) == 0:
        print(f"{tag:<22} sigs={s['n_sig']:>5}  n=0")
        return
    print(f"{tag:<22} sigs={s['n_sig']:>5}  n={s['n']:>4}  fill={s['fill']:>5.1%}  "
          f"win={s['win']:>5.1%}  avgR={s['avgR']:>+.3f}  medR={s['medR']:>+.3f}  "
          f"totR={s['totR']:>+7.1f}  PF={s['pf']:>5.2f}  flat$={s['flat']:>+12,.0f}")


def era_rows(sig, tag):
    out = []
    if sig.empty:
        return out
    dates = pd.to_datetime(sig["Date"])
    for era, a, b in ERAS:
        e = sig[(dates >= a) & (dates <= b)]
        r = (e["PnL"] / e["Risk $"].replace(0, np.nan)).dropna()
        out.append(f"    {era:<10} n={len(e):>4}  win={(e['PnL'] > 0).mean() if len(e) else float('nan'):>5.1%}  "
                   f"avgR={r.mean() if len(r) else float('nan'):>+.3f}  totR={r.sum():>+7.1f}")
    return out


def main():
    t0 = time.time()
    print("=" * 78)
    print("OLV ENTRY WHAT-IF: gap-down-gated limit at T+1 open - 0.25 ATR")
    print("=" * 78)

    full_book = build_full_strategy_book()
    base_book = copy.deepcopy([s for s in full_book if s["name"] == OLV_NAME])
    assert len(base_book) == 2, f"expected OLV liquid+overflow, got {len(base_book)}"
    print(f"OLV book entries: {len(base_book)} "
          f"(universes: {[len(s['universe_tickers']) for s in base_book]})")

    sznl_map = load_seasonal_map()
    atr_sznl_map = load_atr_seasonal_map()

    tickers = set()
    for s in base_book:
        tickers.update(s["universe_tickers"])
    tickers.update(["SPY", "^VIX"])
    print(f"Loading {len(tickers)} tickers ...")
    md = data_provider.get_history(sorted(tickers), start=DATA_START)

    vix_df = md.get("^VIX")
    vix_series = None
    if vix_df is not None and not vix_df.empty:
        vd = vix_df.copy()
        if isinstance(vd.columns, pd.MultiIndex):
            vd.columns = vd.columns.get_level_values(0)
        vd.columns = [c.capitalize() for c in vd.columns]
        vix_series = vd["Close"]

    print("Precomputing indicators (runs once) ...")
    processed = precompute_all_indicators(md, base_book, sznl_map, vix_series, atr_sznl_map)
    print(f"  done in {time.time() - t0:.0f}s")

    def make_book(gate=False, open_anchor=False):
        book = copy.deepcopy(base_book)
        for s in book:
            if gate:
                s["settings"]["use_t1_open_filter"] = True
                s["settings"]["t1_open_filters"] = copy.deepcopy(GAP_DOWN_FILTER)
            if open_anchor:
                s["settings"]["entry_type"] = DAY_LIMIT_ENTRY
        return book

    variants = [
        ("baseline", make_book()),
        ("gate_only", make_book(gate=True)),
        ("proposal", make_book(gate=True, open_anchor=True)),
        ("open_anchor_nogate", make_book(open_anchor=True)),
    ]

    results = {}
    for tag, book in variants:
        t = time.time()
        candidates, signal_data = generate_candidates_fast(processed, book, sznl_map, BT_START)
        sig = process_signals_fast(
            candidates, signal_data, processed, book, ACCOUNT_VALUE,
            cap_bps=250, overflow_active=True, flat_sizing=True,
        )
        results[tag] = (len(candidates), sig)
        print(f"  [{tag}] {len(candidates)} candidates -> {len(sig)} trades "
              f"({time.time() - t:.0f}s)")

    print("\n" + "=" * 78)
    print("WHOLE-VARIANT COMPARISON (R sizing-invariant; flat $750k basis)")
    print("=" * 78)
    for tag, (n_cand, sig) in results.items():
        print_stats(tag, stats_line(sig, n_cand))

    n_base = results["baseline"][0]
    n_gate = results["gate_only"][0]
    print(f"\nGap-down frequency at T+1: {n_gate}/{n_base} = {n_gate / n_base:.1%} of candidates")

    print("\n" + "=" * 78)
    print("ERA SPLIT")
    print("=" * 78)
    for tag, (_, sig) in results.items():
        print(f"  {tag}:")
        for line in era_rows(sig, tag):
            print(line)

    print("\n" + "=" * 78)
    print("EXIT-TYPE MIX")
    print("=" * 78)
    for tag, (_, sig) in results.items():
        if not sig.empty and "Exit Type" in sig.columns:
            print(f"  {tag:<22} {sig['Exit Type'].value_counts().to_dict()}")

    # per-year totR for baseline vs proposal (robustness read)
    print("\n" + "=" * 78)
    print("PER-YEAR totR: baseline vs proposal")
    print("=" * 78)

    def yearly(sig):
        if sig.empty:
            return pd.Series(dtype=float)
        r = sig["PnL"] / sig["Risk $"].replace(0, np.nan)
        return r.groupby(pd.to_datetime(sig["Date"]).dt.year).sum()

    yb, yp = yearly(results["baseline"][1]), yearly(results["proposal"][1])
    yrs = sorted(set(yb.index) | set(yp.index))
    print(f"{'year':>6} {'baseline':>10} {'proposal':>10} {'diff':>8}")
    wins = 0
    for y in yrs:
        b, p = yb.get(y, 0.0), yp.get(y, 0.0)
        wins += p > b
        print(f"{y:>6} {b:>+10.1f} {p:>+10.1f} {p - b:>+8.1f}")
    print(f"\nproposal beats baseline in {wins}/{len(yrs)} years")
    print(f"\nTotal runtime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
