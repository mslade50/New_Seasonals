"""GRM replay study (2026-07-16): the book at GRM 1.0 / 1.25 / 1.5 / 1.75.

GLOBAL_RISK_MULTIPLIER=1.5 is the single largest sizing decision in the
system and shipped 2026-05-27 with no evidence trail (commit 2fa129e), while
0.75x overlays got LOYO/PIT batteries. This replays the full ledger engine
with the GRM-scaled bps keys rescaled to each level while the CAPS STAY
FIXED at prod (per-strategy 250 bps, pooled 500L/250S, flat $750k basis) —
exactly what changing the constant does live, since the caps are not
GRM-scaled. Note risk_multipliers= can't be used for this: it scales the
per-strategy cap along with the size.

Output: scratch/grm_replay_results.csv + console table.
Run with PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python (streamlit import).
"""
import copy
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import scripts.build_trade_ledger as btl
from strategy_config import ACCOUNT_VALUE, GLOBAL_RISK_MULTIPLIER

GRMS = [1.0, 1.25, 1.5, 1.75]
SCALED_KEYS = ("risk_bps", "path1_bps", "path2_bps", "path2_daily_cap_pct")


def rescale_book(book, factor):
    """Rescale exactly the keys strategy_config's GRM loop scales."""
    out = copy.deepcopy(book)
    for s in out:
        exe = s.get("execution", {})
        for k in SCALED_KEYS:
            if k in exe:
                exe[k] = exe[k] * factor
        eo = exe.get("earnings_size_override")
        if eo and "risk_bps" in eo:
            eo["risk_bps"] = eo["risk_bps"] * factor
    return out


def metrics(pnl: pd.Series, label):
    pnl = pnl.fillna(0.0)
    equity = ACCOUNT_VALUE + pnl.cumsum()
    dd = equity - equity.cummax()
    ann_pnl = pnl.mean() * 252
    ann_vol = pnl.std() * np.sqrt(252)
    return {
        "grm": label,
        "total_pnl": round(float(pnl.sum())),
        "ann_pnl": round(float(ann_pnl)),
        "ann_vol": round(float(ann_vol)),
        "sharpe": round(float(ann_pnl / ann_vol), 3) if ann_vol > 0 else np.nan,
        "max_dd": round(float(dd.min())),
        "max_dd_pct_nav": round(float(dd.min() / ACCOUNT_VALUE * 100), 2),
        "ann_over_dd": round(float(ann_pnl / abs(dd.min())), 3) if dd.min() < 0 else np.nan,
        "worst_day": round(float(pnl.min())),
        "best_day": round(float(pnl.max())),
    }


def main():
    print(f"GRM replay: {GRMS} (current {GLOBAL_RISK_MULTIPLIER}), caps fixed 250/500L/250S, flat ${ACCOUNT_VALUE:,}")
    full_book = btl.build_full_strategy_book()
    sznl_map = btl.load_seasonal_map()
    atr_sznl_map = btl.load_atr_seasonal_map()
    all_tickers = set()
    for s in full_book:
        all_tickers.update(s["universe_tickers"])
    all_tickers.update(["SPY", "^VIX"])
    md = btl.load_data(all_tickers)
    vix_df = md.get("^VIX")
    vix_series = None
    if vix_df is not None and not vix_df.empty:
        vd = vix_df.copy()
        if isinstance(vd.columns, pd.MultiIndex):
            vd.columns = vd.columns.get_level_values(0)
        vd.columns = [c.capitalize() for c in vd.columns]
        vix_series = vd["Close"]
    print("Precomputing indicators (once)...")
    processed = btl.precompute_all_indicators(md, full_book, sznl_map, vix_series, atr_sznl_map)
    candidates, signal_data = btl.generate_candidates_fast(processed, full_book, sznl_map, btl.BT_START)
    print(f"{len(candidates)} candidates — running {len(GRMS)} sizing passes")

    rows = []
    for g in GRMS:
        book_g = rescale_book(full_book, g / GLOBAL_RISK_MULTIPLIER)
        sig = btl.process_signals_fast(
            candidates, signal_data, processed, book_g, ACCOUNT_VALUE,
            cap_bps=250, overflow_active=True, flat_sizing=True,
            max_long_risk_bps=btl.POOLED_LONG_CAP_BPS,
            max_short_risk_bps=btl.POOLED_SHORT_CAP_BPS,
        )
        pnl = btl.get_daily_mtm_series(sig, md, start_date=btl.BT_START)
        m = metrics(pnl, g)
        m["trades"] = len(sig)
        rows.append(m)
        print(f"  GRM {g}: {len(sig)} trades, total ${m['total_pnl']:,}, "
              f"Sharpe {m['sharpe']}, maxDD ${m['max_dd']:,} ({m['max_dd_pct_nav']}% NAV)")

    out = pd.DataFrame(rows)
    dest = os.path.join(ROOT, "scratch", "grm_replay_results.csv")
    out.to_csv(dest, index=False)
    print("\n" + out.to_string(index=False))
    print(f"\nWrote {dest}")
    print("\nReading guide: flat-basis Sharpe is scale-invariant EXCEPT where the "
          "fixed caps bind — a Sharpe that decays with GRM means cluster-day "
          "trimming is eating the best days; maxDD%/NAV scaling faster than "
          "linearly means tail days compound. The honest comparison is "
          "ann_over_dd and maxDD%_NAV at equal cap settings.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
