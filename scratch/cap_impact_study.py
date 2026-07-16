"""Cap impact study (2026-07-16): what the daily risk caps cost.

Replays the full ledger engine (current book, GRM 1.5, flat $750k) under
four cap regimes and measures total return, Sharpe, Sortino, maxDD:

    prod        per-strategy 250 bps + pooled 500L/250S   (live behavior)
    strat-only  per-strategy 250 bps, no pooled caps
    pooled-only no per-strategy cap, pooled 500L/250S
    none        no caps at all

Sortino uses downside deviation of the daily flat PnL:
    dd_ann = sqrt(mean(min(pnl,0)^2)) * sqrt(252);  sortino = ann_pnl / dd_ann

Output: scratch/cap_impact_results.csv + console table.
Run with PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import scripts.build_trade_ledger as btl
from strategy_config import ACCOUNT_VALUE

VARIANTS = [
    ("prod",        dict(cap_bps=250, max_long_risk_bps=500, max_short_risk_bps=250)),
    ("strat-only",  dict(cap_bps=250, max_long_risk_bps=None, max_short_risk_bps=None)),
    ("pooled-only", dict(cap_bps=0,   max_long_risk_bps=500, max_short_risk_bps=250)),
    ("none",        dict(cap_bps=0,   max_long_risk_bps=None, max_short_risk_bps=None)),
]


def metrics(pnl: pd.Series, label):
    pnl = pnl.fillna(0.0)
    equity = ACCOUNT_VALUE + pnl.cumsum()
    dd = equity - equity.cummax()
    ann_pnl = pnl.mean() * 252
    ann_vol = pnl.std() * np.sqrt(252)
    downside = np.sqrt(np.mean(np.minimum(pnl.values, 0.0) ** 2)) * np.sqrt(252)
    return {
        "variant": label,
        "total_pnl": round(float(pnl.sum())),
        "ann_pnl": round(float(ann_pnl)),
        "sharpe": round(float(ann_pnl / ann_vol), 3) if ann_vol > 0 else np.nan,
        "sortino": round(float(ann_pnl / downside), 3) if downside > 0 else np.nan,
        "max_dd": round(float(dd.min())),
        "max_dd_pct_nav": round(float(dd.min() / ACCOUNT_VALUE * 100), 2),
        "worst_day": round(float(pnl.min())),
    }


def main():
    print(f"Cap impact replay (flat ${ACCOUNT_VALUE:,}, current book/GRM)")
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
    print(f"{len(candidates)} candidates — running {len(VARIANTS)} cap variants")

    rows = []
    for label, kw in VARIANTS:
        sig = btl.process_signals_fast(
            candidates, signal_data, processed, full_book, ACCOUNT_VALUE,
            overflow_active=True, flat_sizing=True, **kw)
        pnl = btl.get_daily_mtm_series(sig, md, start_date=btl.BT_START)
        m = metrics(pnl, label)
        m["trades"] = len(sig)
        rows.append(m)
        print(f"  {label}: total ${m['total_pnl']:,}, Sharpe {m['sharpe']}, "
              f"Sortino {m['sortino']}, maxDD {m['max_dd_pct_nav']}% NAV")

    out = pd.DataFrame(rows)
    dest = os.path.join(ROOT, "scratch", "cap_impact_results.csv")
    out.to_csv(dest, index=False)
    print("\n" + out.to_string(index=False))
    print(f"\nWrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
