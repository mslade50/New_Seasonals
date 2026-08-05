"""Dip-buy family trades conditioned on the equity P/C fear state.

Pre-stated design:
  Universe: ledger trades of the dip-buy family — FAMILY4 (Weak Close Decent
    Sznls, SPY QQQ MonFri Reversion, Monday Dip, Indices Oversold Bounce) +
    family-analogy carriers 3x Bear ETF Overbot Fade. (Monthly Weak Close
    shipped 2026-07-31, not yet in the ledger.)
  Condition: pct252 of the 10d-MA equity P/C as of the SIGNAL date (ffill
    limit 3). PRIMARY: fear ON = pct > 85 vs OFF. Secondary: complacency
    pct < 15 bucket.
  Metric: R_Multiple. Inference: date-clustered (family trades co-fire) —
    Welch t on signal-date-level mean R, ON dates vs OFF dates.
  Also reported: per-strategy split, ON-trade year distribution, overlap of
    fear-ON with the incumbent frag dial >= 50 (whose 0.25x band already
    downsizes these strategies).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as sps

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from scratch.putcall_dial_study import load_spy, rolling_pct_rank  # noqa: E402

FAMILY = ["Weak Close Decent Sznls", "SPY QQQ MonFri Reversion", "Monday Dip",
          "Indices Oversold Bounce", "3x Bear ETF Overbot Fade"]


def cluster_stats(df: pd.DataFrame, mask: pd.Series) -> tuple[int, int, float, float]:
    sub = df[mask]
    dates = sub.groupby("sig_date")["R_Multiple"].mean()
    return len(sub), len(dates), sub["R_Multiple"].mean(), dates


def main() -> None:
    pc = pd.read_parquet(os.path.join(ROOT, "data", "cboe_putcall.parquet"))
    equity = pc["equity"].dropna().sort_index()
    spy = load_spy()
    cal = spy.index[(spy.index >= equity.index.min()) & (spy.index <= equity.index.max())]
    eq = equity.reindex(cal).ffill(limit=3)
    pct = rolling_pct_rank(eq.rolling(10, min_periods=10).mean(), 252).dropna()

    led = pd.read_parquet(os.path.join(ROOT, "data", "backtest_trades_full.parquet"))
    fam = led[led["Strategy"].isin(FAMILY)].copy()
    fam["sig_date"] = pd.to_datetime(fam["Signal Date"])
    fam["pc_pct"] = fam["sig_date"].map(pct.reindex(
        pd.DatetimeIndex(sorted(fam["sig_date"].unique()))).ffill(limit=3))
    n_all = len(fam)
    fam = fam.dropna(subset=["pc_pct", "R_Multiple"])
    print(f"family trades: {n_all} total, {len(fam)} with P/C state "
          f"({fam['sig_date'].min().date()} -> {fam['sig_date'].max().date()})")

    fam["state"] = np.where(fam["pc_pct"] > 85, "fear>85",
                    np.where(fam["pc_pct"] < 15, "complacent<15", "mid"))

    print("\n=== family, by P/C state ===")
    rows = []
    for st, g in fam.groupby("state"):
        dates = g.groupby("sig_date")["R_Multiple"].mean()
        rows.append({"state": st, "n_trades": len(g), "n_dates": len(dates),
                     "avgR": g["R_Multiple"].mean(),
                     "medR": g["R_Multiple"].median(),
                     "win%": 100 * (g["R_Multiple"] > 0).mean(),
                     "date_avgR": dates.mean()})
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    on_d = fam[fam["state"] == "fear>85"].groupby("sig_date")["R_Multiple"].mean()
    off_d = fam[fam["state"] != "fear>85"].groupby("sig_date")["R_Multiple"].mean()
    t, p = sps.ttest_ind(on_d, off_d, equal_var=False)
    print(f"\nPRIMARY fear ON vs OFF, date-clustered: ON {len(on_d)} dates "
          f"avg {on_d.mean():.3f} vs OFF {len(off_d)} dates avg {off_d.mean():.3f} "
          f"-> Welch t={t:.2f} p={p:.3f}")
    lo_d = fam[fam["state"] == "complacent<15"].groupby("sig_date")["R_Multiple"].mean()
    md_d = fam[fam["state"] == "mid"].groupby("sig_date")["R_Multiple"].mean()
    t2, p2 = sps.ttest_ind(lo_d, md_d, equal_var=False)
    print(f"SECONDARY complacent vs mid: {lo_d.mean():.3f} vs {md_d.mean():.3f} "
          f"t={t2:.2f} p={p2:.3f}")

    print("\n=== per strategy, fear ON vs rest ===")
    rows = []
    for s, g in fam.groupby("Strategy"):
        on = g[g["state"] == "fear>85"]; off = g[g["state"] != "fear>85"]
        rows.append({"strategy": s, "n_on": len(on), "avgR_on": on["R_Multiple"].mean(),
                     "win_on%": 100 * (on["R_Multiple"] > 0).mean() if len(on) else np.nan,
                     "n_off": len(off), "avgR_off": off["R_Multiple"].mean(),
                     "win_off%": 100 * (off["R_Multiple"] > 0).mean()})
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    on = fam[fam["state"] == "fear>85"]
    print("\nfear-ON trades by year:")
    yr = on.groupby(on["sig_date"].dt.year)["R_Multiple"].agg(["count", "mean"]).round(2)
    print(yr.to_string())
    print("\nleave-one-year-out avgR of fear-ON trades:")
    for y in yr.index:
        rest = on[on["sig_date"].dt.year != y]["R_Multiple"]
        print(f"  drop {y}: {rest.mean():.3f}")

    # overlap with the incumbent frag dial (research recompute)
    frag_p = os.path.join(ROOT, "data", "rd2_fragility_ts.parquet")
    if os.path.exists(frag_p):
        frag = pd.read_parquet(frag_p)
        frag.index = pd.to_datetime(frag.index)
        dial = frag["63d"].rolling(10, min_periods=1).mean()
        on2 = on.copy()
        on2["dial"] = on2["sig_date"].map(dial)
        both = on2.dropna(subset=["dial"])
        hi = both[both["dial"] >= 50]
        print(f"\nfear-ON trades with dial history: {len(both)} "
              f"({both['sig_date'].min().date()}+); dial>=50 (0.25x band): "
              f"{len(hi)} trades, avgR {hi['R_Multiple'].mean():.3f} "
              f"vs dial<50 avgR {both[both['dial'] < 50]['R_Multiple'].mean():.3f}")


if __name__ == "__main__":
    main()
