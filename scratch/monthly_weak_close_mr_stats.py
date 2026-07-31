"""Significance + robustness follow-up for monthly_weak_close_mr.py."""
from __future__ import annotations

import numpy as np
import pandas as pd

from monthly_weak_close_mr import (
    load_data, month_signals, run_trades, summarize, baseline_5d,
)


def clustered_t(t: pd.DataFrame) -> tuple[float, int]:
    """t-stat on month-cluster mean returns (SPY/QQQ same-month = 1 obs)."""
    t = t.copy()
    t["ym"] = pd.to_datetime(t["sig_day"]).dt.to_period("M")
    cl = t.groupby("ym")["ret"].mean()
    return float(cl.mean() / (cl.std(ddof=1) / np.sqrt(len(cl)))), len(cl)


def excess_t(t: pd.DataFrame, data: dict, hold: int) -> float:
    """t-stat of trade returns minus the unconditional same-hold mean."""
    base = baseline_5d(data, hold)
    x = t["ret"] - base
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


def main() -> None:
    data = load_data()
    thresh = 0.15

    print("=== Significance (t1open, thresh=0.15, tgt=2 ATR) by hold ===")
    rows = []
    for hold in [5, 10, 21]:
        agg = []
        for tk, df in data.items():
            agg += run_trades(df, month_signals(df, thresh), "t1open", hold, 2.0)
        t = pd.DataFrame(agg)
        ct, ncl = clustered_t(t)
        rows.append({
            "hold": hold, "N": len(t), "clusters": ncl,
            "avg_ret%": round(100 * t.ret.mean(), 3),
            "baseline%": round(100 * baseline_5d(data, hold), 3),
            "t_clustered": round(ct, 2),
            "t_excess": round(excess_t(t, data, hold), 2),
        })
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n=== Equities only (SPY+QQQ) vs TLT, close entry, hold=5, tgt=2 ===")
    for label, tks in [("SPY+QQQ", ["SPY", "QQQ"]), ("TLT", ["TLT"])]:
        agg, n_sigs = [], 0
        for tk in tks:
            sigs = month_signals(data[tk], thresh)
            n_sigs += len(sigs)
            agg += run_trades(data[tk], sigs, "close", 5, 2.0)
        t = pd.DataFrame(agg)
        ct, ncl = clustered_t(t)
        s = summarize(agg, label, n_sigs)
        s["t_clustered"] = round(ct, 2)
        print(pd.DataFrame([s]).to_string(index=False))

    print("\n=== Close entry, no target, hold=21 (ride the full next month) ===")
    for label, tks in [("SPY+QQQ", ["SPY", "QQQ"]), ("all3", list(data))]:
        agg, n_sigs = [], 0
        for tk in tks:
            sigs = month_signals(data[tk], thresh)
            n_sigs += len(sigs)
            agg += run_trades(data[tk], sigs, "close", 21, None)
        t = pd.DataFrame(agg)
        ct, ncl = clustered_t(t)
        s = summarize(agg, label, n_sigs)
        s["t_clustered"] = round(ct, 2)
        print(pd.DataFrame([s]).to_string(index=False))

    print("\n=== Decade split (close, hold=5, tgt=2, SPY+QQQ) ===")
    agg = []
    for tk in ["SPY", "QQQ"]:
        agg += run_trades(data[tk], month_signals(data[tk], thresh), "close", 5, 2.0)
    t = pd.DataFrame(agg)
    t["era"] = np.where(pd.to_datetime(t.sig_day).dt.year < 2013, "2000-2012", "2013-2026")
    for era, sub in t.groupby("era"):
        ct, ncl = clustered_t(sub)
        print(f"{era}: N={len(sub)} avg={100*sub.ret.mean():+.2f}% "
              f"win={100*(sub.ret>0).mean():.0f}% t_cl={ct:.2f}")

    print("\n=== Limit entry unfilled-signal check (thresh=0.15, hold=5, tgt=2) ===")
    # what did the signals the limit never filled go on to do (t1open basis)?
    filled_days, all_tr = set(), []
    for tk, df in data.items():
        sigs = month_signals(df, thresh)
        lim = run_trades(df, sigs, "limit", 5, 2.0)
        filled_days |= {(tk, x["sig_day"]) for x in lim}
        for x in run_trades(df, sigs, "t1open", 5, 2.0):
            x["ticker"] = tk
            all_tr.append(x)
    t = pd.DataFrame(all_tr)
    t["filled"] = t.apply(lambda r: (r.ticker, r.sig_day) in filled_days, axis=1)
    for f, sub in t.groupby("filled"):
        print(f"limit {'filled' if f else 'MISSED'}: N={len(sub)} "
              f"t1open avg={100*sub.ret.mean():+.2f}% win={100*(sub.ret>0).mean():.0f}%")


if __name__ == "__main__":
    main()
