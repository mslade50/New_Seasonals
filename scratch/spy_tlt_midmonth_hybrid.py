"""Hybrid SPY/TLT mid-month divergence: long SPY naked when SPY lags;
long TLT + short SPY (pairs) when TLT lags.

Reports per-trade return on the PRIMARY leg's notional. Pairs hedge sized
two ways: equal notional, and equal ATR-risk (each leg's notional chosen so
one ATR move = same dollars; short notional = long notional x
ATRpct_long/ATRpct_short).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from monthly_weak_close_mr import load_data as _load
from spy_tlt_midmonth_gap import month_frames, run
from spy_tlt_midmonth_gap_excess import tstat


def leg_ret(df, sig_day, exit_day, entry_mode):
    i = df.index.get_loc(sig_day)
    if entry_mode == "t1open":
        if i + 1 >= len(df):
            return None
        e = df["Open"].iloc[i + 1]
    else:
        e = df["Close"].loc[sig_day]
    return df["Close"].loc[exit_day] / e - 1


def build(spy, tlt, k=10, thresh=4.0, entry_mode="close"):
    frames = {"SPY": spy, "TLT": tlt}
    t = run(spy, tlt, k=k, thresh=thresh)
    rows = []
    for _, x in t.iterrows():
        lagf, leadf = frames[x.lag], frames["TLT" if x.lag == "SPY" else "SPY"]
        r_lag = leg_ret(lagf, x.sig_day, x.exit_day, entry_mode)
        r_lead = leg_ret(leadf, x.sig_day, x.exit_day, entry_mode)
        if r_lag is None or r_lead is None:
            continue
        atrp_lag = lagf["ATR"].loc[x.sig_day] / lagf["Close"].loc[x.sig_day]
        atrp_lead = leadf["ATR"].loc[x.sig_day] / leadf["Close"].loc[x.sig_day]
        hedge = atrp_lag / atrp_lead          # equal-ATR-risk short notional
        if x.lag == "SPY":
            r_eq = r_ab = r_lag               # naked long SPY
        else:
            r_eq = r_lag - r_lead             # equal-notional pair
            r_ab = r_lag - hedge * r_lead     # ATR-balanced pair
        rows.append({"month": x.month, "sig_day": x.sig_day, "yr":
                     pd.Timestamp(x.sig_day).year, "lag": x.lag,
                     "r_eq": r_eq, "r_ab": r_ab, "hedge": hedge})
    return pd.DataFrame(rows)


def stats(x, label):
    x = np.asarray(x, dtype=float)
    up, dn = x[x > 0].sum(), -x[x < 0].sum()
    return {"variant": label, "N": len(x),
            "win%": round(100 * (x > 0).mean(), 1),
            "avg%": round(100 * x.mean(), 2),
            "tot%": round(100 * x.sum(), 1),
            "PF": round(up / dn, 2) if dn > 0 else np.inf,
            "worst%": round(100 * x.min(), 2),
            "t": round(tstat(x), 2)}


def main():
    data = _load()
    spy, tlt = month_frames(data)

    for mode in ["close", "t1open"]:
        h = build(spy, tlt, entry_mode=mode)
        print(f"=== Hybrid (k=10, thresh=4, entry={mode}) ===")
        rows = [stats(h.r_eq, "hybrid eq-notional hedge"),
                stats(h.r_ab, "hybrid ATR-balanced hedge"),
                stats(h[h.lag == "SPY"].r_eq, "  SPY naked leg"),
                stats(h[h.lag == "TLT"].r_eq, "  TLT pair leg (eq)"),
                stats(h[h.lag == "TLT"].r_ab, "  TLT pair leg (ATR-bal)")]
        print(pd.DataFrame(rows).to_string(index=False))
        for era in [False, True]:
            sub = h[(h.yr >= 2022) == era]
            lab = "2022+" if era else "2002-2021"
            print(f"  {lab}: eq {100*sub.r_eq.mean():+.2f}% (t={tstat(sub.r_eq):.2f}) "
                  f"ATR-bal {100*sub.r_ab.mean():+.2f}% (t={tstat(sub.r_ab):.2f}) N={len(sub)}")
        if mode == "close":
            print(f"  avg hedge ratio on TLT-pair trades: "
                  f"{h[h.lag=='TLT'].hedge.mean():.2f} "
                  f"(range {h[h.lag=='TLT'].hedge.min():.2f}-{h[h.lag=='TLT'].hedge.max():.2f})")
        print()

    h = build(spy, tlt, entry_mode="t1open")
    print("=== Per-year (t1open, ATR-balanced) ===")
    yr = h.groupby("yr").agg(n=("r_ab", "count"), avg=("r_ab", "mean"), tot=("r_ab", "sum"))
    yr[["avg", "tot"]] = (100 * yr[["avg", "tot"]]).round(2)
    print(yr.to_string())

    print("\n=== Sensitivity thresh x k (t1open, ATR-balanced): avg% (t) / N ===")
    for th in [3.0, 4.0, 5.0]:
        cells = []
        for k in [8, 10, 12, 14]:
            s = build(spy, tlt, k=k, thresh=th, entry_mode="t1open")
            cells.append(f"k{k}: {100*s.r_ab.mean():+.2f} (t={tstat(s.r_ab):.1f}, N={len(s)})")
        print(f" >={th:.0f}: " + "  ".join(cells))


if __name__ == "__main__":
    main()
