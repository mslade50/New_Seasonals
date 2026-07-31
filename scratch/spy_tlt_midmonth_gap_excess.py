"""Excess-return decomposition for the SPY/TLT mid-month divergence fade.

Is the edge real convergence, or just the turn-of-month drift both assets
earn from day-k to EOM anyway? For every signal month compare:
  - laggard's hold return vs that TICKER's unconditional day-k->EOM mean
  - the LEADER's return over the same window (control: buy the winner)
  - the long/short spread (laggard minus leader)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from monthly_weak_close_mr import load_data as _load
from spy_tlt_midmonth_gap import month_frames, run


def window_rets(spy, tlt, k=10):
    """Every month's day-k->EOM return per ticker (the unconditional pool)."""
    per = spy.index.to_period("M")
    out = {"SPY": {}, "TLT": {}}
    for p in per.unique():
        days = spy.index[per == p]
        if len(days) < k + 2:
            continue
        for name, df in [("SPY", spy), ("TLT", tlt)]:
            out[name][str(p)] = df["Close"].loc[days[-1]] / df["Close"].loc[days[k - 1]] - 1
    return out


def tstat(x):
    x = np.asarray(x, dtype=float)
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) if len(x) > 2 else np.nan


def main() -> None:
    data = _load()
    spy, tlt = month_frames(data)
    pool = window_rets(spy, tlt)

    for th in [3.0, 4.0, 5.0]:
        t = run(spy, tlt, k=10, thresh=th)
        t["leader"] = np.where(t.lag == "SPY", "TLT", "SPY")
        t["lead_ret"] = [pool[ld][m] for ld, m in zip(t.leader, t.month)]
        t["base"] = [np.mean(list(pool[lg].values())) for lg in t.lag]
        t["excess"] = t.ret - t.base
        t["spread"] = t.ret - t.lead_ret
        print(f"=== k=10, thresh={th:.0f} (N={len(t)}) ===")
        print(f"  laggard avg {100*t.ret.mean():+.2f}%  vs own-ticker baseline "
              f"{100*t.base.mean():+.2f}%  -> excess {100*t.excess.mean():+.2f}% "
              f"(t={tstat(t.excess):.2f}, win {100*(t.excess>0).mean():.0f}%)")
        print(f"  leader control avg {100*t.lead_ret.mean():+.2f}%   "
              f"laggard-minus-leader spread {100*t.spread.mean():+.2f}% "
              f"(t={tstat(t.spread):.2f}, win {100*(t.spread>0).mean():.0f}%)")
        for leg in ["SPY", "TLT"]:
            s = t[t.lag == leg]
            if len(s) > 2:
                print(f"  long-{leg} leg: N={len(s)} excess {100*s.excess.mean():+.2f}% "
                      f"(t={tstat(s.excess):.2f}) spread {100*s.spread.mean():+.2f}% "
                      f"(t={tstat(s.spread):.2f})")
        print()

    # Regime split for the TLT leg: bond bull (<=2021) vs bond bear (2022+)
    t = run(spy, tlt, k=10, thresh=3.0)
    t["base"] = [np.mean(list(pool[lg].values())) for lg in t.lag]
    t["excess"] = t.ret - t.base
    t["yr"] = pd.to_datetime(t.sig_day).dt.year
    print("=== TLT-laggard leg by era (thresh=3) ===")
    for era, sub in t[t.lag == "TLT"].groupby(t.yr >= 2022):
        label = "2022+ (bond bear)" if era else "2002-2021 (bond bull)"
        print(f"  {label}: N={len(sub)} avg {100*sub.ret.mean():+.2f}% "
              f"excess {100*sub.excess.mean():+.2f}% win {100*(sub.ret>0).mean():.0f}%")
    print("\n=== SPY-laggard leg by era (thresh=3) ===")
    for era, sub in t[t.lag == "SPY"].groupby(t.yr >= 2022):
        label = "2022+" if era else "2002-2021"
        print(f"  {label}: N={len(sub)} avg {100*sub.ret.mean():+.2f}% "
              f"excess {100*sub.excess.mean():+.2f}% win {100*(sub.ret>0).mean():.0f}%")


if __name__ == "__main__":
    main()
