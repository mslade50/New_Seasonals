"""E3: "Vol floor into a fragile tape" -- SPY forward 5/10td when VIX<16 AND VIX 5d-rank<=25
AND SPY within 1% of its 52w high.

^VIX is not tradeable and the registry kills puts/VXX, so the only expression is a SHORT
index position. Test both signs; the bar is 'big enough to short SPY after cost'.
Real order: trigger day D close -> ENTER MOC D+1 -> EXIT MOC D+1+h.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _common import *  # noqa: F401,F403

px = load_prices(["SPY", "^VIX"])
spy = px["SPY"]["Close"]
idx = spy.index
vix = px["^VIX"]["Close"].reindex(idx).ffill()

rk5v = pct_rank(vix, 5)
dist = (spy / spy.rolling(252).max() - 1) * 100


def fwd_entry_next(s, h):
    return s.shift(-(h + 1)) / s.shift(-1) - 1.0


F = {h: fwd_entry_next(spy, h) for h in (5, 10, 21)}

cA = vix < 16
cB = rk5v <= 25
cC = dist >= -1.0
full = cA & cB & cC

for H in (5, 10):
    f = F[H]
    valid = rk5v.notna() & dist.notna() & f.notna() & vix.notna()
    d = idx[full & valid]
    ep = declusters(d, H + 1, idx)
    v = f[ep].dropna().values
    print(f"\n########## E3 h={H} ##########")
    print(f"day-level N={len(d)}  episodes N={len(ep)}   "
          f"first={d[0].date() if len(d) else '-'} last={d[-1].date() if len(d) else '-'}")
    show([summarize(f[d].values, "TRIGGER day-level"),
          summarize(v, "TRIGGER episode-level"),
          summarize(f[valid].values, "ctrl A: SPY uncond same window"),
          summarize(f[f.notna()].values, "ctrl B: SPY all-days")],
         f"E3 h={H} vs controls")
    mu_c = np.nanmean(f[valid].values)
    print(f"edge vs same-window control: {100*np.mean(v)-mu_c*100:+.3f}%")
    a, b = v, f[valid].dropna().values
    tw = (a.mean() - b.mean()) / np.sqrt(a.var(ddof=1)/len(a) + b.var(ddof=1)/len(b))
    print(f"Welch t vs control: {tw:+.2f}")
    print(f"bootstrap P(mean<=0) LONG : {bootstrap_p_le0(v):.4f}")
    print(f"bootstrap P(mean<=0) SHORT: {bootstrap_p_le0(-v):.4f}")
    j = int(np.argmax(v)); k = int(np.argmin(v))
    print(f"best {ep[j].date()} {100*v[j]:+.2f}%   worst {ep[k].date()} {100*v[k]:+.2f}%")
    show([summarize(np.delete(v, j), "drop-BEST"), summarize(np.delete(v, k), "drop-WORST")],
         f"E3 h={H} drop-one")
    show(era_split(ep, v, "2018-01-01"), f"E3 h={H} era 2018")
    show(era_split(ep, v, "2013-01-01"), f"E3 h={H} era 2013")
    for lo, hi in [(2000, 2008), (2008, 2013), (2013, 2018), (2018, 2020),
                   (2020, 2023), (2023, 2027)]:
        m = (ep >= pd.Timestamp(f"{lo}-01-01")) & (ep < pd.Timestamp(f"{hi}-01-01"))
        if m.sum():
            ss = summarize(v[m], f"{lo}-{hi}")
            print(f"  {ss['label']:>10s} n={ss['n']:3d} mean={ss['mean_pct']:+.3f}% "
                  f"hit={ss['hit']:.0f}% worst={ss['worst_pct']:+.2f}%")
    yrs = pd.Series(ep.year).value_counts().sort_index()
    print(f"  episodes per year: {dict(yrs)}")

    # ---- marginal contribution ----
    combos = [("VIX<16 only", cA), ("VIXrk5<=25 only", cB), ("SPY near-high only", cC),
              ("VIX<16 + rk5", cA & cB), ("VIX<16 + near-high", cA & cC),
              ("rk5 + near-high", cB & cC), ("FULL TRIPLE", full)]
    rows = []
    for lab, m in combos:
        dd = declusters(idx[m & valid], H + 1, idx)
        rows.append(summarize(f[dd].dropna().values, lab))
    rows.append(summarize(f[valid].values, "-- control --"))
    show(rows, f"E3 h={H} marginal contribution (episodes)")

    # ---- sensitivity ----
    rows = []
    for a_ in (14, 16, 18):
        for b_ in (15, 25, 40):
            for c_ in (-0.5, -1.0, -2.0):
                m = (vix < a_) & (rk5v <= b_) & (dist >= c_) & valid
                dd = declusters(idx[m], H + 1, idx)
                vv = f[dd].dropna().values
                if len(vv) < 3:
                    continue
                s = summarize(vv, "")
                rows.append(dict(vix_lt=a_, vixrk=b_, dist=c_, n=s["n"],
                                 mean=round(s["mean_pct"], 3), t=round(s["t"], 2),
                                 hit=round(s["hit"], 0), worst=round(s["worst_pct"], 2)))
    print(f"\n--- E3 h={H} sensitivity grid (episodes, LONG sign) ---")
    print(pd.DataFrame(rows).to_string(index=False))

    # ---- CPI/PPI in window ----
    cpi = pd.DatetimeIndex(load_events(["cpi"])["date"])
    ppi = pd.DatetimeIndex(load_events(["ppi"])["date"])
    both = pd.DatetimeIndex(sorted(set(cpi) | set(ppi)))
    pos = pd.Series(range(len(idx)), index=idx)
    mk = []
    for dd_ in ep:
        p = pos[dd_]
        if p + 1 + H >= len(idx):
            mk.append(False); continue
        lo, hi = idx[p + 1], idx[p + 1 + H]
        mk.append(bool(((both > lo) & (both <= hi)).any()))
    mk = np.array(mk, dtype=bool)
    show([summarize(v[mk], "CPI/PPI in hold"), summarize(v[~mk], "neither")],
         f"E3 h={H} CPI/PPI split ({mk.sum()}/{len(mk)})")
