"""E1: "Exhaustion at the high" -- SPY 5d-rank>=95 + within 0.5% of 52w high + VIX 5d-rank<=25.

Real order modelled: trigger on day D's close, ENTER MOC D+1, EXIT MOC D+1+h.
So the forward return is close[D+1+h]/close[D+1]-1, aligned to D.
Direction is left to the data (long numbers reported; short = negate).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _common import *  # noqa: F401,F403

H = 5
px = load_prices(["SPY", "^VIX"])
spy = px["SPY"]["Close"]
vix = px["^VIX"]["Close"].reindex(spy.index).ffill()
idx = spy.index

rk5 = pct_rank(spy, 5)
rk5v = pct_rank(vix, 5)
dist = (spy / spy.rolling(252).max() - 1) * 100


def fwd_entry_next(s: pd.Series, h: int) -> pd.Series:
    """Enter at close D+1, exit at close D+1+h; aligned to D."""
    return s.shift(-(h + 1)) / s.shift(-1) - 1.0


F = {h: fwd_entry_next(spy, h) for h in (1, 3, 5, 10)}
f = F[H]

full = (rk5 >= 95) & (dist >= -0.5) & (rk5v <= 25)
valid = rk5.notna() & rk5v.notna() & dist.notna() & f.notna()

d_full = idx[full & valid]
print(f"sample: {idx[valid][0].date()} .. {idx[valid][-1].date()}   trigger days N={len(d_full)}")
print(f"most recent 8 trigger days: {[str(d.date()) for d in d_full[-8:]]}")

# ---------- 1. pattern vs TWO controls ----------
rows = [summarize(f[d_full].values, "TRIGGER (day-level)")]
# control A: SPY unconditional over the SAME window (all valid days)
rows.append(summarize(f[valid].values, "ctrl A: SPY uncond, same window"))
# control B: all days, whole history, no rank-warmup restriction
rows.append(summarize(f[f.notna()].values, "ctrl B: all-days baseline"))
show(rows, f"E1 h={H}td (enter D+1 close, exit D+1+{H})")

mu_c = np.nanmean(f[valid].values)
mu_t = np.nanmean(f[d_full].values)
print(f"\nedge vs same-window control: {100*(mu_t-mu_c):+.3f}%  "
      f"(trigger {100*mu_t:+.3f}% vs control {100*mu_c:+.3f}%)")
# Welch t of trigger vs same-window control
a, b = f[d_full].dropna().values, f[valid].dropna().values
tw = (a.mean() - b.mean()) / np.sqrt(a.var(ddof=1)/len(a) + b.var(ddof=1)/len(b))
print(f"Welch t (trigger vs control): {tw:+.2f}")

# ---------- marginal contribution of each condition ----------
cA = rk5 >= 95
cB = dist >= -0.5
cC = rk5v <= 25
combos = [
    ("rk5>=95 only", cA),
    ("near-high only", cB),
    ("VIXrk5<=25 only", cC),
    ("rk5 + near-high (no VIX)", cA & cB),
    ("rk5 + VIX (no high)", cA & cC),
    ("near-high + VIX (no rk5)", cB & cC),
    ("FULL TRIPLE", full),
]
rows = [summarize(f[idx[m & valid]].values, lab) for lab, m in combos]
rows.append(summarize(f[valid].values, "-- control --"))
show(rows, "E1 marginal contribution of each condition (day-level, h=5)")

# ---------- 2/3. decluster ----------
ep = declusters(d_full, H + 1, idx)
v_ep = f[ep].dropna().values
print(f"\nepisodes (min gap {H+1} td): {len(ep)}  of {len(d_full)} day-level")
show([summarize(f[d_full].values, "day-level"), summarize(v_ep, "episode-level")],
     "E1 day vs episode")
print(f"bootstrap P(mean<=0) LONG side: {bootstrap_p_le0(v_ep):.4f}")
print(f"bootstrap P(mean<=0) SHORT side (negated): {bootstrap_p_le0(-v_ep):.4f}")
if len(v_ep):
    j = int(np.argmax(v_ep))
    k = int(np.argmin(v_ep))
    print(f"best episode {ep[j].date()} {100*v_ep[j]:+.2f}%   "
          f"worst episode {ep[k].date()} {100*v_ep[k]:+.2f}%")
    show([summarize(np.delete(v_ep, j), "drop-BEST episode"),
          summarize(np.delete(v_ep, k), "drop-WORST episode")], "E1 drop-one")

# ---------- era stability ----------
show(era_split(ep, v_ep, "2018-01-01"), "E1 era split (episodes) 2018")
show(era_split(ep, v_ep, "2013-01-01"), "E1 era split (episodes) 2013")
for lo, hi in [(2000, 2008), (2008, 2013), (2013, 2018), (2018, 2022), (2022, 2027)]:
    m = (ep >= pd.Timestamp(f"{lo}-01-01")) & (ep < pd.Timestamp(f"{hi}-01-01"))
    if m.sum():
        s = summarize(v_ep[m], f"{lo}-{hi}")
        print(f"  {s['label']:>10s} n={s['n']:3d} mean={s['mean_pct']:+.3f}% "
              f"hit={s['hit']:.0f}% worst={s['worst_pct']:+.2f}%")

# ---------- 4. sensitivity grid ----------
print("\n=== E1 sensitivity grid (episode-level, h=5, LONG sign) ===")
out = []
for a_ in (90, 95, 98):
    for b_ in (-0.25, -0.5, -1.0):
        for c_ in (15, 25, 40, 101):
            m = (rk5 >= a_) & (dist >= b_) & (rk5v <= c_) & valid
            dd = declusters(idx[m], H + 1, idx)
            vv = f[dd].dropna().values
            if len(vv) < 3:
                continue
            s = summarize(vv, "")
            out.append(dict(rk5=a_, dist=b_, vixrk=("none" if c_ > 100 else c_),
                            n=s["n"], mean=round(s["mean_pct"], 3),
                            t=round(s["t"], 2), hit=round(s["hit"], 0)))
print(pd.DataFrame(out).to_string(index=False))

# ---------- 5. horizons ----------
rows = []
for h in (1, 3, 5, 10):
    fh = F[h]
    vld = rk5.notna() & rk5v.notna() & dist.notna() & fh.notna()
    dd = declusters(idx[full & vld], h + 1, idx)
    rows.append(summarize(fh[dd].dropna().values, f"h={h} trigger"))
    rows.append(summarize(fh[vld].values, f"h={h} control"))
show(rows, "E1 horizon scan (episodes vs same-window control)")

# ---------- 6. CPI-in-window ----------
ev = load_events(["cpi"])
cpi = pd.DatetimeIndex(ev["date"])
pos = pd.Series(range(len(idx)), index=idx)


def cpi_in_window(d, h):
    p = pos[d]
    if p + 1 + h >= len(idx):
        return None
    lo, hi = idx[p + 1], idx[p + 1 + h]
    return bool(((cpi > lo) & (cpi <= hi)).any())


flag = [cpi_in_window(d, H) for d in ep]
mk = np.array([f is True for f in flag], dtype=bool)
print(f"\nCPI-in-window: {mk.sum()} of {len(mk)} episodes (None/out-of-range: "
      f"{sum(1 for f in flag if f is None)})")
show([summarize(v_ep[mk], "CPI inside hold"), summarize(v_ep[~mk], "no CPI in hold")],
     "E1 CPI-in-window split (episodes)")

# also PPI, and the pair (CPI or PPI) since both land inside this week's hold
ppi = pd.DatetimeIndex(load_events(["ppi"])["date"])


def any_in_window(dates, d, h):
    p = pos[d]
    if p + 1 + h >= len(idx):
        return None
    lo, hi = idx[p + 1], idx[p + 1 + h]
    return bool(((dates > lo) & (dates <= hi)).any())


both = pd.DatetimeIndex(sorted(set(cpi) | set(ppi)))
mk2 = np.array([any_in_window(both, d, H) is True for d in ep], dtype=bool)
print(f"CPI-or-PPI in window: {mk2.sum()} of {len(mk2)}")
show([summarize(v_ep[mk2], "CPI/PPI inside hold"), summarize(v_ep[~mk2], "neither")],
     "E1 CPI-or-PPI split (episodes)")
