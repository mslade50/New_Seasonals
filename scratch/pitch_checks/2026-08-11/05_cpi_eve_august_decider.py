"""The decider for the CPI-eve short-vol trade: does TODAY'S cell work?

04_ found two conditioners that both describe today and both point the wrong
way:
  - today's ^VIX sits at the 16.3rd percentile of its trailing year (63d rank
    8.1), the bucket where the h=3 VIX decline shrinks from -2.6/-3.2% to
    -0.41%
  - AUGUST is the single worst month in the table: ^VIX h=3 is +2.674% (it
    RISES) and SVXY h=3 is -0.258%, the only negative month of the twelve

This script asks the three questions that decide the idea:
  Q1  year by year, what does the August CPI-eve trade actually do, and is the
      negative mean one episode or a pattern?
  Q2  in today's LOW-VIX bucket, does the CPI anchor add anything at all over
      simply being long SVXY on a random low-VIX day?  If not, the "event
      crush" thesis is false in today's state and what is left is carry.
  Q3  the joint cell (August AND low VIX): what is on the tape today?
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    close_panel, load_events, fwd_lag, declusters, summarize, sign_test,
    bootstrap_p_le0,
)

warnings.filterwarnings("ignore")

px = close_panel(["^VIX", "SVXY", "SPY"])
idx = px.index
ev = load_events(["cpi"])
cpi = pd.DatetimeIndex(sorted(ev.loc[ev["event"] == "cpi", "date"].unique()))

mask = []
for d in cpi:
    loc = idx.searchsorted(d)
    if 0 <= loc - 2 and loc < len(idx):
        mask.append(idx[loc - 2])
mask = declusters(pd.DatetimeIndex(sorted(set(mask))), 5, idx)

vix = px["^VIX"].dropna()
svxy = px["SVXY"].dropna()
vix_pct = vix.rolling(252).apply(lambda w: (w[-1] > w[:-1]).mean() * 100, raw=True)

H = 3

print("=" * 90)
print("Q1  AUGUST CPI-eve, year by year")
print("=" * 90)
aug = mask[mask.month == 8]
fv = fwd_lag(vix, H, lag=1)
fs = fwd_lag(svxy, H, lag=1)
print(" year   eve date     VIX pctile   VIX h3      SVXY h3")
for d in aug:
    p = vix_pct.get(d, np.nan)
    v = fv.get(d, np.nan)
    s = fs.get(d, np.nan)
    print(f" {d.year}   {d.date()}   {p:9.1f}   {100*v:+8.2f}%   "
          f"{('%+8.2f%%' % (100*s)) if not pd.isna(s) else '      n/a'}")

av = fv.reindex(aug).dropna()
asv = fs.reindex(aug).dropna()
print()
print(f"  ^VIX August h={H}: N={len(av)} mean={100*av.mean():+.3f}% "
      f"median={100*av.median():+.3f}% up={100*(av>0).mean():.1f}% "
      f"sign p(up)={sign_test(int((av>0).sum()), len(av)):.4f}")
print(f"  SVXY August h={H}: N={len(asv)} mean={100*asv.mean():+.3f}% "
      f"median={100*asv.median():+.3f}% up={100*(asv>0).mean():.1f}% "
      f"sign p(up)={sign_test(int((asv>0).sum()), len(asv)):.4f} "
      f"worst={100*asv.min():+.2f}%")
print(f"  SVXY August drop-worst mean: {100*asv.drop(asv.idxmin()).mean():+.3f}%")
print(f"  SVXY non-August h={H}: N={len(fs.reindex(mask[mask.month!=8]).dropna())} "
      f"mean={100*fs.reindex(mask[mask.month!=8]).dropna().mean():+.3f}%")

print()
print("=" * 90)
print("Q2  in TODAY'S low-VIX bucket, does the CPI anchor add anything?")
print("=" * 90)
today_pct = vix_pct.iloc[-1]
print(f"today's ^VIX 252d percentile = {today_pct:.1f}\n")
for name, s in (("^VIX", vix), ("SVXY", svxy)):
    f = fwd_lag(s, H, lag=1)
    lowmask = (vix_pct.reindex(s.index) <= 33)
    trig = mask[mask.isin(s.index)]
    trig_low = trig[(vix_pct.reindex(trig) <= 33).values]
    # control: every low-VIX day that is NOT a CPI eve
    ctl_idx = s.index[lowmask.fillna(False).values]
    ctl_idx = ctl_idx.difference(trig_low)
    a = f.reindex(trig_low).dropna()
    b = f.reindex(ctl_idx).dropna()
    print(f"  {name} h={H}  CPI eve & low VIX : N={len(a):<4} mean={100*a.mean():+7.3f}% "
          f"up={100*(a>0).mean():5.1f}%")
    print(f"  {name} h={H}  low VIX, no event : N={len(b):<4} mean={100*b.mean():+7.3f}% "
          f"up={100*(b>0).mean():5.1f}%")
    print(f"  {name} h={H}  --> anchor adds     {100*(a.mean()-b.mean()):+7.3f} pp\n")

print("=" * 90)
print("Q3  the joint cell that describes TODAY: August AND low VIX")
print("=" * 90)
for name, s in (("^VIX", vix), ("SVXY", svxy)):
    f = fwd_lag(s, H, lag=1)
    trig = mask[mask.isin(s.index)]
    sel = trig[((trig.month == 8) & (vix_pct.reindex(trig) <= 33)).values]
    v = f.reindex(sel).dropna()
    if len(v) == 0:
        print(f"  {name}: joint cell EMPTY")
        continue
    print(f"  {name} h={H} August & VIX pctile<=33: N={len(v)} "
          f"mean={100*v.mean():+.3f}% up={100*(v>0).mean():.1f}% "
          f"sign p(up)={sign_test(int((v>0).sum()), len(v)):.4f}")
    print(f"      episodes: " + ", ".join(f"{d.year}:{100*x:+.2f}%" for d, x in v.items()))

print()
print("=" * 90)
print("Q4  is August special, or is every month noisy? (placebo across months)")
print("=" * 90)
f = fwd_lag(svxy, H, lag=1)
rows = []
for mo in range(1, 13):
    v = f.reindex(mask[mask.month == mo]).dropna()
    if len(v) < 4:
        continue
    rows.append((mo, len(v), 100 * v.mean(), 100 * v.median(), 100 * (v > 0).mean()))
rows.sort(key=lambda r: r[2])
print("  SVXY h=3 by month, sorted worst first")
for mo, n, mean, med, up in rows:
    star = "   <-- AUGUST, the only negative month" if mo == 8 else ""
    print(f"   month {mo:>2}  N={n:<3} mean={mean:+7.3f}  median={med:+7.3f}  up={up:5.1f}%{star}")

v_all = f.reindex(mask).dropna()
print(f"\n  all months: N={len(v_all)} mean={100*v_all.mean():+.3f}% "
      f"bootstrap P(mean<=0)={bootstrap_p_le0(v_all.values):.4f}")
