"""The CPI-day vol crush when VIX enters with nothing to crush.

Engine cell: VIX on the print session, n=317, -0.847%, 120-194 down, t=-2.12,
edge -1.105 vs all days. Tonight VIX closes 15.28, 50.8% under its 52-week
high and 17.7% under its 200d. The question the engine did not ask is whether
the crush is a premium-release effect (needs premium) or unconditional.

Also settles the Aug-doy confound: the seasonal cell says VIX is 0-for-6 on
this trading day of the year in midterm years. If those six days are mostly
CPI prints, it is one effect, not two.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

px = load_prices(["^VIX", "SPY"])
vix = px["^VIX"]["Close"].dropna()
ev = load_events(["cpi"])
cpi = pd.DatetimeIndex(sorted(pd.to_datetime(ev["date"]).unique()))

d = vix.index
anc, prn = [], []
for x in cpi:
    pos = d.searchsorted(x)
    if pos <= 0 or pos >= len(d) or d[pos] != x:
        continue
    anc.append(d[pos - 1])
    prn.append(d[pos])
anc = pd.DatetimeIndex(anc)
prn = pd.DatetimeIndex(prn)

r = vix.pct_change()
nxt = r.shift(-1)                       # the print session's VIX move
pctile = vix.rolling(252, min_periods=252).rank(pct=True) * 100
sma200 = vix.rolling(200, min_periods=200).mean()

base = nxt.reindex(anc).dropna()
s = summarize(base.values, "all CPI")
print(f"ALL CPI prints: n={s['n']}  VIX {s['mean_pct']:+.3f}%  "
      f"down {(base<0).sum()}-{(base>0).sum()} up  t={s['t']:+.2f}  "
      f"sign p {sign_test(int((base<0).sum()), int(len(base))):.4f}")
allr = r.dropna()
allr = allr[allr.index >= base.index[0]]
print(f"  every session same span: n={len(allr)}  {allr.mean()*100:+.3f}%  "
      f"down {(allr<0).sum()}-{(allr>0).sum()}  ({(allr<0).mean()*100:.1f}% down)")

print("\n--- by VIX percentile entering the print (trailing 252d) ---")
pc = pctile.reindex(anc)
for lo, hi, lab in [(0, 25, "bottom quartile"), (25, 50, "2nd"), (50, 75, "3rd"),
                    (75, 101, "top quartile"), (0, 33.4, "bottom third"), (0, 10, "bottom decile")]:
    sel = pd.DatetimeIndex([x for x in anc if pd.notna(pc.get(x, np.nan)) and lo <= pc[x] < hi])
    v = nxt.reindex(sel).dropna()
    if len(v) < 5:
        print(f"  {lab:<16} n={len(v)} too small")
        continue
    ss = summarize(v.values, lab)
    print(f"  {lab:<16} n={ss['n']:<4} mean {ss['mean_pct']:+7.3f}%  "
          f"down {(v<0).sum():>3}-{(v>0).sum():<3}  {(v<0).mean()*100:5.1f}% down  "
          f"t={ss['t']:+5.2f}  sign p {sign_test(int((v<0).sum()), int(len(v))):.4f}")

print("\n--- VIX below its 200d entering the print ---")
below = pd.DatetimeIndex([x for x in anc if pd.notna(sma200.get(x, np.nan)) and vix[x] < sma200[x]])
above = anc.difference(below)
for sel, lab in [(below, "VIX < 200d"), (above, "VIX > 200d")]:
    v = nxt.reindex(sel).dropna()
    ss = summarize(v.values, lab)
    print(f"  {lab:<12} n={ss['n']:<4} mean {ss['mean_pct']:+7.3f}%  "
          f"down {(v<0).sum()}-{(v>0).sum()}  {(v<0).mean()*100:.1f}% down  t={ss['t']:+.2f}  "
          f"sign p {sign_test(int((v<0).sum()), int(len(v))):.4f}")
    print(f"    era: {[(e['label'], e['n'], round(e['mean_pct'],2), round((100-e['hit']),1)) for e in era_split(v.index, v.values)]}")

# tonight's exact state: below the 200d AND bottom third of the trailing year
tight = pd.DatetimeIndex([x for x in below if pd.notna(pc.get(x, np.nan)) and pc[x] < 33.4])
v = nxt.reindex(tight).dropna()
ss = summarize(v.values, "tight")
print(f"\n  TONIGHT'S STATE (VIX < 200d and bottom third of its year): n={ss['n']}  "
      f"mean {ss['mean_pct']:+.3f}%  down {(v<0).sum()}-{(v>0).sum()}  "
      f"{(v<0).mean()*100:.1f}% down  t={ss['t']:+.2f}  sign p {sign_test(int((v<0).sum()), int(len(v))):.4f}")
print("  concentration:", cluster_note(v.index, v.values))
print("  years:", sorted(set(v.index.year)))
# control: same VIX state, no print the next session
allq = pd.DatetimeIndex([x for x in vix.index
                         if pd.notna(sma200.get(x, np.nan)) and pd.notna(pctile.get(x, np.nan))
                         and vix[x] < sma200[x] and pctile[x] < 33.4])
ctrl = allq.difference(anc)
vc = nxt.reindex(ctrl).dropna()
sc = summarize(vc.values, "ctrl")
print(f"  same VIX state, NO print next: n={sc['n']}  mean {sc['mean_pct']:+.3f}%  "
      f"{(vc<0).mean()*100:.1f}% down")
print(f"  edge over that control: {ss['mean_pct']-sc['mean_pct']:+.3f}%")

# ---- the Aug-doy confound ----
print("\n--- Aug trading-day-of-year cell vs the CPI cell ---")
aug = pd.DatetimeIndex([x for x in vix.index if x.month == 8 and 10 <= x.day <= 14])
mid = pd.DatetimeIndex([x for x in aug if x.year % 4 == 2])
print(f"  mid-Aug (10-14th) sessions in midterm years: {len(mid)}")
hit_cpi = [x for x in mid if x in prn]
print(f"  of those, how many ARE a CPI print session: {len(hit_cpi)}")
print(f"  CPI prints falling Aug 10-14: {sorted(str(x.date()) for x in prn if x.month==8 and 10<=x.day<=14)}")
