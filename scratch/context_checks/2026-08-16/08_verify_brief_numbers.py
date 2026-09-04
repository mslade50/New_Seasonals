"""Pin every number that goes in tonight's brief, in one place, so the text is transcribed and
not remembered. Anything printed here appears in data/context_briefs/2026-08-16.md verbatim.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from pitch_lab import close_panel, load_prices, fwd_ret, summarize, sign_test, era_split, declusters, load_events  # noqa

px = close_panel(["SPY", "QQQ", "IWM", "^GSPC", "^VIX", "^VIX3M", "GC=F", "^NYA"])
idx = px.index
ev = load_events(["vix_expiry"])
vx = pd.DatetimeIndex(sorted(set(ev.loc[ev["event"] == "vix_expiry", "date"])))
anch = []
for e in vx:
    loc = idx.searchsorted(e)
    if loc < len(idx) and idx[loc] == e and loc >= 3:
        anch.append(idx[loc - 3])
anch = pd.DatetimeIndex(sorted(set(anch)))
nxt = pd.Series(idx, index=idx).shift(-1)
mon = (nxt.dt.weekday == 0).fillna(False)
aug = (nxt.dt.month == 8).fillna(False)
isa = pd.Series(idx.isin(anch), index=idx)
cell = (isa & mon & aug).values
allexp = (isa & mon).values
othmon = (~isa & mon).values
augoth = (~isa & mon & aug).values

print("### 1. August expiry-week Monday")
for t in ["SPY", "QQQ", "IWM", "^GSPC"]:
    v = fwd_ret(px[t], 1)[cell].dropna()
    up = int((v > 0).sum())
    s = summarize(v.values)
    print(f"  {t:6s} n {len(v)}  {up}-{len(v)-up}  mean {s['mean_pct']:+.2f}%  t {s['t']:+.2f}"
          f"  signp {sign_test(up, len(v)):.4f}")
v = fwd_ret(px["SPY"], 1)[cell].dropna()
e = era_split(v.index, v.values)
print(f"  SPY era: pre-2018 n {e[0]['n']} up {e[0]['hit']:.1f}%   2018+ n {e[1]['n']} up {e[1]['hit']:.1f}%"
      f"  ({int((v[v.index.year>=2018]>0).sum())} of {int((v.index.year>=2018).sum())})")
print(f"  SPY worst three: {sorted(100*v.values)[:3]}")
print(f"  SPY worst excluding 2008 and 2009: "
      f"{sorted(100*v[~v.index.year.isin([2008,2009])].values)[:2]}")
for name, m in [("expiry-week Monday, all months", allexp), ("every other Monday", othmon),
                ("August, other Mondays", augoth)]:
    v2 = fwd_ret(px["SPY"], 1)[m].dropna()
    up = int((v2 > 0).sum())
    print(f"  SPY {name:32s} n {len(v2):4d} {up}-{len(v2)-up} ({100*up/len(v2):.1f}%) "
          f"mean {100*v2.mean():+.3f}%")
mid = v[v.index.year % 4 == 2]
print(f"  SPY midterm n {len(mid)} up {int((mid>0).sum())};  "
      f"^GSPC midterm up {int((fwd_ret(px['^GSPC'],1)[cell].dropna()[lambda s: s.index.year%4==2]>0).sum())}/6;  "
      f"IWM midterm up {int((fwd_ret(px['IWM'],1)[cell].dropna()[lambda s: s.index.year%4==2]>0).sum())}/6")
ts = (px["^VIX"] / px["^VIX3M"]).dropna()
sub = fwd_ret(px["SPY"], 1)[cell].dropna()
lo = sub[ts.reindex(sub.index) < 0.90]
print(f"  VIX/VIX3M Friday {ts.iloc[-1]:.3f};  anchors below 0.90: n {len(lo)} "
      f"{int((lo>0).sum())}-{len(lo)-int((lo>0).sum())} mean {100*lo.mean():+.3f}%")

print("\n### 2. VIX on that session")
for name, m in [("August expiry-week Monday", cell), ("expiry-week Monday, all months", allexp),
                ("every other Monday", othmon)]:
    v2 = fwd_ret(px["^VIX"], 1)[m].dropna()
    up = int((v2 > 0).sum())
    s = summarize(v2.values)
    print(f"  {name:32s} n {len(v2):4d} up {up}-{len(v2)-up} ({s['hit']:.1f}%) "
          f"mean {s['mean_pct']:+.2f}% t {s['t']:+.2f} signp {sign_test(up, len(v2)):.4f}")
v2 = fwd_ret(px["^VIX"], 1)[cell].dropna()
e = era_split(v2.index, v2.values)
print(f"  era up rates: {e[0]['hit']:.1f}% / {e[1]['hit']:.1f}%")
joint = ((fwd_ret(px["SPY"], 1) > 0) & (fwd_ret(px["^VIX"], 1) < 0))
for name, m in [("cell", cell), ("other Mondays", othmon)]:
    j = joint[m].dropna()
    print(f"  SPY up and VIX down, {name}: {int(j.sum())}/{len(j)} = {100*j.mean():.1f}%")
jb = joint.dropna()
print(f"  SPY up and VIX down, every session: {100*jb.mean():.1f}%")

print("\n### 3. midterm doy window, ^GSPC")
g = px["^GSPC"]
anchors = []
for y in range(2000, 2026):
    yr = idx[idx.year == y]
    pick = min(yr, key=lambda d: abs((d - pd.Timestamp(f"{y}-08-17")).days))
    loc = idx.get_loc(pick)
    anchors.append(idx[loc - 1])
anchors = pd.DatetimeIndex(anchors)
midd = anchors[anchors.year % 4 == 2]
for h in (1, 42):
    a = fwd_ret(g, h).reindex(anchors).dropna()
    m = fwd_ret(g, h).reindex(midd).dropna()
    print(f"  h{h:<3d} all years n {len(a)} mean {100*a.mean():+.2f}%  |  midterm n {len(m)} "
          f"{int((m>0).sum())}-{int((m<0).sum())} mean {100*m.mean():+.2f}%")
m42 = fwd_ret(g, 42).reindex(midd).dropna()
print("  midterm h42 by year:", ", ".join(f"{d.year}:{100*x:+.1f}" for d, x in zip(m42.index, m42.values)))
m1 = fwd_ret(g, 1).reindex(midd).dropna()
print(f"  midterm h1 {int((m1>0).sum())}-{int((m1<0).sum())}, best year share "
      f"{100*m1.max()/m1.sum():.0f}%")

print("\n### 4. gold")
gc = px["GC=F"].dropna()
ga = []
for y in range(2000, 2026):
    yr = gc.index[gc.index.year == y]
    if len(yr) < 20:
        continue
    pick = min(yr, key=lambda d: abs((d - pd.Timestamp(f"{y}-08-17")).days))
    loc = gc.index.get_loc(pick)
    ga.append(gc.index[loc - 1])
ga = pd.DatetimeIndex(ga)
for h in (5, 10):
    v3 = fwd_ret(gc, h).reindex(ga).dropna()
    base = fwd_ret(gc, h).dropna()
    print(f"  h{h:<3d} n {len(v3)} {int((v3>0).sum())}-{int((v3<0).sum())} mean {100*v3.mean():+.2f}% "
          f"signp {sign_test(int((v3>0).sum()), len(v3)):.4f}   control mean {100*base.mean():+.2f}% "
          f"up {100*(base>0).mean():.1f}%")
r1 = gc.pct_change()
z10 = (gc / gc.shift(10) - 1) / (r1.rolling(21).std() * np.sqrt(10))
print(f"  Friday close {gc.iloc[-1]:,.1f}  z10 {z10.iloc[-1]:+.2f}  "
      f"21d rank {100*(gc.pct_change(21).iloc[-252:] < gc.pct_change(21).iloc[-1]).mean():.0f}")
trig = declusters(z10.index[(z10 >= 2).fillna(False)], 5, gc.index)
both = trig[(trig.month == 8) | ((trig.month == 9) & (trig.day <= 5))]
for h in (1, 5, 10):
    v3 = fwd_ret(gc, h).reindex(both).dropna()
    print(f"  stretched August entries h{h:<3d} n {len(v3)} {int((v3>0).sum())}-{int((v3<0).sum())} "
          f"mean {100*v3.mean():+.2f}%")

print("\n### 5. bitcoin")
b = load_prices(["BTC-USD"])["BTC-USD"]["Close"].dropna()
s = px["SPY"].dropna()
run, vals = 0, []
for x in (b.pct_change() < 0).astype(int).values:
    run = run + 1 if x else 0
    vals.append(run)
streak = pd.Series(vals, index=b.index)
spy_b = s.reindex(b.index).ffill()
near = (spy_b / spy_b.rolling(252).max() - 1) >= -0.005
trig = declusters(streak.index[streak >= 5], 5, b.index)
dates = pd.DatetimeIndex([d for d in trig if bool(near.get(d, False))])
v4 = fwd_ret(b, 21).reindex(dates).dropna()
up = int((v4 > 0).sum())
print(f"  streak tonight {int(streak.iloc[-1])}; BTC {b.iloc[-1]:,.0f}; "
      f"{100*(b.iloc[-1]/b.rolling(365).max().iloc[-1]-1):.1f}% below its 12-month high; "
      f"SPY {100*(spy_b.iloc[-1]/spy_b.rolling(252).max().iloc[-1]-1):+.2f}% from its high")
print(f"  h21 n {len(v4)} {up}-{len(v4)-up} mean {100*v4.mean():+.2f}% med {100*v4.median():+.2f}% "
      f"down-side signp {sign_test(len(v4)-up, len(v4)):.4f}  years {sorted(set(v4.index.year))}")
print(f"  drop worst: mean {100*v4.drop(v4.idxmin()).mean():+.2f}%")
base = fwd_ret(b, 21).dropna()
nostreak = b.index[near.fillna(False).values].difference(streak.index[streak >= 5])
print(f"  control every session {100*base.mean():+.2f}%;  SPY at a high without a streak "
      f"{100*fwd_ret(b,21).reindex(nostreak).dropna().mean():+.2f}%")

print("\n### 6. NYSE composite streak")
nya = px["^NYA"].dropna()
run, vals = 0, []
for x in (nya.pct_change() > 0).astype(int).values:
    run = run + 1 if x else 0
    vals.append(run)
st = pd.Series(vals, index=nya.index)
at_high = nya >= nya.rolling(252).max() * 0.999
trig = declusters(nya.index[(st >= 5).values], 5, nya.index)
hi = pd.DatetimeIndex([d for d in trig if bool(at_high.get(d, False))])
print(f"  tonight streak {int(st.iloc[-1])}, at a 52w high {bool(at_high.iloc[-1])}")
for name, dts in [("all 5+ streaks", trig), ("streaks at a 52-week high", hi)]:
    v5 = fwd_ret(nya, 21).reindex(dts).dropna()
    s5 = summarize(v5.values)
    up = int((v5 > 0).sum())
    print(f"  {name:28s} n {len(v5):3d} {up}-{len(v5)-up} mean {s5['mean_pct']:+.2f}% "
          f"t {s5['t']:+.2f} signp {sign_test(up, len(v5)):.4f}")
base = fwd_ret(nya, 21).dropna()
print(f"  control h21 n {len(base)} mean {100*base.mean():+.2f}% up {100*(base>0).mean():.1f}%")
