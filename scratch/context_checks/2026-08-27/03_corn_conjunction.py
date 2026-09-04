"""ZC=F corn is the only subject firing four price triggers at once.

Tonight: +3.40%, z10 3.38, 5d/21d/63d return ranks all 100.0, a 5+ session
up streak, closing exactly at a 52-week high, volume 2.06x its 63d norm.
Every individual engine cell for corn is weak (h1 t 0.6 to 2.1, every sign p
above 0.25). The question is whether the CONJUNCTION behaves differently from
any single trigger, or whether stacking weak cells just shrinks n.

Anchor = the session the state printed, h1 = the next session.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    load_prices, fwd_ret, summarize, show, sign_test, era_split,
    cluster_note, declusters, rolling_on_valid, local_control,
)

px = load_prices(["ZC=F"])["ZC=F"]
c = px["Close"].dropna()
dates = c.index
print(f"ZC=F {dates.min().date()} .. {dates.max().date()}, n={len(c)}, "
      f"last {c.iloc[-1]:.2f}")

r1 = c.pct_change()
r5 = c / c.shift(5) - 1.0
r21 = c / c.shift(21) - 1.0
r63 = c / c.shift(63) - 1.0
rk = lambda s: s.rolling(252).rank(pct=True) * 100
k5, k21, k63 = rk(r5), rk(r21), rk(r63)

# z10 as defined by build_pitch_state._metrics_for / the context engine
z10 = (c / c.shift(10) - 1.0) / (r1.rolling(21).std() * np.sqrt(10))

hi252 = c.rolling(252).max()
at_high = c >= hi252 * 0.9999999

streak = r1.gt(0).astype(int)
run = streak * (streak.groupby((streak == 0).cumsum()).cumcount() + 1)

print("\ntonight's readings: "
      f"z10 {z10.iloc[-1]:.2f}  k5 {k5.iloc[-1]:.1f}  k21 {k21.iloc[-1]:.1f}  "
      f"k63 {k63.iloc[-1]:.1f}  streak {int(run.iloc[-1])}  "
      f"at52wh {bool(at_high.iloc[-1])}")

singles = {
    "z10 >= 2":        (z10 >= 2),
    "5d rank >= 95":   (k5 >= 95),
    "21d rank >= 95":  (k21 >= 95),
    "up streak >= 5":  (run >= 5),
    "at a 52w high":   at_high,
}
conj = None
for m in singles.values():
    conj = m.fillna(False) if conj is None else (conj & m.fillna(False))

rows = []
for name, m in singles.items():
    idx = dates[m.fillna(False)]
    v = fwd_ret(c, 1).reindex(idx).dropna()
    s = summarize(v.values, name)
    k = int((v.values > 0).sum())
    s["record"] = f"{k}-{s['n']-k}"
    s["sign_p"] = round(sign_test(k, s["n"]), 4)
    rows.append(s)

idx_c = dates[conj]
v_c = fwd_ret(c, 1).reindex(idx_c).dropna()
s = summarize(v_c.values, "ALL FIVE at once")
k = int((v_c.values > 0).sum())
s["record"] = f"{k}-{s['n']-k}"
s["sign_p"] = round(sign_test(k, s["n"]), 4)
rows.append(s)

base = fwd_ret(c, 1).dropna()
b = summarize(base.values, "all days")
b["record"] = f"{int((base.values>0).sum())}-{int((base.values<=0).sum())}"
b["sign_p"] = ""
rows.append(b)
show(rows, "ZC=F h1 by trigger")

print(f"\nconjunction sessions: n={len(idx_c)}")
if len(idx_c):
    print("  years:", pd.Series(idx_c.year).value_counts().sort_index().to_dict())
    dc = declusters(idx_c, 10, dates)
    vd = fwd_ret(c, 1).reindex(dc).dropna()
    sd_ = summarize(vd.values, "declustered 10td")
    kd = int((vd.values > 0).sum())
    print(f"  declustered n={sd_['n']} mean {sd_['mean_pct']:+.3f}% "
          f"median {sd_['median_pct']:+.3f}% record {kd}-{sd_['n']-kd} up "
          f"sign p {sign_test(kd, sd_['n']):.4f}")
    print(f"  era: {[(r['label'], r['n'], round(r['mean_pct'],3)) for r in era_split(v_c.index, v_c.values)]}")
    print(f"  {cluster_note(v_c.index, v_c.values)}")
    lc = local_control(dates, idx_c, 126)
    vl = fwd_ret(c, 1).reindex(lc).dropna()
    print(f"  local +/-126td control: n={len(vl)} mean {100*vl.mean():+.3f}% "
          f"hit {100*(vl.values>0).mean():.1f}%")
    for h in (5, 10, 21):
        v = fwd_ret(c, h).reindex(idx_c).dropna()
        kk = int((v.values > 0).sum())
        print(f"  h{h}: n={len(v)} mean {100*v.mean():+.3f}% "
              f"median {100*np.median(v.values):+.3f}% record {kk}-{len(v)-kk} up")
    print("\n  every conjunction session:")
    f1 = fwd_ret(c, 1)
    f5 = fwd_ret(c, 5)
    f21 = fwd_ret(c, 21)
    for d in idx_c:
        print(f"    {d.date()}  close {c[d]:7.2f}  "
              f"h1 {100*f1.get(d, np.nan):+6.2f}%  "
              f"h5 {100*f5.get(d, np.nan):+6.2f}%  "
              f"h21 {100*f21.get(d, np.nan):+7.2f}%")

# how often is corn simultaneously at 100 on all three horizon ranks?
trip = ((k5 >= 99.99) & (k21 >= 99.99) & (k63 >= 99.99)).fillna(False)
print(f"\nsessions with 5d, 21d AND 63d rank all at 100.0: {int(trip.sum())}")
if trip.sum():
    ti = dates[trip]
    print("  years:", pd.Series(ti.year).value_counts().sort_index().to_dict())
    vt = fwd_ret(c, 1).reindex(ti).dropna()
    kt = int((vt.values > 0).sum())
    st = summarize(vt.values, "triple-100 h1")
    print(f"  h1 n={st['n']} mean {st['mean_pct']:+.3f}% median {st['median_pct']:+.3f}% "
          f"record {kt}-{st['n']-kt} up sign p {sign_test(kt, st['n']):.4f}")
    v21 = fwd_ret(c, 21).reindex(ti).dropna()
    k21_ = int((v21.values > 0).sum())
    print(f"  h21 n={len(v21)} mean {100*v21.mean():+.3f}% "
          f"median {100*np.median(v21.values):+.3f}% record {k21_}-{len(v21)-k21_} up")
