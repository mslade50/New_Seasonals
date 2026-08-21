"""C3/C4/C5 round 2c — the three attacks that decide the survivors.

  1. IMPULSE vs DRIFT. An event effect decays; an exposure grows linearly in h.
     Every cell that looked alive in b1 peaked at h=10, the EDGE of the scanned
     grid, which is the drift signature. Split the hold into days 1-5 and 6-10.
  2. LIVE-STATE HONESTY. SLV is 41.6% below its 252d high after a blowoff and
     a +14.4% 21-day bounce; XLE closed 2026-08-20 AT its 252d high. Ask what
     the historical trigger population looked like.
  3. REGISTRY COLLISION with 2026-08-20's parked crude entry: JH-6 in 2026 is
     2026-08-20 and this morning's opex-close entry is JH-5, one session apart.
     Count the calendar overlap and apply that entry's own state block.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

VEH = ["GLD", "SLV", "TLT", "IEF", "HYG", "LQD", "USO", "XLE", "UUP", "FXI"]
px = close_panel(VEH + ["CL=F"])
idx = px.index
pos = pd.Series(range(len(idx)), index=idx)
opex = pd.DatetimeIndex([d for d in load_events(["opex"])["date"] if d in pos.index])
jh = load_events(["jackson_hole"])["date"]
aA = pd.DatetimeIndex([idx[pos[d] - 1] for d in opex if pos[d] >= 1])
augA = pd.DatetimeIndex([d for d in aA if idx[pos[d] + 1].month == 8])

print("===== 1. IMPULSE vs DRIFT: days 1-5 against days 6-10 of the hold =====")
print("(entry MOC on the opex close; leg1 = entry..+5, leg2 = +5..+10)")
rows = []
for v in VEH:
    s = px[v].dropna()
    for lbl, anc in [("all", aA), ("Aug", augA)]:
        r5 = fwd_lag(s, 5, 1)
        r10 = fwd_lag(s, 10, 1)
        leg2 = (s.shift(-11) / s.shift(-6) - 1.0)
        a = pd.DatetimeIndex(anc).intersection(r10.dropna().index)
        base = r5.dropna()
        base = base[(base.index >= a[0]) & (base.index <= a[-1])]
        if lbl == "Aug":
            base = base[base.index.month == 8]
        rows.append({"v": v, "set": lbl, "n": len(a),
                     "leg1_d1_5": 100 * r5.loc[a].mean(),
                     "leg2_d6_10": 100 * leg2.loc[a].mean(),
                     "base_5d": 100 * base.mean(),
                     "leg1_x": 100 * (r5.loc[a].mean() - base.mean()),
                     "leg2_x": 100 * (leg2.loc[a].mean() - base.mean())})
print(pd.DataFrame(rows).round(3).to_string(index=False))
print("\nread: leg2_x >= leg1_x means the 'post-opex' return keeps arriving a")
print("fortnight later, i.e. it is a WINDOW/exposure, not an expiry impulse.")

print("\n\n===== 2. LIVE-STATE HONESTY =====")
for v in ("SLV", "XLE", "USO", "GLD", "FXI"):
    s = px[v].dropna()
    hi = rolling_on_valid(s, lambda x: x.rolling(252).max())
    dd = s / hi - 1.0
    r21 = rolling_on_valid(s, lambda x: x / x.shift(21) - 1.0)
    a = pd.DatetimeIndex(augA).intersection(s.index)
    print(f"\n{v}: today dd-from-252d-high {100*dd.iloc[-1]:+.1f}%, "
          f"trailing 21d {100*r21.iloc[-1]:+.1f}%")
    print(f"  August anchors: dd median {100*dd.loc[a].median():+.1f}%, "
          f"min {100*dd.loc[a].min():+.1f}%, "
          f"N with dd <= -20% = {int((dd.loc[a] <= -0.20).sum())} of {len(a)}, "
          f"N with dd >= -5% = {int((dd.loc[a] >= -0.05).sum())}")
    r10 = fwd_lag(s, 10, 1)
    aa = a.intersection(r10.dropna().index)
    deep = dd.loc[aa] <= dd.loc[aa].median()
    hot = r21.loc[aa] >= r21.loc[aa].median()
    print(f"  h=10 by drawdown half: deep {100*r10.loc[aa][deep.values].mean():+.2f}% "
          f"(N={int(deep.sum())}) vs shallow "
          f"{100*r10.loc[aa][~deep.values].mean():+.2f}%")
    print(f"  h=10 by trailing-21d half: hot {100*r10.loc[aa][hot.values].mean():+.2f}% "
          f"(N={int(hot.sum())}) vs cool "
          f"{100*r10.loc[aa][~hot.values].mean():+.2f}%")

print("\n\n===== 3. REGISTRY COLLISION with the parked crude JH-6 entry =====")
ov = []
for d in augA:
    y = d.year
    jd = [x for x in jh if x.year == y]
    if not jd:
        continue
    p_jh = int(idx.searchsorted(jd[0]))
    if p_jh >= len(idx):
        continue  # registry 2026-08-11: a future event date mints a fake anchor
    ov.append((y, pos[d] + 1 - p_jh))   # entry session position vs JH session
s = pd.Series({y: k for y, k in ov})
print("entry session (opex close) minus the JH session, in trading days, by year:")
print("  " + "  ".join(f"{y}:{k:+d}" for y, k in s.items()))
print(f"  |offset| <= 3 in {int((s.abs() <= 3).sum())} of {len(s)} years; "
      f"median {s.median():+.1f}")
print("2026: opex close 2026-08-21 is JH-5; the parked entry's anchor JH-6 is")
print("2026-08-20, ONE session earlier. Same window, same vehicles.")
print("\nparked entry's own state block: 'do not take it with XLE within 5% of")
print("its 52-week high'.")
sx = px["XLE"].dropna()
hi = rolling_on_valid(sx, lambda x: x.rolling(252).max())
print(f"  XLE 2026-08-20 close {sx.iloc[-1]:.2f}, 252d high {hi.iloc[-1]:.2f}, "
      f"distance {100*(sx.iloc[-1]/hi.iloc[-1]-1):+.2f}% -> BLOCK "
      f"{'FIRES' if sx.iloc[-1]/hi.iloc[-1] >= 0.95 else 'clear'}")

print("\n\n===== 4. the dead classes, priced against cost so the kill is numeric =====")
COST = {"HYG": 4, "LQD": 4, "IEF": 3, "TLT": 2, "UUP": 6}
tdom = pd.Series(pd.Series(idx, index=idx).groupby([idx.year, idx.month])
                 .cumcount().values + 1, index=idx)
ATD = [9, 10, 11, 12, 13, 14]
for v, c in COST.items():
    r = fwd_lag(px[v].dropna(), 10, 1).dropna()
    a = pd.DatetimeIndex(aA).intersection(r.index)
    ctrl = r.index[(tdom.reindex(r.index).isin(ATD)) & (r.index >= a[0])
                   & (r.index <= a[-1])].difference(a)
    x = 100 * (r.loc[a].mean() - r.loc[ctrl].mean())
    print(f"{v}: h=10 excess-vs-tdom {x:+.3f}pp = {x*100:+.1f} bps against a "
          f"{c} bp round trip = {x*100/c:+.1f}x (need >=5x)")

print("\n\n===== 5. SLV modern-era survival, the only cell still standing =====")
s = px["SLV"].dropna()
r10 = fwd_lag(s, 10, 1)
a = pd.DatetimeIndex(augA).intersection(r10.dropna().index)
v = r10.loc[a]
for cut in (2013, 2016, 2018):
    m = v[v.index.year >= cut]
    o = np.argsort(-m.values)
    print(f"SLV Aug h=10, {cut}+: mean {100*m.mean():+.3f}% N={len(m)} "
          f"{int((m>0).sum())}-{int((m<=0).sum())} sign p="
          f"{sign_test(int((m>0).sum()), len(m)):.4f} | drop1 "
          f"{100*np.delete(m.values,o[:1]).mean():+.3f}% drop2 "
          f"{100*np.delete(m.values,o[:2]).mean():+.3f}% | worst "
          f"{100*m.min():+.2f}% ({m.idxmin().year})")
    base = r10.dropna()
    base = base[(base.index.month == 8) & (base.index.year >= cut)]
    print(f"    vs its own August all-sessions base {100*base.mean():+.3f}% "
          f"(N={len(base)}) -> excess {100*(m.mean()-base.mean()):+.3f}pp")
