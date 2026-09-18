import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

TK = ["SPY", "^VIX", "SVXY", "UVXY", "VIXY", "^VIX3M"]
raw = close_panel(TK)
cal = raw["SPY"].dropna().index
px = raw.reindex(cal)
for t in TK:
    if t in px:
        s = px[t].dropna()
        print(t, "first", s.index[0].date() if len(s) else None, "last", s.index[-1].date() if len(s) else None, "n", len(s))
px["^VIX"] = px["^VIX"].ffill(limit=2)

vix = px["^VIX"]
vchg = vix / vix.shift(1) - 1
fomc = pd.DatetimeIndex(load_events(["fomc_decision"])["date"])
pos, kept = anchor_positions(cal, fomc, 0)
fpos = np.array(pos)

# k = sessions until next FOMC decision (1..3 means decision at d+k)
nxt = np.full(len(cal), 999)
j = 0
fps = np.sort(fpos)
for i in range(len(cal)):
    while j < len(fps) and fps[j] <= i:
        j += 1
    if j < len(fps):
        nxt[i] = fps[j] - i
k_to_fomc = pd.Series(nxt, index=cal)

crush = (vchg <= -0.10)
pre3 = k_to_fomc.between(1, 3)
live = pd.Timestamp("2026-09-11")
print("live: vchg", round(vchg.loc[live], 4), "k_to_fomc", k_to_fomc.loc[live])

post18 = cal >= pd.Timestamp("2018-03-01")
mid = pd.Series(cal.year % 4 == 2, index=cal)


def cells(tkr, w, h, lag=1, era=None):
    r = vehicle_ret(px, [(tkr, w)], h, lag)
    base = r.notna()
    if era is not None:
        base &= era
    out = []
    for lbl, m in [("CELL crush & FOMC in 1..3", crush & pre3),
                   ("crush & FOMC k=3 exact", crush & (k_to_fomc == 3)),
                   ("crush NOT pre-FOMC (gate complement)", crush & ~pre3),
                   ("FOMC 1..3 NO crush", pre3 & ~crush),
                   ("FOMC k=3 exact, no crush", (k_to_fomc == 3) & ~crush),
                   ("all days", pd.Series(True, index=cal))]:
        mm = (m & base).reindex(cal, fill_value=False)
        d = cal[mm.values]
        if lbl.startswith("CELL") or "crush" in lbl and "NOT" not in lbl and "NO" not in lbl:
            d = declusters(d, max(h, 3), cal)
        v = r.loc[d].values
        s = summarize(v, lbl)
        w_ = int((v > 0).sum())
        s["rec"] = f"{w_}-{len(v)-w_}"
        s["sign_p"] = round(sign_test(w_, len(v)), 4) if len(v) else np.nan
        out.append(s)
    return out


for h in [1, 2, 3]:
    show(cells("^VIX", 1.0, h), f"LONG ^VIX level change (sign info only), h={h}, full history")
    show(cells("SVXY", -1.0, h, era=pd.Series(post18, index=cal)), f"SHORT SVXY 2018-03+, h={h}")
    show(cells("UVXY", 1.0, h, era=pd.Series(post18, index=cal)), f"LONG UVXY 2018-03+, h={h}")
    pass

# SPY residual rule: regress SVXY on SPY over same h-holds, 2018-03+, all days
for h in [1, 2, 3]:
    rs = vehicle_ret(px, [("SVXY", 1.0)], h)
    rspy = vehicle_ret(px, [("SPY", 1.0)], h)
    ok = rs.notna() & rspy.notna() & pd.Series(post18, index=cal)
    b = np.polyfit(rspy[ok].values, rs[ok].values, 1)[0]
    res = rs - b * rspy
    m = (crush & (k_to_fomc == 3) & ok)
    d = declusters(cal[m.reindex(cal, fill_value=False).values], 3, cal)
    comp = (crush & ~pre3 & ok)
    dc = declusters(cal[comp.reindex(cal, fill_value=False).values], 3, cal)
    print(f"\nh={h} SVXY beta on SPY {b:.2f}; cell N={len(d)} SVXY {100*rs.loc[d].mean():+.3f}% SPY {100*rspy.loc[d].mean():+.3f}% "
          f"residual(long SVXY) {100*res.loc[d].mean():+.3f}% t={summarize(res.loc[d].values)['t']:.2f}; "
          f"complement resid {100*res.loc[dc].mean():+.3f}% (N={len(dc)}); all-day resid {100*res[ok].mean():+.3f}%")

# cell dates + midterm split, ^VIX and SVXY
m = crush & pre3
d = declusters(cal[m.values], 3, cal)
r2 = vehicle_ret(px, [("SVXY", -1.0)], 2)
rv2 = vehicle_ret(px, [("^VIX", 1.0)], 2)
print("\ncell episodes (short SVXY h2 / long VIX h2 / k_to_fomc / vchg):")
for x in d:
    print(f"  {x.date()} k={k_to_fomc.loc[x]} vchg {100*vchg.loc[x]:+.1f}% VIX {vix.loc[x]:.1f}  "
          f"shortSVXY {100*r2.loc[x]:+.2f}%  VIX {100*rv2.loc[x]:+.2f}%  mid={x.year%4==2}")
v = rv2.loc[d].values
show(era_split(d, v), "VIX h2 era split")
