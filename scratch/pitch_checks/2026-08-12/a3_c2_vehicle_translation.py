"""C2 round 1 -- instrument translation. Is a lower-duration vehicle actually
better after cost, or is the edge simply proportional to duration (in which case
TLT wins and C2 is not a separate idea)?

Round trips priced honestly: TLT ~2.5 bps, IEF ~2 bps, LQD ~3 bps, HYG ~3 bps.
Everything tdom-matched, and reported per unit of the instrument's OWN daily
vol so 'proportional to duration' is testable rather than asserted.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

COST = {"TLT": 2.5, "IEF": 2.0, "LQD": 3.0, "HYG": 3.0, "SPY": 1.0, "AGG": 2.0,
        "SHY": 2.0, "TLH": 4.0, "VCIT": 3.0}
TK = ["TLT", "IEF", "LQD", "HYG", "SPY", "AGG", "SHY", "TLH", "VCIT"]
px = close_panel(TK)
TK = [t for t in TK if t in px.columns]
print(f"vehicles available in cache: {TK}")

ev = load_events()
ref = px["TLT"].dropna()
ridx = ref.index
sp = lambda k: sorted({int(ridx.searchsorted(x, "left"))
                       for x in ev[ev.event == k]["date"]
                       if 0 <= int(ridx.searchsorted(x, "left")) < len(ridx)})
ppi_l = [p for p in sp("ppi") if 1 <= p < len(ridx)]
cpi_all = set(sp("cpi"))
prn_d = pd.DatetimeIndex([ridx[p] for p in ppi_l])
eve_cpi = np.array([(p - 1) in cpi_all for p in ppi_l])


def cell_for(t):
    s = px[t].dropna()
    ii, cc = s.index, s.values
    d1 = np.full(len(cc), np.nan)
    d1[1:] = cc[1:] / cc[:-1] - 1.0
    ym = pd.Series(ii.year * 100 + ii.month, index=ii)
    tdom = ym.groupby(ym.values).cumcount().values + 1
    prn_set = set()
    for d in prn_d:
        if d in ii:
            prn_set.add(ii.get_loc(d))
    buck = {}
    for j in range(1, 25):
        m = (tdom == j) & ~np.isnan(d1) & ~np.isin(np.arange(len(cc)), list(prn_set))
        if m.sum() >= 10:
            buck[j] = d1[m].mean()
    raw, exc, flag, yy = [], [], [], []
    for i, d in enumerate(prn_d):
        if d not in ii:
            continue
        a = ii.get_loc(d)
        if a < 1 or np.isnan(d1[a]):
            continue
        b = buck.get(int(tdom[a]))
        raw.append(d1[a])
        exc.append(d1[a] - b if b is not None else np.nan)
        flag.append(eve_cpi[i])
        yy.append(d.year)
    return (np.array(raw), np.array(exc), np.array(flag, bool),
            np.array(yy), np.nanmean(d1), np.nanstd(d1, ddof=1),
            float((d1[~np.isnan(d1)] > 0).mean()))


print("=" * 116)
print("1. PARENT PPI-PRINT CELL BY VEHICLE (tdom-matched excess, cost-adjusted)")
print("=" * 116)
rows = []
store = {}
for t in TK:
    raw, exc, fl, yy, drift, sd, bh = cell_for(t)
    if len(raw) < 30:
        rows.append({"tkr": t, "N": len(raw), "note": "too short a history"})
        continue
    store[t] = (raw, exc, fl, yy, drift, sd, bh)
    e = exc[~np.isnan(exc)]
    bps = 100 * 100 * e.mean()
    w = int((raw > 0).sum())
    rows.append({"tkr": t, "N": len(raw),
                 "raw_pct": round(100 * raw.mean(), 4),
                 "drift_pct": round(100 * drift, 4),
                 "tdom_exc_bps": round(bps, 2),
                 "hit": round(100 * w / len(raw), 1),
                 "signp": round(sign_test(w, len(raw), bh), 4),
                 "daily_sd_pct": round(100 * sd, 3),
                 "exc_per_sd": round(e.mean() / sd, 4),
                 "cost_bps": COST.get(t, 3.0),
                 "x_cost": round(bps / COST.get(t, 3.0), 2),
                 "net_bps": round(bps - COST.get(t, 3.0), 2)})
print(pd.DataFrame(rows).to_string(index=False))

print("\n" + "=" * 116)
print("2. THE LIVE CELL (CPI printed on the eve) BY VEHICLE")
print("=" * 116)
rows = []
for t, (raw, exc, fl, yy, drift, sd, bh) in store.items():
    r2, e2 = raw[fl], exc[fl]
    if len(r2) < 10:
        continue
    e2 = e2[~np.isnan(e2)]
    bps = 100 * 100 * e2.mean()
    w = int((r2 > 0).sum())
    rows.append({"tkr": t, "N": len(r2), "raw_pct": round(100 * r2.mean(), 4),
                 "tdom_exc_bps": round(bps, 2),
                 "hit": round(100 * w / len(r2), 1),
                 "signp": round(sign_test(w, len(r2), bh), 4),
                 "exc_per_sd": round(e2.mean() / sd, 4),
                 "cost_bps": COST.get(t, 3.0),
                 "x_cost": round(bps / COST.get(t, 3.0), 2),
                 "net_bps": round(bps - COST.get(t, 3.0), 2)})
print(pd.DataFrame(rows).to_string(index=False))

print("\n" + "=" * 116)
print("3. IS THE EDGE PROPORTIONAL TO DURATION? regress excess on daily vol")
print("=" * 116)
xs, ys, labs = [], [], []
for t in ["SHY", "IEF", "TLH", "TLT", "AGG", "LQD", "VCIT", "HYG", "SPY"]:
    if t not in store:
        continue
    raw, exc, fl, yy, drift, sd, bh = store[t]
    e = exc[~np.isnan(exc)]
    xs.append(sd)
    ys.append(e.mean())
    labs.append(t)
xs, ys = np.array(xs), np.array(ys)
if len(xs) >= 3:
    b, a = np.polyfit(xs, ys, 1)
    pred = a + b * xs
    print(f"  excess = {100*100*a:+.2f} bps + {b:.3f} * daily_sd")
    print(f"  R^2 = {1 - ((ys-pred)**2).sum()/((ys-ys.mean())**2).sum():.3f}")
    for i, t in enumerate(labs):
        print(f"    {t:5s} sd {100*xs[i]:.3f}%  excess {100*100*ys[i]:+6.2f} bps  "
              f"fitted {100*100*pred[i]:+6.2f}  resid {100*100*(ys[i]-pred[i]):+6.2f}")
    print("\n  A high R^2 with a near-zero intercept means the cell is a pure")
    print("  DURATION beta: no vehicle choice adds anything and the credit names")
    print("  are carrying rates, not spread.")

print("\n" + "=" * 116)
print("4. SHARPE-EQUIVALENT: excess per unit of the SAME-DAY realised risk")
print("   (the real question -- lower duration wins only if net/sd is higher)")
print("=" * 116)
rows = []
for t, (raw, exc, fl, yy, drift, sd, bh) in store.items():
    e = exc[~np.isnan(exc)]
    net = e.mean() - COST.get(t, 3.0) / 10000.0
    e2 = exc[fl]
    e2 = e2[~np.isnan(e2)]
    net2 = e2.mean() - COST.get(t, 3.0) / 10000.0 if len(e2) else np.nan
    rows.append({"tkr": t, "parent_net_bps": round(100 * 100 * net, 2),
                 "parent_net_per_sd": round(net / sd, 4),
                 "live_net_bps": round(100 * 100 * net2, 2),
                 "live_net_per_sd": round(net2 / sd, 4),
                 "sd_pct": round(100 * sd, 3)})
print(pd.DataFrame(rows).sort_values("live_net_per_sd", ascending=False)
      .to_string(index=False))

print("\n" + "=" * 116)
print("5. LQD SPREAD COMPONENT: LQD excess residual against IEF over the cell")
print("   (registry: at h=5 the LQD-vs-IEF residual was +0.000pp -- no credit)")
print("=" * 116)
if "LQD" in store and "IEF" in store:
    li = px[["LQD", "IEF"]].dropna()
    dl = li["LQD"].pct_change()
    di = li["IEF"].pct_change()
    both = pd.concat([dl, di], axis=1).dropna()
    beta = np.polyfit(both["IEF"], both["LQD"], 1)[0]
    print(f"  full-sample beta(LQD on IEF) = {beta:.3f}")
    res = both["LQD"] - beta * both["IEF"]
    cells = [d for d in prn_d if d in res.index]
    rv = res.loc[cells].values
    lv = res.loc[[d for i, d in enumerate(prn_d)
                  if d in res.index and eve_cpi[i]]].values
    print(f"  parent cell LQD residual vs IEF: N={len(rv)} "
          f"{100*100*rv.mean():+.2f} bps  hit {100*(rv>0).mean():.1f}%")
    print(f"  live cell  LQD residual vs IEF: N={len(lv)} "
          f"{100*100*lv.mean():+.2f} bps  hit {100*(lv>0).mean():.1f}%")
    print("  -> a residual near zero means LQD is a worse-costed TLT, not a")
    print("     separate credit idea.")
