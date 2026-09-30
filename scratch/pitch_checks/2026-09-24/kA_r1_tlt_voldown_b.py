"""kA R1 round 2: concentration, definition-neighbour grid, era/regime,
gate attribution (volume leg, at-low leg as FILTER vs RE-ANCHOR, the MOVE
spike as the mechanism's own signature), vehicle neighbour IEF.
"""
import sys
from itertools import product
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

TK = ["TLT", "IEF", "^TNX", "^MOVE", "SPY"]
raw = load_prices(TK)
tlt = raw["TLT"]
IDX = tlt.index
px = close_panel(TK).reindex(IDX)
c = tlt["Close"]
v = tlt["Volume"].astype(float)
r1 = c.pct_change()
vr = v / v.rolling(63).mean().shift(1)
lo = c.rolling(252).min()
dist = c / lo - 1.0
at_low = dist <= 1e-9
move_chg = rolling_on_valid(px["^MOVE"], lambda x: x.pct_change()).reindex(IDX)
spy1 = px["SPY"].pct_change()
tnx = px["^TNX"].ffill()


def mk(d, vv, lw):
    return ((r1 <= d) & (vr >= vv) & (dist <= lw + 1e-9)).fillna(False)


cell = mk(-0.0125, 1.5, 0.0)


def epis(mask, h):
    ret = vehicle_ret(px, [("TLT", 1.0)], h, 1)
    s = IDX[mask.values & ret.notna().values]
    e = declusters(s, h, IDX)
    return e, ret.loc[e].values, ret


# 1. concentration / LOYO / ex-top
for H in (2, 3, 5):
    e, ep, _ = epis(cell, H)
    o = np.argsort(-ep)
    w = int((ep > 0).sum())
    print(f"\nh={H}: N={len(ep)} mean {100*ep.mean():+.3f}% median {100*np.median(ep):+.3f}% "
          f"rec {w}-{len(ep)-w} p {sign_test(w, len(ep)):.4f}")
    print(f"  {cluster_note(e, ep)}")
    print(f"  ex-top2 mean {100*ep[o[2:]].mean():+.3f}% (N={len(ep)-2});  ex-2022 "
          f"{100*ep[e.year != 2022].mean():+.3f}% (N={int((e.year != 2022).sum())}, "
          f"rec {int((ep[e.year != 2022] > 0).sum())}-{int((ep[e.year != 2022] <= 0).sum())})")
    loyo = {y: 100 * ep[e.year != y].mean() for y in sorted(set(e.year))}
    print("  LOYO:", {k: round(x, 3) for k, x in loyo.items()})

# 2. neighbour grid (27 cells) at h=2,3,5
grid = []
for d, vv, lw in product((-0.010, -0.0125, -0.015), (1.25, 1.5, 2.0), (0.0, 0.005, 0.01)):
    row = {"down": d, "vol": vv, "low": lw}
    for H in (2, 3, 5):
        e, ep, _ = epis(mk(d, vv, lw), H)
        row[f"n{H}"] = len(ep)
        row[f"h{H}"] = round(100 * ep.mean(), 3) if len(ep) else np.nan
        if H == 5:
            row["rec5"] = f"{int((ep>0).sum())}-{int((ep<=0).sum())}"
    grid.append(row)
g = pd.DataFrame(grid)
print("\n=== 2. neighbour grid (episode means, %) ===")
print(g.to_string(index=False))
for H in (2, 3, 5):
    col = g[f"h{H}"]
    print(f"  h={H}: {int((col > 0).sum())}/27 positive, median {col.median():+.3f}%, "
          f"min {col.min():+.3f}%, max {col.max():+.3f}%")

# 3. gate attribution at h=5 (and h=3)
for H in (3, 5):
    ret = vehicle_ret(px, [("TLT", 1.0)], H, 1)
    base = (r1 <= -0.0125).fillna(False)
    sets = {
        "cell": cell,
        "no vol gate (down & at low)": (base & at_low).fillna(False),
        "vol leg complement (at low, vol<1.5)": (base & at_low & (vr < 1.5)).fillna(False),
        "no low gate (down & vol)": (base & (vr >= 1.5)).fillna(False),
        "low leg complement (down & vol, >1% off low)": (base & (vr >= 1.5) & (dist > 0.01)).fillna(False),
        "cell & MOVE >= 96.7 pctile": (cell & (move_chg >= move_chg.quantile(0.967))),
        "cell & MOVE < 96.7 pctile": (cell & (move_chg < move_chg.quantile(0.967))),
        "cell & MOVE >= +10%": (cell & (move_chg >= 0.10)),
        "cell & SPY down same day": (cell & (spy1 < 0)),
        "cell & SPY up same day": (cell & (spy1 >= 0)),
    }
    rows = []
    for k, m in sets.items():
        s = IDX[m.reindex(IDX, fill_value=False).values & ret.notna().values]
        e = declusters(s, H, IDX)
        ep = ret.loc[e].values
        r = summarize(ep, k)
        r["rec"] = f"{int((ep>0).sum())}-{int((ep<=0).sum())}"
        rows.append(r)
    show(rows, f"3. gate attribution h={H}")

    par = declusters(IDX[((base & (vr >= 1.5)).fillna(False)).values & ret.notna().values], H, IDX)
    chi = declusters(IDX[cell.values & ret.notna().values], H, IDX)
    pm = pd.Series(False, index=IDX); pm.loc[par] = True
    cm = pd.Series(False, index=IDX); cm.loc[chi] = True
    fr = filter_vs_reanchor(ret, pm, cm, IDX, window_td=21,
                            label=f"at-low leg on down&vol parent, h={H}")
    if fr["shifts"]:
        nz = [s for s in fr["shifts"]]
        print(f"  shifts: {nz}")
        kept_par = [a for a, _, _ in fr["pairs"]]
        rn = reanchor_null(ret, kept_par, fr["shifts"], IDX,
                           child_mean=fr["kept_at_child_pct"] / 100)
        print(f"  reanchor_null p {rn['p']:.3f} (null mean {rn.get('null_mean_pct', np.nan):+.3f}%)")

# 4. era / regime at h=5
e, ep, _ = epis(cell, 5)
t252 = (tnx - tnx.shift(252)).loc[e].values
show(era_split(e, ep), "4a. era split h=5")
show([summarize(ep[t252 >= 1.0], "TNX +100bp or more over 252d"),
      summarize(ep[t252 < 1.0], "TNX < +100bp over 252d")], "4b. rate-trend strength")
mid = np.array([d.year % 4 == 2 for d in e])
show([summarize(ep[mid], "midterm"), summarize(ep[~mid], "non-midterm")], "4c. midterm")

# 5. vehicle neighbour: IEF on the TLT trigger, and IEF on its own trigger
iv = raw["IEF"]
ic = iv["Close"]; ir1 = ic.pct_change()
ivr = iv["Volume"].astype(float) / iv["Volume"].astype(float).rolling(63).mean().shift(1)
ilow = ic <= ic.rolling(252).min() * (1 + 1e-9)
rows = []
for H in (2, 3, 5):
    ret = vehicle_ret(px, [("IEF", 1.0)], H, 1)
    for lbl, m in (("TLT trigger, IEF vehicle", cell),
                   ("IEF own: r1<=-0.6% & vol>=1.5 & low", (ir1 <= -0.006) & (ivr >= 1.5) & ilow),
                   ("IEF own: r1<=-0.75% & vol>=1.5 & low", (ir1 <= -0.0075) & (ivr >= 1.5) & ilow)):
        m = m.reindex(IDX, fill_value=False).fillna(False)
        s = IDX[m.values & ret.notna().values]
        ee = declusters(s, H, IDX)
        x = ret.loc[ee].values
        r = summarize(x, f"h={H} {lbl}")
        r["rec"] = f"{int((x>0).sum())}-{int((x<=0).sum())}"
        r["ctl_all"] = round(100 * ret.mean(), 3)
        rows.append(r)
show(rows, "5. vehicle neighbour IEF")
print(f"\nLIVE IEF: r1 {100*ir1.iloc[-1]:+.2f}% vol {ivr.iloc[-1]:.2f}x at low {bool(ilow.iloc[-1])}")
print(f"LIVE TNX 252d change {100*(tnx.iloc[-1]-tnx.iloc[-253]):+.1f} bp; MOVE chg {100*move_chg.iloc[-1]:+.1f}%"
      f" (spike thr {100*move_chg.quantile(0.967):.2f}%); SPY {100*spy1.iloc[-1]:+.2f}%")
e, ep, _ = epis(cell, 5)
spk = (move_chg.loc[e] >= move_chg.quantile(0.967)).values
for d, x, s in zip(e, ep, spk):
    print(f"  {d.date()} h5 {100*x:+.3f}%  MOVE spike {bool(s)}  MOVE chg {100*move_chg.loc[d]:+.1f}%")
