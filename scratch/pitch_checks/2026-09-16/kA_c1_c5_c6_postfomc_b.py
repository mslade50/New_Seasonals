"""kA round 2 (kill confirmation) for c1/c5/c6 post-decision-close cells.

Round 1 found every gated form WRONG-SIGNED on the tdom-matched excess.
This script checks the kill is not an artefact:
  A. placebo ladder k=-5..+5 around the decision close (entry shifted, episode
     set fixed by the eve gate), raw mean and tdom-matched excess, h=1 and h=3
  B. regime vs gate: hike-regime decisions NOT near the 252 max
  C. where does the Hillenbrand window live? eve close -> decision close (the
     announcement session, NOT tradeable at the decision close) vs D -> D+h
  D. 2022 and 2026 decisions individually (TLT, DX short, GLD, h=1/h=3)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

VEH = {"TLT long": ("TLT", 1.0), "DX short": ("DX-Y.NYB", -1.0), "GLD long": ("GLD", 1.0)}
pxd = load_prices(["TLT", "GLD", "DX-Y.NYB", "^TNX", "^IRX"])
tnx = pxd["^TNX"]["Close"].dropna()
irx = pxd["^IRX"]["Close"].dropna()
mx = tnx.rolling(252).max()
dist = tnx / mx - 1.0
chg21 = tnx - tnx.shift(21)
irx_chg = irx - irx.shift(126)
fomc = load_events(["fomc_decision"])["date"]
fomc = fomc[fomc <= pd.Timestamp("2026-09-15")]

rows = {}
for d in fomc:
    e = tnx.index[tnx.index < d]
    if len(e) == 0 or np.isnan(mx.get(e[-1], np.nan)):
        continue
    e = e[-1]
    ir, ic = irx.asof(e), irx_chg.asof(e)
    reg = "zirp" if ir < 0.30 else ("hike" if ic > 0.25 else ("cut" if ic < -0.25 else "flat"))
    rows[d] = {"dist": dist[e], "chg21": chg21[e], "regime": reg}
st = pd.DataFrame(rows).T
st["dist"] = st["dist"].astype(float)
st["chg21"] = st["chg21"].astype(float)
SETS = {"ALL": st.index, "at_max": st.index[st.dist >= -1e-9],
        "w2.0": st.index[st.dist >= -0.02],
        "hike_notw2": st.index[(st.regime == "hike") & (st.dist < -0.02)],
        "hike_all": st.index[st.regime == "hike"],
        "cut+zirp+flat": st.index[st.regime != "hike"]}
print({k: len(v) for k, v in SETS.items()})


def tdom_of(idx):
    ym = pd.Series(idx.year * 100 + idx.month, index=idx)
    return ym.groupby(ym.values).cumcount().values + 1


for lbl, (tk, w) in VEH.items():
    s = pxd[tk]["Close"].dropna()
    idx = s.index
    td = tdom_of(idx)
    pos = pd.Series(np.arange(len(idx)), index=idx)
    dpos = {d: pos[d] for d in st.index if d in pos.index}
    excl = np.zeros(len(idx), bool)
    for p in dpos.values():
        excl[max(0, p - 6):p + 12] = True
    print(f"\n{'=' * 100}\n{lbl} ({tk})\n{'=' * 100}")
    for h in (1, 3):
        r = (w * (s.shift(-h) / s - 1.0)).values
        ok = ~np.isnan(r)
        bucket = {j: np.nanmean(r[(td == j) & ~excl & ok]) for j in np.unique(td)}
        out = []
        for sname in ("ALL", "at_max", "w2.0"):
            for k in range(-5, 6):
                v, x = [], []
                for d in SETS[sname]:
                    if d not in dpos:
                        continue
                    p = dpos[d] + k
                    if 0 <= p < len(r) and ok[p]:
                        v.append(r[p])
                        x.append(r[p] - bucket[td[p]])
                v, x = np.array(v), np.array(x)
                out.append({"set": sname, "k": k, "n": len(v), "mean": 100 * v.mean(),
                            "tdomX": 100 * x.mean(), "hit": 100 * (v > 0).mean()})
        df = pd.DataFrame(out)
        print(f"\n A. placebo ladder h={h} (k=0 = decision close entry)")
        for sname, g in df.groupby("set", sort=False):
            g = g.reset_index(drop=True)
            rk_m = int((g["mean"] > g.loc[g.k == 0, "mean"].iloc[0]).sum()) + 1
            rk_x = int((g["tdomX"] > g.loc[g.k == 0, "tdomX"].iloc[0]).sum()) + 1
            print(f"  {sname:7s} n={g.n.iloc[5]:3d}  k0 mean {g['mean'].iloc[5]:+.3f}% "
                  f"tdomX {g['tdomX'].iloc[5]:+.3f}%  rank by mean {rk_m}/11, by tdomX {rk_x}/11")
            print("    means k=-5..+5: " + " ".join(f"{m:+.2f}" for m in g["mean"]))
        # B. regime vs gate
        rb = []
        for sname in ("hike_all", "hike_notw2", "w2.0", "cut+zirp+flat"):
            v = np.array([r[dpos[d]] for d in SETS[sname] if d in dpos and ok[dpos[d]]])
            x = np.array([r[dpos[d]] - bucket[td[dpos[d]]] for d in SETS[sname]
                          if d in dpos and ok[dpos[d]]])
            wn = int((x > 0).sum())
            rb.append({"set": sname, "n": len(v), "mean": round(100 * v.mean(), 3),
                       "tdomX": round(100 * x.mean(), 3), "X_rec": f"{wn}-{len(x) - wn}"})
        print(f"\n B. regime vs gate h={h}")
        print(pd.DataFrame(rb).to_string(index=False))
    # C. announcement session eve->D
    r1 = (w * (s / s.shift(1) - 1.0)).values
    cc = []
    for sname in ("ALL", "at_max", "w2.0", "hike_all"):
        v = np.array([r1[dpos[d]] for d in SETS[sname] if d in dpos and not np.isnan(r1[dpos[d]])])
        wn = int((v > 0).sum())
        cc.append({"set": sname, "n": len(v), "eve->D mean": round(100 * v.mean(), 3),
                   "rec": f"{wn}-{len(v) - wn}", "sign_p": round(sign_test(wn, len(v)), 3)})
    print("\n C. announcement session (eve close -> decision close; not tradeable at D close)")
    print(pd.DataFrame(cc).to_string(index=False), f"  all-days daily mean {100*np.nanmean(r1):+.3f}%")
    # D. 2022 + 2026
    dd = [d for d in st.index if d.year in (2022, 2026) and d in dpos]
    tab = []
    for d in dd:
        p = dpos[d]
        tab.append({"date": d.date(), "dist%": round(100 * st.loc[d, "dist"], 2),
                    "chg21bp": round(100 * st.loc[d, "chg21"], 1),
                    "eve->D": round(100 * r1[p], 3),
                    "h1": round(100 * w * (s.iloc[p + 1] / s.iloc[p] - 1), 3) if p + 1 < len(s) else np.nan,
                    "h3": round(100 * w * (s.iloc[p + 3] / s.iloc[p] - 1), 3) if p + 3 < len(s) else np.nan})
    print("\n D. 2022 and 2026 decisions")
    print(pd.DataFrame(tab).to_string(index=False))
