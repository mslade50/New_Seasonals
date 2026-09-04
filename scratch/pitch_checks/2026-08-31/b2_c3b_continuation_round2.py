"""C3 round 2 -- the two sub-cells round 1 left standing.

Round 1 killed the plain short (episodes gap5 h=5: SHORT GLD -0.102%, SHORT
GDX -0.042%, SHORT EW +0.069% at 0.5x cost) and the reference class came back
homogeneous (I^2 = 0.0%, common excess +0.056% at h=5, permutation max-of-6
p 0.66).  Two things still looked alive and both get charged properly here:

  S1  SHORT SLV at short horizons (h=1 +0.531% t 1.98, h=3 +0.772% t 1.90).
      Era, per-year, drop-2008, hit-vs-mean decomposition, cost, and TODAY's
      own SLV 21d-rank bucket.
  S2  The ^TNX-up split (SHORT GDX h=3 in the TNX-up half: +1.356%, 65.7%,
      sign p 0.006).  It was found by scanning 7 splits x 4 vehicles x 2
      horizons; charge the scan with a permutation max-of-N.
  S3  TODAY'S CELL, stated exactly: GDX 21d rank >= 95 on a complex-break day.
      Both sides reported, including what it says about the OPEN 2026-08-27
      long, which cannot be re-pitched either way.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa: F401,F403

import numpy as np
import pandas as pd

pd.set_option("display.width", 230)

FAMILIES = {
    "metals   GLD/SLV/GDX": ["GLD", "SLV", "GDX"],
    "energy   XLE/XOP/USO": ["XLE", "XOP", "USO"],
    "semis    SMH/AMAT/NVDA": ["SMH", "AMAT", "NVDA"],
    "banks    XLF/KRE/BAC": ["XLF", "KRE", "BAC"],
    "hombld   XHB/ITB/DHI": ["XHB", "ITB", "DHI"],
    "materls  XME/FCX/NUE": ["XME", "FCX", "NUE"],
}
BASE = ["GLD", "SLV", "GDX", "DX-Y.NYB", "^TNX"]


def mkpanel(t, start=None):
    p = close_panel(t).dropna()
    return p.loc[start:] if start else p


def dret(s):
    return s / s.shift(1) - 1.0


def cell(ret, px, mask, label, gap=5):
    s = px.index[mask.reindex(px.index, fill_value=False).values & ret.notna().values]
    if not len(s):
        return {"label": label, "n": 0}
    e = declusters(s, gap, px.index)
    r = summarize(ret.loc[e].values, label)
    r["n_days"] = len(s)
    w = int((ret.loc[e].values > 0).sum())
    r["sign_p"] = round(sign_test(w, len(e)), 4)
    return r


def main():
    px = mkpanel(BASE, "2006-05-22")
    r = {t: dret(px[t]) for t in BASE}
    trig = (r["GLD"] <= -0.02) & (r["SLV"] <= -0.02) & (r["GDX"] <= -0.02)
    dxu, tnu = r["DX-Y.NYB"] > 0, r["^TNX"] > 0
    yr = pd.Series(px.index.year, index=px.index)
    slv_r21 = pct_rank(px["SLV"], 21)
    gdx_r21 = pct_rank(px["GDX"], 21)

    # ---------------- S1  SHORT SLV -----------------------------------------
    print("########## S1  SHORT SLV, the one cell with a positive absolute return ##########")
    legs = [("SLV", -1.0)]
    for h in (1, 3, 5):
        ret = vehicle_ret(px, legs, h, lag=1)
        rows = [
            cell(ret, px, trig, "all triggers"),
            cell(ret, px, trig & (yr < 2018), "pre-2018"),
            cell(ret, px, trig & (yr >= 2018), "2018+"),
            cell(ret, px, trig & (yr != 2008), "drop 2008"),
            cell(ret, px, trig & ~yr.between(2008, 2011), "drop 2008-2011"),
            cell(ret, px, trig & (yr >= 2018) & dxu & tnu, "2018+ AND today's DX/TNX cfg"),
            cell(ret, px, trig & (slv_r21 >= 60), "SLV r21 >= 60 (today 65.9)"),
            cell(ret, px, trig & (gdx_r21 >= 95), "GDX r21 >= 95 (today 97.2)"),
        ]
        show(rows, f"S1 SHORT SLV h={h} (episodes gap5)")
        e = declusters(px.index[trig.values & ret.notna().values], 5, px.index)
        v = ret.loc[e]
        by = v.groupby(v.index.year).agg(["size", "mean"])
        by["mean"] = (100 * by["mean"]).round(3)
        pos = int((by["mean"] > 0).sum())
        print(f"  per-year: positive {pos}/{len(by)} sign p {sign_test(pos, len(by)):.4f}"
              f"   worst yr {by['mean'].min():+.2f}% ({by['mean'].idxmin()})"
              f"   best yr {by['mean'].max():+.2f}% ({by['mean'].idxmax()})")
        print("  " + cluster_note(e, v.values, k=3))
        # median vs mean: is it a right-tail artefact?
        vv = v.values
        print(f"  mean {100*vv.mean():+.3f}%  median {100*np.median(vv):+.3f}%  "
              f"trimmed-10% mean "
              f"{100*np.mean(np.sort(vv)[int(.05*len(vv)):len(vv)-int(.05*len(vv))]):+.3f}%"
              f"   wins {(vv>0).sum()}/{len(vv)}")

    # ---------------- S2  TNX-up split multiplicity --------------------------
    print("\n\n########## S2  THE ^TNX-UP SPLIT, CHARGED FOR THE SCAN ##########")
    vehicles = {"SHORT GLD": [("GLD", -1.0)], "SHORT GDX": [("GDX", -1.0)],
                "SHORT SLV": [("SLV", -1.0)],
                "SHORT EW": [("GLD", -1 / 3), ("SLV", -1 / 3), ("GDX", -1 / 3)]}
    splits = {"DX up": trig & dxu, "DX down": trig & ~dxu, "TNX up": trig & tnu,
              "TNX down": trig & ~tnu, "both up": trig & dxu & tnu,
              "neither up": trig & ~dxu & ~tnu, "all": trig}
    obs, cells = [], []
    for vn, lg in vehicles.items():
        for h in (1, 3, 5):
            ret = vehicle_ret(px, lg, h, lag=1)
            base = ret.dropna()
            for sn, m in splits.items():
                s = px.index[m.values & ret.notna().values]
                if len(s) < 10:
                    continue
                e = declusters(s, 5, px.index)
                edge = 100 * (ret.loc[e].mean() - base.mean())
                obs.append((f"{vn} h={h} {sn}", edge, len(e)))
                cells.append((base.values, len(e)))
    obs_sorted = sorted(obs, key=lambda x: -x[1])
    print(f"  {len(obs)} split-cells scanned. top 6 by excess over all-days:")
    for lbl, e, n in obs_sorted[:6]:
        print(f"    {lbl:34s} excess {e:+.3f}%  (N_epi {n})")
    rng = np.random.default_rng(11)
    B = 3000
    nm = np.zeros(B)
    for b in range(B):
        nm[b] = max(100 * (rng.choice(pool, size=k, replace=False).mean() - pool.mean())
                    for pool, k in cells)
    best = obs_sorted[0][1]
    print(f"  permutation max-of-{len(obs)} p = {(nm >= best).mean():.4f}  "
          f"(best observed {best:+.3f}%, null max median {np.median(nm):+.3f}%, "
          f"95th {np.percentile(nm, 95):+.3f}%)")

    # ---------------- S3  TODAY'S EXACT CELL ---------------------------------
    print("\n\n########## S3  TODAY'S EXACT CELL: complex break with GDX r21 >= 95 ##########")
    m_today = trig & (gdx_r21 >= 95)
    dates = px.index[m_today.values]
    print(f"  trigger days with GDX r21 >= 95: {len(dates)} -> "
          f"{[str(d.date()) for d in dates]}")
    for lbl, lg in [("SHORT GDX", [("GDX", -1.0)]), ("LONG GDX", [("GDX", 1.0)]),
                    ("SHORT EW", [("GLD", -1 / 3), ("SLV", -1 / 3), ("GDX", -1 / 3)])]:
        rows = []
        for h in (1, 2, 3, 5, 7, 10):
            ret = vehicle_ret(px, lg, h, lag=1)
            rows.append(cell(ret, px, m_today, f"h={h}"))
        show(rows, f"S3 {lbl} on GDX r21 >= 95 complex-break days")
    # loosen the rung so the cell is not just N=3
    print("\n  rung loosening (does the sign hold as the bucket fills?):")
    ret3 = vehicle_ret(px, [("GDX", -1.0)], 3, lag=1)
    ret5 = vehicle_ret(px, [("GDX", -1.0)], 5, lag=1)
    rows = []
    for rung in (70, 80, 85, 90, 95):
        rows.append(cell(ret3, px, trig & (gdx_r21 >= rung), f"SHORT GDX h=3, r21>={rung}"))
        rows.append(cell(ret5, px, trig & (gdx_r21 >= rung), f"SHORT GDX h=5, r21>={rung}"))
    show(rows, "S3b rung sensitivity on the run conditioner")

    # ---------------- S4  reference class for the SLV form -------------------
    print("\n\n########## S4  REFERENCE CLASS for 'short the highest-beta member' ##########")
    HIGH_BETA = {"metals   GLD/SLV/GDX": "SLV", "energy   XLE/XOP/USO": "USO",
                 "semis    SMH/AMAT/NVDA": "NVDA", "banks    XLF/KRE/BAC": "BAC",
                 "hombld   XHB/ITB/DHI": "DHI", "materls  XME/FCX/NUE": "FCX"}
    from scipy import stats as st
    for h in (1, 3):
        res, fam_data = [], []
        for fam, members in FAMILIES.items():
            p = mkpanel(members)
            rr = {t: dret(p[t]) for t in members}
            m = None
            for t in members:
                mm = rr[t] <= -0.02
                m = mm if m is None else (m & mm)
            lg = [(HIGH_BETA[fam], -1.0)]
            ret = vehicle_ret(p, lg, h, lag=1)
            valid = ret.notna()
            s = p.index[m.values & valid.values]
            if len(s) < 5:
                continue
            e = declusters(s, 5, p.index)
            spanidx = p.index[(p.index >= s[0]) & (p.index <= s[-1]) & valid.values]
            cond, ctrl = ret.loc[e].values, ret.loc[spanidx].values
            se = np.sqrt(cond.var(ddof=1) / len(cond) + ctrl.var(ddof=1) / len(ctrl))
            res.append({"family": fam, "leg": HIGH_BETA[fam], "n_epi": len(e),
                        "cond_pct": 100 * cond.mean(), "ctrl_pct": 100 * ctrl.mean(),
                        "excess_pct": 100 * (cond.mean() - ctrl.mean()),
                        "se_pct": 100 * se, "t": (cond.mean() - ctrl.mean()) / se,
                        "hit": 100 * (cond > 0).mean()})
            fam_data.append((fam, len(e), ctrl))
        show(res, f"S4 SHORT the high-beta member after a complex break, h={h}")
        y = np.array([x["excess_pct"] for x in res])
        se = np.array([x["se_pct"] for x in res])
        w = 1 / se ** 2
        fe = (w * y).sum() / w.sum()
        Q = float((w * (y - fe) ** 2).sum())
        df = len(y) - 1
        I2 = max(0.0, (Q - df) / Q) * 100 if Q > 0 else 0.0
        print(f"  Cochran Q = {Q:.2f} (df {df}, p {1-st.chi2.cdf(Q, df):.3f})  I^2 = {I2:.1f}%"
              f"   fixed-effect common excess = {fe:+.3f}% (se {1/np.sqrt(w.sum()):.3f}, "
              f"t {fe*np.sqrt(w.sum()):+.2f})")
        rng = np.random.default_rng(5)
        nm = np.zeros(3000)
        for b in range(3000):
            nm[b] = 100 * max(rng.choice(pool, size=k, replace=False).mean() - pool.mean()
                              for _, k, pool in fam_data)
        print(f"  permutation max-of-{len(fam_data)} p = {(nm >= y.max()).mean():.4f} "
              f"(observed max {y.max():+.3f}% = {res[int(np.argmax(y))]['family']})")


if __name__ == "__main__":
    main()
