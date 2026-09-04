"""C2 round 2 -- the three seams round 1 opened.

Round 1 left the LONG DX cell alive on the full sample (episodes gap5, h=5:
+0.182%, 12x cost, metals clause worth +0.432pp over bare DX momentum), but
three things looked wrong:

  R1  ERA. 2018+ episodes = +0.003% (h=5) / +0.070% (h=3) against pre-2018
      +0.257 / +0.174. Test whether ANYTHING survives 2018+, including the
      DX-up subset and today's exact configuration, and what dropping the
      2008-2011 dollar-crisis block does.
  R2  IS IT A COMPLEX CELL AT ALL? "GLD alone <= -2%" paid +0.181% on 144
      episodes against the three-name +0.182% on 119. Measure the INCREMENTAL
      value of the SLV+GDX clause directly (GLD-break days split on whether
      the other two joined).
  R3  REFERENCE CLASS. Does the dollar rally after ANY complex break? Run the
      identical "3 members each <= -2%" -> long DX rule on energy, semis,
      banks, homebuilders, materials. Cochran Q / I^2 / fixed-effect common
      excess / permutation max-of-N.
  R4  HORIZON MULTIPLICITY: h was scanned 1..10, so charge for it.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa: F401,F403

import numpy as np
import pandas as pd
from scipy import stats as st

pd.set_option("display.width", 220)

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
    dxu = r["DX-Y.NYB"] > 0
    tnu = r["^TNX"] > 0
    yr = pd.Series(px.index.year, index=px.index)
    legs = [("DX-Y.NYB", 1.0)]

    # ---------------- R1  ERA -------------------------------------------------
    print("########## R1  ERA ATTRIBUTION ##########")
    for h in (3, 5):
        ret = vehicle_ret(px, legs, h, lag=1)
        rows = [
            cell(ret, px, trig, "all triggers"),
            cell(ret, px, trig & (yr < 2018), "pre-2018"),
            cell(ret, px, trig & (yr >= 2018), "2018+"),
            cell(ret, px, trig & (yr >= 2018) & dxu, "2018+ AND DX up"),
            cell(ret, px, trig & (yr >= 2018) & dxu & tnu, "2018+ AND DX up AND TNX up (today's cfg)"),
            cell(ret, px, trig & ~yr.between(2008, 2011), "drop 2008-2011"),
            cell(ret, px, trig & ~yr.between(2008, 2011) & dxu, "drop 2008-2011 AND DX up"),
            cell(ret, px, trig & (yr >= 2013), "2013+"),
            cell(ret, px, trig & (yr >= 2020), "2020+"),
        ]
        show(rows, f"R1 era attribution, LONG DX h={h} (episodes gap5)")
        # per-year episode means
        ret5 = ret
        s = px.index[trig.values & ret5.notna().values]
        e = declusters(s, 5, px.index)
        v = ret5.loc[e]
        by = v.groupby(v.index.year).agg(["size", "mean"])
        by["mean"] = (100 * by["mean"]).round(3)
        print(f"  per-year episode mean %, h={h}:")
        print("   " + by.to_string().replace("\n", "\n   "))
        pos_years = int((by["mean"] > 0).sum())
        print(f"   positive years {pos_years}/{len(by)}  sign p = "
              f"{sign_test(pos_years, len(by)):.4f}")

    # ---------------- R2  IS IT A COMPLEX CELL? -------------------------------
    print("\n\n########## R2  INCREMENTAL VALUE OF THE SLV+GDX CLAUSE ##########")
    gld_break = r["GLD"] <= -0.02
    for h in (3, 5):
        ret = vehicle_ret(px, legs, h, lag=1)
        rows = [
            cell(ret, px, gld_break, "GLD <= -2% (any)"),
            cell(ret, px, gld_break & trig, "GLD <= -2% AND both others too (= the C2 cell)"),
            cell(ret, px, gld_break & ~trig, "GLD <= -2% and the others did NOT join"),
            cell(ret, px, (r["SLV"] <= -0.02) & (r["GDX"] <= -0.02) & ~gld_break,
                 "SLV+GDX broke but GLD did NOT"),
        ]
        show(rows, f"R2 incremental value of the complex clause, h={h}")
        a = rows[1]
        b = rows[2]
        if a.get("n") and b.get("n"):
            se = np.sqrt((a["sd_pct"] ** 2) / a["n"] + (b["sd_pct"] ** 2) / b["n"])
            print(f"  complex clause incremental over a bare GLD break = "
                  f"{a['mean_pct'] - b['mean_pct']:+.3f} pp  (welch t "
                  f"{(a['mean_pct']-b['mean_pct'])/se:+.2f}, N {a['n']} vs {b['n']})")

    # ---------------- R3  REFERENCE CLASS -------------------------------------
    print("\n\n########## R3  REFERENCE CLASS: does the dollar rally after ANY complex break? ##########")
    for h in (3, 5):
        res, fam_data = [], []
        for fam, members in FAMILIES.items():
            p = mkpanel(members + ["DX-Y.NYB"])
            rr = {t: dret(p[t]) for t in members}
            m = None
            for t in members:
                mm = rr[t] <= -0.02
                m = mm if m is None else (m & mm)
            ret = vehicle_ret(p, legs, h, lag=1)
            valid = ret.notna()
            s = p.index[m.values & valid.values]
            if len(s) < 5:
                continue
            e = declusters(s, 5, p.index)
            spanidx = p.index[(p.index >= s[0]) & (p.index <= s[-1]) & valid.values]
            cond, ctrl = ret.loc[e].values, ret.loc[spanidx].values
            se = np.sqrt(cond.var(ddof=1) / len(cond) + ctrl.var(ddof=1) / len(ctrl))
            res.append({"family": fam, "n_days": len(s), "n_epi": len(e),
                        "cond_pct": 100 * cond.mean(), "ctrl_pct": 100 * ctrl.mean(),
                        "excess_pct": 100 * (cond.mean() - ctrl.mean()),
                        "se_pct": 100 * se, "t": (cond.mean() - ctrl.mean()) / se,
                        "hit": 100 * (cond > 0).mean()})
            fam_data.append((fam, len(e), ctrl))
        show(res, f"R3 LONG DX after a complex break, by family, h={h}")
        y = np.array([x["excess_pct"] for x in res])
        se = np.array([x["se_pct"] for x in res])
        w = 1 / se ** 2
        fe = (w * y).sum() / w.sum()
        fe_se = 1 / np.sqrt(w.sum())
        Q = float((w * (y - fe) ** 2).sum())
        df = len(y) - 1
        I2 = max(0.0, (Q - df) / Q) * 100 if Q > 0 else 0.0
        print(f"  Cochran Q = {Q:.2f} (df {df}, p {1-st.chi2.cdf(Q, df):.3f})  "
              f"I^2 = {I2:.1f}%   fixed-effect common excess = {fe:+.3f}% "
              f"(se {fe_se:.3f}, t {fe/fe_se:+.2f})")
        rng = np.random.default_rng(42)
        B = 4000
        nm = np.zeros(B)
        for b in range(B):
            nm[b] = 100 * max(rng.choice(pool, size=k, replace=False).mean() - pool.mean()
                              for _, k, pool in fam_data)
        obs = y.max()
        print(f"  permutation max-of-{len(fam_data)} p = {(nm >= obs).mean():.4f} "
              f"(observed max {obs:+.3f}% = {res[int(np.argmax(y))]['family']}, "
              f"null 95th {np.percentile(nm, 95):+.3f}%)")

    # ---------------- R4  HORIZON MULTIPLICITY --------------------------------
    print("\n\n########## R4  HORIZON MULTIPLICITY (h was scanned 1..10) ##########")
    ret_by_h = {}
    for h in range(1, 11):
        ret_by_h[h] = vehicle_ret(px, legs, h, lag=1)
    s = px.index[trig.values]
    obs = []
    for h in range(1, 11):
        ret = ret_by_h[h]
        e = declusters(px.index[trig.values & ret.notna().values], 5, px.index)
        cond = ret.loc[e].values
        base = ret.dropna()
        obs.append((h, 100 * (cond.mean() - base.mean()), len(e)))
    best_h, best_edge, k = max(obs, key=lambda x: x[1])
    rng = np.random.default_rng(7)
    B = 4000
    nm = np.zeros(B)
    pools = {h: ret_by_h[h].dropna().values for h in range(1, 11)}
    for b in range(B):
        nm[b] = max(100 * (rng.choice(pools[h], size=k, replace=False).mean()
                           - pools[h].mean()) for h in range(1, 11))
    print(f"  best horizon h={best_h} edge {best_edge:+.3f}% over all days; "
          f"permutation max-of-10-horizons p = {(nm >= best_edge).mean():.4f} "
          f"(null 95th {np.percentile(nm, 95):+.3f}%)")
    print("  (edges by h: " + ", ".join(f"h{h}={e:+.3f}" for h, e, _ in obs) + ")")


if __name__ == "__main__":
    main()
