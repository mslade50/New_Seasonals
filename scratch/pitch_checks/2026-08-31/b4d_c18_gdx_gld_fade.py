"""C18 - FADE the miners' 21-day outperformance of the metal.

Live: GDX r21 +29.79% vs GLD +8.41% -> GDX-minus-GLD 21d spread +21.4pp.
Candidate: short GDX / long GLD (beta-adjusted), or short GDX outright.

Agenda:
  0. premise re-derivation + PIT percentile of today's spread.
  1. per-leg attribution FIRST (registry: if one leg carries it, the pair is
     worse than the outright).
  2. beta-neutral pair with a PIT trailing-252 beta (equal-dollar GDX-GLD is
     NOT beta-neutral, registry 2026-08-10).
  3. threshold ladder on the spread extreme, with TODAY'S value marked.
  4. battery: decluster, concentration, era, midterm, local control, cost.
  5. REFERENCE CLASS across miner/metal pairs (GDX, GDXJ, NEM, AEM, KGC, AU
     against GLD) - Cochran Q, I^2, common excess, permutation max-of-N.
  6. Direct engagement with the 2026-08-27 registry entry: the LONG side of
     the miners after a flush was CONFIRMED on a NEM/AEM/AU/KGC class.  Run
     that same class here on the FADE side.
  7. repetition: the 2026-08-27 pitch was LONG GDX MOC ~5d.  This is the
     opposite sign, so the fingerprint differs, but state it.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
BAR = pd.Timestamp("2026-08-28")
GAP = 5
NAMES = ["GDX", "GLD", "SLV", "GDXJ", "NEM", "AEM", "KGC", "AU", "SPY"]
px = close_panel(NAMES)
px = px.loc[:BAR]
core = px[["GDX", "GLD", "SPY"]].dropna()
print(f"core panel {core.index[0].date()} .. {core.index[-1].date()} n={len(core)}")


def r_n(s, n):
    return s / s.shift(n) - 1.0


# ------------------------------------------------------------------ 0. premise
print("\n" + "=" * 100)
print("0. PREMISE")
print("=" * 100)
sp21 = r_n(core["GDX"], 21) - r_n(core["GLD"], 21)
pit = sp21.rolling(252).rank(pct=True) * 100
print(f"  GDX r21 {100*r_n(core['GDX'],21).iloc[-1]:+.2f}%  GLD r21 "
      f"{100*r_n(core['GLD'],21).iloc[-1]:+.2f}%  spread "
      f"{100*sp21.iloc[-1]:+.2f}pp   PIT pctile {pit.iloc[-1]:.1f}   "
      f"full-sample pctile {100.0*(sp21.dropna() <= sp21.iloc[-1]).mean():.1f}")

TODAY_SP = float(sp21.iloc[-1])
TODAY_PIT = float(pit.iloc[-1])


def cellstats(pxf, mask, legs, h, label, gap=GAP, lag=1):
    r = vehicle_ret(pxf, legs, h, lag)
    v = r.notna()
    days = pxf.index[mask.reindex(pxf.index, fill_value=False).values & v.values]
    if len(days) == 0:
        return {"label": label, "n": 0}, pd.DatetimeIndex([]), np.array([])
    epi = declusters(days, gap, pxf.index)
    vals = r.loc[epi].values
    base = r[v]
    w = int((vals > 0).sum())
    p0 = float((base > 0).mean())
    out = summarize(vals, label)
    out["n_days"] = len(days)
    out["ctrl_pct"] = round(100 * base.mean(), 3)
    out["edge_pp"] = round(out["mean_pct"] - 100 * base.mean(), 3)
    out["rec"] = f"{w}-{len(vals)-w}"
    out["p_vs_base"] = round(sign_test(w, len(vals), p0), 4)
    return out, epi, vals


TRIG = (pit >= 97.5) & pit.notna()
print(f"  trigger (PIT >= 97.5) day count: {int(TRIG.sum())}   "
      f"today PIT {TODAY_PIT:.1f} -> fires: {bool(TRIG.iloc[-1])}")

# --------------------------------------------------- 1. per-leg attribution
print("\n" + "=" * 100)
print("1. PER-LEG ATTRIBUTION (do this before crediting the pair)")
print("=" * 100)
FORMS = {
    "SHORT GDX outright": ([("GDX", -1.0)], 4.0),
    "LONG GLD outright": ([("GLD", 1.0)], 4.0),
    "PAIR short GDX / long GLD (equal $)": ([("GDX", -1.0), ("GLD", 1.0)], 8.0),
    "LONG GDX outright (the 08-27 side)": ([("GDX", 1.0)], 4.0),
}
for h in (1, 3, 5, 10):
    rows = []
    for lbl, (legs, _c) in FORMS.items():
        rows.append(cellstats(core, TRIG, legs, h, f"{lbl}")[0])
    show(rows, f"h={h}, trigger PIT>=97.5")

# ------------------------------------------------------- 2. beta-neutral pair
print("\n" + "=" * 100)
print("2. BETA-NEUTRAL pair (PIT trailing-252 beta of GDX on GLD)")
print("=" * 100)
d_gdx, d_gld = core["GDX"].pct_change(), core["GLD"].pct_change()
beta = (d_gdx.rolling(252).cov(d_gld) / d_gld.rolling(252).var()).shift(1)
print(f"  mean beta {beta.dropna().mean():.2f}   today {beta.iloc[-1]:.2f}")
rows = []
for h in (1, 3, 5, 10):
    f_g, f_l = fwd_lag(core["GDX"], h, 1), fwd_lag(core["GLD"], h, 1)
    resid = -(f_g - beta * f_l)          # short GDX, hedged with beta x GLD
    rv = resid.notna()
    days = core.index[TRIG.reindex(core.index, fill_value=False).values & rv.values]
    epi = declusters(days, GAP, core.index)
    rr = summarize(resid.loc[epi].values, f"h={h} beta-neutral SHORT GDX/GLD")
    rr["ctrl_pct"] = round(100 * resid[rv].mean(), 3)
    rr["edge_pp"] = round(rr["mean_pct"] - 100 * resid[rv].mean(), 3)
    w = int((resid.loc[epi].values > 0).sum())
    rr["rec"] = f"{w}-{len(epi)-w}"
    rows.append(rr)
show(rows, "beta-neutral fade")

# ------------------------------------------------------------- 3. ladder
print("\n" + "=" * 100)
print(f"3. THRESHOLD LADDER on the spread extreme (today PIT {TODAY_PIT:.1f}, "
      f"spread {100*TODAY_SP:+.1f}pp)")
print("=" * 100)
for h in (1, 3, 5, 10):
    rows = []
    for thr in (90.0, 95.0, 97.5, 99.0, 99.5):
        m = (pit >= thr) & pit.notna()
        mark = " ***" if abs(thr - 97.5) < 1e-9 else ""
        rows.append(cellstats(core, m, [("GDX", -1.0)], h,
                              f"PIT>={thr}{mark} SHORT GDX")[0])
    for lvl in (0.10, 0.15, 0.20, 0.25):
        m = sp21 >= lvl
        mark = " ***TODAY (+21.4pp)" if abs(lvl - 0.20) < 1e-9 else ""
        rows.append(cellstats(core, m, [("GDX", -1.0)], h,
                              f"spread>={100*lvl:.0f}pp{mark}")[0])
    show(rows, f"ladder, h={h}")

# ------------------------------------------------------------ 4. full battery
print("\n" + "=" * 100)
print("4. BATTERY on the two live forms")
print("=" * 100)
variants = {f"PIT>={t}": ((pit >= t) & pit.notna()) for t in (90, 95, 97.5, 99)}
battery(core, TRIG, [("GDX", -1.0)], 5, "C18 SHORT GDX at spread extreme", 4.0,
        variants=variants, min_gap=GAP)
battery(core, TRIG, [("GDX", -1.0), ("GLD", 1.0)], 5,
        "C18 PAIR short GDX / long GLD", 4.0, variants=variants, min_gap=GAP)

print("\n  MIDTERM + concentration, SHORT GDX at PIT>=97.5:")
for h in (1, 3, 5, 10):
    _, epi, vals = cellstats(core, TRIG, [("GDX", -1.0)], h, "")
    if not len(vals):
        continue
    yr = pd.DatetimeIndex(epi).year
    mid = (yr % 4) == 2
    r = vehicle_ret(core, [("GDX", -1.0)], h, 1).dropna()
    b_mid = (pd.DatetimeIndex(r.index).year % 4) == 2
    show([summarize(vals[mid], f"h={h} MIDTERM (N={int(mid.sum())})"),
          summarize(vals[~mid], f"h={h} non-midterm"),
          summarize(r.values[b_mid], f"h={h} CTRL midterm days")])
    order = np.argsort(-vals)
    print(f"   h={h} drop-best-2 {100*np.delete(vals, order[:2]).mean():+.3f}%  "
          f"| {cluster_note(epi, vals)}")

# ---------------------------------------------------------- 5. reference class
print("\n" + "=" * 100)
print("5. REFERENCE CLASS: the identical fade across miner/metal pairs")
print("=" * 100)
PAIRS = [("GDX", "GLD"), ("GDXJ", "GLD"), ("NEM", "GLD"), ("AEM", "GLD"),
         ("KGC", "GLD"), ("AU", "GLD"), ("SLV", "GLD")]
for h in (1, 3, 5):
    rows = []
    for child, parent in PAIRS:
        sub = px[[child, parent]].dropna()
        if len(sub) < 800:
            continue
        s = r_n(sub[child], 21) - r_n(sub[parent], 21)
        p = s.rolling(252).rank(pct=True) * 100
        m = (p >= 97.5) & p.notna()
        st, epi, vals = cellstats(sub, m, [(child, -1.0)], h, f"SHORT {child}")
        if st["n"] < 5:
            continue
        r = vehicle_ret(sub, [(child, -1.0)], h, 1).dropna()
        rows.append({"pair": f"{child}/{parent}", "n_epi": st["n"],
                     "cond_pct": round(st["mean_pct"], 3),
                     "ctrl_pct": round(100 * r.mean(), 3),
                     "excess_pct": round(st["mean_pct"] - 100 * r.mean(), 3),
                     "se_pct": round(100 * vals.std(ddof=1) / np.sqrt(len(vals)), 3),
                     "hit": round(st["hit"], 1)})
    df = pd.DataFrame(rows)
    if df.empty:
        continue
    wgt = 1.0 / df["se_pct"] ** 2
    common = float((df["excess_pct"] * wgt).sum() / wgt.sum())
    se_c = float(np.sqrt(1.0 / wgt.sum()))
    Q = float((wgt * (df["excess_pct"] - common) ** 2).sum())
    dfree = len(df) - 1
    I2 = max(0.0, 100.0 * (Q - dfree) / Q) if Q > 0 else 0.0
    print(f"\n  h={h}")
    print(df.to_string(index=False))
    from scipy.stats import chi2  # noqa
    print(f"  Cochran Q = {Q:.2f} (df {dfree}, p {1-chi2.cdf(Q, dfree):.3f})  "
          f"I^2 = {I2:.1f}%   fixed-effect common excess = {common:+.3f}% "
          f"(se {se_c:.3f}, t {common/se_c:+.2f})")
    rng = np.random.default_rng(42)
    nulls = rng.normal(0.0, df["se_pct"].values[None, :], size=(20000, len(df)))
    obs_max = df["excess_pct"].max()
    print(f"  permutation max-of-{len(df)} p = "
          f"{float((nulls.max(axis=1) >= obs_max).mean()):.4f}  "
          f"(observed max {obs_max:+.3f}% = "
          f"{df.loc[df['excess_pct'].idxmax(),'pair']}, null max median "
          f"{np.median(nulls.max(axis=1)):+.3f}%)")

# ---------------------------------------- 6. the 2026-08-27 confirmed LONG side
print("\n" + "=" * 100)
print("6. THE OPPOSITE SIDE: what the miners actually did after these extremes")
print("=" * 100)
print("  registry 2026-08-27 CONFIRMED the LONG miners side after a flush on a")
print("  NEM/AEM/AU/KGC class (+2.228%, 30 episodes, 22-8).  Run the same names")
print("  on THIS trigger (miner outperformance extreme) to see which sign wins.")
for h in (3, 5, 10):
    rows = []
    for nm in ("GDX", "NEM", "AEM", "AU", "KGC"):
        sub = px[[nm, "GLD"]].dropna()
        s = r_n(sub[nm], 21) - r_n(sub["GLD"], 21)
        p = s.rolling(252).rank(pct=True) * 100
        m = (p >= 97.5) & p.notna()
        rows.append(cellstats(sub, m, [(nm, 1.0)], h, f"LONG {nm} (continuation)")[0])
        rows.append(cellstats(sub, m, [(nm, -1.0)], h, f"SHORT {nm} (the fade)")[0])
    show(rows, f"h={h}: continuation vs fade on each miner")
