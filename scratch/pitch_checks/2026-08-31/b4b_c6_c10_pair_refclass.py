"""Shared pair-reference-class engine for C6 and C10.

Both candidates are the same OBJECT in different clothes: a relative-value
statement between two members of one complex, taken at an extreme of a
trailing spread.  The registry closed that form on 2026-08-27 for the 63-day
sector-vs-sector case (132 ordered pairs, Cochran Q p 0.995, I^2 0.0%,
permutation max-of-132 median ABOVE every observed pair).  Run the class
BEFORE round 2, per the instruction, and see whether either candidate is
distinguishable from its family.

  A. C6 class - parent/child 63d return-spread FLOOR, long the CHILD outright,
     h=10.  Named pairs plus the full 132-ordered-pair enumeration.
  B. C10 class - "A is thrusting while B is flushing", long A outright, h=5/10.
     Full 132-ordered-pair enumeration from the same 12-name pool.
  C. C10's own live cell measured directly (XME/FCX long vs SLV/GLD short),
     per-leg attribution, beta neutralisation, ladder, era, midterm, cost.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd
from scipy.stats import chi2

pd.set_option("display.width", 250)
BAR = pd.Timestamp("2026-08-28")
POOL = ["XME", "XLB", "XLE", "XOP", "SMH", "XLK", "XLF", "XLV", "XLI", "XLY",
        "GLD", "SLV"]
EXTRA = ["OIH", "IHI", "XBI", "IBB", "KRE", "ITB", "XHB", "XRT", "GDX", "GDXJ",
         "FCX", "SPY"]
px_all = close_panel(sorted(set(POOL + EXTRA))).loc[:BAR]
print(f"panel {px_all.index[0].date()} .. {px_all.index[-1].date()}  "
      f"n={len(px_all)}  cols={len(px_all.columns)}")


def r_n(s, n):
    return s / s.shift(n) - 1.0


def cellstats(pxf, mask, legs, h, label, gap=None, lag=1):
    gap = gap or h
    r = vehicle_ret(pxf, legs, h, lag)
    v = r.notna()
    days = pxf.index[mask.reindex(pxf.index, fill_value=False).values & v.values]
    if len(days) == 0:
        return {"label": label, "n": 0}, pd.DatetimeIndex([]), np.array([])
    epi = declusters(days, gap, pxf.index)
    vals = r.loc[epi].values
    base = r[v]
    w = int((vals > 0).sum())
    out = summarize(vals, label)
    out["n_days"] = len(days)
    out["ctrl_pct"] = round(100 * base.mean(), 3)
    out["edge_pp"] = round(out["mean_pct"] - 100 * base.mean(), 3)
    out["rec"] = f"{w}-{len(vals)-w}"
    out["p_vs_base"] = round(sign_test(w, len(vals), float((base > 0).mean())), 4)
    return out, epi, vals


def meta(rows, name):
    df = pd.DataFrame(rows)
    df = df[df["se_pct"] > 0]
    if len(df) < 3:
        print(f"  {name}: too few cells")
        return
    wgt = 1.0 / df["se_pct"] ** 2
    common = float((df["excess_pct"] * wgt).sum() / wgt.sum())
    se_c = float(np.sqrt(1.0 / wgt.sum()))
    Q = float((wgt * (df["excess_pct"] - common) ** 2).sum())
    dfree = len(df) - 1
    I2 = max(0.0, 100.0 * (Q - dfree) / Q) if Q > 0 else 0.0
    print(f"\n  --- {name}  ({len(df)} cells) ---")
    top = df.sort_values("excess_pct", ascending=False)
    print(top.head(8).to_string(index=False))
    print(f"  Cochran Q = {Q:.2f} (df {dfree}, p {1-chi2.cdf(Q, dfree):.4f})  "
          f"I^2 = {I2:.1f}%   fixed-effect common excess = {common:+.3f}% "
          f"(se {se_c:.3f}, t {common/se_c:+.2f})")
    rng = np.random.default_rng(42)
    nulls = rng.normal(0.0, df["se_pct"].values[None, :], size=(20000, len(df)))
    nmax = nulls.max(axis=1)
    obs = df["excess_pct"].max()
    lbl = top.iloc[0].get("pair", top.iloc[0].get("label", "?"))
    print(f"  permutation max-of-{len(df)} p = {float((nmax >= obs).mean()):.4f} "
          f" (observed max {obs:+.3f}% = {lbl}; null max median "
          f"{np.median(nmax):+.3f}%, 95th {np.percentile(nmax,95):+.3f}%)")
    return df


# =========================================================== A. C6 class
print("\n" + "=" * 100)
print("A. C6 REFERENCE CLASS - parent/child 63d spread FLOOR, long the CHILD, h=10")
print("=" * 100)
NAMED = [("OIH", "XOP"), ("XOP", "XLE"), ("OIH", "XLE"), ("SMH", "XLK"),
         ("KRE", "XLF"), ("IHI", "XLV"), ("XBI", "IBB"), ("XME", "XLB"),
         ("ITB", "XHB"), ("XRT", "XLY"), ("GDX", "GLD"), ("GDXJ", "GLD")]
for H in (10,):
    rows = []
    for child, parent in NAMED:
        sub = px_all[[child, parent]].dropna()
        if len(sub) < 800:
            continue
        s = r_n(sub[child], 63) - r_n(sub[parent], 63)
        p = s.rolling(252).rank(pct=True) * 100
        m = (p <= 2.5) & p.notna()
        st, epi, vals = cellstats(sub, m, [(child, 1.0)], H, "", gap=H)
        if st["n"] < 6:
            continue
        r = vehicle_ret(sub, [(child, 1.0)], H, 1).dropna()
        rows.append({"pair": f"{child}/{parent}", "n_epi": st["n"],
                     "cond_pct": round(st["mean_pct"], 3),
                     "ctrl_pct": round(100 * r.mean(), 3),
                     "excess_pct": round(st["mean_pct"] - 100 * r.mean(), 3),
                     "se_pct": round(100 * vals.std(ddof=1) / np.sqrt(len(vals)), 3),
                     "hit": round(st["hit"], 1), "rec": st["rec"]})
    meta(rows, f"C6 named parent/child pairs, PIT<=2.5, h={H}")

print("\n  full ordered-pair enumeration from the 12-name pool (the registry form):")
sub_all = px_all[POOL].dropna()
rows = []
for a in POOL:
    for b in POOL:
        if a == b:
            continue
        s = r_n(sub_all[a], 63) - r_n(sub_all[b], 63)
        p = s.rolling(252).rank(pct=True) * 100
        m = (p <= 2.5) & p.notna()
        st, epi, vals = cellstats(sub_all, m, [(a, 1.0)], 10, "", gap=10)
        if st["n"] < 6:
            continue
        r = vehicle_ret(sub_all, [(a, 1.0)], 10, 1).dropna()
        rows.append({"pair": f"{a}/{b}", "n_epi": st["n"],
                     "cond_pct": round(st["mean_pct"], 3),
                     "ctrl_pct": round(100 * r.mean(), 3),
                     "excess_pct": round(st["mean_pct"] - 100 * r.mean(), 3),
                     "se_pct": round(100 * vals.std(ddof=1) / np.sqrt(len(vals)), 3),
                     "hit": round(st["hit"], 1)})
df6 = meta(rows, "C6 form, all ordered pairs, PIT<=2.5, h=10")
if df6 is not None:
    for target in ("OIH/XOP",):
        if (df6["pair"] == target).any():
            row = df6[df6["pair"] == target].iloc[0]
            rk = int((df6["excess_pct"] > row["excess_pct"]).sum()) + 1
            print(f"  {target}: excess {row['excess_pct']:+.3f}%, rank {rk} of "
                  f"{len(df6)}")

# =========================================================== B. C10 class
print("\n" + "=" * 100)
print("B. C10 REFERENCE CLASS - 'A thrusting while B flushing', long A outright")
print("=" * 100)
rk21 = {t: pct_rank(sub_all[t], 21) for t in POOL}
r63 = {t: r_n(sub_all[t], 63) for t in POOL}
sma = {t: sub_all[t].rolling(200).mean() for t in POOL}
for H in (5, 10):
    rows = []
    for a in POOL:
        for b in POOL:
            if a == b:
                continue
            m = (rk21[a] >= 85) & (r63[b] <= -0.10) & (sub_all[b] < sma[b])
            st, epi, vals = cellstats(sub_all, m, [(a, 1.0)], H, "", gap=H)
            if st["n"] < 6:
                continue
            r = vehicle_ret(sub_all, [(a, 1.0)], H, 1).dropna()
            rows.append({"pair": f"long {a} | {b} flushing", "n_epi": st["n"],
                         "cond_pct": round(st["mean_pct"], 3),
                         "ctrl_pct": round(100 * r.mean(), 3),
                         "excess_pct": round(st["mean_pct"] - 100 * r.mean(), 3),
                         "se_pct": round(100 * vals.std(ddof=1) / np.sqrt(len(vals)), 3),
                         "hit": round(st["hit"], 1)})
    d = meta(rows, f"C10 form, all ordered pairs, h={H}")
    if d is not None:
        for tgt in ("long XME | SLV flushing", "long XME | GLD flushing"):
            if (d["pair"] == tgt).any():
                row = d[d["pair"] == tgt].iloc[0]
                rk = int((d["excess_pct"] > row["excess_pct"]).sum()) + 1
                print(f"  {tgt}: excess {row['excess_pct']:+.3f}%, "
                      f"n_epi {row['n_epi']}, rank {rk} of {len(d)}")

# also the PAIR form (long A short B) across the same enumeration, h=5
print("\n  PAIR form (long A / short B, equal dollar), h=5:")
rows = []
for a in POOL:
    for b in POOL:
        if a == b:
            continue
        m = (rk21[a] >= 85) & (r63[b] <= -0.10) & (sub_all[b] < sma[b])
        st, epi, vals = cellstats(sub_all, m, [(a, 1.0), (b, -1.0)], 5, "", gap=5)
        if st["n"] < 6:
            continue
        r = vehicle_ret(sub_all, [(a, 1.0), (b, -1.0)], 5, 1).dropna()
        rows.append({"pair": f"L {a} / S {b}", "n_epi": st["n"],
                     "cond_pct": round(st["mean_pct"], 3),
                     "ctrl_pct": round(100 * r.mean(), 3),
                     "excess_pct": round(st["mean_pct"] - 100 * r.mean(), 3),
                     "se_pct": round(100 * vals.std(ddof=1) / np.sqrt(len(vals)), 3),
                     "hit": round(st["hit"], 1)})
dp = meta(rows, "C10 PAIR form, all ordered pairs, h=5")
if dp is not None:
    for tgt in ("L XME / S SLV", "L XME / S GLD"):
        if (dp["pair"] == tgt).any():
            row = dp[dp["pair"] == tgt].iloc[0]
            rk = int((dp["excess_pct"] > row["excess_pct"]).sum()) + 1
            print(f"  {tgt}: excess {row['excess_pct']:+.3f}%, rank {rk} of {len(dp)}")

# =========================================================== C. C10 live cell
print("\n" + "=" * 100)
print("C. C10's OWN LIVE CELL, measured directly")
print("=" * 100)
c10 = px_all[["XME", "FCX", "SLV", "GLD", "SPY", "XLB"]].dropna()
print(f"  c10 panel {c10.index[0].date()} .. {c10.index[-1].date()} n={len(c10)}")
xme_r21 = pct_rank(c10["XME"], 21)
fcx_z10 = zscore(c10["FCX"], 10)
slv_r63 = r_n(c10["SLV"], 63)
slv_200 = c10["SLV"].rolling(200).mean()
print(f"  TODAY: XME r21 rank {xme_r21.iloc[-1]:.1f}  FCX z10 "
      f"{fcx_z10.iloc[-1]:+.2f}  SLV r63 {100*slv_r63.iloc[-1]:+.2f}%  "
      f"SLV vs 200d {100*(c10['SLV'].iloc[-1]/slv_200.iloc[-1]-1):+.2f}%")
TRIG = ((xme_r21 >= 85) & (slv_r63 <= -0.10) & (c10["SLV"] < slv_200))
print(f"  trigger days: {int(TRIG.sum())}   fires today: {bool(TRIG.iloc[-1])}")

FORMS = {
    "LONG XME outright": [("XME", 1.0)],
    "LONG FCX outright": [("FCX", 1.0)],
    "SHORT SLV outright": [("SLV", -1.0)],
    "SHORT GLD outright": [("GLD", -1.0)],
    "PAIR L XME / S SLV": [("XME", 1.0), ("SLV", -1.0)],
    "PAIR L XME / S GLD": [("XME", 1.0), ("GLD", -1.0)],
    "PAIR L FCX / S SLV": [("FCX", 1.0), ("SLV", -1.0)],
}
for h in (3, 5, 10):
    rows = [cellstats(c10, TRIG, legs, h, lbl, gap=5)[0] for lbl, legs in FORMS.items()]
    show(rows, f"C10 per-leg attribution, h={h}")

print("\n  threshold ladder on the two conditioners (today XME r21 85.3, SLV r63 -12.2):")
for h in (5, 10):
    rows = []
    for a_thr in (75, 80, 85, 90, 95):
        m = (xme_r21 >= a_thr) & (slv_r63 <= -0.10) & (c10["SLV"] < slv_200)
        rows.append(cellstats(c10, m, [("XME", 1.0)], h,
                              f"XME r21>={a_thr}{' ***' if a_thr==85 else ''}",
                              gap=5)[0])
    for b_thr in (-0.05, -0.10, -0.15, -0.20):
        m = (xme_r21 >= 85) & (slv_r63 <= b_thr) & (c10["SLV"] < slv_200)
        rows.append(cellstats(c10, m, [("XME", 1.0)], h,
                              f"SLV r63<={100*b_thr:.0f}%"
                              f"{' ***' if b_thr==-0.10 else ''}", gap=5)[0])
    rows.append(cellstats(c10, (xme_r21 >= 85), [("XME", 1.0)], h,
                          "PARENT: XME r21>=85 ALONE (no SLV gate)", gap=5)[0])
    show(rows, f"ladder + gate attribution, h={h}")

print("\n  era / midterm / concentration on the best live form:")
for lbl, legs in (("LONG XME", [("XME", 1.0)]),
                  ("PAIR L XME / S SLV", [("XME", 1.0), ("SLV", -1.0)])):
    for h in (5, 10):
        st, epi, vals = cellstats(c10, TRIG, legs, h, "", gap=5)
        if st["n"] < 3:
            continue
        yr = pd.DatetimeIndex(epi).year
        mid = (yr % 4) == 2
        show([summarize(vals[mid], f"{lbl} h={h} MIDTERM (N={int(mid.sum())})"),
              summarize(vals[~mid], f"{lbl} h={h} non-midterm")]
             + era_split(epi, vals))
        order = np.argsort(-vals)
        print(f"   drop-best-2 {100*np.delete(vals, order[:2]).mean():+.3f}%  "
              f"| {cluster_note(epi, vals)}")
        print(f"   episode dates: {', '.join(str(d.date()) for d in epi)}")
