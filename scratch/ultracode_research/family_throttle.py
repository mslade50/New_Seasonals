"""Family-level fragility throttle study (track: family-throttle, 2026-07-02).

Question: should the fragility throttle apply to the short-horizon index
mean-reversion FAMILY instead of the whole book?

Family defined EX-ANTE by mechanism (see md for the argument):
  CORE3  = Weak Close Decent Sznls, SPY QQQ MonFri Reversion, Monday Dip
  FAMILY4 = CORE3 + Indices Oversold Bounce (same mechanism: long index/broad-ETF
            dip-buy, 2d hold) — included by mechanism, NOT by its high-frag returns.
  3x ETF Overbot Fade EXCLUDED: short direction, fades strength (anti-dip-buy).

Live basis: 63d fragility, 10d MA, as-of signal date. OVS exempt throughout.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]

trades = pd.read_parquet(ROOT / "data" / "backtest_trades_full.parquet")
frag = pd.read_parquet(ROOT / "data" / "rd2_fragility.parquet")

frag_ma = frag["63d"].dropna().rolling(10, min_periods=1).mean()
frag_ma.index = pd.to_datetime(frag_ma.index).normalize()

trades["Signal Date"] = pd.to_datetime(trades["Signal Date"]).dt.normalize()
start = frag_ma.index.min() + pd.Timedelta(days=20)
t = trades[trades["Signal Date"] >= start].copy()
t = t.sort_values("Signal Date")
t["frag"] = pd.merge_asof(
    t[["Signal Date"]], frag_ma.rename("frag").reset_index(),
    left_on="Signal Date", right_on="Date",
)["frag"].values
t = t.dropna(subset=["frag", "R_Multiple"]).copy()
t["ym"] = t["Signal Date"].dt.to_period("M")
t["yr"] = t["Signal Date"].dt.year

is_ovs = t["Strategy"].eq("Overbot Vol Spike")
nb = t[~is_ovs].copy()
print(f"non-OVS trades in frag window: {len(nb)} "
      f"({nb['Signal Date'].min().date()} .. {nb['Signal Date'].max().date()})")

CORE3 = ["Weak Close Decent Sznls", "SPY QQQ MonFri Reversion", "Monday Dip"]
FAMILY4 = CORE3 + ["Indices Oversold Bounce"]
nb["fam3"] = nb["Strategy"].isin(CORE3)
nb["fam4"] = nb["Strategy"].isin(FAMILY4)

def cell(df):
    if len(df) == 0:
        return "-"
    return f"{df.R_Multiple.mean():+.3f} ({len(df)})"

def clustered_t(a, b):
    """monthly-clustered Welch t between two trade subsets."""
    am = a.groupby("ym")["R_Multiple"].mean()
    bm = b.groupby("ym")["R_Multiple"].mean()
    if len(am) < 3 or len(bm) < 3:
        return np.nan, np.nan, len(am), len(bm)
    tt = stats.ttest_ind(am, bm, equal_var=False)
    return tt.statistic, tt.pvalue, len(am), len(bm)

# ---------------------------------------------------------------- 1. anatomy
print("\n=== per-strategy avgR by frag band (non-OVS) ===")
bands = [(0, 50, "<50"), (50, 101, ">=50"), (50, 55, "50-55"), (55, 101, "55+")]
rows = []
for s, g in nb.groupby("Strategy"):
    r = {"Strategy": s, "fam": "CORE3" if s in CORE3 else ("FAM4" if s in FAMILY4 else "")}
    for lo, hi, lab in bands:
        r[lab] = cell(g[(g.frag >= lo) & (g.frag < hi)])
    rows.append(r)
print(pd.DataFrame(rows).to_string(index=False))

for famcol, famname in [("fam3", "CORE3"), ("fam4", "FAMILY4")]:
    fam = nb[nb[famcol]]
    rest = nb[~nb[famcol]]
    print(f"\n=== {famname} vs REST ===")
    for lo, hi, lab in [(0, 50, "<50"), (50, 101, ">=50")]:
        f = fam[(fam.frag >= lo) & (fam.frag < hi)]
        r = rest[(rest.frag >= lo) & (rest.frag < hi)]
        print(f"  frag {lab:>5}: family {cell(f):>14}  rest {cell(r):>14}")
    # family-vs-rest difference AT >=50, monthly clustered
    fhi = fam[fam.frag >= 50]; rhi = rest[rest.frag >= 50]
    ts, p, nm1, nm2 = clustered_t(fhi, rhi)
    print(f"  family-vs-rest at >=50: t={ts:+.2f} p={p:.3f} ({nm1} vs {nm2} months)")
    # family hi-vs-lo (the damage itself), clustered
    ts, p, nm1, nm2 = clustered_t(fhi, fam[fam.frag < 50])
    print(f"  family >=50 vs family <50: t={ts:+.2f} p={p:.3f} ({nm1} vs {nm2} months)")
    ts, p, nm1, nm2 = clustered_t(rhi, rest[rest.frag < 50])
    print(f"  rest   >=50 vs rest   <50: t={ts:+.2f} p={p:.3f} ({nm1} vs {nm2} months)")

# ---------------------------------------------------------------- 2. LOYO
print("\n=== LOYO: family(FAM4) minus rest avgR at frag>=50, dropping each year ===")
fam_col = "fam4"
hi = nb[nb.frag >= 50]
years = sorted(hi.yr.unique())
rows = []
for drop in [None] + years:
    sub = hi if drop is None else hi[hi.yr != drop]
    f = sub[sub[fam_col]]; r = sub[~sub[fam_col]]
    ts, p, _, _ = clustered_t(f, r)
    rows.append({"drop": drop or "none", "famR": f.R_Multiple.mean(), "famN": len(f),
                 "restR": r.R_Multiple.mean(), "restN": len(r),
                 "diff": f.R_Multiple.mean() - r.R_Multiple.mean(),
                 "t_clust": ts, "p": p})
print(pd.DataFrame(rows).round(3).to_string(index=False))

print("\n=== LOYO same, CORE3 ===")
rows = []
for drop in [None] + years:
    sub = hi if drop is None else hi[hi.yr != drop]
    f = sub[sub["fam3"]]; r = sub[~sub["fam3"]]
    ts, p, _, _ = clustered_t(f, r)
    rows.append({"drop": drop or "none", "famR": f.R_Multiple.mean(), "famN": len(f),
                 "restR": r.R_Multiple.mean(), "restN": len(r),
                 "diff": f.R_Multiple.mean() - r.R_Multiple.mean(),
                 "t_clust": ts, "p": p})
print(pd.DataFrame(rows).round(3).to_string(index=False))

# family damage (>=50 vs <50 within family) excluding episodes
print("\n=== FAMILY4 high-frag damage under exclusions (>=50 vs <50, clustered) ===")
fam = nb[nb[fam_col]]
for label, mask in [
    ("full sample", pd.Series(True, index=fam.index)),
    ("ex 2020", fam.yr != 2020),
    ("ex 2021", fam.yr != 2021),
    ("ex 2022", fam.yr != 2022),
    ("ex 2023", fam.yr != 2023),
    ("ex 2024", fam.yr != 2024),
    ("ex 2025", fam.yr != 2025),
    ("ex 2026", fam.yr != 2026),
]:
    g = fam[mask]
    f_hi, f_lo = g[g.frag >= 50], g[g.frag < 50]
    ts, p, _, _ = clustered_t(f_hi, f_lo)
    print(f"  {label:12}: >=50 {cell(f_hi):>14}  <50 {cell(f_lo):>14}  t={ts:+.2f} p={p:.3f}")

print("\n=== FAMILY4 trades at frag>=50 by year (avgR, N, totR) ===")
fhi = fam[fam.frag >= 50]
yr_tab = fhi.groupby("yr")["R_Multiple"].agg(["mean", "size", "sum"]).round(3)
print(yr_tab.to_string())

# ---------------------------------------------------------------- 3. designs
def book_taper(f):
    # 1.0 through 50, linear to 0.5 at 60, floor 0.5
    if f < 50:
        return 1.0
    return max(0.5, 1.0 - 0.05 * (f - 50))

designs = {}
designs["baseline (no throttle)"] = pd.Series(1.0, index=nb.index)
designs["(a) book taper 1.0->0.5 over 50-60"] = nb.frag.map(book_taper)

for fam_mult in (0.25, 0.5):
    m = pd.Series(1.0, index=nb.index)
    cut = nb[fam_col] & (nb.frag >= 50)
    m[cut] = fam_mult
    designs[f"(c) family-only cut {fam_mult}x at >=50, rest untouched"] = m

    mb = nb.frag.map(book_taper)
    mb[cut] = np.minimum(mb[cut], fam_mult)
    designs[f"(b) family cut {fam_mult}x + book taper on rest"] = mb

# CORE3 variant of the pure family cut
m = pd.Series(1.0, index=nb.index)
m[nb["fam3"] & (nb.frag >= 50)] = 0.25
designs["(c') CORE3-only cut 0.25x at >=50"] = m

print("\n=== design replay (non-OVS, R-weighted; risk-normalized avgR = sumR_adj/sum mult) ===")
rows = []
for name, m in designs.items():
    radj = nb.R_Multiple * m
    daily = radj.groupby(nb["Exit Date"]).sum().sort_index().cumsum()
    dd = (daily - daily.cummax()).min()
    rows.append({"design": name, "totR": radj.sum(),
                 "risk_units": m.sum(),
                 "avgR/unit": radj.sum() / m.sum(),
                 "worstDD_R": dd})
print(pd.DataFrame(rows).round(3).to_string(index=False))

# LOYO of design deltas vs baseline (does the improvement come from one year?)
print("\n=== design totR delta vs baseline, per year (which years pay?) ===")
rows = []
for yr, g in nb.groupby("yr"):
    r = {"yr": yr, "N": len(g)}
    base = g.R_Multiple.sum()
    r["baseR"] = base
    for name, m in designs.items():
        if name.startswith("baseline"):
            continue
        r[name.split(" ", 1)[0] + ("0.25" if "0.25" in name else ("0.5" if "0.5x" in name or "0.5 at" in name else ""))] = \
            (g.R_Multiple * m.loc[g.index]).sum() - base
    rows.append(r)
df = pd.DataFrame(rows).round(2)
print(df.to_string(index=False))

# 2026 YTD check (established: inverted year)
print("\n=== 2026 YTD detail ===")
g26 = nb[nb.yr == 2026]
f26 = g26[g26[fam_col]]
print(f"  non-OVS 2026: >=50 {cell(g26[g26.frag>=50])}  <50 {cell(g26[g26.frag<50])}")
print(f"  FAMILY4 2026: >=50 {cell(f26[f26.frag>=50])}  <50 {cell(f26[f26.frag<50])}")

# 3x ETF Overbot Fade at high frag, for the mechanism argument record
x3 = nb[nb.Strategy == "3x ETF Overbot Fade"]
print(f"\n3x ETF Overbot Fade: >=50 {cell(x3[x3.frag>=50])}  <50 {cell(x3[x3.frag<50])}")
iob = nb[nb.Strategy == "Indices Oversold Bounce"]
print(f"Indices Oversold Bounce: >=50 {cell(iob[iob.frag>=50])}  <50 {cell(iob[iob.frag<50])}")
