"""Adversarial verification of the family-throttle findings.

Independent recompute: join ledger to live fragility basis (63d, 10d MA,
as-of signal date, ffill limit 5), window 2016-08-01..2026-06-30, OVS excluded.
All clustered tests: monthly means of trade R within group, Welch t across months.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

TR = pd.read_parquet("data/backtest_trades_full.parquet")
FR = pd.read_parquet("data/rd2_fragility.parquet")

live = FR["63d"].rolling(10, min_periods=1).mean()
# as-of signal date with ffill limit 5 business days
cal = pd.date_range(live.index.min(), live.index.max() + pd.Timedelta(days=7), freq="D")
live_daily = live.reindex(cal).ffill(limit=5)

t = TR.copy()
t["Signal Date"] = pd.to_datetime(t["Signal Date"])
t["frag"] = t["Signal Date"].map(live_daily)
t = t[(t["Signal Date"] >= "2016-08-01") & (t["Signal Date"] <= "2026-06-30")]
t = t[t["Strategy"] != "Overbot Vol Spike"].copy()
t = t.dropna(subset=["frag", "R_Multiple"])
print(f"non-OVS joined N = {len(t)}")

FAMILY4 = {
    "Weak Close Decent Sznls",
    "SPY QQQ MonFri Reversion",
    "Monday Dip",
    "Indices Oversold Bounce",
}
t["fam"] = t["Strategy"].isin(FAMILY4)
t["hi"] = t["frag"] >= 50
t["ym"] = t["Signal Date"].dt.to_period("M")


def clustered_welch(a: pd.DataFrame, b: pd.DataFrame) -> tuple[float, float, int, int]:
    ma = a.groupby("ym")["R_Multiple"].mean()
    mb = b.groupby("ym")["R_Multiple"].mean()
    if len(ma) < 2 or len(mb) < 2:
        return np.nan, np.nan, len(ma), len(mb)
    res = stats.ttest_ind(ma, mb, equal_var=False)
    return res.statistic, res.pvalue, len(ma), len(mb)


def cell(df: pd.DataFrame) -> str:
    return f"{df['R_Multiple'].mean():+.3f} (N={len(df)}, totR {df['R_Multiple'].sum():+.1f})"


fam, rest = t[t.fam], t[~t.fam]
print("\n== Claim 1: FAMILY4 >=50 vs <50 ==")
hi, lo = fam[fam.hi], fam[~fam.hi]
tt, pp, nh, nl = clustered_welch(hi, lo)
print(f"  hi {cell(hi)} vs lo {cell(lo)}  t={tt:.2f} p={pp:.3f}  months hi/lo={nh}/{nl}")
print(f"  hi win% = {(hi['R_Multiple']>0).mean()*100:.0f}%")

print("\n== Claim 2: REST non-OVS >=50 vs <50 ==")
rhi, rlo = rest[rest.hi], rest[~rest.hi]
tt2, pp2, nh2, nl2 = clustered_welch(rhi, rlo)
print(f"  hi {cell(rhi)} vs lo {cell(rlo)}  t={tt2:.2f} p={pp2:.3f}  months hi/lo={nh2}/{nl2}")

print("\n== Claim 3: shortfall decomposition ==")
base_lo = t[~t.hi]["R_Multiple"].mean()
fam_short = len(hi) * base_lo - hi["R_Multiple"].sum()
rest_short = len(rhi) * base_lo - rhi["R_Multiple"].sum()
tot_short = fam_short + rest_short
print(f"  pooled <50 baseline avgR = {base_lo:+.4f}")
print(f"  family shortfall {fam_short:+.1f}R, rest {rest_short:+.1f}R, total {tot_short:+.1f}R")
print(f"  family share = {fam_short/tot_short*100:.0f}%  trade share = {len(hi)/(len(hi)+len(rhi))*100:.0f}%")

print("\n== Claim 4: family-vs-rest interaction at >=50 ==")
tt3, pp3, nh3, nl3 = clustered_welch(hi, rhi)
print(f"  diff = {hi['R_Multiple'].mean()-rhi['R_Multiple'].mean():+.3f}  t={tt3:.2f} p={pp3:.3f}  months {nh3}/{nl3}")
for yr in range(2016, 2027):
    sub = t[t["Signal Date"].dt.year != yr]
    h = sub[sub.fam & sub.hi]; r = sub[~sub.fam & sub.hi]
    if len(h) == 0:
        continue
    d = h["R_Multiple"].mean() - r["R_Multiple"].mean()
    tt4, pp4, _, _ = clustered_welch(h, r)
    print(f"  LOYO ex-{yr}: diff {d:+.3f}  t={tt4:.2f} p={pp4:.3f}")

print("\n== Claim 5: family damage under exclusions ==")
for yr in [2020, 2021, 2022, 2023, 2024, 2025, 2026]:
    sub = fam[fam["Signal Date"].dt.year != yr]
    h, l = sub[sub.hi], sub[~sub.hi]
    tt5, pp5, _, _ = clustered_welch(h, l)
    print(f"  ex-{yr}: hi {h['R_Multiple'].mean():+.3f} (N={len(h)}) lo {l['R_Multiple'].mean():+.3f} (N={len(l)})  t={tt5:.2f} p={pp5:.3f}")
print("  family hi-frag damage totR by year (vs lo baseline avg within family):")
fam_lo_avg = lo["R_Multiple"].mean()
for yr, g in hi.groupby(hi["Signal Date"].dt.year):
    print(f"    {yr}: totR {g['R_Multiple'].sum():+.1f} (N={len(g)})")

print("\n== Claim 6: Indices Oversold Bounce alone ==")
iob = t[t["Strategy"] == "Indices Oversold Bounce"]
print(f"  <50 {cell(iob[~iob.hi])}  >=50 {cell(iob[iob.hi])}")
i55 = iob[iob.frag >= 55]
print(f"  55+ {cell(i55)}")

print("\n== Per-strategy band table cross-check ==")
for s, g in t.groupby("Strategy"):
    h, l = g[g.hi], g[~g.hi]
    hs = f"{h['R_Multiple'].mean():+.3f} ({len(h)})" if len(h) else "—"
    print(f"  {s:28s} <50 {l['R_Multiple'].mean():+.3f} ({len(l)})  >=50 {hs}")

print("\n== Claims 7-8: design replay ==")

def taper_mult(f: float) -> float:
    if f < 50: return 1.0
    if f >= 60: return 0.5
    return 1.0 - 0.5 * (f - 50) / 10

designs = {
    "baseline": lambda r: 1.0,
    "(a) book taper 50-60": lambda r: taper_mult(r.frag),
    "(c) fam 0.25x @>=50": lambda r: 0.25 if (r.fam and r.hi) else 1.0,
    "(c) fam 0.5x @>=50": lambda r: 0.5 if (r.fam and r.hi) else 1.0,
    "(b) fam0.25 + taper rest": lambda r: 0.25 if (r.fam and r.hi) else taper_mult(r.frag),
}
mults = {}
for name, fn in designs.items():
    m = t.apply(fn, axis=1)
    mults[name] = m
    totR = (t["R_Multiple"] * m).sum()
    units = m.sum()
    print(f"  {name:26s} totR {totR:7.1f}  units {units:7.1f}  avgR/unit {totR/units:.4f}")

print("\n  yearly delta vs baseline (fam 0.25x):")
d = (t["R_Multiple"] * (mults["(c) fam 0.25x @>=50"] - 1.0))
for yr, g in d.groupby(t["Signal Date"].dt.year):
    if abs(g.sum()) > 0.05:
        print(f"    {yr}: {g.sum():+.1f}")
print("  yearly delta vs baseline (book taper):")
da = (t["R_Multiple"] * (mults["(a) book taper 50-60"] - 1.0))
for yr, g in da.groupby(t["Signal Date"].dt.year):
    if abs(g.sum()) > 0.05:
        print(f"    {yr}: {g.sum():+.1f}")

print("\n== Claims 9-10: drawdowns (realized R cumsum by exit date) ==")
t["Exit Date"] = pd.to_datetime(t["Exit Date"])

def dd_series(weighted_r: pd.Series) -> pd.DataFrame:
    daily = weighted_r.groupby(t["Exit Date"]).sum().sort_index()
    cum = daily.cumsum()
    peak = cum.cummax()
    return pd.DataFrame({"cum": cum, "dd": cum - peak})

def top_dds(dds: pd.DataFrame, k: int = 3):
    out = []
    dd = dds["dd"]
    used = pd.Series(False, index=dd.index)
    for _ in range(k):
        cand = dd[~used]
        if cand.empty or cand.min() >= 0:
            break
        trough = cand.idxmin()
        # find the peak before the trough
        pre = dds.loc[:trough]
        peak_date = pre["cum"][pre["cum"] == pre["cum"].cummax()].index[-1]
        peaks = pre["cum"].cummax()
        peak_date = peaks.idxmax() if False else pre["cum"].idxmax()
        out.append((peak_date, trough, dd[trough]))
        used[(used.index >= peak_date) & (used.index <= trough)] = True
    return out

for name in ["baseline", "(c) fam 0.25x @>=50", "(a) book taper 50-60"]:
    dds = dd_series(t["R_Multiple"] * mults[name])
    print(f"  {name}: worst DDs:")
    for pk, trough, v in top_dds(dds):
        print(f"    {pk.date()} -> {trough.date()}: {v:+.1f}R")

# June 2026 episode composition
dds_b = dd_series(t["R_Multiple"])
worst_trough = dds_b["dd"].idxmin()
pre = dds_b.loc[:worst_trough]
pk = pre["cum"].idxmax()
ep = t[(t["Exit Date"] > pk) & (t["Exit Date"] <= worst_trough)]
print(f"\n  worst-DD window {pk.date()} -> {worst_trough.date()}: N={len(ep)}, avg frag {ep['frag'].mean():.1f}, max frag {ep['frag'].max():.1f}, n at >=50 = {(ep.frag>=50).sum()}, totR {ep['R_Multiple'].sum():+.1f}")

# second-worst window under baseline; recompute under family cut
print("\n  2021-11..2022-01 window check:")
w = t[(t["Exit Date"] >= "2021-11-01") & (t["Exit Date"] <= "2022-01-31")]
print(f"    trades exiting in window: {len(w)}, totR {w['R_Multiple'].sum():+.1f}")
EOF_SENTINEL_NOT_USED = None
