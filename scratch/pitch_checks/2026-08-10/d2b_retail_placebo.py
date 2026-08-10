"""D2 round 2 -- the calendar placebo that decides it.

Round 1 (d2_retail_cluster.py) produced the number that matters and it is not
close: the August retail-cluster basket returns +0.698% over the tradeable
h=4 window against a control of ALL AUGUST 4-session windows at +0.681%.
Edge = +0.017pp. The apparent +0.436pp "edge vs all-days" is the retail
complex's August seasonality, not the earnings cluster.

This script nails that down three ways, because "the control ate it" deserves
more than one framing:

  1. ANCHOR PLACEBO. Slide the cluster anchor by +/- 5, 10, 15, 20, 25
     sessions and re-measure the identical k=-6/h=4 window. If fake anchors
     in the same weeks pay the same, the earnings cluster is doing no work.
  2. SEASON MULTIPLICITY. August is 1 of the 4 annual clusters. The pooled
     cell across all 103 clusters is NEGATIVE. Picking August is a search
     over 4 and the other 3 are the evidence against.
  3. WHAT ACTUALLY PAYS. Decompose the August window into SPY beta + residual.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

NAMES = ["HD", "LOW", "TGT", "TJX", "ROST", "WMT", "M", "KSS", "BBY",
         "DG", "DLTR"]
BIG6 = ["HD", "LOW", "TGT", "TJX", "ROST", "WMT"]
px = close_panel(NAMES + ["XLY", "SPY", "XRT"])
px = px.loc[px.index >= "2000-01-03"]
idx = px.index

ec = pd.read_parquet(ROOT / "data" / "earnings_calendar.parquet")
ec = ec[ec["ticker"].isin(NAMES)][["ticker", "date"]].copy()
ec["date"] = pd.to_datetime(ec["date"])
# 2000-06-01 floor: earlier prints clip to session 0 and make a fake cluster
ec = ec[ec["date"] >= "2000-06-01"].drop_duplicates().sort_values("date")
ec["p"] = ec["date"].map(lambda d: int(idx.searchsorted(d))
                         if idx.searchsorted(d) < len(idx) else np.nan)
ec = ec.dropna(subset=["p"])
ec["p"] = ec["p"].astype(int)

clusters = []
for yr, g in ec.groupby(ec["date"].dt.year):
    for lo, hi, tag in [(2, 3, "Feb"), (5, 6, "May"), (8, 9, "Aug"),
                        (11, 12, "Nov")]:
        gg = g[(g["date"].dt.month >= lo) & (g["date"].dt.month <= hi)
               ].sort_values("p")
        if gg.empty:
            continue
        ps = gg["p"].values
        best = next((i for i in range(len(ps))
                     if np.searchsorted(ps, ps[i] + 2, side="right") - i >= 4),
                    None)
        if best is None:
            continue
        j = np.searchsorted(ps, ps[best] + 2, side="right")
        clusters.append({"year": yr, "season": tag, "anchor_p": int(ps[best]),
                         "anchor_date": idx[int(ps[best])],
                         "reporters": sorted(gg.iloc[best:j]["ticker"].tolist())})
cl = pd.DataFrame(clusters).sort_values("anchor_p").reset_index(drop=True)
aug = cl[cl["season"] == "Aug"].reset_index(drop=True)
print(f"clusters {len(cl)}  (Aug {len(aug)}, span "
      f"{aug['anchor_date'].min().date()} .. {aug['anchor_date'].max().date()})")

K_SIG, LAG, H = -6, 1, 4


def window_ret(rows, shift=0, k_sig=K_SIG, h=H, names=None):
    out = []
    for _, c in rows.iterrows():
        pe = c["anchor_p"] + shift + k_sig + LAG
        pex = pe + h
        if pe < 0 or pex >= len(idx):
            out.append({"year": c["year"], "basket": np.nan, "XLY": np.nan,
                        "SPY": np.nan})
            continue
        nm = names or c["reporters"]
        per = []
        for t in nm:
            a, b = px[t].iloc[pe], px[t].iloc[pex]
            if np.isfinite(a) and np.isfinite(b) and a > 0:
                per.append(b / a - 1.0)
        rec = {"year": c["year"],
               "basket": float(np.mean(per)) if per else np.nan}
        for t in ("XLY", "SPY"):
            a, b = px[t].iloc[pe], px[t].iloc[pex]
            rec[t] = (b / a - 1.0) if np.isfinite(a) and np.isfinite(b) else np.nan
        out.append(rec)
    return pd.DataFrame(out)


print("\n" + "=" * 78)
print("1. ANCHOR PLACEBO -- slide the anchor, keep the window identical")
print("   (a real earnings-cluster effect dies when the anchor is fake)")
print("=" * 78)
real = window_ret(aug)
print(f"  {'shift':>7s} {'basket %':>9s} {'record':>8s} {'signp':>7s} "
      f"{'XLY %':>8s} {'SPY %':>8s}")
placebo_b, placebo_x = [], []
for sh in (-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25):
    d = window_ret(aug, shift=sh).dropna(subset=["basket"])
    v = d["basket"].values
    w = int((v > 0).sum())
    tag = "  <== REAL" if sh == 0 else ""
    print(f"  {sh:+7d} {100*v.mean():+9.3f} {w:4d}-{len(v)-w:<3d} "
          f"{sign_test(w, len(v)):7.3f} {100*d['XLY'].mean():+8.3f} "
          f"{100*d['SPY'].mean():+8.3f}{tag}")
    if sh != 0:
        placebo_b.append(100 * v.mean())
        placebo_x.append(100 * d["XLY"].mean())
pb = np.array(placebo_b)
print(f"\n  placebo anchors (10 fakes): basket mean {pb.mean():+.3f}%  "
      f"sd {pb.std(ddof=1):.3f}  range [{pb.min():+.3f}, {pb.max():+.3f}]")
print(f"  REAL anchor {100*real['basket'].mean():+.3f}% sits at percentile "
      f"{100*(pb < 100*real['basket'].mean()).mean():.0f} of its own placebos "
      f"-> z = {(100*real['basket'].mean()-pb.mean())/pb.std(ddof=1):+.2f}")
px_ = np.array(placebo_x)
print(f"  XLY: REAL {100*real['XLY'].mean():+.3f}% vs placebo mean "
      f"{px_.mean():+.3f}% (sd {px_.std(ddof=1):.3f}) -> z = "
      f"{(100*real['XLY'].mean()-px_.mean())/px_.std(ddof=1):+.2f}")

print("\n" + "=" * 78)
print("2. SEASON MULTIPLICITY -- August is 1 of 4 and the pool is negative")
print("=" * 78)
tot = []
for s in ("Feb", "May", "Aug", "Nov"):
    d = window_ret(cl[cl["season"] == s]).dropna(subset=["basket"])
    v = d["basket"].values
    w = int((v > 0).sum())
    tot.append(v)
    print(f"  {s}: basket {100*v.mean():+.3f}% (N={len(v)}, {w}-{len(v)-w}, "
          f"sign p {sign_test(w, len(v)):.3f})  minus SPY "
          f"{100*(d['basket']-d['SPY']).mean():+.3f}%")
allv = np.concatenate(tot)
w = int((allv > 0).sum())
print(f"  POOLED across all four clusters: {100*allv.mean():+.3f}% "
      f"(N={len(allv)}, {w}-{len(allv)-w}, sign p "
      f"{sign_test(w, len(allv)):.3f}, boot P<=0 {bootstrap_p_le0(allv):.3f})")
print(f"  -> if the mechanism is 'drift into a predictable retail earnings")
print(f"     cluster', it must appear in Feb/May/Nov too. It does not.")

print("\n" + "=" * 78)
print("3. DECOMPOSITION -- what is the August window actually paying?")
print("=" * 78)
d = real.dropna(subset=["basket"])
beta = np.cov(d["basket"], d["SPY"])[0, 1] / np.var(d["SPY"], ddof=1)
alpha = d["basket"].mean() - beta * d["SPY"].mean()
resid = d["basket"] - beta * d["SPY"]
w = int((resid > 0).sum())
print(f"  basket = {beta:.2f} x SPY + alpha;  alpha = {100*alpha:+.3f}% over "
      f"the 4-session window")
print(f"  residual (beta-hedged) {100*resid.mean():+.3f}%  N={len(resid)}  "
      f"{w}-{len(resid)-w}  sign p {sign_test(w, len(resid)):.3f}  "
      f"boot P<=0 {bootstrap_p_le0(resid.values):.3f}")
print(f"  hedged cost: 7 legs x 3 bps = 21 bps vs edge "
      f"{100*resid.mean()*100:.1f} bps -> {resid.mean()*10000/21:.2f}x")

print("\n" + "=" * 78)
print("4. THE AUGUST-SEASONALITY CONTROL, stated once more, cleanly")
print("=" * 78)
bd = px[BIG6].pct_change().mean(axis=1)
cum = (1 + bd.fillna(0)).cumprod()
fwd4 = cum.shift(-H) / cum - 1.0
augmask = pd.Series(idx.month == 8, index=idx)
a_all = fwd4[augmask & fwd4.notna()]
print(f"  ALL August 4-session windows, retail basket: {100*a_all.mean():+.3f}% "
      f"(N={len(a_all)})")
print(f"  Cluster-anchored August window:              "
      f"{100*d['basket'].mean():+.3f}% (N={len(d)})")
print(f"  EDGE OF THE CLUSTER ANCHOR OVER PLAIN AUGUST: "
      f"{100*(d['basket'].mean()-a_all.mean()):+.3f}pp")
print(f"  cost of the 6-name basket: 18 bps round trip -> "
      f"{(d['basket'].mean()-a_all.mean())*10000/18:.2f}x cost")

print("\n" + "=" * 78)
print("5. TODAY'S STATE vs THE HISTORICAL SIGNAL-DAY DISTRIBUTION")
print("=" * 78)
zt = {t: zscore(px[t]) for t in BIG6 + ["XLY"]}
hist = []
for _, c in aug.iterrows():
    p = c["anchor_p"] + K_SIG
    if p < 260:
        continue
    hist.append({"year": c["year"], "xly_z10": zt["XLY"].iloc[p],
                 "basket_z10": float(np.nanmean([zt[t].iloc[p] for t in BIG6]))})
h = pd.DataFrame(hist)
tz, bz = zt["XLY"].iloc[-1], float(np.nanmean([zt[t].iloc[-1] for t in BIG6]))
print(f"  historical signal-day XLY z10: min {h['xly_z10'].min():+.2f}  "
      f"median {h['xly_z10'].median():+.2f}  max {h['xly_z10'].max():+.2f} "
      f"(in {int(h.loc[h['xly_z10'].idxmax(),'year'])})")
print(f"  TODAY XLY z10 = {tz:+.2f}  -> "
      f"{'ABOVE EVERY HISTORICAL ANALOGUE' if tz > h['xly_z10'].max() else 'inside the range'}")
print(f"  historical signal-day basket z10: max {h['basket_z10'].max():+.2f}; "
      f"TODAY {bz:+.2f}")
top = h.nlargest(3, "xly_z10")
rr = real.set_index("year")
for _, r_ in top.iterrows():
    y = int(r_["year"])
    print(f"    nearest analogue {y}: XLY z10 {r_['xly_z10']:+.2f} -> basket "
          f"{100*rr.loc[y,'basket']:+.2f}%, XLY {100*rr.loc[y,'XLY']:+.2f}%")
