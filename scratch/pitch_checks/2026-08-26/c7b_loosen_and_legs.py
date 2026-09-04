"""C7 round 1b: the literal rung has ONE day in 20 years (today).

"Unmeasurable is a kill, not a pass" (registry 2026-08-07), but the honest
move is to loosen to the nearest reasonable definitions and see whether the
SHAPE works at all, then do leg attribution (trap 8) on whatever survives.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

T = ["KRE", "XLF", "SPY"]
px = load_prices(T)
common = px["KRE"].index.intersection(px["XLF"].index).intersection(px["SPY"].index)
P = pd.DataFrame({t: px[t]["Close"].reindex(common) for t in T}).dropna()

r5k = pct_rank(P["KRE"], 5)
r63f = pct_rank(P["XLF"], 63)
hi = rolling_on_valid(P["XLF"], lambda x: x.rolling(252).max())
offhi = P["XLF"] / hi - 1.0
z10k = zscore(P["KRE"], 10)

print("=" * 78)
print("0. WHICH CONDITION BINDS?  day counts, 2007+ (rank warm-up excluded)")
print("=" * 78)
warm = P.index >= "2007-07-01"
print(f"  sessions in sample            : {int(warm.sum())}")
print(f"  KRE r5 <= 10                  : {int((warm & (r5k<=10)).sum())}")
print(f"  XLF r63 >= 95                 : {int((warm & (r63f>=95)).sum())}")
print(f"  XLF within 0.5% of 52w high   : {int((warm & (offhi>=-0.005)).sum())}")
print(f"  KRE r5<=10 & XLF r63>=95      : {int((warm & (r5k<=10) & (r63f>=95)).sum())}")
print(f"  KRE r5<=10 & XLF near-high    : {int((warm & (r5k<=10) & (offhi>=-0.005)).sum())}")
print(f"  all three                     : {int((warm & (r5k<=10) & (r63f>=95) & (offhi>=-0.005)).sum())}")

RUNGS = {
    "LITERAL r5<=10, r63>=95, hi 0.5%": (r5k <= 10) & (r63f >= 95) & (offhi >= -0.005),
    "r5<=10, r63>=95, hi 2%": (r5k <= 10) & (r63f >= 95) & (offhi >= -0.02),
    "r5<=10, r63>=90, hi 2%": (r5k <= 10) & (r63f >= 90) & (offhi >= -0.02),
    "r5<=15, r63>=85, hi 3%": (r5k <= 15) & (r63f >= 85) & (offhi >= -0.03),
    "r5<=20, r63>=80, hi 5%": (r5k <= 20) & (r63f >= 80) & (offhi >= -0.05),
    "r5<=20, r63>=80 (no hi gate)": (r5k <= 20) & (r63f >= 80),
    "GATE-OFF: r5<=20 alone": (r5k <= 20),
    "GATE-ONLY: r63>=80 & hi 5% (no KRE)": (r63f >= 80) & (offhi >= -0.05),
}

print("\n" + "=" * 78)
print("1. LOOSENING LADDER, equal-dollar pair LONG KRE / SHORT XLF, episodes")
print("=" * 78)
for h in (3, 5, 10):
    rows = []
    for lbl, m in RUNGS.items():
        m = m & pd.Series(warm, index=P.index)
        ret = vehicle_ret(P, [("KRE", 1.0), ("XLF", -1.0)], h)
        sig = P.index[m.fillna(False).values & ret.notna().values]
        if len(sig) == 0:
            rows.append({"label": lbl, "n": 0, "n_days": 0})
            continue
        epi = declusters(sig, 5, P.index)
        r = summarize(ret.loc[epi].values, lbl)
        r["n_days"] = len(sig)
        r["x_cost"] = round(100 * ret.loc[epi].mean() * 100 / 12, 1)
        rows.append(r)
    show(rows, f"h={h}  (12 bp two-leg round trip, need >= 5x)")

print("\n" + "=" * 78)
print("2. LEG ATTRIBUTION on the loosest measurable rung (r5<=20, r63>=80, hi 5%)")
print("=" * 78)
m = RUNGS["r5<=20, r63>=80, hi 5%"] & pd.Series(warm, index=P.index)
sig = P.index[m.fillna(False).values]
dk = P["KRE"].pct_change().dropna()
dfp = P["XLF"].pct_change().reindex(dk.index)
beta = np.polyfit(dfp.values[1:], dk.values[1:], 1)[0]
print(f"beta(KRE on XLF), daily, full sample = {beta:.3f}")
for h in (3, 5, 10):
    epi = declusters(sig[fwd_lag(P['KRE'], h).loc[sig].notna().values], 5, P.index)
    k = fwd_lag(P["KRE"], h).loc[epi].values
    f = fwd_lag(P["XLF"], h).loc[epi].values
    s = fwd_lag(P["SPY"], h).loc[epi].values
    kb = fwd_lag(P["KRE"], h).dropna().values
    fb = fwd_lag(P["XLF"], h).dropna().values
    show([summarize(k, f"h={h} NAKED LONG KRE (N={len(k)})"),
          summarize(kb, "  KRE own drift, all days"),
          summarize(-f, f"h={h} NAKED SHORT XLF"),
          summarize(-fb, "  -XLF own drift, all days"),
          summarize(k - f, f"h={h} equal-dollar PAIR"),
          summarize(k - beta * f, f"h={h} beta-neutral resid"),
          summarize(k - s, f"h={h} KRE - SPY")])
    print(f"  long-leg excess over own drift {100*(k.mean()-kb.mean()):+.3f}pp; "
          f"short-leg contribution {100*(-f.mean()):+.3f}pp; "
          f"pair {100*(k-f).mean()*100:.1f} bp = "
          f"{100*(k-f).mean()*100/12:.1f}x cost; naked long "
          f"{100*k.mean()*100:.1f} bp = {100*k.mean()*100/6:.1f}x cost")
    ok = np.array([d.year not in (2008, 2009, 2020) for d in epi])
    v = (k - f)[ok]
    print(f"  ex-crisis pair (N={int(ok.sum())}): {100*v.mean():+.3f}% = "
          f"{100*v.mean()*100/12:.1f}x cost;  ex-crisis naked long "
          f"{100*k[ok].mean():+.3f}%")
    mid = np.array([d.year % 4 == 2 for d in epi])
    print(f"  midterm (today) N={int(mid.sum())}: pair "
          f"{100*(k-f)[mid].mean():+.3f}%  vs non-midterm "
          f"{100*(k-f)[~mid].mean():+.3f}%")
    show(era_split(epi, (k - f)), "  pair era split")
    print(f"  {cluster_note(epi, k - f)}\n")

print("=" * 78)
print("3. REFERENCE CLASS: identical rule on 10 sub-industry / parent-sector pairs")
print("=" * 78)
PAIRS = [("KRE", "XLF"), ("XRT", "XLY"), ("IHI", "XLV"), ("XME", "XLB"),
         ("OIH", "XLE"), ("XOP", "XLE"), ("SMH", "XLK"), ("ITB", "XLY"),
         ("XHB", "XLY"), ("XBI", "XLV")]
allt = sorted({x for p in PAIRS for x in p})
px2 = load_prices(allt)
idx = None
for t in allt:
    idx = px2[t].index if idx is None else idx.intersection(px2[t].index)
Q = pd.DataFrame({t: px2[t]["Close"].reindex(idx) for t in allt}).dropna()
Q = Q[Q.index >= "2007-07-01"]
rows = []
for a, b in PAIRS:
    ra = pct_rank(Q[a], 5)
    rb = pct_rank(Q[b], 63)
    hb = rolling_on_valid(Q[b], lambda x: x.rolling(252).max())
    ob = Q[b] / hb - 1.0
    mm = (ra <= 20) & (rb >= 80) & (ob >= -0.05)
    ret = vehicle_ret(Q, [(a, 1.0), (b, -1.0)], 5)
    sg = Q.index[mm.fillna(False).values & ret.notna().values]
    if len(sg) < 5:
        rows.append({"label": f"{a}/{b}", "n": 0})
        continue
    ep = declusters(sg, 5, Q.index)
    r = summarize(ret.loc[ep].values, f"{a}/{b}")
    naked = fwd_lag(Q[a], 5).loc[ep].values
    r["naked_long_pct"] = round(100 * naked.mean(), 3)
    rows.append(r)
df = pd.DataFrame(rows).sort_values("mean_pct", ascending=False)
for c in df.columns:
    if df[c].dtype.kind == "f":
        df[c] = df[c].round(3)
print(df.to_string(index=False))
kre_t = float(df.loc[df["label"] == "KRE/XLF", "t"].iloc[0])
ts = df["t"].dropna().values
print(f"\nKRE/XLF t = {kre_t:+.2f}, rank {1+int((ts>kre_t).sum())} of {len(ts)} by t")
# Cochran Q over the pairs
mus = df["mean_pct"].values / 100
ses = (df["sd_pct"].values / 100) / np.sqrt(df["n"].values)
w = 1 / ses ** 2
mfe = (w * mus).sum() / w.sum()
Qstat = (w * (mus - mfe) ** 2).sum()
from math import erfc
print(f"fixed-effect common pair excess = {100*mfe:+.3f}%; Cochran Q = {Qstat:.2f} "
      f"on {len(mus)-1} df")
