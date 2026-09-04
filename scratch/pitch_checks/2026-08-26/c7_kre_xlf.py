"""C7 round 1: LONG KRE against SHORT XLF on the intra-financials gap.

Trigger (today's state): KRE 5d PIT rank <= 10 (today 9.1) AND XLF 63d PIT
rank >= 95 (96.8) AND XLF within 0.5% of its 52-week high (0.00%).

Watchlist #19 is the OPPOSITE side (short KRE / long XLF) on a bank-breadth
washout, parked at 1.5x cost ex-crisis.  Trap 8 is mandatory here: measure the
naked long KRE and the naked short XLF separately and beta-neutralise.
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
close = {t: px[t]["Close"].dropna() for t in T}
common = close["KRE"].index.intersection(close["XLF"].index).intersection(
    close["SPY"].index)
P = pd.DataFrame({t: close[t].reindex(common) for t in T}).dropna()
print(f"panel {P.index[0].date()} .. {P.index[-1].date()}  N={len(P)}")

r5_kre = pct_rank(P["KRE"], 5)
r63_xlf = pct_rank(P["XLF"], 63)
hi52_xlf = rolling_on_valid(P["XLF"], lambda x: x.rolling(252).max())
off_hi = P["XLF"] / hi52_xlf - 1.0
z10_kre = zscore(P["KRE"], 10)

print(f"\nTODAY: KRE r5={r5_kre.iloc[-1]:.1f}  z10={z10_kre.iloc[-1]:+.2f}  "
      f"XLF r63={r63_xlf.iloc[-1]:.1f}  XLF off-high={100*off_hi.iloc[-1]:+.2f}%")

MASK = (r5_kre <= 10) & (r63_xlf >= 95) & (off_hi >= -0.005)
print(f"trigger days: {int(MASK.sum())}")

VARIANTS = {
    "r5<=5, r63>=95, hi<=0.5%": (r5_kre <= 5) & (r63_xlf >= 95) & (off_hi >= -0.005),
    "r5<=20, r63>=95, hi<=0.5%": (r5_kre <= 20) & (r63_xlf >= 95) & (off_hi >= -0.005),
    "r5<=10, r63>=90, hi<=0.5%": (r5_kre <= 10) & (r63_xlf >= 90) & (off_hi >= -0.005),
    "r5<=10, r63>=95, hi<=2%": (r5_kre <= 10) & (r63_xlf >= 95) & (off_hi >= -0.02),
    "r5<=10 alone (no XLF gate)": (r5_kre <= 10),
    "XLF gate alone (no KRE leg)": (r63_xlf >= 95) & (off_hi >= -0.005),
}

for h in (3, 5, 10):
    battery(P, MASK, [("KRE", 1.0), ("XLF", -1.0)], h,
            f"C7 LONG KRE / SHORT XLF (equal dollar)", 6.0,
            variants=VARIANTS if h == 5 else None, min_gap=5)

print("\n" + "=" * 78)
print("LEG ATTRIBUTION (trap 8) + beta-neutral residual")
print("=" * 78)
sig = P.index[MASK.values]
epi = declusters(sig, 5, P.index)
print(f"episodes (min gap 5 td): N={len(epi)}  "
      f"{[str(d.date()) for d in epi]}")
for h in (3, 5, 10):
    k = fwd_lag(P["KRE"], h).loc[epi].values
    f = fwd_lag(P["XLF"], h).loc[epi].values
    s = fwd_lag(P["SPY"], h).loc[epi].values
    # beta of KRE on XLF, daily returns, full sample
    dk = P["KRE"].pct_change().dropna()
    df_ = P["XLF"].pct_change().reindex(dk.index)
    beta = np.polyfit(df_.values[1:], dk.values[1:], 1)[0]
    resid = k - beta * f
    kb = fwd_lag(P["KRE"], h).dropna()
    fb = fwd_lag(P["XLF"], h).dropna()
    rows = [summarize(k, f"h={h} NAKED LONG KRE"),
            summarize(kb.values, f"  KRE all-days drift"),
            summarize(-f, f"h={h} NAKED SHORT XLF"),
            summarize(-fb.values, "  -XLF all-days drift"),
            summarize(k - f, f"h={h} equal-dollar PAIR"),
            summarize(resid, f"h={h} beta-neutral (beta={beta:.2f})"),
            summarize(k - s, f"h={h} KRE minus SPY")]
    show(rows)
    print(f"  cost: naked leg needs 30 bps (5x6); pair needs 60 bps (5x12). "
          f"long {100*k.mean()*100:.1f} bps -> {100*k.mean()*100/6:.1f}x ; "
          f"pair {100*(k-f).mean()*100:.1f} bps -> {100*(k-f).mean()*100/12:.1f}x ; "
          f"beta-neutral {100*resid.mean()*100:.1f} bps -> "
          f"{100*resid.mean()*100/12:.1f}x")
    print(f"  excess over own drift: long KRE "
          f"{100*(k.mean()-kb.mean()):+.3f}pp ; short XLF "
          f"{100*(-f.mean()+fb.mean()):+.3f}pp\n")

print("=" * 78)
print("EX-CRISIS SUBSET (drop 2008/2009/2020), the watchlist #19 arithmetic")
print("=" * 78)
ok = np.array([d.year not in (2008, 2009, 2020) for d in epi])
for h in (3, 5, 10):
    k = fwd_lag(P["KRE"], h).loc[epi].values
    f = fwd_lag(P["XLF"], h).loc[epi].values
    show([summarize((k - f)[ok], f"h={h} pair ex-crisis (N={int(ok.sum())})"),
          summarize(k[ok], f"h={h} naked long KRE ex-crisis"),
          summarize((k - f)[~ok], f"h={h} pair crisis years only")])
    v = (k - f)[ok]
    if len(v):
        print(f"  ex-crisis pair {100*v.mean()*100:.1f} bps -> "
              f"{100*v.mean()*100/12:.1f}x a 12 bp two-leg round trip "
              f"({100*v.mean()*100/7:.1f}x at the watchlist's 7 bp)")
        print(f"  {cluster_note(epi[ok], v)}\n")
