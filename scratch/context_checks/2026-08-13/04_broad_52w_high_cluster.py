"""No trigger fired on the most specific thing on tonight's tape.

SPY, ^GSPC, IWM, ^RUT, ^NYA and HYG all closed AT a 52-week high on the same
session. QQQ did not, 1.78% under its own. P1 only fires on a FIRST high in 30+
days, and these have been printing highs, so the sweep never looked.

The 08-10 brief already published SPY-at-a-high-with-TLT-on-the-floor and found
nothing in it. This is a different cell: small caps, the broad NYSE composite
and high yield credit confirming together while the growth index lags. Does
that breadth of confirmation say anything the bare SPY high does not?
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (close_panel, declusters, era_split,  # noqa: E402
                       fwd_ret, sign_test, summarize, cluster_note)

TK = ["SPY", "QQQ", "IWM", "^NYA", "^RUT", "HYG"]
px = close_panel(TK)
px = px[px.index >= "1999-01-01"]

at_high, dist = {}, {}
for t in TK:
    s = px[t]
    hi = s.rolling(252, min_periods=200).max()
    dist[t] = s / hi - 1.0
    at_high[t] = dist[t] >= -0.0005          # within 5bp of the trailing high

print("tonight's distances to a 52-week high:")
for t in TK:
    print(f"   {t:<6} {100 * dist[t].iloc[-1]:+6.2f}%   at_high={bool(at_high[t].iloc[-1])}")

spy = px["SPY"].dropna()
start = max(px[t].dropna().index[0] for t in TK)
print(f"\njoint history starts {start.date()} (HYG is the binding constraint)\n")


def rep(name, mask_idx, subject="SPY", h=(1, 5, 10, 21), gap=10, era=False):
    s = px[subject].dropna()
    dc = declusters(pd.DatetimeIndex(mask_idx), gap, s.index)
    out = []
    for hh in h:
        r = fwd_ret(s, hh).reindex(dc).dropna()
        if len(r) < 4:
            continue
        d = summarize(r.to_numpy(), name)
        up = int((r > 0).sum())
        out.append((hh, d, up, len(r), r))
        print(f"   {name:<42} h{hh:<3} n={len(r):>4} mean={d['mean_pct']:+6.2f}% "
              f"med={d['median_pct']:+6.2f}% up={up}-{len(r) - up} "
              f"({100 * up / len(r):4.1f}%) t={d['t']:+5.2f} signp={sign_test(up, len(r)):.4f}")
    if era and out:
        hh, d, up, n, r = out[1] if len(out) > 1 else out[0]
        for e in era_split(r.index, r.to_numpy()):
            if e["n"]:
                print(f"        era {e['label']}: n={e['n']} mean={e['mean_pct']:+.2f}% "
                      f"hit={e['hit']:.1f}%")
        print("        ", cluster_note(r.index, r.to_numpy()))
    return out


common = px[TK].dropna().index

broad = common[(at_high["SPY"].reindex(common).fillna(False)
                & at_high["IWM"].reindex(common).fillna(False)
                & at_high["HYG"].reindex(common).fillna(False)).to_numpy()]
broad_nya = common[(at_high["SPY"].reindex(common).fillna(False)
                    & at_high["IWM"].reindex(common).fillna(False)
                    & at_high["HYG"].reindex(common).fillna(False)
                    & at_high["^NYA"].reindex(common).fillna(False)).to_numpy()]
spy_only = common[at_high["SPY"].reindex(common).fillna(False).to_numpy()]
spy_no_iwm = common[(at_high["SPY"].reindex(common).fillna(False)
                     & ~at_high["IWM"].reindex(common).fillna(False)).to_numpy()]

print(f"raw session counts since {start.date()}: "
      f"SPY at a high {len(spy_only)}, SPY+IWM+HYG {len(broad)}, "
      f"+^NYA {len(broad_nya)}, SPY high without IWM {len(spy_no_iwm)}")

print("\n=== A. SPY forward, the broad cell vs the bare SPY high ===")
rep("SPY+IWM+HYG all at a 52w high", broad, era=True)
print()
rep("SPY at a 52w high (bare)", spy_only)
print()
rep("SPY at a high, IWM NOT", spy_no_iwm)
print()
rep("all sessions in the joint window", common, gap=10)

print("\n=== B. add the NYSE composite ===")
rep("SPY+IWM+HYG+^NYA all at a 52w high", broad_nya, era=True)

print("\n=== C. tonight's extra condition: QQQ is NOT at its high ===")
q_lag = common[(at_high["SPY"].reindex(common).fillna(False)
                & at_high["IWM"].reindex(common).fillna(False)
                & at_high["HYG"].reindex(common).fillna(False)
                & (dist["QQQ"].reindex(common) < -0.01).fillna(False)).to_numpy()]
print(f"   sessions with the broad cell AND QQQ more than 1% off its high: {len(q_lag)}")
rep("broad high, QQQ lagging 1%+", q_lag, era=True)
print()
rep("broad high, QQQ also at its high",
    common[(at_high["SPY"].reindex(common).fillna(False)
            & at_high["IWM"].reindex(common).fillna(False)
            & at_high["HYG"].reindex(common).fillna(False)
            & at_high["QQQ"].reindex(common).fillna(False)).to_numpy()])

print("\n=== D. what does the broad cell do to IWM, the laggard that just caught up ===")
rep("SPY+IWM+HYG high -> IWM forward", broad, subject="IWM", era=True)
print()
rep("all sessions -> IWM forward", common, subject="IWM")

print("\n=== E. how rare is the cluster, and when did it last print ===")
dc = declusters(pd.DatetimeIndex(broad), 21, spy.index)
yrs = pd.Series(1, index=pd.DatetimeIndex(broad)).groupby(
    pd.DatetimeIndex(broad).year).sum()
print("   sessions per year:", {int(k): int(v) for k, v in yrs.items()})
print(f"   declustered episodes (21td gap): {len(dc)}")
print("   last 12 episode dates:", [str(d.date()) for d in dc[-12:]])
