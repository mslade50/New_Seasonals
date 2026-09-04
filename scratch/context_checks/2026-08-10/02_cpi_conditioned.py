"""CPI at k=2, conditioned on the state this one is walking into.

Anchor = the session 2 td before a CPI (today's analogue). So h1 is TOMORROW,
the eve, and h2 is the CPI session's own close-to-close move.

Live conditions: CL=F 21d +15.25% (rank 86.1), SPY 0.03% off its 52w high,
TLT 0.17% off its 52w low, VIX 15.46 and 50% below its 52w high, midterm August.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    close_panel, load_events, fwd_ret, summarize, era_split, sign_test,
    cluster_note,
)

TIX = ["SPY", "^GSPC", "^TNX", "TLT", "^VIX", "GC=F", "CL=F", "QQQ"]
px = close_panel(TIX)
px = px[px.index >= "1999-01-01"]
dates = px.index

ev = load_events(["cpi"])
ev = ev[(ev["date"] >= dates[0]) & (ev["date"] <= dates[-1])]
print(f"CPI events in panel: {len(ev)}  {ev['date'].min().date()} -> {ev['date'].max().date()}")

# anchor = 2 td before each CPI
anchors = []
for d in ev["date"]:
    pos = dates.searchsorted(pd.Timestamp(d))
    if pos >= len(dates) or dates[pos] != pd.Timestamp(d):
        continue                       # CPI on a non-session, skip
    if pos - 2 < 0:
        continue
    anchors.append((dates[pos - 2], dates[pos]))
anch = pd.DatetimeIndex([a for a, _ in anchors])
print(f"anchors (2 td before a CPI session): {len(anch)}")

cl21 = px["CL=F"] / px["CL=F"].shift(21) - 1.0
cl21_rank = cl21.rolling(252).rank(pct=True) * 100
spy_dh = px["SPY"] / px["SPY"].rolling(252).max() - 1.0
vix_dh = px["^VIX"] / px["^VIX"].rolling(252).max() - 1.0

print("\nlive: CL 21d %+.2f%% (rank %.1f)  SPY dist-high %+.3f%%  VIX dist-high %+.1f%%"
      % (100 * cl21.iloc[-1], cl21_rank.iloc[-1], 100 * spy_dh.iloc[-1], 100 * vix_dh.iloc[-1]))


def report(idx, label, subjects=("SPY", "^TNX", "^VIX", "TLT", "GC=F"), hs=(1, 2, 5)):
    idx = pd.DatetimeIndex(sorted(set(idx)))
    print(f"\n=== {label}   n_anchors={len(idx)}")
    if len(idx) < 4:
        print("   too few, skipping")
        return
    print(f"   years: {sorted(set(idx.year))}")
    for sub in subjects:
        s = px[sub].dropna()
        for h in hs:
            f = fwd_ret(s, h)
            v = f.reindex(idx).dropna()
            if len(v) < 4:
                continue
            r = summarize(v.values, "")
            base = summarize(f.dropna().values, "")
            up = int((v.values > 0).sum())
            hlab = {1: "h1 eve", 2: "h2 CPIday", 5: "h5"}.get(h, f"h{h}")
            print(f"   {sub:6s} {hlab:10s} n={r['n']:3d} mean {r['mean_pct']:+.2f}% "
                  f"med {r['median_pct']:+.2f}% hit {r['hit']:.0f}% t={r['t']:+.2f} | "
                  f"{up}-{len(v) - up} up-p {sign_test(up, len(v)):.4f} "
                  f"dn-p {sign_test(len(v) - up, len(v)):.4f} | all {base['mean_pct']:+.2f}%")
        if sub in ("SPY", "^TNX"):
            v2 = fwd_ret(px[sub].dropna(), 2).reindex(idx).dropna()
            if len(v2) >= 4:
                print(f"          h2 {cluster_note(v2.index, v2.values)}")
                for e in era_split(v2.index, v2.values):
                    print(f"          h2 era n={e['n']:3d} mean {e['mean_pct']:+.2f}% "
                          f"hit {e['hit']:.0f}% t={e['t']:+.2f}")


report(anch, "ALL CPI anchors")

# --- crude running hard into the print ---------------------------------------
for thr in (0.10, 0.15):
    m = cl21.reindex(anch) >= thr
    report(anch[m.fillna(False).values], f"CPI with CL=F 21d >= +{100*thr:.0f}%")

m = cl21_rank.reindex(anch) >= 85
report(anch[m.fillna(False).values], "CPI with CL=F 21d return in the top 15% of its year")

# --- index pinned at a 52w high ----------------------------------------------
m = spy_dh.reindex(anch) > -0.005
report(anch[m.fillna(False).values], "CPI with SPY within 0.5% of its 52w high")

m = (spy_dh.reindex(anch) > -0.005) & (cl21.reindex(anch) >= 0.10)
report(anch[m.fillna(False).values], "CPI with SPY at its high AND crude 21d >= +10%")

# --- calendar conditionings ---------------------------------------------------
report(anch[anch.month == 8], "August CPI")
report(anch[anch.year % 4 == 2], "midterm-year CPI")
report(anch[(anch.month == 8) & (anch.year % 4 == 2)], "August CPI in midterm years")

# --- vol starting low ---------------------------------------------------------
m = vix_dh.reindex(anch) < -0.45
report(anch[m.fillna(False).values], "CPI with VIX more than 45% below its 52w high",
       subjects=("^VIX", "SPY"))
