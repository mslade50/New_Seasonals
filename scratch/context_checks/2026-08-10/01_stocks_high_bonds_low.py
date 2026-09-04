"""Stocks pinned at a 52w high while long bonds sit on a 52w low.

Today: SPY closed 0.03% off its 252d max, TLT 0.17% above its 252d min.
The P9 family keys on joint MOVES and stayed silent because today's moves were
tiny. The joint LEVEL is the thing. Anchor = the session the state printed,
h=1 = the next session, lag=0 close-to-close.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    close_panel, fwd_ret, declusters, local_control, summarize, era_split,
    sign_test, cluster_note,
)

TIX = ["SPY", "TLT", "IEF", "^GSPC", "^TNX", "QQQ", "IWM", "^NYA"]
px = close_panel(TIX)
px = px[px.index >= "1999-01-01"]
print("panel", px.index.min().date(), "->", px.index.max().date())
for t in TIX:
    s = px[t].dropna()
    print(f"  {t:8s} {len(s):5d} bars from {s.index.min().date()}")


def dist_high(s, win=252):
    return s / s.rolling(win).max() - 1.0


def dist_low(s, win=252):
    return s / s.rolling(win).min() - 1.0


spy_dh = dist_high(px["SPY"])
tlt_dl = dist_low(px["TLT"])
ief_dl = dist_low(px["IEF"])

print("\ntoday: SPY dist-high %.3f%%  TLT dist-low %.3f%%  IEF dist-low %.3f%%"
      % (100 * spy_dh.iloc[-1], 100 * tlt_dl.iloc[-1], 100 * ief_dl.iloc[-1]))

all_dates = px.index


def run(mask, label, subjects=("SPY", "TLT", "^TNX"), gap=10):
    trig = all_dates[mask.reindex(all_dates).fillna(False).values]
    trig = trig[trig <= all_dates[-2]]          # need at least one forward bar
    if len(trig) == 0:
        print(f"\n=== {label}: NO OCCURRENCES")
        return
    dc = declusters(trig, gap, all_dates)
    print(f"\n=== {label}")
    print(f"  raw sessions {len(trig)}, declustered episodes {len(dc)} "
          f"(min gap {gap} td)")
    print(f"  years: {sorted(set(pd.DatetimeIndex(dc).year))}")
    ctrl_idx = local_control(all_dates, dc, win=126)
    for sub in subjects:
        s = px[sub].dropna()
        for h in (1, 5, 21):
            f = fwd_ret(s, h)
            v = f.reindex(dc).dropna()
            if len(v) < 3:
                continue
            r = summarize(v.values, f"{sub} h{h}")
            base = summarize(f.dropna().values, "all")
            loc = summarize(f.reindex(ctrl_idx).dropna().values, "local")
            up = int((v.values > 0).sum())
            p = sign_test(up, len(v))
            print(f"  {sub:6s} h{h:<3d} n={r['n']:3d} mean {r['mean_pct']:+.2f}% "
                  f"med {r['median_pct']:+.2f}% hit {r['hit']:.0f}% "
                  f"t={r['t']:+.2f} | {up}-{len(v) - up} sign p {p:.4f} | "
                  f"all {base['mean_pct']:+.2f}% local {loc['mean_pct']:+.2f}% "
                  f"(n={loc['n']})")
            if h == 1:
                print(f"         {cluster_note(dc[:len(v)], v.values)}")
                for e in era_split(v.index, v.values):
                    print(f"         era n={e['n']:3d} mean {e['mean_pct']:+.2f}% "
                          f"hit {e['hit']:.0f}% t={e['t']:+.2f}")


# --- the live state, three widths -------------------------------------------
run((spy_dh > -0.005) & (tlt_dl < 0.015), "SPY within 0.5% of 52w high AND TLT within 1.5% of 52w low")
run((spy_dh > -0.010) & (tlt_dl < 0.020), "SPY within 1.0% of high AND TLT within 2.0% of low")
run((spy_dh > -0.020) & (tlt_dl < 0.030), "SPY within 2.0% of high AND TLT within 3.0% of low")

# --- the halves on their own, so the joint claim has a reference -------------
run(spy_dh > -0.005, "SPY within 0.5% of its 52w high (bonds ignored)", subjects=("SPY",))
run(tlt_dl < 0.015, "TLT within 1.5% of its 52w low (stocks ignored)", subjects=("TLT", "SPY"))

# --- IEF version, for the 'is it a TLT duration quirk' question --------------
run((spy_dh > -0.005) & (ief_dl < 0.015), "SPY within 0.5% of high AND IEF within 1.5% of low",
    subjects=("SPY", "IEF"))

# --- secondary: NYSE composite prints a 52w high while SPY/QQQ/IWM all fall --
nya_hi = dist_high(px["^NYA"]) > -0.0005
down3 = (px["SPY"].pct_change() < 0) & (px["QQQ"].pct_change() < 0) & (px["IWM"].pct_change() < 0)
run(nya_hi & down3, "^NYA at a 52w high while SPY, QQQ and IWM all close lower",
    subjects=("SPY", "IWM"), gap=5)
