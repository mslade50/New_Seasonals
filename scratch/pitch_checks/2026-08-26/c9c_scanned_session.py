"""C9 round 2b: the ME-3 -> ME-2 session is a SCANNED session.

c9b showed 93% (IWM) / 100% (SPY) of the h=3 Aug-end return sits in the single
session right after entry.  The 2026-08-24 registry entry made the same finding
on SPY ME-5 and named the multiplicity problem: "3 of 16 scanned sessions
clearing |t|>=2 against 0.8 expected".  Here that count is redone on the August
subset, which is what C9 is actually built on.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

px = load_prices(["IWM", "SPY"])
ser = {t: px[t]["Close"].dropna() for t in ("IWM", "SPY")}


def ltds(idx):
    per = pd.Series(idx.to_period("M"), index=range(len(idx)))
    return [int(g.index.max()) for _, g in per.groupby(per.values)][:-1]


for t in ("IWM", "SPY"):
    s = ser[t]; idx = s.index; v = s.values
    print(f"\n### {t}: every session offset ME-8..ME+7, AUGUST anchors only")
    rows = []
    for off in range(-8, 8):
        vals = []
        for p in ltds(idx):
            if idx[p].month != 8:
                continue
            a, b = p + off, p + off + 1
            if a < 1 or b >= len(idx):
                continue
            vals.append(v[b] / v[a] - 1.0)
        x = np.asarray(vals)
        tt = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
        rows.append({"session": f"ME{off:+d}->ME{off+1:+d}", "n": len(x),
                     "bp": round(100 * 100 * x.mean(), 2),
                     "hit": round(100 * (x > 0).mean(), 1), "t": round(tt, 2)})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    n_sig = int((df["t"].abs() >= 2).sum())
    print(f"  sessions scanned: {len(df)};  |t| >= 2: {n_sig} "
          f"(expected under the null ~{0.05*len(df):.1f})")
    # the pitched window's total vs its single best session
    h3 = []
    for p in ltds(idx):
        if idx[p].month != 8:
            continue
        e, x = p - 3, p
        if e < 1 or x >= len(idx):
            continue
        h3.append(v[x] / v[e] - 1.0)
    h3 = np.asarray(h3)
    best = df.iloc[int(df["bp"].abs().idxmax())]
    first = df[df["session"] == "ME-3->ME-2"].iloc[0]
    print(f"  ME-3 entry, h=3 total: {100*h3.mean():+.3f}% over N={len(h3)}")
    print(f"  the first session of that hold (ME-3->ME-2) alone: "
          f"{first['bp']:+.2f} bp = {100*first['bp']/(100*100*h3.mean()):.0f}% "
          f"of the whole window, at t={first['t']:+.2f}, hit {first['hit']}%")
    # placebo: same session offset in every OTHER month
    other = []
    for p in ltds(idx):
        if idx[p].month == 8:
            continue
        a, b = p - 3, p - 2
        if a < 1 or b >= len(idx):
            continue
        other.append(v[b] / v[a] - 1.0)
    other = np.asarray(other)
    print(f"  the SAME session in the other 11 months: "
          f"{100*100*other.mean():+.2f} bp over N={len(other)}, hit "
          f"{100*(other>0).mean():.1f}%")
    # bootstrap: how often does a random 16-session scan produce a |t|>=2.5?
    rng = np.random.default_rng(5)
    daily = (s.shift(-1) / s - 1.0).dropna().values
    hits = 0
    for _ in range(4000):
        best_t = 0.0
        for _ in range(16):
            samp = rng.choice(daily, size=len(h3), replace=False)
            tt = abs(samp.mean() / (samp.std(ddof=1) / np.sqrt(len(samp))))
            best_t = max(best_t, tt)
        hits += best_t >= abs(float(first["t"]))
    print(f"  random 16-session scan on {len(h3)} draws: "
          f"P(max |t| >= {abs(float(first['t'])):.2f}) = {hits/4000:.3f}")
