"""Byproduct of C1's gate attribution, measured before it is reported.

Killing C1 turned up a cell I did not pitch: the SKEW leg contributes
nothing, but its two companions -- ^VIX LEVEL in its bottom decile AND SPY
within 0.5% of its 52w high -- gave +0.667% at h=10 over 88 episodes at
t=4.57 against an unconditional +0.382%. Both legs are LIVE today (VIX lvl
pctile 5.6, SPY 0.00% off its high).

This cell was found by MY search, so it owes the full battery plus a
multiplicity discount, not a victory lap. Question: is it a real, still-live
"calm tape at a high" drift, or another pre-2018 fossil / a restatement of
the registry-dead stretched-high family?
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pitch_lab import (battery, close_panel, declusters, horizon_scan,  # noqa: E402
                       show, sign_test, summarize, vehicle_ret,
                       bootstrap_p_le0, cluster_note)

px = close_panel(["SPY", "^VIX"]).dropna()
vx_p = px["^VIX"].rolling(252).rank(pct=True) * 100.0
sp_hi = px["SPY"] / px["SPY"].rolling(252).max() - 1.0
print(f"panel {px.index[0].date()} .. {px.index[-1].date()}  n={len(px)}")
print(f"today: VIX lvl pctile {vx_p.iloc[-1]:.1f} (need <=10), "
      f"SPY {100*sp_hi.iloc[-1]:+.2f}% off 52wh (need >=-0.5%)  -> LIVE")

CELL = (vx_p <= 10) & (sp_hi >= -0.005)
VARIANTS = {
    "VIX<=5  & hi>=-0.5%": (vx_p <= 5) & (sp_hi >= -0.005),
    "VIX<=10 & hi>=-0.5% (cell)": CELL,
    "VIX<=15 & hi>=-0.5%": (vx_p <= 15) & (sp_hi >= -0.005),
    "VIX<=20 & hi>=-0.5%": (vx_p <= 20) & (sp_hi >= -0.005),
    "VIX<=10 & hi>=0.0% (at the high)": (vx_p <= 10) & (sp_hi >= 0.0),
    "VIX<=10 & hi>=-1%": (vx_p <= 10) & (sp_hi >= -0.01),
    "VIX<=10 & hi>=-2%": (vx_p <= 10) & (sp_hi >= -0.02),
    "GATE-OFF: VIX<=10 alone": (vx_p <= 10),
    "GATE-OFF: hi>=-0.5% alone": (sp_hi >= -0.005),
}

battery(px, CELL, [("SPY", 1.0)], 10,
        "BYPRODUCT: VIX level bottom-decile + SPY within 0.5% of 52w high",
        cost_bps=2.0, variants=VARIANTS, min_gap=21,
        event_kinds=("cpi", "fomc_decision"))

print("\n" + "=" * 78)
print("HORIZON SCAN (the horizon must come from here, not be assumed)")
print("=" * 78)
epi = declusters(px.index[CELL.values], 21, px.index)
show(horizon_scan(px, epi, [("SPY", 1.0)], hs=(1, 2, 3, 5, 7, 10, 15, 21),
                  lag=1, min_gap=21), "episode-level, entry lag=1")

print("\n" + "=" * 78)
print("ERA — the question that killed C1. Is this cell still alive post-2018?")
print("=" * 78)
for h in (5, 10, 21):
    ret = vehicle_ret(px, [("SPY", 1.0)], h, 1)
    e = pd.DatetimeIndex(epi).intersection(ret.dropna().index)
    base = ret.dropna()
    rows = []
    for lbl, lo, hi in [("2001-2009", 2001, 2010), ("2010-2017", 2010, 2018),
                        ("2018-2025", 2018, 2026)]:
        m = np.array([lo <= x.year < hi for x in e])
        r = summarize(ret.loc[e[m]].values, f"h={h} {lbl}")
        bm = np.array([lo <= x.year < hi for x in base.index])
        r["base_pct"] = round(100 * base[bm].mean(), 3)
        r["edge_pct"] = round(r.get("mean_pct", np.nan) - r["base_pct"], 3) \
            if r.get("n") else np.nan
        rows.append(r)
    show(rows, f"h={h} by era, vs that era's own unconditional drift")

print("\n" + "=" * 78)
print("MIDTERM cross (2026 is year %% 4 == 2)")
print("=" * 78)
for h in (5, 10):
    ret = vehicle_ret(px, [("SPY", 1.0)], h, 1)
    e = pd.DatetimeIndex(epi).intersection(ret.dropna().index)
    mid = np.array([x.year % 4 == 2 for x in e])
    base = ret.dropna()
    bmid = np.array([x.year % 4 == 2 for x in base.index])
    show([summarize(ret.loc[e[mid]].values, f"h={h} midterm episodes"),
          summarize(base[bmid].values, f"h={h} midterm ALL days"),
          summarize(ret.loc[e[~mid]].values, f"h={h} non-midterm episodes"),
          summarize(base[~bmid].values, f"h={h} non-midterm ALL days")],
         f"h={h} cycle-year cross")

print("\n" + "=" * 78)
print("MULTIPLICITY — this cell came out of a search over the C1 grid")
print("=" * 78)
print("  the a1b gate-attribution table scored 6 cells x 2 horizons = 12,")
print("  the a1b threshold grid scored 7 x 4 x 2 = 56, and this script's")
print("  variant table adds 9 x 1. Total cells my search touched ~ 77.")
print("  Bonferroni at alpha=0.05 over 77 cells needs p < 0.00065,")
print("  i.e. |t| > 3.42 on an independent sample. Quote the episode t.")

print("\n" + "=" * 78)
print("CONCENTRATION + the strongest argument against")
print("=" * 78)
for h in (5, 10):
    ret = vehicle_ret(px, [("SPY", 1.0)], h, 1)
    e = pd.DatetimeIndex(epi).intersection(ret.dropna().index)
    v = ret.loc[e].values
    print(f"  h={h}: {cluster_note(e, v, k=5)}")
    yrs = pd.Series(v).groupby(pd.DatetimeIndex(e).year.values).agg(['size', 'mean'])
    yrs['mean'] = (100 * yrs['mean']).round(2)
    print(f"    episodes by year: {dict(zip(yrs.index, zip(yrs['size'], yrs['mean'])))}")
