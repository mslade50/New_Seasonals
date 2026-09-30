"""d11c: does the MIDTERM conditioner filter, or is it the election-year cell (registry
2026-09-21 kC_c7, closed)? Sept QE-2 entry, h=5/10, split midterm / presidential / odd,
on ^VIX %, SHORT SPY and the SPY-hedged long-vol residual (synthetic -0.5x SVXY)."""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

px = close_panel(["^VIX", "SPY", "SVXY"])
px = px[px["SPY"].notna()]
idx = px.index
BREAK = pd.Timestamp("2018-02-28")
r_sv = px["SVXY"].pct_change(fill_method=None)
S = (1 + r_sv.where(idx >= BREAK, 0.5 * r_sv).fillna(0)).cumprod()
S[idx <= px["SVXY"].first_valid_index()] = np.nan
MID = {2002, 2006, 2010, 2014, 2018, 2022}
PRES = {2000, 2004, 2008, 2012, 2016, 2020, 2024}

rows = []
for y in range(2000, 2026):
    md = idx[(idx.year == y) & (idx.month == 9)]
    ent = idx.get_loc(md[-1]) - 2
    for h in (5, 10):
        ex = ent + h
        spy = px["SPY"].values[ex] / px["SPY"].values[ent] - 1
        s = S.values[ex] / S.values[ent] - 1 if not np.isnan(S.values[ent]) else np.nan
        b = 1.48 if idx[ent] > BREAK else 1.98
        rows.append({"year": y, "h": h, "cyc": "mid" if y in MID else ("pres" if y in PRES else "odd"),
                     "vix": px["^VIX"].values[ex] / px["^VIX"].values[ent] - 1,
                     "short_spy": -spy, "long_vol_res": -(s - b * spy)})
D = pd.DataFrame(rows)
out = []
for h in (5, 10):
    for c in ("mid", "pres", "odd"):
        for col in ("vix", "short_spy", "long_vol_res"):
            v = D[(D.h == h) & (D.cyc == c)][col].dropna().values
            if len(v) == 0:
                continue
            w = int((v > 0).sum())
            out.append({"h": h, "cycle": c, "object": col, "n": len(v), "mean_pct": round(100 * v.mean(), 2),
                        "median_pct": round(100 * np.median(v), 2), "rec": f"{w}-{len(v)-w}",
                        "sign_p": round(sign_test(w, len(v)), 3)})
show(out, "Sept QE-2 entry by cycle year")
