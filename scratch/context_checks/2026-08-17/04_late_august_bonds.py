"""The late-August bond seasonal.

E:seasonal_doy flagged TLT h5 at 18 up of 23, sign p 0.0053, mean +0.672%, with
^TNX h5 at 18 down of 25, sign p 0.0378, the same claim from the yield side. Live
tonight because TLT closed at a 252-day low.

Is it era-stable, is it concentrated, does it hold from the level TLT is at, and does
it survive being asked at a slightly different day-of-year (a robustness check the
doy cell cannot do for itself)?
"""
import sys
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from pitch_lab import close_panel, load_prices, summarize, sign_test, era_split  # noqa: E402

spec = importlib.util.spec_from_file_location("se", ROOT / "scripts" / "seasonal_edge.py")
se = importlib.util.module_from_spec(spec)
sys.modules["se"] = se
spec.loader.exec_module(se)

ASOF = pd.Timestamp("2026-08-17")
px = close_panel(["TLT", "IEF", "^TNX", "LQD", "SPY"]).dropna(how="all")
raw = load_prices(["TLT", "IEF", "^TNX"])

print("=== seasonal_window_returns, anchored at today's day-of-year ===")
for t in ["TLT", "IEF", "^TNX"]:
    df = raw[t]
    for h in [5, 10, 21]:
        r = se.seasonal_window_returns(df, ASOF, h, doy_tol=2)
        if not r or r.get("insufficient"):
            print(f"{t:6s} h{h:<3d} insufficient")
            continue
        k = r["n_up"]
        p = se.binom_p_greater(k, r["n"])
        print(f"{t:6s} h{h:<3d} n={r['n']:2d} mean={100*r['mean']:+.3f}% "
              f"med={100*r['median']:+.3f}% rec={r['n_up']}-{r['n_down']} "
              f"binom_p_up={p:.4f}")

print("\n=== midterm-only (cycle_phase_filter=2) ===")
for t in ["TLT", "IEF", "^TNX"]:
    for h in [5, 10, 21]:
        r = se.seasonal_window_returns(raw[t], ASOF, h, cycle_phase_filter=2, doy_tol=2)
        if not r or r.get("insufficient"):
            print(f"{t:6s} h{h:<3d} insufficient")
            continue
        print(f"{t:6s} h{h:<3d} n={r['n']:2d} mean={100*r['mean']:+.3f}% "
              f"rec={r['n_up']}-{r['n_down']} years={r['years']}")

# --- rebuild the cell by hand so the per-year detail is visible --------------
print("\n=== TLT h5 from the matching session each prior year ===")
tlt = px["TLT"].dropna()
doy = ASOF.dayofyear
rows = []
for yr in sorted(set(tlt.index.year)):
    if yr == ASOF.year:
        continue
    cand = tlt.index[(tlt.index.year == yr) & (abs(tlt.index.dayofyear - doy) <= 2)]
    if len(cand) == 0:
        continue
    d = cand[np.argmin(np.abs(cand.dayofyear - doy))]
    pos = tlt.index.get_loc(d)
    if pos + 5 >= len(tlt):
        continue
    r5 = tlt.iloc[pos + 5] / tlt.iloc[pos] - 1.0
    r21 = tlt.iloc[pos + 21] / tlt.iloc[pos] - 1.0 if pos + 21 < len(tlt) else np.nan
    # where was TLT sitting relative to its own year at that anchor?
    win = tlt.iloc[max(0, pos - 251):pos + 1]
    pct = 100.0 * (win <= tlt.iloc[pos]).mean()
    rows.append((yr, d, r5, r21, pct))

df = pd.DataFrame(rows, columns=["year", "date", "h5", "h21", "pctile_252"])
print(df.assign(h5=lambda x: (100 * x.h5).round(2),
                h21=lambda x: (100 * x.h21).round(2),
                pctile_252=lambda x: x.pctile_252.round(0)).to_string(index=False))

v = df["h5"].values
up = int((v > 0).sum())
print(f"\nTLT h5: n={len(v)} mean={100*v.mean():+.3f}% med={100*np.median(v):+.3f}% "
      f"rec={up}-{len(v)-up} sign_p_up={sign_test(up, len(v)):.4f}")
print("era split:")
for part in era_split(pd.DatetimeIndex(df["date"]), v):
    u = int(round(part["hit"] / 100 * part["n"]))
    print(f"  {part.get('label',''):12s} n={part['n']:2d} mean={part['mean_pct']:+.3f}% "
          f"rec={u}-{part['n']-u}")

srt = np.sort(v)
print(f"top 2 contributions: {100*srt[-2:].sum():+.2f}pp of {100*v.sum():+.2f}pp total")
print(f"drop best year: mean {100*srt[:-1].mean():+.3f}%, "
      f"rec {int((srt[:-1]>0).sum())}-{int((srt[:-1]<=0).sum())}")

print("\n=== does the low matter? split the anchors by TLT's own 252d percentile ===")
lo = df[df.pctile_252 <= 33]
hi = df[df.pctile_252 > 33]
for lbl, sub in [("anchor in bottom third of its year", lo), ("rest", hi)]:
    if len(sub) == 0:
        continue
    u = int((sub.h5 > 0).sum())
    print(f"  {lbl:36s} n={len(sub):2d} mean={100*sub.h5.mean():+.3f}% "
          f"rec={u}-{len(sub)-u} years={list(sub.year)}")

print("\n=== control: TLT h5 over all days, and over all August days ===")
from pitch_lab import fwd_ret  # noqa: E402
f5 = fwd_ret(px["TLT"], 5).dropna()
for lbl, s in [("all days", f5), ("all August days", f5[f5.index.month == 8])]:
    d = summarize(s.values)
    u = int(round(d["hit"] / 100 * d["n"]))
    print(f"  {lbl:18s} n={d['n']:5d} mean={d['mean_pct']:+.3f}% rec={u}-{d['n']-u}")

print("\n=== robustness: same cell asked at neighbouring days-of-year ===")
for off in [-6, -4, -2, 0, 2, 4, 6]:
    a = ASOF + pd.Timedelta(days=off)
    r = se.seasonal_window_returns(raw["TLT"], a, 5, doy_tol=2)
    if r and not r.get("insufficient"):
        print(f"  asof {a.date()} (doy{off:+d}): n={r['n']:2d} "
              f"mean={100*r['mean']:+.3f}% rec={r['n_up']}-{r['n_down']}")
