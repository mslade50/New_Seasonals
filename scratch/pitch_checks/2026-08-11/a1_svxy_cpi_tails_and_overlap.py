"""C1 kill attempt 4 + the mandatory disclosures.

  G. concentration: top-2 episodes, year histogram of the cell, worst window
     and what it costs at a tradeable size, leave-one-year-out floor.
  H. is TODAY in the good bucket?  The cell is conditioned on VIX regime
     (63d rank of the VIX LEVEL) and on SVXY's distance from its 52w high,
     and today's readings are placed inside those distributions.
  I. cost: SVXY spread + the -0.5x ETP's own carry over a 3 td hold.
  J. overlap with the live event sleeve's V4 (long SVXY, opex MOC -> +3 td)
     and V2 (long SVXY, Nov -> year end): historical calendar overlap of the
     CPI-1 window with a live sleeve position, and specifically for 2026-08.

Run: python scratch/pitch_checks/2026-08-11/a1_svxy_cpi_tails_and_overlap.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a1_lab import SVXY_LEV_BREAK, anchor_dates, loyo, tdom_of  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    close_panel, cluster_note, declusters, fwd_lag, load_events, show,
    sign_test, summarize,
)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)

OFFSET, H = 2, 3
px = close_panel(["SVXY", "SPY", "^VIX"])
all_dates = px.index
pos = pd.Series(np.arange(len(all_dates)), index=all_dates)
TDOM = tdom_of(all_dates)
ev_all = load_events(["cpi", "opex"])
anch = declusters(anchor_dates(ev_all, "cpi", OFFSET, all_dates), 5, all_dates)
svxy = px["SVXY"].dropna()
anch = anch[anch.isin(svxy.index)]
f3 = fwd_lag(svxy, H, lag=1)
v3 = f3.reindex(anch).dropna()
post = v3[v3.index >= SVXY_LEV_BREAK]

# ---------------------------------------------------------------------------
print("=" * 78)
print("G. concentration and tails")
print("=" * 78)
print("full sample:", cluster_note(v3.index, v3.values, k=2))
print("\n-0.5x era :", cluster_note(post.index, post.values, k=2))

print("\nG1. year histogram of the h=3 cell (sum of window returns, pp)")
yr = pd.DataFrame({"sum_pp": 100 * v3.groupby(v3.index.year).sum(),
                   "n": v3.groupby(v3.index.year).size(),
                   "mean_pp": 100 * v3.groupby(v3.index.year).mean(),
                   "wins": v3.groupby(v3.index.year).apply(lambda s: int((s > 0).sum()))})
print(yr.round(2).to_string())

print("\nG2. worst windows (full sample), and what a 10%-NAV sleeve-sized "
      "position would have lost")
w = v3.sort_values().head(8)
for d, r in w.items():
    era = "-1x" if d < SVXY_LEV_BREAK else "-0.5x"
    print(f"  entry {d.date()} ({era})  window {100*r:+7.2f}%   "
          f"at 10% NAV = {100*r*0.10:+6.2f}% NAV   at 25% NAV = "
          f"{100*r*0.25:+6.2f}% NAV")

print("\nG3. leave-one-year-out, -0.5x era only")
lo = loyo(post.index, post.values)
print(lo.round(3).to_string(index=False))
if len(lo):
    i = lo["mean_pct"].idxmin()
    print(f"  LOYO FLOOR: dropping {int(lo.loc[i,'drop_year'])} leaves "
          f"{lo.loc[i,'mean_pct']:+.3f}%  (that year alone was "
          f"{lo.loc[i,'in_year_pct']:+.3f}% over {int(lo.loc[i,'in_year_n'])} events)")

# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("H. is TODAY in the good bucket?  VIX regime and SVXY distance-from-high")
print("=" * 78)
vix = px["^VIX"].dropna()
vix_rank63 = vix.rolling(63).rank(pct=True) * 100          # rank of the LEVEL
dist_high = svxy / svxy.rolling(252).max() - 1.0
today = all_dates[-1]
print(f"today ({today.date()}): VIX {vix.loc[today]:.2f}, 63d level rank "
      f"{vix_rank63.loc[today]:.1f}; SVXY {100*dist_high.loc[today]:+.2f}% "
      f"from its 252d high")

rows = []
for lbl, m in (("VIX 63d level rank <= 50 (calm, LIKE TODAY)", vix_rank63 <= 50),
               ("VIX 63d level rank > 50", vix_rank63 > 50),
               ("SVXY within 2% of 252d high (LIKE TODAY)", dist_high >= -0.02),
               ("SVXY more than 2% below 252d high", dist_high < -0.02),
               ("BOTH today-like (calm VIX + SVXY at high)",
                (vix_rank63 <= 50) & (dist_high >= -0.02))):
    sel = anch[anch.isin(m[m].index)]
    for era_lbl, sub in (("full", sel), ("-0.5x era", sel[sel >= SVXY_LEV_BREAK])):
        v = f3.reindex(sub).dropna()
        if len(v) < 4:
            rows.append({"label": f"{lbl} | {era_lbl}", "n": len(v)})
            continue
        st = summarize(v.values, f"{lbl} | {era_lbl}")
        st["signp"] = sign_test(int((v.values > 0).sum()), len(v))
        rows.append(st)
show(rows, "H1. the cell conditioned on today's regime")

# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("I. cost")
print("=" * 78)
adv_note = """SVXY quotes ~1-2 c wide on a ~$50 tape = ~2-4 bps half-spread,
so a round trip is ~5-8 bps plus commission.  The -0.5x ETP also carries its
own daily drift, which for a LONG SVXY position is the short-vol carry - it is
a tailwind, not a cost, in contango, and a fast headwind in backwardation."""
print(adv_note)
r = svxy.pct_change().dropna()
for lbl, lo_, hi_ in (("-1x era", pd.Timestamp("2000-01-01"), SVXY_LEV_BREAK),
                      ("-0.5x era", SVXY_LEV_BREAK, pd.Timestamp("2030-01-01")),
                      ("2021+", pd.Timestamp("2021-01-01"), pd.Timestamp("2030-01-01"))):
    s = r[(r.index >= lo_) & (r.index < hi_)]
    print(f"  {lbl:<10} SVXY unconditional daily {1e4*s.mean():+6.1f} bps "
          f"-> {1e4*s.mean()*3:+6.1f} bps over a 3 td hold "
          f"(this is the DRIFT the cell must beat, not a cost)")
print("\n  -> the honest hurdle: round-trip cost ~6 bps PLUS the unconditional "
      "3 td drift.")

# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("J. overlap with the live event sleeve (V4 opex short-vol, V2 Nov-Dec)")
print("=" * 78)
opex = pd.DatetimeIndex(sorted(ev_all.loc[ev_all.event == "opex", "date"].unique()))
opex_sess = pd.DatetimeIndex(
    [all_dates[all_dates.searchsorted(d)] for d in opex
     if all_dates.searchsorted(d) < len(all_dates)])
# V4: entry MOC at the opex session, exit MOC +3 sessions, skip September
v4_hold = set()
for d in opex_sess:
    p = pos.get(d)
    if p is None or d.month == 9:
        continue
    for k in range(1, 4):
        if p + k < len(all_dates):
            v4_hold.add(all_dates[p + k])
# V2: long SVXY first Nov session -> year end, non-midterm years
v2_hold = set()
for y in sorted(set(all_dates.year)):
    if y % 4 == 2:
        continue
    nov = all_dates[(all_dates.year == y) & (all_dates.month == 11)]
    dec = all_dates[(all_dates.year == y) & (all_dates.month == 12)]
    if len(nov) and len(dec):
        v2_hold |= set(all_dates[(all_dates >= nov[0]) & (all_dates <= dec[-1])])

n_ov4 = n_ov2 = 0
for d in anch:
    p = pos.get(d)
    win = {all_dates[p + k] for k in range(1, 2 + H) if p + k < len(all_dates)}
    if win & v4_hold:
        n_ov4 += 1
    if win & v2_hold:
        n_ov2 += 1
print(f"CPI-1 entry + {H} td hold, N={len(anch)} episodes:")
print(f"  windows touching a V4 (ex-Sep opex +3) holding day : {n_ov4} "
      f"({100*n_ov4/len(anch):.1f}%)")
print(f"  windows touching a V2 (non-midterm Nov-Dec) holding day: {n_ov2} "
      f"({100*n_ov2/len(anch):.1f}%)")
print(f"  either: {sum(1 for d in anch if ({all_dates[pos[d]+k] for k in range(1, 2+H) if pos[d]+k < len(all_dates)}) & (v4_hold | v2_hold))} "
      f"({100*sum(1 for d in anch if ({all_dates[pos[d]+k] for k in range(1, 2+H) if pos[d]+k < len(all_dates)}) & (v4_hold | v2_hold))/len(anch):.1f}%)")
print("\n2026-08 specifically: entry MOC 2026-08-11, exit MOC +3 td = "
      "2026-08-14 (sessions 08-12, 08-13, 08-14).")
print("  opex 2026-08-21 -> V4 holds 08-24..08-26. 2026 is a MIDTERM year so "
      "V2 does not run. NO overlap this month.")
