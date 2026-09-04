"""C11 round 1 -- the month turn on the washed-out rate-sensitive defensives.

Signal: ME-1 close (today) with XLP, XLU and XLRE all at bottom-quintile 21-day
return ranks. Entry lag=1 = MOC on ME-0 (the month's last session), exit h
sessions into the new month.

Everything the brief demands is here:
  (i)   placebo offset ladder ME-5 .. ME+5 -- a level shift is not an event
  (ii)  gate attribution: the bare sector month turn, with no washout gate
  (iii) midterm split (2026 is midterm; the registry has reproduced a midterm
        inversion six times)
  (iv)  the search charge: offsets x sectors x horizons is counted and a
        Sidak max-of-K is applied to the best cell.
XLRE only lists from 2015-10, so a two-sector (XLP+XLU) form back to 1999 is run
beside the three-sector one -- otherwise the three-way join is 11 years long.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pitch_lab import *  # noqa: E402,F403

ASOF = pd.Timestamp("2026-08-27")
DEF3 = ["XLP", "XLU", "XLRE"]
raw = load_prices(DEF3 + ["SPY"])
spy = raw["SPY"]["Close"].dropna()
cal = spy.index                       # NYSE calendar master

px = pd.DataFrame({t: raw[t]["Close"].reindex(cal) for t in DEF3})
px["SPY"] = spy
for t in DEF3:
    print("%-5s first bar %s  (%d sessions)" % (t, px[t].dropna().index[0].date(),
                                                px[t].notna().sum()))

# ranks computed on each ticker's OWN valid series (rolling_on_valid rule)
rk = {t: pct_rank(px[t], 21) for t in DEF3}
spy_dist = px["SPY"] / px["SPY"].rolling(252).max() - 1.0

# ---- month-position index ----------------------------------------------------
pos = pd.Series(range(len(cal)), index=cal)
me0 = {}
for (y, m), g in pd.Series(cal, index=cal).groupby([cal.year, cal.month]):
    me0[(y, m)] = pos[g.index[-1]]
me0_positions = sorted(me0.values())
# offset k -> the session k trading days BEFORE the month's last session
def me_offset_dates(k: int) -> pd.DatetimeIndex:
    out = [cal[p - k] for p in me0_positions if 0 <= p - k < len(cal)]
    return pd.DatetimeIndex(out)

ME1 = me_offset_dates(1)
print("\nME-1 sessions in the sample: %d  (last %s)" % (len(ME1), ME1[-1].date()))
print("today 2026-08-28 is ME-1: %s (Aug's last session is %s)"
      % (bool(pd.Timestamp("2026-08-28") in me_offset_dates(1)) if pd.Timestamp("2026-08-28") in cal
         else "n/a-not-in-cache", me0_positions and cal[me0[(2026, 8)]].date()))
print("LIVE ranks: " + "  ".join("%s %.1f" % (t, rk[t].loc[ASOF]) for t in DEF3)
      + "   SPY dist %+.2f%%" % (100 * spy_dist.loc[ASOF]))

wash3 = pd.Series(True, index=cal)
for t in DEF3:
    wash3 &= (rk[t] <= 20).fillna(False)
wash2 = ((rk["XLP"] <= 20) & (rk["XLU"] <= 20)).fillna(False)

def cell(dates, mask=None):
    d = pd.DatetimeIndex(dates)
    if mask is not None:
        d = d[mask.reindex(d, fill_value=False).values]
    return d

LEGS3 = [("XLP", 1 / 3), ("XLU", 1 / 3), ("XLRE", 1 / 3)]
LEGS2 = [("XLP", 0.5), ("XLU", 0.5)]

# ---- (ii) GATE ATTRIBUTION + (iii) MIDTERM SPLIT ------------------------------
print("\n" + "=" * 84)
print("(ii) gate attribution -- does the washout conditioner do ANY work?")
print("=" * 84)
for h in (3, 5, 10):
    rows = []
    for lbl, d in [("ME-1 bare (no washout gate)", cell(ME1)),
                   ("ME-1 x XLP+XLU+XLRE all r21<=20", cell(ME1, wash3)),
                   ("ME-1 x XLP+XLU r21<=20 (2sec)", cell(ME1, wash2)),
                   ("ALL days (baseline)", cal)]:
        for legname, legs in [("DEF3 basket", LEGS3), ("DEF2 basket", LEGS2),
                              ("SPY", [("SPY", 1.0)])]:
            r = vehicle_ret(px, legs, h, 1).reindex(d).dropna()
            rows.append(summarize(r.values, f"h={h} {lbl} | {legname}"))
    show(rows)

print("\n" + "=" * 84)
print("(iii) midterm split on the conditioned cell (year %% 4 == 2)")
print("=" * 84)
for h in (5, 10):
    for lbl, d, legs in [("ME-1 bare | DEF2", cell(ME1), LEGS2),
                         ("ME-1 x wash2 | DEF2", cell(ME1, wash2), LEGS2),
                         ("ME-1 x wash3 | DEF3", cell(ME1, wash3), LEGS3)]:
        r = vehicle_ret(px, legs, h, 1).reindex(d).dropna()
        mt = pd.DatetimeIndex(r.index).year % 4 == 2
        show([summarize(r.values[mt], f"h={h} {lbl} MIDTERM"),
              summarize(r.values[~mt], f"h={h} {lbl} non-midterm")])

# ---- (i) PLACEBO OFFSET LADDER ----------------------------------------------
print("\n" + "=" * 84)
print("(i) placebo offset ladder ME-5..ME+5, conditioned cell, h=5, DEF2 + DEF3")
print("=" * 84)
for legname, legs, gate in [("DEF2 x wash2", LEGS2, wash2), ("DEF3 x wash3", LEGS3, wash3)]:
    print("  --- %s ---" % legname)
    for k in range(5, -6, -1):
        d = cell(me_offset_dates(k), gate)
        r = vehicle_ret(px, legs, 5, 1).reindex(d).dropna()
        s = summarize(r.values, "")
        if s["n"]:
            print("    ME%+d  n=%3d  mean %+7.3f%%  med %+7.3f%%  hit %5.1f%%  t %+5.2f"
                  % (-k, s["n"], s["mean_pct"], s["median_pct"], s["hit"], s["t"]))

# ---- (iv) the grid charge ----------------------------------------------------
print("\n" + "=" * 84)
print("(iv) the search charge: offsets(11) x sector-baskets(2) x horizons(5)")
print("=" * 84)
best = None
K = 0
for k in range(5, -6, -1):
    for legname, legs, gate in [("DEF2", LEGS2, wash2), ("DEF3", LEGS3, wash3)]:
        for h in (1, 2, 3, 5, 10):
            K += 1
            d = cell(me_offset_dates(k), gate)
            r = vehicle_ret(px, legs, h, 1).reindex(d).dropna()
            s = summarize(r.values, f"ME{-k:+d} {legname} h={h}")
            if s["n"] >= 8 and not np.isnan(s.get("t", np.nan)):
                if best is None or s["t"] > best["t"]:
                    best = s
print("  cells searched K = %d" % K)
print("  best cell: %s  n=%d mean %+0.3f%% t %+0.2f hit %.1f%%"
      % (best["label"], best["n"], best["mean_pct"], best["t"], best["hit"]))
from scipy import stats  # noqa: E402
p1 = 1 - stats.t.cdf(best["t"], best["n"] - 1)
print("  raw one-sided p = %.4f  ->  Sidak max-of-%d p = %.4f"
      % (p1, K, 1 - (1 - p1) ** K))

# ---- round 1 battery on the headline form -----------------------------------
m3 = pd.Series(False, index=cal)
m3.loc[cell(ME1, wash3)] = True
m2 = pd.Series(False, index=cal)
m2.loc[cell(ME1, wash2)] = True
for h in (3, 5):
    battery(px, m2, LEGS2, h, f"C11 ME-1 x XLP+XLU washed out -> DEF2 basket, h={h}",
            cost_bps=6.0, min_gap=15,
            variants={"r21<=10 both": pd.Series(m2.values & ((rk["XLP"] <= 10) & (rk["XLU"] <= 10)).reindex(cal, fill_value=False).values, index=cal),
                      "r21<=30 both": pd.Series(pd.Series(False, index=cal).pipe(lambda s: s).values | np.isin(cal, cell(ME1, ((rk["XLP"] <= 30) & (rk["XLU"] <= 30)).fillna(False))), index=cal),
                      "bare ME-1 (no gate)": pd.Series(np.isin(cal, ME1), index=cal)})
    battery(px, m3, LEGS3, h, f"C11 ME-1 x XLP+XLU+XLRE washed out -> DEF3 basket, h={h}",
            cost_bps=6.0, min_gap=15)
