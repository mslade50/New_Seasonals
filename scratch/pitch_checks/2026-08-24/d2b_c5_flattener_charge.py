"""C5 round 2 -- the ONE cell in C5 that came back positive is the FLATTENER
(long IEF / short 0.52 TLT): h=10 +0.221%, t 2.61, 70% hit, sign p 0.003.
That is the only number in the whole candidate that could survive, so it gets
priced rather than celebrated.

Four tests, in the order that can kill it fastest:
  (1) GATE OFF, the flattener-specific version.  The LEVEL mask is degenerate
      in regime (round 1: the falling-yield cell is N=0), so the plain state
      underneath is "we are in a rising-yield regime".  Does the flattener pay
      the same on ALL rising-regime days?  If yes the trigger adds nothing
  (2) the MULTIPLE-COMPARISON charge.  The candidate pre-declared 3 vehicles
      (TLT, IEF, the spread) x 2 signs x 5 horizons.  Rotation permutation of
      the trigger mask, max |t| over the grid, P(max |t| >= 2.61)
  (3) definition neighbours on the flattener: proximity rung and lookback,
      NEAR neighbours included
  (4) era / midterm / concentration / cost on the flattener
Plus the near-miss arithmetic: what would have to be true for C5's live state
to sit inside its own trigger's support.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

pd.set_option("display.width", 240)

raw = close_panel(["^TNX", "TLT", "IEF"])
px = raw.dropna(how="any")
idx = px.index
tnx = px["^TNX"]
hi252 = rolling_on_valid(tnx, lambda x: x.rolling(252).max())
off_hi = tnx / hi252 - 1.0
LEVEL = off_hi >= -0.0025
sec = tnx - tnx.shift(252)
d = px[["TLT", "IEF"]].pct_change().dropna()
BETA = float(np.polyfit(d["IEF"].values, d["TLT"].values, 1)[0])
FLAT = [("IEF", 1.0), ("TLT", -1.0 / BETA)]
print("panel %s .. %s N=%d ; beta TLT~IEF %.3f ; flattener legs %s"
      % (idx[0].date(), idx[-1].date(), len(idx), BETA, FLAT))


def epi_stats(mask, legs, h, min_gap=10):
    ret = vehicle_ret(px, legs, h, 1)
    valid = ret.notna()
    sig = idx[np.asarray(mask.reindex(idx, fill_value=False).values, bool) & valid.values]
    if len(sig) == 0:
        return None, None, None
    epi = declusters(sig, max(h, min_gap), idx)
    return ret, epi, ret.loc[epi].values


# =============================================================== 1. GATE OFF
print("\n" + "=" * 110)
print("1. GATE OFF for the flattener.  The LEVEL mask cannot exist outside a")
print("   rising-yield regime (round 1: falling-regime cell N=0 at every h).")
print("   So: does the flattener pay the same on ALL rising-regime days?")
print("=" * 110)
RISING = sec > 0
rows = []
for h in (1, 3, 5, 10):
    ret = vehicle_ret(px, FLAT, h, 1)
    valid = ret.notna()
    _, epi, v = epi_stats(LEVEL, FLAT, h)
    r_trig = summarize(v, f"h={h} TRIGGER (TNX at 52wh), episodes")
    m_rise = RISING.reindex(idx).fillna(False).values & valid.values
    r_rise = summarize(ret[m_rise].values, f"h={h} ALL rising-regime days")
    m_all = valid.values
    r_all = summarize(ret[m_all].values, f"h={h} ALL days")
    r_trig["edge_vs_rising"] = round(r_trig["mean_pct"] - r_rise["mean_pct"], 3)
    rows += [r_trig, r_rise, r_all]
show(rows, "1a. flattener: trigger vs the rising-yield regime it lives inside")

# also decluster the RISING control so the comparison is episode-vs-episode
print("\n  episode-matched: rising-regime days declustered to the same 10td spacing")
for h in (1, 3, 5, 10):
    ret = vehicle_ret(px, FLAT, h, 1)
    valid = ret.notna()
    rise_days = idx[RISING.reindex(idx).fillna(False).values & valid.values]
    ep_r = declusters(rise_days, max(h, 10), idx)
    _, epi, v = epi_stats(LEVEL, FLAT, h)
    a, b = v, ret.loc[ep_r].values
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    print("   h=%2d trigger %+.3f%% (N=%d) vs rising-regime episodes %+.3f%% (N=%d)"
          "  diff %+.3f pp  welch t %+.2f"
          % (h, 100 * a.mean(), len(a), 100 * b.mean(), len(b),
             100 * (a.mean() - b.mean()), (a.mean() - b.mean()) / se))

# ================================================ 2. MULTIPLE-COMPARISON CHARGE
print("\n" + "=" * 110)
print("2. ROTATION PERMUTATION over the pre-declared grid")
print("   3 vehicles (TLT, IEF, flattener) x 2 signs x 5 horizons.")
print("   Signs mirror, so 15 unique |t|; the charge is on max |t|.")
print("=" * 110)
GRID_LEGS = {"TLT": [("TLT", 1.0)], "IEF": [("IEF", 1.0)], "FLAT": FLAT}
HS = (1, 2, 3, 5, 10)
rets = {(vk, h): vehicle_ret(px, lg, h, 1) for vk, lg in GRID_LEGS.items() for h in HS}


def grid_max_t(mask_vals):
    best, where = 0.0, None
    for (vk, h), ret in rets.items():
        valid = ret.notna().values
        sig = idx[mask_vals & valid]
        if len(sig) < 8:
            continue
        epi = declusters(sig, max(h, 10), idx)
        v = ret.loc[epi].values
        v = v[~np.isnan(v)]
        if len(v) < 8 or v.std(ddof=1) == 0:
            continue
        t = abs(v.mean() / (v.std(ddof=1) / np.sqrt(len(v))))
        if t > best:
            best, where = t, (vk, h, 100 * v.mean(), len(v))
    return best, where


obs_t, obs_where = grid_max_t(LEVEL.reindex(idx).fillna(False).values)
print("  OBSERVED grid max |t| = %.2f  at vehicle=%s h=%d mean %+.3f%% N=%d"
      % (obs_t, obs_where[0], obs_where[1], obs_where[2], obs_where[3]))
base = LEVEL.reindex(idx).fillna(False).values
rng = np.random.default_rng(42)
NPERM = 400
shifts = rng.integers(126, len(idx) - 126, size=NPERM)
maxes = []
for s in shifts:
    maxes.append(grid_max_t(np.roll(base, int(s)))[0])
maxes = np.array(maxes)
print("  rotation null over %d circular shifts: median max|t| %.2f, 95th %.2f"
      % (NPERM, np.median(maxes), np.percentile(maxes, 95)))
print("  P(rotated grid max |t| >= %.2f) = %.3f" % (obs_t, float((maxes >= obs_t).mean())))

# ================================================== 3. DEFINITION NEIGHBOURS
print("\n" + "=" * 110)
print("3. DEFINITION NEIGHBOURS on the flattener, h=10 (its best cell)")
print("=" * 110)
rows = []
for prox in (0.0000, 0.0010, 0.0025, 0.0050, 0.0100, 0.0200):
    m = off_hi >= -prox
    _, epi, v = epi_stats(m, FLAT, 10)
    r = summarize(v, "prox<=%.2f%%" % (100 * prox))
    r["n_days"] = int((m & vehicle_ret(px, FLAT, 10, 1).notna()).sum())
    w = int((v > 0).sum()); r["signp"] = round(sign_test(w, len(v)), 4)
    rows.append(r)
show(rows, "3a. proximity rung, flattener h=10")
rows = []
for lb in (63, 126, 189, 252, 378, 504):
    hl = rolling_on_valid(tnx, lambda x, L=lb: x.rolling(L).max())
    m = (tnx / hl - 1.0) >= -0.0025
    _, epi, v = epi_stats(m, FLAT, 10)
    r = summarize(v, "lookback %d" % lb)
    r["live_fires"] = bool(m.iloc[-1])
    rows.append(r)
show(rows, "3b. lookback, flattener h=10 (NEAR neighbours 189/378 included)")
rows = []
for h in (1, 2, 3, 4, 5, 7, 10):
    _, epi, v = epi_stats(LEVEL, FLAT, h)
    r = summarize(v, "h=%d" % h)
    w = int((v > 0).sum()); r["signp"] = round(sign_test(w, len(v)), 4)
    ret = vehicle_ret(px, FLAT, h, 1)
    r["ctrl_b"] = round(100 * ret.dropna().mean(), 3)
    r["bps_x_cost"] = round(abs(100 * 100 * v.mean()) / 6.0, 2)
    rows.append(r)
show(rows, "3c. horizon ladder, flattener (cost = 2 legs x 3 bps = 6 bps, need >=5x)")

# ============================================ 4. era / midterm / concentration
print("\n" + "=" * 110)
print("4. ERA / MIDTERM / CONCENTRATION, flattener h=10")
print("=" * 110)
_, epi, v = epi_stats(LEVEL, FLAT, 10)
show(era_split(epi, v), "4a. era")
mid = (pd.DatetimeIndex(epi).year % 4 == 2)
show([summarize(v[mid], "MIDTERM (live)"), summarize(v[~mid], "non-midterm")], "4b. cycle")
print("  concentration:", cluster_note(epi, v))
print("  drop-best-2 mean: %+.3f%%" % (100 * np.sort(v)[:-2].mean()))
print("  bootstrap P(mean<=0) = %.3f" % bootstrap_p_le0(v))

# ================================================== 5. the near-miss arithmetic
print("\n" + "=" * 110)
print("5. NEAR-MISS ARITHMETIC -- what would put today inside the trigger's support?")
print("=" * 110)
chg21 = tnx - tnx.shift(21)
rank21 = pct_rank(tnx, 21, 252)
tr = chg21[LEVEL].dropna()
print("  trigger-day 21d yield change: 10th pct %+.3f, median %+.3f, mean %+.3f"
      % (tr.quantile(0.10), tr.median(), tr.mean()))
print("  today %+.3f pt -> %.0fth percentile of trigger days."
      % (chg21.iloc[-1], 100 * (tr <= chg21.iloc[-1]).mean()))
print("  to reach the trigger population's 10th percentile (%+.3f pt) from today's"
      " 21-session base, ^TNX would have to be at %.3f rather than %.3f "
      "(i.e. +%.0f bp more)."
      % (tr.quantile(0.10), tnx.iloc[-22] + tr.quantile(0.10), tnx.iloc[-1],
         100 * (tnx.iloc[-22] + tr.quantile(0.10) - tnx.iloc[-1])))
print("  BUT: that is the state the registry already killed (rank/thrust forms).")
