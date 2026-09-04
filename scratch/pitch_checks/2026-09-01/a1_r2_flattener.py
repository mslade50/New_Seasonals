"""A1 ROUND 2 -- adversarial re-examination of the parked duration-neutral
flattener (long IEF / short 0.523 TLT, MOC entry when ^TNX closes within 0.25%
of its trailing-252 high).

Parked source of construction: scratch/pitch_checks/2026-08-24/d2b_c5_flattener_charge.py
(same panel, same beta fit, same LEVEL mask, same lag=1 vehicle_ret).

Four jobs, in kill order:
  0. DATA INTEGRITY on the recon's yield-change units (^TNX is quoted in
     PERCENT, so bp = pt * 100, not * 10).
  1. Reproduce the parked cell on today's data: trigger set, episode count,
     h-ladder.  Report drift against the parked numbers.
  3. THE CONDITIONING QUESTION (2026-08-07 registry trap): a percentile gate is
     not a magnitude gate.  Compute the 21d/63d/252d yield CHANGE and the
     trailing-252 RANGE WIDTH at every historical trigger episode, place today
     in that distribution, and split the cell's forward returns by magnitude.
  4. Re-charge the multiplicity over the grid ACTUALLY WALKED (3 vehicles x 10
     horizons x 6 proximity rungs), not the 3x2x5 the parked entry charged.
Plus the standing traps: lag profile 0/1/2, decluster ladder 5/10/21/42,
concentration BY VALUE on the traded side, episode YEAR histogram.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

pd.set_option("display.width", 250)

ASOF = pd.Timestamp("2026-08-31")

raw = close_panel(["^TNX", "TLT", "IEF"])
px = raw.dropna(how="any")
px = px[px.index <= ASOF]
idx = px.index
tnx = px["^TNX"]

hi252 = rolling_on_valid(tnx, lambda x: x.rolling(252).max())
off_hi = tnx / hi252 - 1.0
LEVEL = off_hi >= -0.0025
sec = tnx - tnx.shift(252)

d = px[["TLT", "IEF"]].pct_change().dropna()
BETA = float(np.polyfit(d["IEF"].values, d["TLT"].values, 1)[0])
FLAT = [("IEF", 1.0), ("TLT", -1.0 / BETA)]

print("=" * 112)
print("0. DATA INTEGRITY -- units on the yield change")
print("=" * 112)
print("panel %s .. %s  N=%d | beta(TLT~IEF) %.4f -> TLT weight %.4f"
      % (idx[0].date(), idx[-1].date(), len(idx), BETA, -1.0 / BETA))
print("  ^TNX last %.4f  (this series is quoted in PERCENT: 4.7580 = 4.758%%)"
      % tnx.iloc[-1])
print("  ^TNX 21 sessions ago %.4f -> change %+.4f pt = %+.1f bp"
      % (tnx.iloc[-22], tnx.iloc[-1] - tnx.iloc[-22], 100 * (tnx.iloc[-1] - tnx.iloc[-22])))
print("  ^TNX 252 sessions ago %.4f -> change %+.4f pt = %+.1f bp"
      % (tnx.iloc[-253], tnx.iloc[-1] - tnx.iloc[-253], 100 * (tnx.iloc[-1] - tnx.iloc[-253])))
print("  ^TNX 63 sessions ago %.4f -> change %+.4f pt = %+.1f bp"
      % (tnx.iloc[-64], tnx.iloc[-1] - tnx.iloc[-64], 100 * (tnx.iloc[-1] - tnx.iloc[-64])))
w = tnx.iloc[-252:]
print("  trailing-252 window: min %.4f  max %.4f  RANGE %.1f bp"
      % (w.min(), w.max(), 100 * (w.max() - w.min())))
print("  NOTE: 00_recon.py / 00c_state_recon.py multiply the pt change by 10,")
print("        not 100, so the surface map's '+0.1 bp / +5.5 bp' are 10x LOW.")
print("  percentile conventions on the LEVEL: incl-self %.4f%%  excl-self %.4f%%  off-high %+.5f"
      % (100 * float((w <= w.iloc[-1]).mean()),
         100 * float((w.iloc[:-1] <= w.iloc[-1]).mean()),
         float(off_hi.iloc[-1])))

# ---------------------------------------------------------------- fast helpers
POS = {d_: i for i, d_ in enumerate(idx)}


def fast_decluster(sig_dates, min_gap):
    keep, last = [], -10 ** 9
    for dd in sig_dates:
        p = POS.get(dd)
        if p is None:
            continue
        if p - last >= min_gap:
            keep.append(dd)
            last = p
    return pd.DatetimeIndex(keep)


def cell(mask, legs, h, lag=1, min_gap=None):
    ret = vehicle_ret(px, legs, h, lag)
    valid = ret.notna()
    sig = idx[np.asarray(mask.reindex(idx, fill_value=False).values, bool) & valid.values]
    epi = fast_decluster(sig, max(h, min_gap or 10))
    return ret, sig, epi, ret.loc[epi].values


print("\n" + "=" * 112)
print("1. REPRODUCTION of the parked cell on today's data")
print("=" * 112)
ret10 = vehicle_ret(px, FLAT, 10, 1)
sig_all = idx[LEVEL.reindex(idx, fill_value=False).values & ret10.notna().values]
print("  trigger DAYS (off-high >= -0.25%%): %d   span %s .. %s"
      % (len(sig_all), sig_all[0].date(), sig_all[-1].date()))
print("  live bar fires: %s (off-high %+.5f)" % (bool(LEVEL.iloc[-1]), off_hi.iloc[-1]))

rows = []
for h in range(1, 11):
    _, sg, epi, v = cell(LEVEL, FLAT, h)
    r = summarize(v, "h=%d" % h)
    r["n_days"] = len(sg)
    wins = int((v > 0).sum())
    r["signp"] = round(sign_test(wins, len(v)), 4)
    r["bps"] = round(100 * 100 * v.mean(), 1)
    r["x_at_6bps"] = round(abs(100 * 100 * v.mean()) / 6.0, 2)
    r["x_at_4.4bps"] = round(abs(100 * 100 * v.mean()) / 4.4, 2)
    rows.append(r)
show(rows, "1a. horizon ladder h=1..10, flattener, lag=1, decluster max(h,10)")
print("  PARKED ladder (2026-08-24 entry, x at 6 bps): 1.12 2.12 2.02 1.95 2.58"
      " 3.00 3.21 3.70 3.45 3.68")

_, sg8, epi8, v8 = cell(LEVEL, FLAT, 8)
print("\n  h=8 episodes N=%d  mean %+.4f%% = %.1f bps   parked said 22.2 bps"
      % (len(epi8), 100 * v8.mean(), 100 * 100 * v8.mean()))
print("  h=8 episode dates:", ", ".join(str(x.date()) for x in epi8))
print("  h=8 episode YEAR histogram:",
      dict(pd.Series(pd.DatetimeIndex(epi8).year).value_counts().sort_index()))

print("\n  LAG PROFILE at h=8 (registry: an effect that starts a session late has no shape)")
for lag in (0, 1, 2):
    _, sg, ep, v = cell(LEVEL, FLAT, 8, lag=lag)
    print("    lag=%d  N=%d  mean %+.4f%%  t %+.2f  hit %.1f%%"
          % (lag, len(ep), 100 * v.mean(), v.mean() / (v.std(ddof=1) / np.sqrt(len(v))),
             100 * (v > 0).mean()))

print("\n  DECLUSTER LADDER at h=8 (a result at exactly one gap is an artifact)")
for g in (5, 10, 21, 42):
    _, sg, ep, v = cell(LEVEL, FLAT, 8, min_gap=g)
    print("    min_gap=%2d  N=%2d  mean %+.4f%% = %5.1f bps  t %+.2f  hit %.1f%%"
          % (g, len(ep), 100 * v.mean(), 100 * 100 * v.mean(),
             v.mean() / (v.std(ddof=1) / np.sqrt(len(v))), 100 * (v > 0).mean()))

print("\n  CONCENTRATION BY VALUE on the traded side (h=8, min_gap=10)")
order = np.argsort(v8)
print("    top 2 winners: %s" % [(str(epi8[i].date()), round(100 * v8[i], 3)) for i in order[-2:]])
print("    worst 2      : %s" % [(str(epi8[i].date()), round(100 * v8[i], 3)) for i in order[:2]])
print("    mean %+.4f%% | drop top-2 winners %+.4f%% (%.1f bps) | drop worst-2 %+.4f%%"
      % (100 * v8.mean(), 100 * np.sort(v8)[:-2].mean(), 100 * 100 * np.sort(v8)[:-2].mean(),
         100 * np.sort(v8)[2:].mean()))
print("    cluster_note (NETS, do not trust alone):", cluster_note(epi8, v8))
show(era_split(epi8, v8), "  era split h=8")
mid8 = (pd.DatetimeIndex(epi8).year % 4 == 2)
show([summarize(v8[mid8], "MIDTERM (live)"), summarize(v8[~mid8], "non-midterm")],
     "  cycle split h=8")

# =========================================================== 3. CONDITIONING
print("\n" + "=" * 112)
print("3. THE CONDITIONING QUESTION -- is today's level-high the same ANIMAL?")
print("   registry 2026-08-07: 'a percentile gate is not a magnitude gate;")
print("   check the LEVEL the rank corresponds to today.'")
print("=" * 112)
chg21 = (tnx - tnx.shift(21)) * 100.0      # bp
chg63 = (tnx - tnx.shift(63)) * 100.0
chg252 = (tnx - tnx.shift(252)) * 100.0
rng252 = (rolling_on_valid(tnx, lambda x: x.rolling(252).max())
          - rolling_on_valid(tnx, lambda x: x.rolling(252).min())) * 100.0

TODAY = {"chg21": float(chg21.iloc[-1]), "chg63": float(chg63.iloc[-1]),
         "chg252": float(chg252.iloc[-1]), "rng252": float(rng252.iloc[-1])}
print("  TODAY: 21d %+.1f bp | 63d %+.1f bp | 252d %+.1f bp | 252d RANGE %.1f bp"
      % (TODAY["chg21"], TODAY["chg63"], TODAY["chg252"], TODAY["rng252"]))

for h in (8, 10, 5):
    _, sg, epi, v = cell(LEVEL, FLAT, h)
    print("\n  --- h=%d, %d episodes ---" % (h, len(epi)))
    for key, ser in (("chg21", chg21), ("chg63", chg63),
                     ("chg252", chg252), ("rng252", rng252)):
        e = ser.loc[epi].values
        pctile = 100.0 * float((e <= TODAY[key]).mean())
        print("   %-7s episodes: min %+7.1f  p10 %+7.1f  med %+7.1f  p90 %+7.1f  max %+7.1f"
              "  | TODAY %+7.1f = %.0fth pctile of the cell's own support"
              % (key, np.nanmin(e), np.nanpercentile(e, 10), np.nanmedian(e),
                 np.nanpercentile(e, 90), np.nanmax(e), TODAY[key], pctile))
        # dose response: split episodes by the conditioner
        med = np.nanmedian(e)
        lo, hi = e <= med, e > med
        print("            dose: LOW half (<=%+.1f) %+.4f%% N=%d   HIGH half %+.4f%% N=%d"
              % (med, 100 * v[lo].mean(), int(lo.sum()), 100 * v[hi].mean(), int(hi.sum())))
        # the bucket today actually lands in
        near = e <= np.nanpercentile(e, 20)
        if near.sum() >= 2:
            print("            bottom-quintile-of-conditioner bucket (where TODAY sits if "
                  "pctile<=20): %+.4f%% N=%d  hit %.0f%%"
                  % (100 * v[near].mean(), int(near.sum()), 100 * (v[near] > 0).mean()))

print("\n  MATCHED CONTROL: episodes whose 252d yield change is within +/-40 bp of today's")
for h in (8, 10):
    _, sg, epi, v = cell(LEVEL, FLAT, h)
    e252 = chg252.loc[epi].values
    m = np.abs(e252 - TODAY["chg252"]) <= 40
    if m.sum():
        print("   h=%2d  N=%d matched  mean %+.4f%% (%.1f bps)  hit %.0f%%  vs all-episode %+.4f%%"
              % (h, int(m.sum()), 100 * v[m].mean(), 100 * 100 * v[m].mean(),
                 100 * (v[m] > 0).mean(), 100 * v.mean()))
        print("        matched dates:", [str(x.date()) for x in pd.DatetimeIndex(epi)[m]])
    else:
        print("   h=%2d  ZERO episodes within +/-40bp of today's 252d change -> the live"
              " state is OUTSIDE the cell's support" % h)

print("\n  RANGE-WIDTH split (a 252d high inside a NARROW year is a different object)")
for h in (8, 10):
    _, sg, epi, v = cell(LEVEL, FLAT, h)
    e = rng252.loc[epi].values
    for lab, m in (("range <= today's %.0f bp" % TODAY["rng252"], e <= TODAY["rng252"]),
                   ("range >  today's", e > TODAY["rng252"])):
        if m.sum():
            print("   h=%2d %-28s N=%2d  mean %+.4f%% (%5.1f bps)  hit %.0f%%"
                  % (h, lab, int(m.sum()), 100 * v[m].mean(),
                     100 * 100 * v[m].mean(), 100 * (v[m] > 0).mean()))
        else:
            print("   h=%2d %-28s N=0" % (h, lab))

# ====================================================== 4. MULTIPLICITY RECHARGE
print("\n" + "=" * 112)
print("4. MULTIPLICITY RECHARGE -- charge the grid ACTUALLY WALKED")
print("   parked charge: 3 vehicles x 2 signs x 5 horizons (15 unique |t|), P=0.018")
print("   actually walked in d2b: 3 vehicles x h=1..10 x 6 proximity rungs = 180 cells")
print("=" * 112)
GRID_LEGS = {"TLT": [("TLT", 1.0)], "IEF": [("IEF", 1.0)], "FLAT": FLAT}
HS = tuple(range(1, 11))
PROX = (0.0000, 0.0010, 0.0025, 0.0050, 0.0100, 0.0200)
rets = {(vk, h): vehicle_ret(px, lg, h, 1) for vk, lg in GRID_LEGS.items() for h in HS}
valids = {k: r.notna().values for k, r in rets.items()}
prox_masks = {p: (off_hi >= -p).reindex(idx, fill_value=False).values for p in PROX}


def grid_max_t(shift):
    best, where = 0.0, None
    for p in PROX:
        base = prox_masks[p] if shift == 0 else np.roll(prox_masks[p], shift)
        for (vk, h), r in rets.items():
            m = base & valids[(vk, h)]
            if m.sum() < 8:
                continue
            sig = idx[m]
            epi = fast_decluster(sig, max(h, 10))
            if len(epi) < 8:
                continue
            v = r.loc[epi].values
            v = v[~np.isnan(v)]
            if len(v) < 8 or v.std(ddof=1) == 0:
                continue
            t = abs(v.mean() / (v.std(ddof=1) / np.sqrt(len(v))))
            if t > best:
                best, where = t, (vk, h, p, 100 * v.mean(), len(v))
    return best, where


obs_t, obs_where = grid_max_t(0)
print("  OBSERVED grid max |t| = %.2f at vehicle=%s h=%d prox=%.4f mean %+.3f%% N=%d"
      % (obs_t, obs_where[0], obs_where[1], obs_where[2], obs_where[3], obs_where[4]))
rng = np.random.default_rng(42)
NPERM = 250
shifts = rng.integers(126, len(idx) - 126, size=NPERM)
maxes = np.array([grid_max_t(int(s))[0] for s in shifts])
print("  rotation null over %d circular shifts: median %.2f  p95 %.2f  max %.2f"
      % (NPERM, np.median(maxes), np.percentile(maxes, 95), maxes.max()))
print("  P(rotated 180-cell grid max |t| >= %.2f) = %.3f" % (obs_t, float((maxes >= obs_t).mean())))

# what does the h=8 cell itself score, and where does its t sit in the null?
t8 = v8.mean() / (v8.std(ddof=1) / np.sqrt(len(v8)))
print("  the SHIPPED cell (FLAT, h=8, prox 0.0025) has |t| = %.2f -> P(null max >= that) = %.3f"
      % (abs(t8), float((maxes >= abs(t8)).mean())))

print("\n" + "=" * 112)
print("5. GATE-OFF re-run (the trigger vs the rising-yield regime it lives inside), h=8")
print("=" * 112)
RISING = (sec > 0)
ret8 = vehicle_ret(px, FLAT, 8, 1)
valid8 = ret8.notna()
rise_days = idx[RISING.reindex(idx).fillna(False).values & valid8.values]
ep_r = fast_decluster(rise_days, 10)
b = ret8.loc[ep_r].values
se = np.sqrt(v8.var(ddof=1) / len(v8) + b.var(ddof=1) / len(b))
print("  trigger %+.4f%% (N=%d) vs rising-regime episodes %+.4f%% (N=%d)  diff %+.4f pp  welch t %+.2f"
      % (100 * v8.mean(), len(v8), 100 * b.mean(), len(b),
         100 * (v8.mean() - b.mean()), (v8.mean() - b.mean()) / se))
print("  ALL days h=8: %+.4f%%   (N=%d)" % (100 * ret8[valid8].mean(), int(valid8.sum())))
loc = local_control(idx[valid8.values], pd.DatetimeIndex(sg8))
print("  local +/-126td ex-trigger: %+.4f%% (N=%d)" % (100 * ret8.loc[loc].mean(), len(loc)))
