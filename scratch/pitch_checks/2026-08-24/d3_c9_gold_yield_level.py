"""C9 round 1 -- long gold on GLD r21 rank >= 95 WHILE ^TNX prints a 52-week
high (a LEVEL trigger, claimed to be a different object from W14's RANK form).

Adversarial order:
  (0) live state
  (1) GATE OFF FIRST (2026-08-19): what does the PLAIN state "GLD r21 rank>=95"
      pay?  The loud state is usually the poisoning conditioner
  (2) the yield gate ON -- add, subtract, or nothing?
  (3) THE OVERLAP MEASUREMENT the brief demands: is the LEVEL mask a different
      object from the RANK mask W14 uses, or the same days?  Jaccard, not
      assertion.  If it is the same family it inherits the 168-mask rotation
      charge, P(grid max t >= 2.06) = 0.937
  (4) THE 2026-08-21 KILLER, applied directly: 2018+, GLD more than 10% below
      its 52-week high pays -0.641% over 10 at a 50% hit vs +0.844% at 72% for
      the complement.  GLD IS -14.63% OFF ITS HIGH TODAY.  Re-derive it inside
      THIS cell
  (5) vehicles GLD and GDX, cost 2 / 5 bps, need >=5x
  (6) BOOK OVERLAP measured: the scanner staged Overbot Vol Spike SHORTS in
      NEM, AGI, AU and CGAU this morning
  (7) era, midterm, concentration
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

pd.set_option("display.width", 240)

CORE = ["GLD", "GDX", "^TNX", "SLV", "DX-Y.NYB"]
raw = close_panel(CORE)
px = raw[["GLD", "GDX", "^TNX", "SLV", "DX-Y.NYB"]].dropna(how="any")
idx = px.index
print("panel (intersection) %s .. %s  N=%d" % (idx[0].date(), idx[-1].date(), len(idx)))

gld, gdx, tnx = px["GLD"], px["GDX"], px["^TNX"]
r21_gld = pct_rank(gld, 21, 252)
gld_hi = rolling_on_valid(gld, lambda x: x.rolling(252).max())
gld_off = gld / gld_hi - 1.0
tnx_hi = rolling_on_valid(tnx, lambda x: x.rolling(252).max())
tnx_off = tnx / tnx_hi - 1.0
tnx_chg21 = tnx - tnx.shift(21)
tnx_rank21 = pct_rank(tnx, 21, 252)
dx_rank21 = pct_rank(px["DX-Y.NYB"], 21, 252)

print("\n--- 0. live state, %s ---" % idx[-1].date())
print("  GLD r21 rank %.1f  (r21 %+.2f%%)" % (r21_gld.iloc[-1], 100 * (gld.iloc[-1] / gld.iloc[-22] - 1)))
print("  GLD off its 52w HIGH: %+.2f%%   <-- the 2026-08-21 kill state" % (100 * gld_off.iloc[-1]))
print("  ^TNX off its 52w high: %+.2f%%  (level trigger fires: %s)"
      % (100 * tnx_off.iloc[-1], bool(tnx_off.iloc[-1] >= -0.0025)))
print("  ^TNX 21d change %+.3f pt ; 21d return rank %.1f ; DX 21d rank %.1f"
      % (tnx_chg21.iloc[-1], tnx_rank21.iloc[-1], dx_rank21.iloc[-1]))

# ------------------------------------------------------------------- masks
GOLD = r21_gld >= 95.0
LEVEL = tnx_off >= -0.0025
W14_MAG = tnx_chg21 >= 0.20                 # W14's magnitude rung
W14_RANK = tnx_rank21 >= 65.0               # the return-rank form the registry killed
C9 = GOLD & LEVEL

print("\n--- mask populations ---")
for lab, m in [("GLD r21rank>=95", GOLD), ("TNX at 52wh (LEVEL)", LEVEL),
               ("TNX chg21>=+0.20 (W14 magnitude)", W14_MAG),
               ("TNX rank21>=65 (RANK form)", W14_RANK), ("C9 joint", C9)]:
    print("  %-34s %4d days" % (lab, int(m.sum())))


def cell(mask, legs, hs=(1, 2, 3, 5, 10), lab="", min_gap=None):
    rows = []
    for h in hs:
        ret = vehicle_ret(px, legs, h, 1)
        valid = ret.notna()
        sig = idx[np.asarray(mask.reindex(idx, fill_value=False).values, bool) & valid.values]
        if len(sig) == 0:
            rows.append({"label": f"{lab} h={h}", "n": 0})
            continue
        epi = declusters(sig, min_gap or max(h, 5), idx)
        r = summarize(ret.loc[epi].values, f"{lab} h={h}")
        in_span = (idx >= sig[0]) & (idx <= sig[-1]) & valid.values
        r["ctrl_a"] = round(100 * ret[in_span].mean(), 3)
        r["ctrl_b"] = round(100 * ret[valid].mean(), 3)
        loc = local_control(idx[valid.values], sig)
        r["ctrl_c"] = round(100 * ret.loc[loc].mean(), 3)
        r["edge_a"] = round(r["mean_pct"] - r["ctrl_a"], 3)
        r["n_days"] = len(sig)
        w = int((ret.loc[epi].values > 0).sum())
        r["signp"] = round(sign_test(w, len(epi)), 4)
        rows.append(r)
    return rows


# ============================================================== 1. GATE OFF
print("\n" + "=" * 115)
print("1. GATE OFF FIRST.  Long GLD on r21 rank >= 95, NO yield condition.")
print("=" * 115)
show(cell(GOLD, [("GLD", 1.0)], lab="GLD | r21rank>=95"), "1a. the plain thrust state")
show(cell(GOLD, [("GDX", 1.0)], lab="GDX | GLD r21rank>=95"), "1b. same state, miner vehicle")

# ============================================================== 2. GATE ON
print("\n" + "=" * 115)
print("2. GATE ON.  + ^TNX at a 52-week high (LEVEL).")
print("=" * 115)
show(cell(C9, [("GLD", 1.0)], lab="C9 GLD | thrust+TNXhi"), "2a. the candidate")
show(cell(GOLD & ~LEVEL, [("GLD", 1.0)], lab="COMPLEMENT | thrust, no TNXhi"),
     "2b. the complement -- does the gate REMOVE the good days?")
show(cell(C9, [("GDX", 1.0)], lab="C9 GDX | thrust+TNXhi"), "2c. miner vehicle")

# ================================================= 3. LEVEL vs RANK OVERLAP
print("\n" + "=" * 115)
print("3. IS THE LEVEL TRIGGER A DIFFERENT OBJECT FROM THE RANK TRIGGER?")
print("   Measured as day overlap, not asserted (brief's explicit requirement).")
print("=" * 115)


def jac(a, b):
    A = set(idx[a.values]); B = set(idx[b.values])
    return len(A & B), len(A), len(B), (len(A & B) / len(A | B) if (A | B) else np.nan)


for lab, other in [("TNX chg21>=+0.20 (W14 magnitude rung)", W14_MAG),
                   ("TNX rank21>=65 (the killed RANK form)", W14_RANK)]:
    i, na, nb, j = jac(LEVEL, other)
    print("  LEVEL vs %-38s  overlap %4d of %4d LEVEL days (%.0f%%), Jaccard %.3f"
          % (lab, i, na, 100 * i / na, j))
i, na, nb, j = jac(C9, GOLD & W14_MAG)
print("  C9 joint vs W14 joint (gold thrust + magnitude): overlap %d of %d C9 days "
      "(%.0f%%), Jaccard %.3f" % (i, na, 100 * i / na if na else np.nan, j))
print("\n  distribution of TNX chg21 ON LEVEL DAYS (what the level actually bought):")
print("   ", pd.Series(tnx_chg21[LEVEL]).describe().round(3).to_dict())
print("  today's chg21 %+.3f -> %.0fth percentile of LEVEL days"
      % (tnx_chg21.iloc[-1], 100 * (tnx_chg21[LEVEL] <= tnx_chg21.iloc[-1]).mean()))
print("  distribution of TNX rank21 ON C9 DAYS:")
print("   ", pd.Series(tnx_rank21[C9]).describe().round(1).to_dict())
print("  today's rank21 %.1f -> %.0fth percentile of C9 days"
      % (tnx_rank21.iloc[-1], 100 * (tnx_rank21[C9] <= tnx_rank21.iloc[-1]).mean()))

# ==================================================== 4. THE 08-21 KILL STATE
print("\n" + "=" * 115)
print("4. THE 2026-08-21 KILL STATE APPLIED HERE.  GLD is -14.63%% off its 52w high.")
print("=" * 115)
for h in (3, 5, 10):
    ret = vehicle_ret(px, [("GLD", 1.0)], h, 1)
    valid = ret.notna()
    for mlab, m in [("PARENT gold thrust", GOLD), ("C9 joint", C9)]:
        sig = idx[m.values & valid.values]
        epi = declusters(sig, max(h, 5), idx)
        deep = (gld_off.reindex(epi) <= -0.10).values
        rows = [summarize(ret.loc[epi[deep]].values,
                          f"{mlab} h={h}: GLD >10% BELOW its 52wh  (LIVE STATE)"),
                summarize(ret.loc[epi[~deep]].values,
                          f"{mlab} h={h}: GLD within 10% of its 52wh")]
        show(rows, None)
    # gradient in distance-from-the-high
    sig = idx[GOLD.values & valid.values]
    epi = declusters(sig, max(h, 5), idx)
    off = gld_off.reindex(epi).values
    rows = []
    for lo, hi in [(-1.00, -0.15), (-0.15, -0.10), (-0.10, -0.05), (-0.05, -0.02), (-0.02, 1.0)]:
        m = (off > lo) & (off <= hi)
        rows.append(summarize(ret.loc[epi[m]].values,
                              f"h={h} off-high in ({100*lo:.0f}%,{100*hi:.0f}%]"))
    show(rows, f"4.{h} distance-from-the-52w-high GRADIENT (parent state, h={h}); "
               f"today = -14.63% -> the (-100%,-15%] / (-15%,-10%] boundary")

# ==================================================== 5. cost + 6. book overlap
print("\n" + "=" * 115)
print("5/6. COST and BOOK OVERLAP")
print("=" * 115)
print("  GLD ~2 bps -> need >= 10 bps (0.10%);  GDX ~5 bps -> need >= 25 bps (0.25%)")
MINERS = ["NEM", "AGI", "AU", "CGAU"]
try:
    bpx = close_panel(MINERS + ["GLD", "GDX"]).dropna(how="any")
    r = bpx.pct_change().dropna().tail(504)
    bask = r[MINERS].mean(axis=1)
    cg, cx = bask.corr(r["GLD"]), bask.corr(r["GDX"])
    bg = np.polyfit(r["GLD"].values, bask.values, 1)[0]
    print("\n  book staged SHORT %s (Overbot Vol Spike) this morning." % ", ".join(MINERS))
    print("  equal-weight miner basket vs GLD: corr %.3f, beta %.2f   |  vs GDX: corr %.3f"
          % (cg, bg, cx))
    print("  -> a LONG GLD pitch offsets %.0f%% of the variance of a short the book "
          "just put on." % (100 * cg ** 2))
except Exception as e:                                          # pragma: no cover
    print("  book overlap failed:", e)

# ============================================================ 7. era/midterm
print("\n" + "=" * 115)
print("7. ERA / MIDTERM / CONCENTRATION on the C9 joint cell")
print("=" * 115)
for h in (3, 5, 10):
    ret = vehicle_ret(px, [("GLD", 1.0)], h, 1)
    sig = idx[C9.values & ret.notna().values]
    if len(sig) == 0:
        continue
    epi = declusters(sig, max(h, 5), idx)
    v = ret.loc[epi].values
    show(era_split(epi, v), f"7.{h} C9 long GLD h={h} era split")
    mid = (pd.DatetimeIndex(epi).year % 4 == 2)
    show([summarize(v[mid], f"h={h} MIDTERM (live)"), summarize(v[~mid], f"h={h} non-midterm")],
         None)
    print("  concentration:", cluster_note(epi, v))
    print("  episodes:", ", ".join(str(d.date()) for d in epi))

variants = {
    "GLDrank>=90 + TNXhi": (r21_gld >= 90) & LEVEL,
    "GLDrank>=95 + TNXhi": C9,
    "GLDrank>=98 + TNXhi": (r21_gld >= 98) & LEVEL,
    "GLDrank>=95, no gate": GOLD,
    "GLDrank>=95 + TNXhi<=1%": GOLD & (tnx_off >= -0.01),
}
battery(px, C9, [("GLD", 1.0)], 5, "C9 long GLD, gold thrust + TNX 52wh level",
        2.0, variants=variants, min_gap=5)
