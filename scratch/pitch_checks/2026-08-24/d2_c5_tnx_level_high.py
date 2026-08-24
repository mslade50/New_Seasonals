"""C5 round 1 -- the 10-year yield AT a 52-week HIGH as a LEVEL trigger,
traded on duration (TLT / IEF / the TLT-IEF curve spread).

Adversarial order, and it is deliberate:
  (0) live state, with the level percentile quoted against BOTH the trailing
      252 days AND full history (the 2026-08-14 ^SKEW kill: 2.0th percentile
      trailing, 77.1st full history -- a level percentile on a secularly
      drifting series is still a rank trap)
  (1) what the LEVEL bought in RETURN terms: the 21d yield change and the 21d
      return rank on trigger days, against today's +0.035pt / rank 49.2
  (2) GATE OFF FIRST: the plain state underneath is "TLT near its 52-week low"
      -- price that before adding any yield-level condition
  (3) the level gate ON, BOTH SIGNS in ONE pass, h = 1..10 (the sign is
      decided from this table and not revisited)
  (4) the BOND-BULL FOSSIL test (registry 2026-08-17): split every cell by the
      secular yield regime, trailing-252 yield change up vs down. Today is the
      RISING half
  (5) the curve spread must beat DURATION SCALING, not merely be positive
      (registry 2026-08-12: TLT/IEF excess ratio 2.25 vs a daily-sd ratio 2.10)
  (6) definition neighbours: proximity rung and lookback window
  (7) concentration, era, midterm, cost
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

pd.set_option("display.width", 240)

TKS = ["^TNX", "TLT", "IEF", "LQD", "SPY"]
raw = close_panel(TKS)
px = raw[["^TNX", "TLT", "IEF", "LQD", "SPY"]].dropna(how="any")
idx = px.index
print("panel (intersection) %s .. %s  N=%d  (dropped %d union rows)"
      % (idx[0].date(), idx[-1].date(), len(idx), len(raw) - len(idx)))

tnx = px["^TNX"]
tnx_full = raw["^TNX"].dropna()          # 2000-01-03 onwards, for full-hist pct

# ------------------------------------------------------------ 0. live state
hi252 = rolling_on_valid(tnx, lambda x: x.rolling(252).max())
off_hi = tnx / hi252 - 1.0
lvl_pct_252 = rolling_on_valid(tnx, lambda x: x.rolling(252).rank(pct=True) * 100)
lvl_pct_full = pd.Series(
    [100.0 * (tnx_full.iloc[:i + 1] <= tnx_full.iloc[i]).mean()
     for i in range(len(tnx_full))], index=tnx_full.index).reindex(idx)
chg21 = tnx - tnx.shift(21)
rank21 = pct_rank(tnx, 21, 252)

print("\n--- 0. live state, %s ---" % idx[-1].date())
print("  ^TNX close                       %.3f" % tnx.iloc[-1])
print("  trailing-252 max                 %.3f  -> off high %+.2f%%"
      % (hi252.iloc[-1], 100 * off_hi.iloc[-1]))
print("  LEVEL percentile, trailing 252   %.1f" % lvl_pct_252.iloc[-1])
print("  LEVEL percentile, FULL history   %.1f   <-- the SKEW-kill contrast"
      % lvl_pct_full.iloc[-1])
print("  full-history min / max on file   %.3f / %.3f"
      % (tnx_full.min(), tnx_full.max()))
print("  21-session yield CHANGE          %+.3f pt" % chg21.iloc[-1])
print("  21d RETURN rank (252)            %.1f" % rank21.iloc[-1])
for t in ["TLT", "IEF", "LQD"]:
    lo = rolling_on_valid(px[t], lambda x: x.rolling(252).min())
    print("  %-4s %+.2f%% off its 52w LOW" % (t, 100 * (px[t].iloc[-1] / lo.iloc[-1] - 1)))

# -------------------------------------------------------------- 1. the mask
LEVEL = off_hi >= -0.0025            # AT a 52-week high (within 0.25%)
print("\n--- 1. what the LEVEL trigger bought, in RETURN terms ---")
sub = pd.DataFrame({"chg21": chg21[LEVEL], "rank21": rank21[LEVEL],
                    "lvl252": lvl_pct_252[LEVEL], "lvlfull": lvl_pct_full[LEVEL]}).dropna()
print(sub.describe().loc[["count", "mean", "50%", "min", "max"]].round(2).to_string())
print("  today: chg21 %+.3f (pctile of trigger days %.0f),  rank21 %.1f (pctile %.0f)"
      % (chg21.iloc[-1], 100 * (sub["chg21"] <= chg21.iloc[-1]).mean(),
         rank21.iloc[-1], 100 * (sub["rank21"] <= rank21.iloc[-1]).mean()))
print("  today's FULL-history level pctile %.1f vs trigger-day median %.1f"
      % (lvl_pct_full.iloc[-1], sub["lvlfull"].median()))
print("\n  mask population: %d days" % int(LEVEL.sum()))
for gap in (5, 10, 21, 63):
    print("     declustered at %2d td: %3d episodes" % (gap, len(declusters(idx[LEVEL.values], gap, idx))))
print("  by year:", dict(pd.Series(idx[LEVEL.values]).dt.year.value_counts().sort_index()))


def cell(mask, legs, hs=(1, 2, 3, 5, 10), lab="", min_gap=None, quiet=False):
    rows = []
    for h in hs:
        ret = vehicle_ret(px, legs, h, 1)
        valid = ret.notna()
        sig = idx[np.asarray(mask.reindex(idx, fill_value=False).values, bool) & valid.values]
        if len(sig) == 0:
            rows.append({"label": f"{lab} h={h}", "n": 0})
            continue
        epi = declusters(sig, min_gap or max(h, 10), idx)
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


# ============================================================== 2. GATE OFF
print("\n" + "=" * 110)
print("2. GATE OFF FIRST.  The plain state underneath is 'TLT near its 52-week low'")
print("   (today TLT +0.86% off it).  Price that with NO yield-level condition.")
print("=" * 110)
tlt_lo = rolling_on_valid(px["TLT"], lambda x: x.rolling(252).min())
TLT_NEAR_LOW = (px["TLT"] / tlt_lo - 1.0) <= 0.02
show(cell(TLT_NEAR_LOW, [("TLT", 1.0)], lab="LONG TLT | near 52w low"),
     "2a. long TLT, TLT within 2% of its 52w low, gate OFF")

# ============================================================== 3. GATE ON
print("\n" + "=" * 110)
print("3. GATE ON.  ^TNX AT a 52-week high (level trigger).  BOTH SIGNS, one pass.")
print("   The sign is read off THIS table and not revisited (2026-08-07 rule).")
print("=" * 110)
show(cell(LEVEL, [("TLT", 1.0)], hs=(1, 2, 3, 5, 10), lab="LONG TLT  (exhaustion)"),
     "3a. LONG duration on the yield-level high")
show(cell(LEVEL, [("TLT", -1.0)], hs=(1, 2, 3, 5, 10), lab="SHORT TLT (continuation)"),
     "3b. SHORT duration on the yield-level high  [exact mirror of 3a]")
show(cell(LEVEL, [("IEF", 1.0)], hs=(1, 2, 3, 5, 10), lab="LONG IEF"),
     "3c. the belly")
show(cell(LEVEL & TLT_NEAR_LOW, [("TLT", 1.0)], hs=(1, 3, 5, 10), lab="LONG TLT | level+nearlow"),
     "3d. gate attribution: BOTH conditions (today's literal joint state)")

# ================================================= 4. BOND-BULL FOSSIL TEST
print("\n" + "=" * 110)
print("4. FOSSIL TEST (registry 2026-08-17).  Secular yield regime = sign of the")
print("   trailing-252 session change in ^TNX.  TODAY IS THE RISING HALF.")
print("=" * 110)
sec = tnx - tnx.shift(252)
print("  today's trailing-252 yield change: %+.3f pt -> regime = %s"
      % (sec.iloc[-1], "RISING" if sec.iloc[-1] > 0 else "FALLING"))
for h in (1, 3, 5, 10):
    ret = vehicle_ret(px, [("TLT", 1.0)], h, 1)
    valid = ret.notna()
    sig = idx[LEVEL.values & valid.values]
    epi = declusters(sig, max(h, 10), idx)
    up = (sec.reindex(epi) > 0).values
    rows = [summarize(ret.loc[epi[up]].values, f"h={h} RISING-yield regime (live)"),
            summarize(ret.loc[epi[~up]].values, f"h={h} FALLING-yield regime")]
    base_up = (sec.reindex(idx[valid.values]) > 0).values
    rows.append(summarize(ret[valid][base_up].values, f"h={h} CTRL all days, rising regime"))
    rows.append(summarize(ret[valid][~base_up].values, f"h={h} CTRL all days, falling regime"))
    show(rows, f"4.{h} long TLT, regime split")

# ======================================== 5. CURVE SPREAD vs DURATION SCALING
print("\n" + "=" * 110)
print("5. CURVE SPREAD.  A TLT-vs-IEF spread has to BEAT duration scaling.")
print("=" * 110)
d = px[["TLT", "IEF"]].pct_change().dropna()
sd_ratio = d["TLT"].std() / d["IEF"].std()
beta = np.polyfit(d["IEF"].values, d["TLT"].values, 1)[0]
print("  daily-sd ratio TLT/IEF = %.3f ; OLS beta TLT~IEF = %.3f" % (sd_ratio, beta))
show(cell(LEVEL, [("TLT", 1.0), ("IEF", -beta)], hs=(1, 3, 5, 10),
          lab="STEEPENER long TLT / short %.2f IEF" % beta),
     "5a. duration-neutral curve spread on the level trigger")
show(cell(LEVEL, [("IEF", 1.0), ("TLT", -1.0 / beta)], hs=(1, 3, 5, 10),
          lab="FLATTENER long IEF / short %.2f TLT" % (1 / beta)),
     "5b. the mirror")
for h in (1, 3, 5, 10):
    rt = vehicle_ret(px, [("TLT", 1.0)], h, 1)
    ri = vehicle_ret(px, [("IEF", 1.0)], h, 1)
    valid = rt.notna() & ri.notna()
    sig = idx[LEVEL.values & valid.values]
    epi = declusters(sig, max(h, 10), idx)
    mt, mi = rt.loc[epi].mean(), ri.loc[epi].mean()
    print("  h=%2d  TLT %+.3f%%  IEF %+.3f%%  ratio %6.2f  (duration-scaling would "
          "predict %.2f)" % (h, 100 * mt, 100 * mi,
                             (mt / mi) if mi else np.nan, sd_ratio))

# ================================================== 6. DEFINITION NEIGHBOURS
print("\n" + "=" * 110)
print("6. DEFINITION NEIGHBOURS -- nudge the proximity rung and the lookback.")
print("=" * 110)
rows = []
for prox in (0.0000, 0.0025, 0.0050, 0.0100, 0.0200):
    m = off_hi >= -prox
    ret = vehicle_ret(px, [("TLT", 1.0)], 5, 1)
    sig = idx[m.values & ret.notna().values]
    if len(sig) == 0:
        continue
    epi = declusters(sig, 10, idx)
    r = summarize(ret.loc[epi].values, "prox<=%.2f%%  (long TLT h=5)" % (100 * prox))
    r["n_days"] = len(sig)
    rows.append(r)
show(rows, "6a. proximity-to-the-high rung")
rows = []
for lb in (63, 126, 189, 252, 378, 504):
    h_lb = rolling_on_valid(tnx, lambda x, L=lb: x.rolling(L).max())
    m = (tnx / h_lb - 1.0) >= -0.0025
    ret = vehicle_ret(px, [("TLT", 1.0)], 5, 1)
    sig = idx[m.values & ret.notna().values]
    epi = declusters(sig, 10, idx)
    r = summarize(ret.loc[epi].values, "lookback %d (long TLT h=5)" % lb)
    r["n_days"] = len(sig)
    r["live_fires"] = bool(m.iloc[-1])
    rows.append(r)
show(rows, "6b. lookback window (NEAR neighbours included -- the 2026-08-19 lesson)")

# ==================================================== 7. era / midterm / cost
print("\n" + "=" * 110)
print("7. ERA, MIDTERM, CONCENTRATION, COST")
print("=" * 110)
for h in (3, 5, 10):
    ret = vehicle_ret(px, [("TLT", 1.0)], h, 1)
    sig = idx[LEVEL.values & ret.notna().values]
    epi = declusters(sig, max(h, 10), idx)
    v = ret.loc[epi].values
    show(era_split(epi, v), f"7.{h} long TLT h={h} era split")
    mid = (pd.DatetimeIndex(epi).year % 4 == 2)
    show([summarize(v[mid], f"h={h} MIDTERM years (live)"),
          summarize(v[~mid], f"h={h} non-midterm")], f"  midterm split h={h}")
    print("  concentration:", cluster_note(epi, v))
    print("  cost: TLT ~3 bps. episode mean %.3f%% = %.1f bps -> %.1fx cost"
          % (100 * v.mean(), 100 * v.mean() * 100, abs(100 * v.mean() * 100) / 3.0))

battery(px, LEVEL, [("TLT", 1.0)], 5, "C5 LONG TLT on ^TNX at a 52w high level",
        3.0, min_gap=10)
