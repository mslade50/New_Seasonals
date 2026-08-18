"""C11 -- the 63-day-rank laggard cross-section, built to be killed.

Universe = strategy_config.LIQUID_PLUS_COMMODITIES minus SOXS (corrupt before
2026-05-26). Trigger = a name's trailing-252 percentile rank of its 63d return
<= 5, measured on close D, entered MOC D+1 (lag=1).

Order of operations:
  0. sanity -- today's rank63 values must reproduce the recon (CSCO 0.4,
     SMH 8.3, USO 6.3)
  1. the DENOMINATOR-ROLL decomposition (2026-08-13 IHI lesson): how much of
     the low rank is today's price versus the t-63 reference bar rolling off,
     plus how many trigger names are actually ABOVE their 200d
  2. REGIME OVER-SELECTION: fraction of trigger name-days with SPY below its
     200d against the base rate (the statistic that killed SMH/QQQ twice)
  3. forward 1..10 against (a) each name's own drift over the same span and
     (b) the equal-weight universe on the same dates -- name-episode level and
     basket/date level
  4. era split
  5. overlap with the book's own dip-buy gates (OLV / St OS style perf ranks,
     computed the BOOK's way: expanding rank, min 252)
  6. alphabetical placebo on the tradeable 4-name cut
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import strategy_config as sc

RANK_THR = 5.0
HS = (1, 2, 3, 5, 10)

uni = [t for t in sc.LIQUID_PLUS_COMMODITIES if t != "SOXS"]
print(f"universe requested: {len(uni)} (SOXS excluded)")
px_all = close_panel(sorted(set(uni + ["SPY"])))
have = [t for t in uni if t in px_all.columns]
print(f"universe loaded   : {len(have)}")
P = px_all[have]
idx = P.index
spy = px_all["SPY"]

# ---------------------------------------------------------------------------
# 0. sanity
# ---------------------------------------------------------------------------
rank63 = P.apply(lambda s: pct_rank(s.dropna(), 63)).reindex(idx)
print("\n=== 0. TODAY'S rank63 (must match 01_live_state_recon / tape sort) ===")
today = rank63.iloc[-1].dropna().sort_values()
print("  deepest 12:", ", ".join(f"{t} {v:.1f}" for t, v in today.head(12).items()))
for t, want in [("CSCO", 0.4), ("SMH", 8.3), ("USO", 6.3), ("ADI", 6.7)]:
    if t in today:
        print(f"  {t}: {today[t]:.1f}  (recon {want})")

MASK = (rank63 <= RANK_THR) & rank63.notna()
print(f"\ntrigger name-days total: {int(MASK.values.sum())}")
fire_today = list(MASK.columns[MASK.iloc[-1].values])
print(f"fires today ({len(fire_today)}): {sorted(fire_today)}")

# ---------------------------------------------------------------------------
# 1. denominator roll + trend artifact
# ---------------------------------------------------------------------------
print("\n=== 1. DENOMINATOR ROLL (IHI lesson) + trend artifact ===")
lp = np.log(P)
today_comp = lp.diff(1)                 # today's own move
roll_comp = -lp.diff(1).shift(63)       # the t-63 bar rolling out of the window
sma200 = P.rolling(200).mean()
above200 = P > sma200

m = MASK.values
tc = today_comp.values[m]
rc = roll_comp.values[m]
ok = ~np.isnan(tc) & ~np.isnan(rc)
tc, rc = tc[ok], rc[ok]
print(f"  N trigger name-days with both components: {len(tc)}")
print(f"  mean |today's move|      = {100*np.abs(tc).mean():.3f}%")
print(f"  mean |roll-off contrib|  = {100*np.abs(rc).mean():.3f}%")
print(f"  roll-off dominates today's move on {100*(np.abs(rc) > np.abs(tc)).mean():.1f}% "
      "of trigger days")
a2 = above200.values[m]
a2 = a2[~pd.isna(above200.values[m])]
print(f"  trigger names ABOVE their own 200d SMA: {100*np.nanmean(above200.values[m]):.1f}% "
      "(today's SMH warning generalised)")
tod_a200 = {t: bool(above200.iloc[-1].get(t, False)) for t in sorted(fire_today)}
print(f"  today's firing names above 200d: {tod_a200}")

# ---------------------------------------------------------------------------
# 2. REGIME OVER-SELECTION
# ---------------------------------------------------------------------------
print("\n=== 2. REGIME OVER-SELECTION (the SMH/QQQ kill statistic) ===")
spy_below = (spy < spy.rolling(200).mean()).reindex(idx)
trig_dates_any = idx[MASK.any(axis=1).values]
span = (idx >= trig_dates_any[0]) & (idx <= trig_dates_any[-1])
sb = spy_below.values
w = MASK.values & ~np.isnan(rank63.values)
# name-day weighted
nd_below = np.nansum(w * sb[:, None]) / np.nansum(w)
base = np.nanmean(sb[span & spy_below.notna().values])
print(f"  SPY below 200d on {100*nd_below:.1f}% of trigger NAME-days")
print(f"  SPY below 200d on {100*np.nanmean(sb[np.isin(idx, trig_dates_any)]):.1f}% "
      "of trigger DATES")
print(f"  base rate over the same span: {100*base:.1f}%")
print(f"  today: SPY below 200d = {bool(spy_below.iloc[-1])}")

# ---------------------------------------------------------------------------
# 3. forward returns, name-episode level and basket/date level
# ---------------------------------------------------------------------------
print("\n=== 3. FORWARD RETURNS, lag=1 ===")
rows_name, rows_basket = [], []
for h in HS:
    F = P.shift(-(1 + h)) / P.shift(-1) - 1.0        # per-name fwd, lag 1
    U = F.mean(axis=1)                               # equal-weight universe
    EX = F.sub(U, axis=0)                            # excess vs universe

    # --- name-episode level: decluster each name's own trigger dates by h
    vals_r, vals_x, dts = [], [], []
    for t in have:
        d = idx[(MASK[t].values) & F[t].notna().values]
        if len(d) == 0:
            continue
        epi = declusters(d, h, idx)
        vals_r.append(F.loc[epi, t].values)
        vals_x.append(EX.loc[epi, t].values)
        dts.append(np.asarray(epi))
    vr = np.concatenate(vals_r) if vals_r else np.array([])
    vx = np.concatenate(vals_x) if vals_x else np.array([])
    dd = pd.DatetimeIndex(np.concatenate(dts)) if dts else pd.DatetimeIndex([])

    r = summarize(vr, f"h={h} laggard RAW (name-episodes)")
    r["ctrl_own_drift_pct"] = round(100 * F.values[np.isfinite(F.values)].mean(), 3)
    rows_name.append(r)
    rx = summarize(vx, f"h={h} laggard EXCESS vs universe")
    rows_name.append(rx)

    # --- basket/date level: equal-weight the day's laggards, decluster dates
    bx = EX.where(MASK).mean(axis=1)
    br = F.where(MASK).mean(axis=1)
    bd = idx[bx.notna().values]
    bepi = declusters(bd, h, idx)
    rb = summarize(br.loc[bepi].values, f"h={h} BASKET raw (date-episodes)")
    rows_basket.append(rb)
    rbx = summarize(bx.loc[bepi].values, f"h={h} BASKET excess vs universe")
    rows_basket.append(rbx)
    if h == 5:
        globals()["_EX5"], globals()["_F5"], globals()["_U5"] = EX, F, U
        globals()["_bepi5"], globals()["_bx5"] = bepi, bx
        globals()["_dd5"], globals()["_vx5"] = dd, vx

show(rows_name, "3a. name-episode level (raw and excess)")
show(rows_basket, "3b. basket / date-episode level")

# unconditional universe control
for h in HS:
    F = P.shift(-(1 + h)) / P.shift(-1) - 1.0
    v = F.values[np.isfinite(F.values)]
    print(f"  CTRL all name-days h={h}: mean {100*v.mean():+.3f}%  N={len(v)}")

# ---------------------------------------------------------------------------
# 4. era split (h=5, name-episodes, excess)
# ---------------------------------------------------------------------------
print("\n=== 4. ERA SPLIT, h=5 name-episode EXCESS ===")
show(era_split(_dd5, _vx5), "")
yr = pd.Series(_vx5, index=pd.DatetimeIndex(_dd5)).groupby(
    pd.DatetimeIndex(_dd5).year)
print("  by year (mean excess %, N):")
print("  " + "  ".join(f"{y}:{100*g.mean():+.2f}({len(g)})" for y, g in yr))

print("\n  basket-episode excess h=5:")
show([summarize(_bx5.loc[_bepi5].values, "basket excess h=5")], "")
if len(_bepi5) > 2:
    ep = _bx5.loc[_bepi5].values
    wns = int((ep > 0).sum())
    print(f"  record {wns}-{len(ep)-wns}, sign p={sign_test(wns, len(ep)):.4f}, "
          f"bootstrap P(mean<=0)={bootstrap_p_le0(ep):.3f}")
    print("  concentration:", cluster_note(_bepi5, ep))

# ---------------------------------------------------------------------------
# 5. overlap with the BOOK's dip-buy gates (expanding rank, the book's way)
# ---------------------------------------------------------------------------
print("\n=== 5. OVERLAP with the book's own dip-buy gates ===")


def book_rank(s: pd.Series, w: int) -> pd.Series:
    return s.pct_change(w).expanding(min_periods=252).rank(pct=True) * 100.0


r2 = P.apply(lambda s: book_rank(s.dropna(), 2)).reindex(idx)
r5 = P.apply(lambda s: book_rank(s.dropna(), 5)).reindex(idx)
gates = {
    "OLV-style (r2<25 & r5<33)": (r2 < 25) & (r5 < 33),
    "St OS-style (r2<15)": (r2 < 15),
    "generic 5d washout (r5<10)": (r5 < 10),
}
tot = int(MASK.values.sum())
for lbl, g in gates.items():
    both = int((MASK & g).values.sum())
    print(f"  {lbl:30s}: {both} of {tot} laggard name-days ({100*both/tot:.1f}%) "
          f"also satisfy it; gate base rate {100*np.nanmean(g.values):.1f}%")

# ---------------------------------------------------------------------------
# 6. ALPHABETICAL PLACEBO on the tradeable 4-name cut, h=5
# ---------------------------------------------------------------------------
print("\n=== 6. ALPHABETICAL PLACEBO, h=5, 4-name baskets ===")
EX, F = _EX5, _F5
sel_raw, sel_ex, alp_raw, alp_ex, keep = [], [], [], [], []
alpha_order = sorted(have)
for d in idx[MASK.any(axis=1).values]:
    row = rank63.loc[d]
    fired = [t for t in have if MASK.loc[d, t] and np.isfinite(F.loc[d, t])]
    if len(fired) < 4:
        continue
    pick = list(row[fired].sort_values().index[:4])
    avail = [t for t in alpha_order if np.isfinite(F.loc[d, t])][:4]
    if len(avail) < 4:
        continue
    sel_raw.append(F.loc[d, pick].mean())
    sel_ex.append(EX.loc[d, pick].mean())
    alp_raw.append(F.loc[d, avail].mean())
    alp_ex.append(EX.loc[d, avail].mean())
    keep.append(d)

keep = pd.DatetimeIndex(keep)
if len(keep):
    epi = declusters(keep, 5, idx)
    sel = pd.Series(sel_raw, index=keep)
    selx = pd.Series(sel_ex, index=keep)
    alp = pd.Series(alp_raw, index=keep)
    alpx = pd.Series(alp_ex, index=keep)
    show([summarize(sel.loc[epi].values, "signal-selected 4 deepest, RAW"),
          summarize(alp.loc[epi].values, "alphabetically-first 4, RAW"),
          summarize(selx.loc[epi].values, "signal-selected 4, EXCESS"),
          summarize(alpx.loc[epi].values, "alphabetically-first 4, EXCESS")],
         f"date-episodes N={len(epi)} (dates with >=4 laggards)")
    print(f"  selection premium (raw): {100*(sel.loc[epi].mean()-alp.loc[epi].mean()):+.3f}pp")
else:
    print("  no dates with >=4 simultaneous laggards")

print("\nSURVIVORSHIP CAVEAT: master_prices holds only names alive in today's "
      "universe files, which biases any long-the-laggard result UPWARD.")
