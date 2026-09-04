"""C3 round 1 -- kill attempt on "long XLE on a crude one-day thrust".

Attack surface (from the adversarial brief):
  1. definition fragility -- +5% is a round number. Sweep raw thresholds,
     ATR-normalised forms and percentile forms, and report WHICH bucket
     today's +6.73% actually lands in and what THAT bucket pays.
  2. concentration / era -- top-2 episodes, year histogram, era split at 2018,
     leave-one-year-out, and whether the trigger over-selects crisis tape
     (SPY's contemporaneous behaviour on trigger days).

Everything is lag=1 MOC entry, declustered 5 td, vs the instrument's own
unconditional drift over the same horizon.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (close_panel, load_prices, fwd_lag, declusters, summarize,  # noqa: E402
                       sign_test, bootstrap_p_le0, cluster_note, battery,
                       local_control, wilder_atr)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)

TK = ["USO", "XLE", "DBC", "SPY", "XOP", "OIH"]
px = close_panel(TK)
ohlc = load_prices(["USO"])["USO"]
uso_1d = px["USO"].pct_change()

# Wilder-14 ATR on USO, expressed as a fraction of the prior close, so an
# ATR-normalised thrust is comparable across the 2008 / 2020 vol regimes.
atr = pd.Series(wilder_atr(ohlc["High"], ohlc["Low"], ohlc["Close"]),
                index=ohlc.index)
atr_pct = (atr / ohlc["Close"].shift(1)).reindex(px.index)
thrust_atr = uso_1d / atr_pct

# percentile form: trailing-252d rank of the 1d return
rank1d = uso_1d.rolling(252).rank(pct=True) * 100.0

TODAY_RET = 0.0673
print("=" * 100)
print("TODAY'S READING: USO 1d = +6.73%")
print(f"  today's ATR-normalised thrust would be ~ {TODAY_RET / atr_pct.iloc[-1]:.2f} ATR "
      f"(USO ATR14 = {100*atr_pct.iloc[-1]:.2f}% of price)")
print(f"  today's 1d percentile rank (trailing 252d) = {float((uso_1d.tail(252) < TODAY_RET).mean())*100:.1f}")
print("=" * 100)

# ---------------------------------------------------------------------------
# 1. the pitched cell, full battery
# ---------------------------------------------------------------------------
base = uso_1d >= 0.05
variants = {}
for thr in (0.03, 0.04, 0.045, 0.05, 0.06, 0.07, 0.08):
    variants[f"USO 1d >= {100*thr:.1f}%"] = uso_1d >= thr
for k in (1.0, 1.25, 1.5, 1.75, 2.0):
    variants[f"USO 1d >= {k:.2f} ATR"] = thrust_atr >= k
for r_ in (95, 97, 99):
    variants[f"USO 1d rank >= {r_}"] = rank1d >= r_
# the bucket today actually lands in, and its neighbour
variants["BUCKET [5%,6%)"] = (uso_1d >= 0.05) & (uso_1d < 0.06)
variants["BUCKET [6%,inf)  <-- TODAY"] = uso_1d >= 0.06
variants["BUCKET [4%,5%)"] = (uso_1d >= 0.04) & (uso_1d < 0.05)
variants["BUCKET [3%,4%)"] = (uso_1d >= 0.03) & (uso_1d < 0.04)

battery(px, base, [("XLE", 1.0)], h=3,
        title="C3 PITCHED: long XLE, USO 1d >= +5%, h=3",
        cost_bps=4.0, variants=variants, min_gap=5)

# ---------------------------------------------------------------------------
# 2. every threshold x every horizon, one grid, episode level
# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("2. DEFINITION FRAGILITY GRID -- XLE episode mean %, excess over own drift")
print("=" * 100)
rows = []
defs = {}
for thr in (0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.07):
    defs[f"raw >= {100*thr:.1f}%"] = uso_1d >= thr
for k in (1.0, 1.25, 1.5, 1.75, 2.0, 2.5):
    defs[f"atr >= {k:.2f}"] = thrust_atr >= k
for r_ in (94, 95, 96, 97, 98, 99):
    defs[f"rank >= {r_}"] = rank1d >= r_
defs["bucket [5,6)"] = (uso_1d >= 0.05) & (uso_1d < 0.06)
defs["bucket [6,inf) TODAY"] = uso_1d >= 0.06

s = px["XLE"].dropna()
for lbl, m in defs.items():
    m = m.reindex(s.index).fillna(False)
    trig_all = s.index[m.values]
    row = {"def": lbl, "n_days": len(trig_all)}
    epi = declusters(trig_all, 5, s.index)
    row["n_epi"] = len(epi)
    for h in (1, 2, 3, 5, 10):
        f = fwd_lag(s, h, lag=1)
        v = f.reindex(epi).dropna()
        if len(v) < 4:
            row[f"h{h}"] = np.nan
            continue
        own = f.dropna().mean() * 100
        row[f"h{h}"] = round(100 * v.mean() - own, 3)
        if h == 3:
            row["h3_hit"] = round(100 * float((v > 0).mean()), 1)
            row["h3_signp"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
    rows.append(row)
print(pd.DataFrame(rows).to_string(index=False))

# ---------------------------------------------------------------------------
# 3. concentration: year histogram + LOYO on the pitched definition
# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("3. CONCENTRATION -- year histogram + leave-one-year-out (pitched def, h=3)")
print("=" * 100)
f3 = fwd_lag(s, 3, lag=1)
m = base.reindex(s.index).fillna(False)
epi = declusters(s.index[m.values], 5, s.index)
v = f3.reindex(epi).dropna()
epi = v.index
own3 = f3.dropna().mean()

yr = pd.DataFrame({"r": v.values}, index=epi)
yr["year"] = yr.index.year
agg = yr.groupby("year")["r"].agg(["count", "mean", "sum"])
agg["mean_pct"] = (100 * agg["mean"]).round(3)
agg["sum_pp"] = (100 * agg["sum"]).round(2)
print(agg[["count", "mean_pct", "sum_pp"]].to_string())
print(f"\ntotal episodes {len(v)}, total pp {100*v.sum():+.2f}, "
      f"own-drift baseline per episode {100*own3:+.3f}%")

print("\nLEAVE-ONE-YEAR-OUT (drop each year, re-score the rest):")
loyo = []
for y in sorted(set(epi.year)):
    keep = v[epi.year != y]
    if len(keep) < 5:
        continue
    st = summarize(keep.values)
    loyo.append({"drop": y, "n": st["n"], "mean_pct": round(st["mean_pct"], 3),
                 "excess": round(st["mean_pct"] - 100 * own3, 3),
                 "hit": round(st["hit"], 1), "t": round(st["t"], 2),
                 "signp": round(sign_test(int((keep.values > 0).sum()), len(keep)), 4)})
df_l = pd.DataFrame(loyo)
print(df_l.to_string(index=False))
print(f"\nLOYO excess floor = {df_l['excess'].min():+.3f}%  (worst year to drop: "
      f"{int(df_l.loc[df_l['excess'].idxmin(), 'drop'])})")
print(f"LOYO t floor      = {df_l['t'].min():+.2f}")

print("\nDROP-BEST-EPISODE / DROP-TOP-2:")
order = np.argsort(-v.values)
for k in (1, 2, 3):
    keep = np.delete(v.values, order[:k])
    st = summarize(keep)
    print(f"  drop top {k}: n={st['n']} mean {st['mean_pct']:+.3f}% "
          f"excess {st['mean_pct']-100*own3:+.3f}% hit {st['hit']:.1f} "
          f"t {st['t']:+.2f} signp {sign_test(int((keep>0).sum()), len(keep)):.4f}")

# ---------------------------------------------------------------------------
# 4. does the trigger over-select crisis tape? SPY contemporaneously
# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("4. CRISIS-TAPE SELECTION -- what is SPY doing on trigger days?")
print("=" * 100)
spy = px["SPY"]
spy_1d = spy.pct_change()
spy_21 = spy.pct_change(21)
spy_dd = spy / spy.rolling(252).max() - 1.0
vix_proxy = spy_1d.rolling(21).std() * np.sqrt(252) * 100

tbl = pd.DataFrame({
    "spy_1d_pct": 100 * spy_1d.reindex(epi),
    "spy_21d_pct": 100 * spy_21.reindex(epi),
    "spy_dd_from_52wh_pct": 100 * spy_dd.reindex(epi),
    "spy_realvol_21d": vix_proxy.reindex(epi),
    "uso_1d_pct": 100 * uso_1d.reindex(epi),
    "xle_h3_pct": 100 * v.values,
})
print(tbl.round(2).to_string())
print("\nunconditional medians (all days):")
print(f"  spy_21d {100*spy_21.median():+.2f}%  spy_dd {100*spy_dd.median():+.2f}%  "
      f"realvol21 {vix_proxy.median():.1f}")
print("trigger-day medians:")
print(f"  spy_21d {tbl['spy_21d_pct'].median():+.2f}%  "
      f"spy_dd {tbl['spy_dd_from_52wh_pct'].median():+.2f}%  "
      f"realvol21 {tbl['spy_realvol_21d'].median():.1f}")
print("\nTODAY for comparison:")
print(f"  spy_21d {100*spy_21.iloc[-1]:+.2f}%  spy_dd {100*spy_dd.iloc[-1]:+.2f}%  "
      f"realvol21 {vix_proxy.iloc[-1]:.1f}")

# split the cell by whether the tape was calm or stressed at the trigger
calm = tbl["spy_realvol_21d"] <= vix_proxy.median()
print("\nSPLIT BY TAPE STRESS (21d realised vol vs its own full-sample median):")
for lbl, mm in (("CALM tape (realvol <= median)", calm.values),
                ("STRESSED tape (realvol > median)", ~calm.values)):
    st = summarize(v.values[mm])
    if st["n"]:
        print(f"  {lbl:<38} n={st['n']:<3} mean {st['mean_pct']:+.3f}% "
              f"excess {st['mean_pct']-100*own3:+.3f}% hit {st['hit']:.1f} "
              f"signp {sign_test(int((v.values[mm]>0).sum()), st['n']):.4f}")

# and by SPY drawdown state (today SPY is at/near a 52w high)
near_hi = tbl["spy_dd_from_52wh_pct"] >= -5.0
print("\nSPLIT BY SPY DRAWDOWN STATE (today SPY is ~at its 52w high):")
for lbl, mm in ((">= -5% from 52wh  <-- TODAY'S STATE", near_hi.values),
                ("< -5% from 52wh (crisis tape)", ~near_hi.values)):
    st = summarize(v.values[mm])
    if st["n"]:
        print(f"  {lbl:<38} n={st['n']:<3} mean {st['mean_pct']:+.3f}% "
              f"excess {st['mean_pct']-100*own3:+.3f}% hit {st['hit']:.1f} "
              f"signp {sign_test(int((v.values[mm]>0).sum()), st['n']):.4f} "
              f"worst {100*v.values[mm].min():+.2f}%")

# ---------------------------------------------------------------------------
# 5. midterm-year cut (today is a midterm year)
# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("5. MIDTERM-YEAR CUT (today is a midterm year; the board's own read is de-risk)")
print("=" * 100)
mid = (epi.year % 4) == 2
for lbl, mm in (("midterm years  <-- TODAY", mid), ("non-midterm", ~mid)):
    st = summarize(v.values[mm])
    if st["n"]:
        print(f"  {lbl:<26} n={st['n']:<3} mean {st['mean_pct']:+.3f}% "
              f"excess {st['mean_pct']-100*own3:+.3f}% hit {st['hit']:.1f} "
              f"signp {sign_test(int((v.values[mm]>0).sum()), st['n']):.4f} "
              f"years {sorted(set(epi[mm].year))}")
