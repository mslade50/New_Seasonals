"""C1 round 2 - regime, neighbours, gate attribution, near-high subclass.

Round 1 found the rung ladder inverts as it approaches today's value (>=8pp
+0.555% at h=5, >=9pp -0.408%, >=10pp -0.490%, today 9.98pp) and that the
trigger is a BEAR-tape selector (20% of trigger days above SPY's 200d against
a 71.6% base) while today sits +8.1% above.  This quantifies:

  A. the near-high subclass - the population that actually matches today
  B. the fragility-dial regime of the trigger set vs today's 89.5
  C. neighbour lookbacks 3d / 5d / 10d (definition fragility)
  D. gate attribution - is the XLV leg doing anything, or is this "XLK fell"?
  E. midterm and era splits on the populated rung
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

pd.set_option("display.width", 240)

TK = ["XLK", "XLV", "SPY", "QQQ", "SMH", "XLI", "XLP"]
px = close_panel(TK)
r5 = px.pct_change(5)
SPREAD = (r5["XLV"] - r5["XLK"]) * 100.0

hi252 = rolling_on_valid(px["SPY"], lambda x: x.rolling(252).max())
dist = px["SPY"] / hi252 - 1.0
sma200 = rolling_on_valid(px["SPY"], lambda x: x.rolling(200).mean())
above200 = px["SPY"] > sma200

print(f"today: spread {SPREAD.iloc[-1]:+.2f}pp | SPY {dist.iloc[-1]*100:+.2f}% off 52w high "
      f"| {(px['SPY'].iloc[-1]/sma200.iloc[-1]-1)*100:+.1f}% over 200d")

# ------------------------------------------------------------------ A. near-high
print("\n########## A. NEAR-HIGH SUBCLASS - the population that matches today ##########")
for rung in (5, 6, 7, 8, 9, 10):
    m = (SPREAD >= rung).fillna(False)
    for lbl, sub in (("all", m),
                     ("SPY within 3% of 52w high", m & (dist > -0.03)),
                     ("SPY above 200d", m & above200)):
        d = px.index[sub.fillna(False).values]
        if len(d) == 0:
            print(f"  rung>={rung:>2}pp  {lbl:<28} N=0 DAYS EVER")
            continue
        ret = vehicle_ret(px, [("XLK", 1.0)], 5)
        e = declusters(d.intersection(ret.dropna().index), 5, ret.dropna().index)
        s = summarize(ret.loc[e].values)
        yrs = sorted(set(e.year))
        print(f"  rung>={rung:>2}pp  {lbl:<28} N_days={len(d):>3d} N_epi={s.get('n',0):>2d} "
              f"h=5 mean={s.get('mean_pct', float('nan')):>+7.3f}%  hit={s.get('hit', float('nan')):>5.1f}%  "
              f"years={yrs}")

# ------------------------------------------------------------------ B. dial
print("\n########## B. FRAGILITY DIAL REGIME (ma10 of the 63d column) ##########")
frag = pd.read_parquet("data/rd2_fragility.parquet")
frag.index = pd.to_datetime(frag.index)
dial = frag["63d"].rolling(10).mean()
print(f"  dial series {dial.dropna().index[0].date()} -> {dial.dropna().index[-1].date()}, "
      f"today {dial.dropna().iloc[-1]:.1f}")
print("  NOTE: rows before 2026-07-02 are a RECOMPUTE vintage (CLAUDE.md); used as-is,")
print("        and pre-2016 there is no dial at all, which is where most C1 triggers live.")
for rung in (6, 8, 10):
    d = px.index[(SPREAD >= rung).fillna(False).values]
    dd = dial.reindex(d).dropna()
    print(f"  rung>={rung}pp: {len(dd)} of {len(d)} trigger days have a dial reading; "
          f"median {dd.median() if len(dd) else float('nan'):.1f}, max "
          f"{dd.max() if len(dd) else float('nan'):.1f}  (today 89.5)")
    if len(dd):
        print(f"     days at dial >= 80: {int((dd >= 80).sum())}   >= 85: {int((dd >= 85).sum())}")

# ------------------------------------------------------------------ C. neighbours
print("\n########## C. DEFINITION NEIGHBOURS - lookback 3 / 5 / 10 sessions ##########")
ret5 = vehicle_ret(px, [("XLK", 1.0)], 5)
valid = ret5.dropna().index
base = ret5.loc[valid].mean() * 100
rows = []
for k in (3, 5, 10):
    sp = (px.pct_change(k)["XLV"] - px.pct_change(k)["XLK"]) * 100.0
    cur = sp.iloc[-1]
    # match today's PERCENTILE, so each lookback is compared at its own extreme
    thr = sp.dropna().quantile(0.9964)
    m = (sp >= thr).fillna(False)
    d = px.index[m.values].intersection(valid)
    e = declusters(d, 5, valid)
    s = summarize(ret5.loc[e].values, f"{k}d lookback, 99.64 pctile (thr {thr:.2f}pp)")
    s["today_pp"] = round(cur, 2)
    s["fires_today"] = bool(cur >= thr)
    rows.append(s)
    # and the same at a populated 95th pctile
    thr2 = sp.dropna().quantile(0.95)
    m2 = (sp >= thr2).fillna(False)
    e2 = declusters(px.index[m2.values].intersection(valid), 5, valid)
    s2 = summarize(ret5.loc[e2].values, f"{k}d lookback, 95th pctile (thr {thr2:.2f}pp)")
    s2["today_pp"] = round(cur, 2)
    s2["fires_today"] = bool(cur >= thr2)
    rows.append(s2)
show(rows, f"long XLK h=5 by lookback   (all-days base {base:+.3f}%)")

# ------------------------------------------------------------------ D. attribution
print("\n########## D. GATE ATTRIBUTION - does the XLV leg do ANY work? ##########")
xlk5 = px.pct_change(5)["XLK"] * 100.0
xlv5 = px.pct_change(5)["XLV"] * 100.0
print(f"  today: XLK 5d {xlk5.iloc[-1]:+.2f}%   XLV 5d {xlv5.iloc[-1]:+.2f}%")
# count-matched: pick the XLK-only rung that fires the same number of days as spread>=8
n_target = int((SPREAD >= 8).fillna(False).sum())
thr_xlk = xlk5.dropna().quantile(n_target / len(xlk5.dropna()))
rows = []
for lbl, m in [
    (f"SPREAD >= 8pp (N={n_target})", (SPREAD >= 8).fillna(False)),
    (f"XLK 5d <= {thr_xlk:.2f}% only, count-matched", (xlk5 <= thr_xlk).fillna(False)),
    ("XLK 5d <= -5% only", (xlk5 <= -5.0).fillna(False)),
    ("XLK 5d <= -5% AND XLV 5d > 0 (the rotation)", ((xlk5 <= -5.0) & (xlv5 > 0)).fillna(False)),
    ("XLK 5d <= -5% AND XLV 5d <= 0 (no rotation)", ((xlk5 <= -5.0) & (xlv5 <= 0)).fillna(False)),
]:
    d = px.index[m.values].intersection(valid)
    e = declusters(d, 5, valid)
    s = summarize(ret5.loc[e].values, lbl)
    s["n_days"] = len(d)
    rows.append(s)
show(rows, f"long XLK h=5   (all-days base {base:+.3f}%)")
print("  Today's state: XLK 5d = {:+.2f}%, XLV 5d = {:+.2f}%".format(xlk5.iloc[-1], xlv5.iloc[-1]))

# ------------------------------------------------------------------ E. splits
print("\n########## E. ERA / MIDTERM SPLIT on the populated rung (>=8pp) ##########")
d8 = px.index[(SPREAD >= 8).fillna(False).values].intersection(valid)
e8 = declusters(d8, 5, valid)
v8 = ret5.loc[e8].values
mid = np.array([d.year % 4 == 2 for d in e8])
show([summarize(v8[mid], f"midterm years (N={int(mid.sum())})"),
      summarize(v8[~mid], f"non-midterm (N={int((~mid).sum())})")], "midterm split, h=5")
show(era_split(e8, v8), "era split, h=5")
by_yr = pd.Series(v8 * 100).groupby(e8.year.values).agg(["mean", "count"])
print("\n  by year (h=5, %):")
print(by_yr.round(2).to_string())
