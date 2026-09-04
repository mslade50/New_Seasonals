"""C5 confirmatory -- WHERE DOES TODAY SIT, and is the one strong sub-cell real?

Round-1 killed the parent. Two honest follow-ups:
 (1) partition ratio>=95 into BOTH / VIX3M-only / ratio-only and locate TODAY;
     the only cell that beat its control was ratio-only, and today may not be
     in it (2026-08-27 registry: split at the LIVE value).
 (2) the FULL-HISTORY level basis (the one the "tail premium is rich in
     absolute terms" mechanism actually needs) -- does today trigger at all?
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

pd.set_option("display.width", 220)

px = close_panel(["^SKEW", "^VIX3M", "^VIX", "SPY"])
px = px.loc[px[["^SKEW", "^VIX3M", "^VIX"]].dropna().index]
skew, v3, vix = px["^SKEW"], px["^VIX3M"], px["^VIX"]
ratio3 = skew / v3

def lvl(s, lb=252):
    return rolling_on_valid(s, lambda x: x.rolling(lb).rank(pct=True) * 100.0)

r3p = lvl(ratio3)
r3full = rolling_on_valid(ratio3, lambda x: x.expanding(252).rank(pct=True) * 100.0)
v3p = lvl(v3)

def M(c):
    return c.reindex(px.index).fillna(False)

m_ratio = M(r3p >= 95)
m_v3 = M(v3p <= 5)
parts = {
    "BOTH (ratio>=95 & VIX3M<=5)": m_ratio & m_v3,
    "VIX3M-only (<=5, ratio<95)": m_v3 & ~m_ratio,
    "ratio-only (>=95, VIX3M>5)": m_ratio & ~m_v3,
}
today = px.index[-1]
print("TODAY %s: ratio3 %.4f p252 %.1f pFULL %.1f | VIX3M %.2f lvlpct %.2f"
      % (today.date(), ratio3.iloc[-1], r3p.iloc[-1], r3full.iloc[-1],
         v3.iloc[-1], v3p.iloc[-1]))
for k, v in parts.items():
    print(f"  today in {k}: {bool(v.iloc[-1])}   (n_days {int(v.sum())})")

legs = [("SPY", 1.0)]
for h in (5, 10):
    ret = vehicle_ret(px, legs, h, 1)
    valid = ret.dropna().index
    base = ret.loc[valid].mean()
    rows = []
    for k, v in parts.items():
        s = pd.DatetimeIndex(px.index[v.values]).intersection(valid)
        e = declusters(s, 10, valid)
        lc = local_control(valid, s)
        r = summarize(ret.loc[e].values, k)
        r["excess_vs_alldays_pp"] = round(r["mean_pct"] - 100 * base, 3)
        r["CTRLc_local_pct"] = round(100 * ret.loc[lc].mean(), 3)
        r["excess_vs_CTRLc_pp"] = round(r["mean_pct"] - 100 * ret.loc[lc].mean(), 3)
        r["conc_top2"] = cluster_note(e, ret.loc[e].values)
        rows.append(r)
    show([{k: v for k, v in r.items() if k != "conc_top2"} for r in rows],
         f"h={h} long SPY partition, all-days {100*base:.3f}%")
    for r in rows:
        print("   ", r["label"], "->", r["conc_top2"])

# ---- full-history basis: the mechanism's own basis
print("\n" + "=" * 74)
print("FULL-HISTORY LEVEL BASIS (2026-08-14 trap: SKEW's median has drifted)")
print("=" * 74)
for thr in (90, 95, 98):
    mm = M(r3full >= thr)
    print(f"pFULL>={thr}: n_days {int(mm.sum())}, live today? {bool(mm.iloc[-1])}"
          f"  (today pFULL {r3full.iloc[-1]:.1f})")
mm = M(r3full >= 95)
for h in (5, 10):
    ret = vehicle_ret(px, legs, h, 1)
    valid = ret.dropna().index
    s = pd.DatetimeIndex(px.index[mm.values]).intersection(valid)
    e = declusters(s, 10, valid)
    lc = local_control(valid, s)
    show([summarize(ret.loc[e].values, f"h={h} pFULL>=95 episodes"),
          summarize(ret.loc[valid].values, f"h={h} all days"),
          summarize(ret.loc[lc].values, f"h={h} CTRL-c local")])
    print("  span:", s[0].date(), "..", s[-1].date(),
          "| years:", sorted(set(pd.DatetimeIndex(e).year)))

# ---- absolute-terms check the mechanism claims
print("\nIs tail premium actually RICH in absolute terms today?")
print("  SKEW 149.77 vs 2018+ median %.2f (pctile of 2018+ = %.1f)"
      % (skew.loc["2018":].median(),
         100 * (skew.loc["2018":] <= skew.iloc[-1]).mean()))
print("  SKEW/VIX3M %.4f vs 2018+ median %.4f (pctile of 2018+ = %.1f)"
      % (ratio3.iloc[-1], ratio3.loc["2018":].median(),
         100 * (ratio3.loc["2018":] <= ratio3.iloc[-1]).mean()))
print("  SKEW/VIX3M pctile of 2013-2017 era = %.1f"
      % (100 * (ratio3.loc["2013":"2017"] <= ratio3.iloc[-1]).mean()))
