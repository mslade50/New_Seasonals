"""C5 STEPS 2-5 -- battery BOTH directions, gate attribution, mechanism
falsification inside the window, era/midterm/concentration/cost.

Object: SKEW/VIX3M trailing-252 LEVEL percentile >= 95 (ladder 90..99).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

pd.set_option("display.width", 220)

TK = ["^SKEW", "^VIX3M", "^VIX", "SPY", "SVXY"]
px = close_panel(TK)
core = px[["^SKEW", "^VIX3M", "^VIX"]].dropna()
px = px.loc[core.index]
skew, v3, vix = px["^SKEW"], px["^VIX3M"], px["^VIX"]
ratio3, ratio1 = skew / v3, skew / vix

def lvl_pct(s, lb=252):
    return rolling_on_valid(s, lambda x: x.rolling(lb).rank(pct=True) * 100.0)

def M(c):
    return c.reindex(px.index).fillna(False)

r3p = lvl_pct(ratio3)
r1p = lvl_pct(ratio1)
v3_lvl = lvl_pct(v3)
sk_lvl = lvl_pct(skew)
sk_r5 = pct_rank(skew, 5)
sk_r21 = pct_rank(skew, 21)

mask = M(r3p >= 95)

# ---- which dates carry the high dial? (confirm the out-of-sample point)
frag = pd.read_parquet(ROOT / "data" / "rd2_fragility.parquet")
ma10 = frag["63d"].rolling(10).mean()
ma10.index = pd.to_datetime(ma10.index)
ma10 = ma10.reindex(px.index).ffill(limit=3)
hi = px.index[mask.values & (ma10 >= 70).fillna(False).values]
print("### dial>=70 dates inside ratio3>=95 mask:", [str(d.date()) for d in hi])
m90 = M(r3p >= 90)
hi80 = px.index[m90.values & (ma10 >= 80).fillna(False).values]
print("### dial>=80 dates inside ratio3>=90 mask:", [str(d.date()) for d in hi80])
ex_today = mask & (px.index < pd.Timestamp("2026-01-01"))
sub = pd.DataFrame({"m": ex_today, "dial": ma10}).dropna()
t = sub[sub["m"]]
print("### EXCLUDING 2026: cell dial min/med/max = %.1f / %.1f / %.1f, n=%d"
      % (t["dial"].min(), t["dial"].median(), t["dial"].max(), len(t)))

variants = {
    "r3 p252>=90": M(r3p >= 90),
    "r3 p252>=95 (base)": mask,
    "r3 p252>=97": M(r3p >= 97),
    "r3 p252>=98": M(r3p >= 98),
    "r3 p252>=99": M(r3p >= 99),
    "raw ratio >=7.0": M(ratio3 >= 7.0),
    "raw ratio >=8.0": M(ratio3 >= 8.0),
    "raw ratio >=8.57 (today)": M(ratio3 >= 8.5681),
    "r1 (SKEW/VIX) p252>=95": M(r1p >= 95),
    "all days": pd.Series(True, index=px.index),
}

# ------------------------------------------------- STEP 2: horizon scan first
print("\n" + "=" * 74)
print("STEP 2a -- HORIZON SCAN (episodes, min_gap=10, lag=1)")
print("=" * 74)
trg = px.index[mask.values]
for legs, nm in (([("SPY", 1.0)], "LONG SPY"), ([("SPY", -1.0)], "SHORT SPY")):
    show(horizon_scan(px, trg, legs, hs=(1, 2, 3, 5, 10, 21), min_gap=10), nm)

# ------------------------------------------------- STEP 2b: battery both ways
for legs, nm in (([("SPY", 1.0)], "C5 LONG SPY"), ([("SPY", -1.0)], "C5 SHORT SPY")):
    for h in (3, 5, 10):
        battery(px, mask, legs, h, f"{nm} | SKEW/VIX3M p252>=95", 2.0,
                variants=variants, min_gap=10, event_kinds=("fomc_decision",))

# ------------------------------------------------- STEP 3: gate attribution
print("\n" + "=" * 74)
print("STEP 3 -- GATE ATTRIBUTION (h=5 and h=10, long SPY, episodes min_gap=10)")
print("=" * 74)
legs = [("SPY", 1.0)]
LEGS = {
    "SKEW leg alone: lvlpct>=80": M(sk_lvl >= 80),
    "SKEW leg alone: lvlpct>=90": M(sk_lvl >= 90),
    "SKEW leg alone: rank21>=90": M(sk_r21 >= 90),
    "SKEW leg alone: rank5>=95": M(sk_r5 >= 95),
    "VIX3M leg alone: lvlpct<=5": M(v3_lvl <= 5),
    "VIX3M leg alone: lvlpct<=2": M(v3_lvl <= 2),
    "RATIO r3 p252>=95": mask,
    "RATIO r3 p252>=98": M(r3p >= 98),
    "CONJ SKEWlvl>=80 & VIX3Mlvl<=5": M((sk_lvl >= 80) & (v3_lvl <= 5)),
    "LIVE CONJ SKEWr21>=90 & VIX3Mlvl<=2": M((sk_r21 >= 90) & (v3_lvl <= 2)),
    "all days": pd.Series(True, index=px.index),
}
for h in (5, 10):
    ret = vehicle_ret(px, legs, h, 1)
    valid = ret.dropna().index
    base = ret.loc[valid].mean()
    rows = []
    for lbl, mm in LEGS.items():
        s = pd.DatetimeIndex(px.index[mm.values]).intersection(valid)
        e = declusters(s, 10, valid)
        r = summarize(ret.loc[e].values, lbl)
        if r["n"]:
            r["excess_pp"] = round(r["mean_pct"] - 100 * base, 3)
            r["n_days"] = len(s)
            r["signp"] = round(sign_test(int((ret.loc[e].values > 0).sum()),
                                         r["n"], p=float((ret.loc[valid] > 0).mean())), 4)
        rows.append(r)
    show(rows, f"h={h} long SPY, all-days base = {100*base:.3f}%")

    # what does the ratio ADD over the better single leg?
    best_leg = M(v3_lvl <= 5)
    s_leg = pd.DatetimeIndex(px.index[best_leg.values]).intersection(valid)
    s_rat = pd.DatetimeIndex(px.index[mask.values]).intersection(valid)
    only_leg = s_leg.difference(s_rat)
    both = s_leg.intersection(s_rat)
    only_rat = s_rat.difference(s_leg)
    print(f"  VIX3M<=5 days {len(s_leg)}; ratio days {len(s_rat)}; "
          f"BOTH {len(both)}; VIX3M-only {len(only_leg)}; ratio-only {len(only_rat)}")
    for lbl, ss in (("BOTH", both), ("VIX3M-only", only_leg), ("ratio-only", only_rat)):
        e = declusters(ss, 10, valid)
        print("   ", summarize(ret.loc[e].values, lbl))

# ------------------------------------------------- STEP 4: mechanism inside window
print("\n" + "=" * 74)
print("STEP 4 -- FALSIFY THE MECHANISM INSIDE ITS OWN WINDOW")
print("(story: tail premium rich + ATM vol at a floor -> complacency repriced)")
print("=" * 74)
for h in (3, 5, 10):
    print(f"\n-- h={h}, entry lag=1 (change from entry close to exit close) --")
    rows = []
    for tk in ("^SKEW", "^VIX", "^VIX3M"):
        f = fwd_lag(px[tk], h, 1)
        valid = f.dropna().index
        e = declusters(pd.DatetimeIndex(px.index[mask.values]).intersection(valid),
                       10, valid)
        rows.append(summarize(f.loc[e].values, f"{tk} COND episodes"))
        rows.append(summarize(f.loc[valid].values, f"{tk} all days"))
    # ratio itself
    f = fwd_lag(ratio3, h, 1)
    valid = f.dropna().index
    e = declusters(pd.DatetimeIndex(px.index[mask.values]).intersection(valid), 10, valid)
    rows.append(summarize(f.loc[e].values, "SKEW/VIX3M COND episodes"))
    rows.append(summarize(f.loc[valid].values, "SKEW/VIX3M all days"))
    show(rows)

# ------------------------------------------------- STEP 5: midterm split
print("\n" + "=" * 74)
print("STEP 5 -- MIDTERM (year%4==2) split, episodes, long SPY")
print("=" * 74)
for h in (5, 10):
    ret = vehicle_ret(px, legs, h, 1)
    valid = ret.dropna().index
    e = declusters(pd.DatetimeIndex(px.index[mask.values]).intersection(valid), 10, valid)
    mt = (pd.DatetimeIndex(e).year % 4 == 2)
    show([summarize(ret.loc[e].values[mt], f"h={h} MIDTERM"),
          summarize(ret.loc[e].values[~mt], f"h={h} non-midterm"),
          summarize(ret.loc[valid].values[(pd.DatetimeIndex(valid).year % 4 == 2)],
                    f"h={h} all midterm days")])
    print("  midterm episode dates:", [str(d.date()) for d in pd.DatetimeIndex(e)[mt]])

# ------------------------------------------------- month/august control
print("\nAugust-only + last-td-of-month sanity (today is both):")
ret5 = vehicle_ret(px, legs, 5, 1)
valid = ret5.dropna().index
e = declusters(pd.DatetimeIndex(px.index[mask.values]).intersection(valid), 10, valid)
aug = pd.DatetimeIndex(e).month == 8
show([summarize(ret5.loc[e].values[aug], "h=5 August episodes"),
      summarize(ret5.loc[e].values[~aug], "h=5 non-August episodes")])
print("  August episode dates:", [str(d.date()) for d in pd.DatetimeIndex(e)[aug]])
