"""C5 round 1: megacap-growth complex breaking hard while the index holds.

Trigger (all three legs, measured on close D):
  1. SMH one-day return <= -1.0 * (Wilder-14 ATR / close)   [the break]
  2. SPY within 1.5% of its 252-session closing high          [index holds]
  3. SPY one-day return > 0.5 * SMH one-day return            [falls < half]

Forward SMH, QQQ, SPY and the SMH-minus-SPY residual, both directions, from
ONE measurement.  Gate attribution runs each leg out separately, because the
2026-08-14 registry finding is that an intact-trend gate INVERTS rather than
filters.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

pd.set_option("display.width", 240)

TK = ["SMH", "QQQ", "SPY", "XLK"]
raw = load_prices(TK)
px = close_panel(TK)
r1 = px.pct_change()
today = px.index[-1]

atr = {}
for t in TK:
    g = raw[t]                      # per-ticker frame, NO reindex: a leading
    a = wilder_atr(g["High"], g["Low"], g["Close"], 14)   # NaN poisons Wilder
    atr[t] = pd.Series(a, index=g.index).reindex(px.index)
atr = pd.DataFrame(atr)
atr_pct = atr / px

hi252 = px["SPY"].rolling(252).max()
dist = px["SPY"] / hi252 - 1.0

break_smh = r1["SMH"] <= -1.0 * atr_pct["SMH"]
holds = dist >= -0.015
half = r1["SPY"] > 0.5 * r1["SMH"]

print(f"live {today.date()}:")
print(f"  SMH 1d {100*r1['SMH'].loc[today]:+.2f}%  ATR%% {100*atr_pct['SMH'].loc[today]:.2f}%"
      f"  -> ATR multiples {r1['SMH'].loc[today]/atr_pct['SMH'].loc[today]:+.2f}")
print(f"  SPY dist from 52w high {100*dist.loc[today]:+.2f}%  (gate >= -1.5%)")
print(f"  SPY 1d {100*r1['SPY'].loc[today]:+.2f}% vs half of SMH "
      f"{100*0.5*r1['SMH'].loc[today]:+.2f}%")
print(f"  legs: break={bool(break_smh.loc[today])} holds={bool(holds.loc[today])} "
      f"half={bool(half.loc[today])}")

FULL = (break_smh & holds & half).fillna(False)
print(f"\nfull trigger n_days = {int(FULL.sum())}   FIRES TODAY: {bool(FULL.loc[today])}")

variants = {
    "FULL (3 legs)": FULL,
    "break only": break_smh.fillna(False),
    "break + holds": (break_smh & holds).fillna(False),
    "break + half": (break_smh & half).fillna(False),
    "break 1.25 ATR + holds + half":
        ((r1["SMH"] <= -1.25 * atr_pct["SMH"]) & holds & half).fillna(False),
    "break 0.75 ATR + holds + half":
        ((r1["SMH"] <= -0.75 * atr_pct["SMH"]) & holds & half).fillna(False),
    "holds gate at -1.0%": (break_smh & (dist >= -0.010) & half).fillna(False),
    "holds gate at -2.5%": (break_smh & (dist >= -0.025) & half).fillna(False),
    "XLK instead of SMH":
        ((r1["XLK"] <= -1.0 * atr_pct["XLK"]) & holds
         & (r1["SPY"] > 0.5 * r1["XLK"])).fillna(False),
}
for k, m in variants.items():
    mm = m.reindex(px.index, fill_value=False)
    print(f"  {k:32s} n_days={int(mm.sum()):5d}  FIRES TODAY: {bool(mm.loc[today])}")

BASE = FULL.reindex(px.index, fill_value=False)

for legs, name in ([("SMH", 1.0)], "SMH long"), ([("SPY", 1.0)], "SPY long"), \
                  ([("QQQ", 1.0)], "QQQ long"), \
                  ([("SMH", 1.0), ("SPY", -1.0)], "SMH-SPY residual"):
    for h in (3, 5, 10):
        battery(px, BASE, legs, h, f"C5 CELL: {name}", cost_bps=2.0,
                variants={k: v.reindex(px.index, fill_value=False)
                          for k, v in variants.items()},
                event_kinds=("cpi", "fomc"))

print("\n\n########## HORIZON SCAN, ALL FOUR VEHICLES ##########")
d = px.index[BASE.values]
for legs, name in ([("SMH", 1.0)], "SMH"), ([("QQQ", 1.0)], "QQQ"), \
                  ([("SPY", 1.0)], "SPY"), \
                  ([("SMH", 1.0), ("SPY", -1.0)], "SMH-SPY"):
    show(horizon_scan(px, d, legs, hs=(1, 2, 3, 5, 7, 10)), f"{name} long")

print("\n\n########## PER-LEG ATTRIBUTION for the SMH-SPY residual ##########")
rows = []
for h in (1, 2, 3, 5, 10):
    ret_sp = vehicle_ret(px, [("SMH", 1.0), ("SPY", -1.0)], h)
    valid = ret_sp.dropna().index
    sig = px.index[BASE.values].intersection(valid)
    epi = declusters(sig, h, valid)
    row = {"h": h, "n_epi": len(epi)}
    for tkr in ("SMH", "SPY", "QQQ"):
        leg = fwd_lag(px[tkr], h, 1)
        row[f"{tkr}_cond"] = round(100 * leg.loc[epi].mean(), 3)
        row[f"{tkr}_base"] = round(100 * leg.dropna().mean(), 3)
        row[f"{tkr}_exc"] = round(100 * (leg.loc[epi].mean() - leg.dropna().mean()), 3)
    row["resid_cond"] = round(100 * ret_sp.loc[epi].mean(), 3)
    rows.append(row)
print(pd.DataFrame(rows).to_string(index=False))

print("\n\n########## GATE ATTRIBUTION (does each gate move anything?) ##########")
for h in (3, 5, 10):
    for legs, name in ([("SMH", 1.0)], "SMH"), ([("SMH", 1.0), ("SPY", -1.0)], "SMH-SPY"):
        ret = vehicle_ret(px, legs, h)
        valid = ret.dropna().index
        rows = []
        for k, m in variants.items():
            s = px.index[m.reindex(px.index, fill_value=False).values].intersection(valid)
            e = declusters(s, h, valid)
            rr = summarize(ret.loc[e].values, k)
            rr["n_days"] = len(s)
            rows.append(rr)
        rows.append(summarize(ret.loc[valid].values, "CTRL all days"))
        show(rows, f"h={h} {name}: gate ladder")

print("\n\n########## MAGNITUDE GRADIENT at today's reading ##########")
mult = (r1["SMH"] / atr_pct["SMH"])
live_mult = mult.loc[today]
print(f"today's SMH break = {live_mult:+.2f} ATR")
for h in (3, 5, 10):
    for legs, name in ([("SMH", 1.0)], "SMH"), ([("SMH", 1.0), ("SPY", -1.0)], "SMH-SPY"):
        ret = vehicle_ret(px, legs, h)
        valid = ret.dropna().index
        s = px.index[(break_smh & holds & half).fillna(False).values].intersection(valid)
        e = declusters(s, h, valid)
        x = mult.loc[e].values
        y = ret.loc[e].values
        if len(e) < 6:
            print(f"h={h} {name}: N={len(e)} too few for a gradient")
            continue
        b1, b0 = np.polyfit(x, y, 1)
        print(f"h={h:2d} {name:8s} N={len(e):3d} slope={b1:+.4f} corr="
              f"{np.corrcoef(x, y)[0,1]:+.3f} fitted at {live_mult:+.2f} ATR = "
              f"{100*(b0+b1*live_mult):+.3f}%")
