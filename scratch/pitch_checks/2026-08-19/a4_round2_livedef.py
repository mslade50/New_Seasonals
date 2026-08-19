"""Round 2 on the ONE trigger definition that actually fires today.

a1b showed the trailing-252-rank>=99 version of the XLV-XLK gap does NOT fire
on 2026-08-18 (today's trailing rank is 97.2).  What fires is the ABSOLUTE
magnitude: +4.07pp, 99.3rd pctile of the full sample.  So the absolute
definition gets the full round-2 treatment it was only sampled for in the
a1b threshold ladder -- and the ladder's numbers for it pointed the OTHER
way (C2, the snap-back), so this is where C2 gets its real hearing.

Also re-runs C3 on the XLV-XLK-absolute trigger with the near-a-high gate,
since a2 killed the max-min version on reference class and the XLV-XLK
version was the one showing +0.733% at h=3.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

pd.set_option("display.width", 250)

TK = ["XLV", "XLK", "SPY", "QQQ"]
raw = load_prices(TK)
px = close_panel(TK)
r1 = px.pct_change()
vk = (r1["XLV"] - r1["XLK"]).dropna()
today = px.index[-1]
LIVE = vk.loc[today]

hi252 = px["SPY"].rolling(252).max()
dist = px["SPY"] / hi252 - 1.0
sma200 = px["SPY"].rolling(200).mean()
spy_atr = pd.Series(wilder_atr(raw["SPY"]["High"], raw["SPY"]["Low"],
                               raw["SPY"]["Close"], 14),
                    index=raw["SPY"].index).reindex(px.index)
spy_atrp = spy_atr / px["SPY"]

M35 = (vk >= 0.035).reindex(px.index, fill_value=False)
M40 = (vk >= LIVE).reindex(px.index, fill_value=False)
M30 = (vk >= 0.030).reindex(px.index, fill_value=False)

print("################ PART A: C1/C2 on the LIVE (absolute) definition ###########")
print(f"today XLV-XLK = {100*LIVE:+.3f}pp   n_days>=3.5pp {int(M35.sum())}  "
      f">=4.07pp {int(M40.sum())}")

for h in (1, 2, 3, 5, 10):
    battery(px, M40, [("XLV", 1.0), ("XLK", -1.0)], h,
            f"C1 direction, XLV-XLK >= {100*LIVE:.2f}pp (the live definition)",
            cost_bps=2.0,
            variants={"abs>=3.0pp": M30, "abs>=3.5pp": M35, "abs>=4.07pp": M40},
            event_kinds=("cpi", "fomc"))

print("\n\n########## A2. PER-LEG ATTRIBUTION on the live definition ##########")
rows = []
for h in (1, 2, 3, 5, 10):
    ret_sp = vehicle_ret(px, [("XLV", 1.0), ("XLK", -1.0)], h)
    valid = ret_sp.dropna().index
    epi = declusters(px.index[M40.values].intersection(valid), h, valid)
    row = {"h": h, "n_epi": len(epi)}
    for tkr in ("XLV", "XLK", "SPY"):
        leg = fwd_lag(px[tkr], h, 1)
        row[f"{tkr}_cond"] = round(100 * leg.loc[epi].mean(), 3)
        row[f"{tkr}_exc"] = round(100 * (leg.loc[epi].mean()
                                         - leg.dropna().mean()), 3)
    row["spread_cond"] = round(100 * ret_sp.loc[epi].mean(), 3)
    rows.append(row)
print(pd.DataFrame(rows).to_string(index=False))

print("\n\n########## A3. BETA-NEUTRAL on the live definition ##########")
beta = r1["XLV"].rolling(252).cov(r1["XLK"]) / r1["XLK"].rolling(252).var()
print(f"live PIT beta {beta.loc[today]:.3f}  mean over trigger days "
      f"{beta[M40.values].mean():.3f}  median hist {beta.median():.3f}")
for h in (1, 3, 5):
    ret_eq = vehicle_ret(px, [("XLV", 1.0), ("XLK", -1.0)], h)
    ret_bn = fwd_lag(px["XLV"], h, 1) - beta * fwd_lag(px["XLK"], h, 1)
    valid = ret_eq.dropna().index.intersection(ret_bn.dropna().index)
    epi = declusters(px.index[M40.values].intersection(valid), h, valid)
    show([summarize(ret_eq.loc[epi].values, f"h={h} eq-$ COND"),
          summarize(ret_eq.loc[valid].values, f"h={h} eq-$ all days"),
          summarize(ret_bn.loc[epi].values, f"h={h} beta-neutral COND"),
          summarize(ret_bn.loc[valid].values, f"h={h} beta-neutral all days")],
         f"beta-neutral, live definition, h={h}")

print("\n\n########## A4. REFERENCE CLASS -- is TODAY in this set? ##########")
sig = px.index[M40.values]
tbl = pd.DataFrame({
    "SPY_1d_pct": 100 * r1["SPY"].loc[sig],
    "XLV_1d_pct": 100 * r1["XLV"].loc[sig],
    "XLK_1d_pct": 100 * r1["XLK"].loc[sig],
    "SPY_dist52wh_pct": 100 * dist.loc[sig],
    "SPY_ATR_pct": 100 * spy_atrp.loc[sig],
    "above200d": (px["SPY"] > sma200).loc[sig],
})
print(tbl.round(2).to_string())
print("\nTODAY for comparison:")
print(f"  SPY_1d {100*r1['SPY'].loc[today]:+.2f}  XLV_1d {100*r1['XLV'].loc[today]:+.2f}  "
      f"XLK_1d {100*r1['XLK'].loc[today]:+.2f}  dist52wh {100*dist.loc[today]:+.2f}  "
      f"SPY_ATR% {100*spy_atrp.loc[today]:.2f}  above200d "
      f"{bool((px['SPY'] > sma200).loc[today])}")
print("\nsummary of the historical set:")
print(tbl.describe().round(2).to_string())
print(f"\nfrac of trigger days with SPY down: "
      f"{100*(r1['SPY'].loc[sig] < 0).mean():.1f}%")
print(f"frac with SPY within 3% of its 52w high: "
      f"{100*(dist.loc[sig] > -0.03).mean():.1f}%")
print(f"frac with SPY ATR%% below today's {100*spy_atrp.loc[today]:.2f}%: "
      f"{100*(spy_atrp.loc[sig] < spy_atrp.loc[today]).mean():.1f}%")

print("\n\n########## A5. TODAY'S SUBCLASS: calm tape, index near a high ##########")
calm = (dist > -0.03) & (spy_atrp < 0.012)
for h in (1, 2, 3, 5, 10):
    ret = vehicle_ret(px, [("XLV", 1.0), ("XLK", -1.0)], h)
    valid = ret.dropna().index
    s_all = px.index[M30.values].intersection(valid)     # widen to keep N alive
    s_calm = px.index[(M30 & calm.reindex(px.index, fill_value=False)).values
                      ].intersection(valid)
    e_all = declusters(s_all, h, valid)
    e_calm = declusters(s_calm, h, valid)
    r_calm = ret.loc[e_calm].values
    w = int((r_calm > 0).sum())
    show([summarize(ret.loc[e_all].values, "all >=3.0pp triggers"),
          summarize(r_calm, "calm + near-high subclass  <-- today"),
          summarize(ret.loc[valid].values, "CTRL all days")],
         f"h={h} C1 direction, today's subclass")
    if len(r_calm):
        print(f"   subclass sign test C1: {w}-{len(r_calm)-w}, "
              f"p={sign_test(w, len(r_calm)):.4f}  (C2 is the mirror)")
        print("   subclass dates:", ", ".join(str(d.date()) for d in e_calm))

print("\n\n################ PART B: C3 on the XLV-XLK live definition ###########")
for h in (3, 5, 10):
    ret = fwd_lag(px["SPY"], h, 1)
    valid = ret.dropna().index
    near = (dist > -0.03).reindex(px.index, fill_value=False)
    rows = []
    for lbl, m in (("XLV-XLK >=4.07pp", M40), ("XLV-XLK >=3.5pp", M35),
                   ("XLV-XLK >=3.0pp", M30)):
        e = declusters(px.index[m.values].intersection(valid), h, valid)
        rows.append(summarize(ret.loc[e].values, lbl))
        e2 = declusters(px.index[(m & near).values].intersection(valid), h, valid)
        rows.append(summarize(ret.loc[e2].values, f"  {lbl} + SPY near 52wh"))
    rows.append(summarize(ret.loc[valid].values, "CTRL all days"))
    show(rows, f"h={h} C3 long SPY, XLV-XLK trigger, near-high gate")
