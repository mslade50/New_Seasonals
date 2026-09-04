"""C4 round 1: is a COUNT of industrial/rail names at a 5-day rank floor worth
anything beyond XLI's own rank, and does the relative (XLI - beta*SPY) form beat
the outright?

Registry priors this must clear (all negative):
  2026-08-24 XLI-washout-vs-peer-high pair loses to the naked long at every h
  2026-08-26 the "intact trend" clause is a NEGATIVE-value filter
  2026-09-02 the pooled sector triple-rank-floor cell IS THE BOOK (153 signals)
  2026-09-02 "index near its high" gates are bull-tape selectors

Live state 2026-09-02: 12 of 13 complex names at r5 <= 5.2, XLI r5 0.8.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import (battery, close_panel, declusters, era_split, horizon_scan,
                       load_prices, local_control, pct_rank, show, sign_test,
                       summarize, vehicle_ret, bootstrap_p_le0, cluster_note)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 240)

COMPLEX = ["NSC", "UNP", "CSX", "DOV", "ITW", "PH", "MMM", "HON", "SNA", "IP",
           "GE", "CAT", "EMR"]
VEH = ["XLI", "SPY"]
FLOOR = 5.0          # r5 percentile floor, live: 12 of 13 sit at or below 5.2
LIVE_COUNT = 12

print("=" * 78)
print("C4  industrial/rail rank floor as a COUNT  (h focus = 10 td)")
print("=" * 78)

px_ind = load_prices(COMPLEX + VEH)
xli = px_ind["XLI"]["Close"].dropna()
cal = xli.index

# --- per-name 5d rank on OWN calendar, then aligned to XLI's ---------------
ranks = {}
for t in COMPLEX:
    s = px_ind[t]["Close"].dropna()
    ranks[t] = pct_rank(s, 5, 252).reindex(cal)
R = pd.DataFrame(ranks, index=cal)
avail = R.notna().sum(axis=1)
count = (R <= FLOOR).sum(axis=1)
frac = count / avail.replace(0, np.nan)

xli_r5 = pct_rank(xli, 5, 252)

print(f"\nlive 2026-09-02: avail={int(avail.iloc[-1])}  count(r5<={FLOOR})="
      f"{int(count.iloc[-1])}  frac={frac.iloc[-1]:.2f}  XLI r5={xli_r5.iloc[-1]:.1f}")
print("count history head/tail of availability:", cal[0].date(), "->", cal[-1].date(),
      " first date with 13 avail:", str(avail[avail >= 13].index[0].date()) if (avail >= 13).any() else "never")

usable = avail >= 10
COUNT_HI = (count >= 10) & usable            # >= 10 of >= 10 names at the floor
XLI_LO = xli_r5 <= FLOOR

print("\ntrigger day counts (full history):")
print(f"  COUNT_HI (>=10 names at r5<={FLOOR}) : {int(COUNT_HI.sum())}")
print(f"  XLI_LO   (XLI r5<={FLOOR})           : {int(XLI_LO.sum())}")
print(f"  JOINT (live state)                   : {int((COUNT_HI & XLI_LO).sum())}")
print(f"  XLI_LO & NOT COUNT_HI                : {int((XLI_LO & ~COUNT_HI).sum())}")
print(f"  COUNT_HI & NOT XLI_LO                : {int((COUNT_HI & ~XLI_LO).sum())}")

px = close_panel(VEH)
px = px.loc[px.index.isin(cal)]

# --- the regression beta the brief demands ---------------------------------
rx = px["XLI"].pct_change()
rs = px["SPY"].pct_change()
both = pd.concat([rx, rs], axis=1).dropna()
BETA = float(np.polyfit(both["SPY"], both["XLI"], 1)[0])
corr = float(both.corr().iloc[0, 1])
print(f"\nXLI on SPY daily-return regression: beta = {BETA:.3f}  corr = {corr:.3f}"
      f"  (n={len(both)})")
# trailing 252d beta for a "live" number
b252 = float(np.polyfit(both["SPY"].tail(252), both["XLI"].tail(252), 1)[0])
print(f"  trailing-252d beta = {b252:.3f}")

OUT = [("XLI", 1.0)]
REL = [("XLI", 1.0), ("SPY", -BETA)]

# --- round 1 battery on the JOINT (live) cell, outright --------------------
joint = (COUNT_HI & XLI_LO).reindex(px.index, fill_value=False)
variants = {
    f"count>=6": ((count >= 6) & usable & XLI_LO).reindex(px.index, fill_value=False),
    f"count>=8": ((count >= 8) & usable & XLI_LO).reindex(px.index, fill_value=False),
    f"count>=10 (live)": joint,
    f"count>=12": ((count >= 12) & usable & XLI_LO).reindex(px.index, fill_value=False),
    "XLI_LO alone": XLI_LO.reindex(px.index, fill_value=False),
}
battery(px, joint, OUT, h=10, title="C4 JOINT count>=10 & XLI r5<=5 -> LONG XLI",
        cost_bps=3.0, variants=variants, min_gap=10, event_kinds=("cpi", "ppi", "nfp"))

# --- GATE ATTRIBUTION: does the count separate? ----------------------------
print("\n" + "=" * 78)
print("GATE ATTRIBUTION -- run it WITHOUT the count gate, and without XLI's rank")
print("=" * 78)
for h in (3, 5, 10):
    ret_o = vehicle_ret(px, OUT, h, 1)
    ret_r = vehicle_ret(px, REL, h, 1)
    valid = ret_o.notna()
    rows = []
    cells = {
        "A XLI_LO alone": XLI_LO,
        "B JOINT (count>=10 & XLI_LO) = LIVE": COUNT_HI & XLI_LO,
        "C XLI_LO & count<10 (count OFF)": XLI_LO & ~COUNT_HI,
        "D COUNT_HI & XLI r5>5 (XLI OFF)": COUNT_HI & ~XLI_LO,
        "E COUNT_HI alone": COUNT_HI,
    }
    for lbl, m in cells.items():
        d = px.index[m.reindex(px.index, fill_value=False).values & valid.values]
        epi = declusters(d, h, px.index)
        r = summarize(ret_o.loc[epi].values, f"{lbl} OUTRIGHT")
        r["n_days"] = len(d)
        rows.append(r)
        r2 = summarize(ret_r.loc[epi].values, f"{lbl} RELATIVE(b={BETA:.2f})")
        r2["n_days"] = len(d)
        rows.append(r2)
    rows.append(summarize(ret_o[valid].values, "CTRL-b XLI all days"))
    rows.append(summarize(ret_r[valid].values, "CTRL-b RELATIVE all days"))
    show(rows, f"gate attribution, h={h}")

    # explicit B-vs-C separation test (does the count add on top of XLI_LO?)
    ret = ret_o
    dB = px.index[(COUNT_HI & XLI_LO).reindex(px.index, fill_value=False).values & valid.values]
    dC = px.index[(XLI_LO & ~COUNT_HI).reindex(px.index, fill_value=False).values & valid.values]
    eB, eC = declusters(dB, h, px.index), declusters(dC, h, px.index)
    vB, vC = ret.loc[eB].values, ret.loc[eC].values
    if len(vB) > 1 and len(vC) > 1:
        se = np.sqrt(vB.var(ddof=1) / len(vB) + vC.var(ddof=1) / len(vC))
        print(f"  h={h}: count-ON {100*vB.mean():+.3f}% (n={len(vB)}) vs count-OFF "
              f"{100*vC.mean():+.3f}% (n={len(vC)})  diff {100*(vB.mean()-vC.mean()):+.3f}pp "
              f"welch t {(vB.mean()-vC.mean())/se:+.2f}")
    # outright vs relative on the SAME episodes
    if len(eB) > 1:
        o, r = ret_o.loc[eB].values, ret_r.loc[eB].values
        d = o - r
        se = d.std(ddof=1) / np.sqrt(len(d))
        print(f"  h={h}: OUTRIGHT {100*o.mean():+.3f}% vs RELATIVE {100*r.mean():+.3f}% "
              f"on the same {len(eB)} episodes; paired diff {100*d.mean():+.3f}pp "
              f"t {d.mean()/se:+.2f}  (positive = outright wins)")

# --- horizon scan, both forms ---------------------------------------------
epi_live = declusters(px.index[joint.values & vehicle_ret(px, OUT, 10, 1).notna().values],
                      10, px.index)
show(horizon_scan(px, epi_live, OUT, hs=(1, 2, 3, 5, 7, 10), min_gap=10),
     "horizon scan OUTRIGHT (live cell episodes)")
show(horizon_scan(px, epi_live, REL, hs=(1, 2, 3, 5, 7, 10), min_gap=10),
     "horizon scan RELATIVE (live cell episodes)")

# --- the 200d base-rate split the registry demands -------------------------
print("\n" + "=" * 78)
print("SPY 200d base-rate split (registry: index-near-high gates are bull selectors)")
print("=" * 78)
spy = px["SPY"]
sma200 = spy.rolling(200).mean()
above = spy > sma200
dist_hi = spy / spy.rolling(252).max() - 1.0
for h in (5, 10):
    ret = vehicle_ret(px, OUT, h, 1)
    valid = ret.notna()
    d = px.index[joint.values & valid.values]
    epi = declusters(d, h, px.index)
    a = above.reindex(epi).fillna(False).values
    show([summarize(ret.loc[epi].values[a], f"h={h} SPY ABOVE 200d (N={int(a.sum())})"),
          summarize(ret.loc[epi].values[~a], f"h={h} SPY BELOW 200d (N={int((~a).sum())})"),
          summarize(ret[valid & above].values, f"h={h} ALL DAYS above 200d"),
          summarize(ret[valid & ~above].values, f"h={h} ALL DAYS below 200d")],
         f"200d split, h={h}")
    print(f"  live SPY dist to 52wh = {100*dist_hi.iloc[-1]:.2f}%  above200d="
          f"{bool(above.iloc[-1])}")
    near = (dist_hi.reindex(epi) > -0.03).fillna(False).values
    if near.sum() and (~near).sum():
        show([summarize(ret.loc[epi].values[near], f"h={h} SPY within 3% of 52wh (N={int(near.sum())})"),
              summarize(ret.loc[epi].values[~near], f"h={h} SPY >3% off 52wh (N={int((~near).sum())})")],
             f"index-near-high split, h={h}")

# --- BOOK OVERLAP: how much of this is already the scanner? ----------------
print("\n" + "=" * 78)
print("BOOK OVERLAP -- systematic signals inside [-1,+11] td of each episode")
print("=" * 78)
led = pd.read_parquet(ROOT / "data" / "backtest_trades_full.parquet")
led["Signal Date"] = pd.to_datetime(led["Signal Date"])
pos = pd.Series(range(len(px.index)), index=px.index)
led["pos"] = led["Signal Date"].map(pos)
led = led.dropna(subset=["pos"])
led["pos"] = led["pos"].astype(int)

epi10 = declusters(px.index[joint.values], 10, px.index)
win = set()
for d in epi10:
    p = int(pos[d])
    win.update(range(max(0, p - 1), min(len(px.index), p + 12)))
inwin = led[led["pos"].isin(win)]
print(f"episodes (min_gap 10td): {len(epi10)}   ledger rows in window: {len(inwin)} "
      f"of {len(led)} ({100*len(inwin)/max(1,len(led)):.1f}%)")
print("\nby strategy:")
print(inwin["Strategy"].value_counts().to_string())
print(f"\nwindow covers {len(win)} of {len(px.index)} sessions "
      f"({100*len(win)/len(px.index):.1f}%) -- if the strategy share ~ the calendar "
      f"share the overlap is not special")
print("\nledger rows in window whose TICKER is in the complex or XLI:")
sub = inwin[inwin["Ticker"].isin(COMPLEX + ["XLI"])]
print(f"  {len(sub)} rows; by strategy:")
print(sub["Strategy"].value_counts().to_string() if len(sub) else "  (none)")
print("\ncomplex-ticker ledger rows OVERALL (any date):",
      len(led[led["Ticker"].isin(COMPLEX + ["XLI"])]))
print(led[led["Ticker"].isin(COMPLEX + ["XLI"])]["Strategy"].value_counts().to_string())

print("\nDONE C4")
