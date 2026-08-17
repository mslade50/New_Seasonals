"""C3 round 1 -- VIX/VIX3M ratio at its 1.2nd percentile, BOTH directions.

Today: VIX 14.25 / VIX3M 18.46 = 0.772, trailing-252d rank ~1.2.
That is the SAME STATE as the 2026-08-13 kill "98th-percentile VIX3M/VIX
contango" (a1_c1_termspread.py) read from the reciprocal side, so this run
is a re-measurement of a registry cell, not a new one. Re-measured anyway,
because a re-skin kill still owes today's numbers.

Directions measured on the SAME trigger days:
  (a) carry continuation  -> long SVXY (POST-BREAK ONLY, 2018-03-01+)
  (b) complacency fade    -> long UVXY / short SPY / short QQQ
Plus gate attribution: does the RATIO add anything over "VIX level is low"?
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

TICKERS = ["^VIX", "^VIX3M", "SVXY", "UVXY", "SPY", "QQQ"]
SVXY_BREAK = pd.Timestamp("2018-03-01")   # -1x -> -0.5x on 2018-02-28

px = close_panel(TICKERS)
vix, vix3m = px["^VIX"], px["^VIX3M"]
ratio = (vix / vix3m).dropna()


def level_rank(s: pd.Series, lb: int = 252) -> pd.Series:
    """Trailing percentile of the LEVEL (not of a return). pct_rank() in the
    lab takes a PRICE series and ranks pct_change(n) -- wrong tool here."""
    return s.rolling(lb).rank(pct=True) * 100.0


rr252 = level_rank(ratio, 252)
vixr252 = level_rank(vix, 252)

print("=" * 78)
print("0. LIVE STATE HONESTY (the 2026-08-14 secular-drift trap)")
print("=" * 78)
last = ratio.index[-1]
print(f"  asof {last.date()}  VIX {vix.iloc[-1]:.2f}  VIX3M {vix3m.iloc[-1]:.2f} "
      f" ratio {ratio.iloc[-1]:.4f}")
print(f"  ratio trailing-252d pctile : {rr252.iloc[-1]:.2f}")
print(f"  ratio FULL-HISTORY pctile  : {100*(ratio < ratio.iloc[-1]).mean():.2f}  "
      f"(2006-07+ , N={len(ratio)})")
print(f"  ratio 2018+ pctile         : "
      f"{100*(ratio[ratio.index >= '2018-01-01'] < ratio.iloc[-1]).mean():.2f}")
print(f"  ratio median 2006-2012 {ratio[:'2012-12-31'].median():.4f}  "
      f"2018-2026 {ratio['2018-01-01':].median():.4f}  (drift check)")
print(f"  VIX level trailing-252d pctile: {vixr252.iloc[-1]:.2f}   "
      f"full-history pctile {100*(vix < vix.iloc[-1]).mean():.2f}")

# ---------------------------------------------------------------- triggers
MAIN = 2.0
trig = (rr252 <= MAIN)
trig_days = ratio.index[trig.reindex(ratio.index, fill_value=False).values]
print(f"\n  trigger  ratio rank252 <= {MAIN}: {len(trig_days)} days, "
      f"{trig_days.min().date()} .. {trig_days.max().date()}")
print("  by year:", dict(pd.Series(1, index=trig_days).groupby(trig_days.year).sum()))

variants = {f"rank<={k}": (rr252 <= k) for k in (1.0, 2.0, 3.0, 5.0, 10.0, 20.0)}

# ------------------------------------------------------------ (a) long SVXY
px_post = px.loc[px.index >= SVXY_BREAK]
print("\n" + "=" * 78)
print("A. CARRY CONTINUATION -- long SVXY, POST-BREAK ONLY (2018-03-01+)")
print("=" * 78)
n_post = int(trig.reindex(px_post.index, fill_value=False).sum())
n_pre = len(trig_days) - n_post
print(f"  trigger days pre-break {n_pre} / post-break {n_post}  "
      f"-> pooling would put {100*n_pre/max(1,len(trig_days)):.0f}% of the "
      f"sample on the security that no longer exists")
for h in (3, 5, 10):
    battery(px_post, trig, [("SVXY", 1.0)], h,
            f"A long SVXY h={h} (post-break)", cost_bps=10.0,
            variants=variants, min_gap=max(h, 5))

# ------------------------------------------------------- (b) complacency fade
print("\n" + "=" * 78)
print("B. COMPLACENCY FADE -- long UVXY / short SPY / short QQQ (same days)")
print("=" * 78)
for legs, cost in ((([("UVXY", 1.0)]), 12.0), (([("SPY", -1.0)]), 2.0),
                   (([("QQQ", -1.0)]), 2.0)):
    frame = px_post if legs[0][0] == "UVXY" else px
    for h in (5, 10):
        battery(frame, trig, legs, h,
                f"B {legs} h={h}", cost_bps=cost, variants=variants,
                min_gap=max(h, 5))

# --------------------------------------------------------- gate attribution
print("\n" + "=" * 78)
print("C. GATE ATTRIBUTION -- does the RATIO add anything over 'VIX is low'?")
print("=" * 78)
vix_low = (vixr252 <= 10)
cells = {
    "ratio<=2 ONLY (not vix-low)": trig & ~vix_low,
    "vix-low ONLY (not ratio<=2)": vix_low & ~trig,
    "BOTH": trig & vix_low,
    "ratio<=2 (all)": trig,
    "vix-low (all)": vix_low,
}
print(f"  overlap: ratio<=2 days that are also vix-low: "
      f"{int((trig & vix_low).sum())} of {int(trig.sum())} "
      f"({100*(trig & vix_low).sum()/max(1,trig.sum()):.0f}%)")
print(f"  today: ratio rank {rr252.iloc[-1]:.2f}, VIX level rank "
      f"{vixr252.iloc[-1]:.2f} -> live cell is "
      f"{'BOTH' if vixr252.iloc[-1] <= 10 else 'ratio-ONLY'}")

for h in (5, 10):
    for name, legs, frame, in (("SVXY", [("SVXY", 1.0)], px_post),
                               ("SPY", [("SPY", 1.0)], px)):
        rows = []
        ret = vehicle_ret(frame, legs, h, 1)
        valid = ret.dropna().index
        for lbl, m in cells.items():
            d = frame.index[m.reindex(frame.index, fill_value=False).values]
            d = pd.DatetimeIndex(d).intersection(valid)
            if len(d) == 0:
                rows.append({"label": lbl, "n": 0})
                continue
            e = declusters(d, max(h, 5), valid)
            r = summarize(ret.loc[e].values, lbl)
            r["n_days"] = len(d)
            rows.append(r)
        rows.append(summarize(ret.loc[valid].values, "ALL DAYS"))
        show(rows, f"C. gate attribution long {name} h={h} (episodes)")

# ------------------------------------- the 08-13 lagging-marker + 08-11 beta
print("\n" + "=" * 78)
print("D. IS THE STATE A LAGGING MARKER? (2026-08-13 trap)")
print("=" * 78)
for tkr in ("SVXY", "SPY"):
    s = px_post[tkr] if tkr == "SVXY" else px[tkr]
    tr21 = s.pct_change(21)
    d = pd.DatetimeIndex(s.index).intersection(trig_days)
    print(f"  {tkr} trailing-21d return on trigger days: median "
          f"{100*tr21.loc[d].median():+.2f}%  mean {100*tr21.loc[d].mean():+.2f}%"
          f"   (all days median {100*tr21.median():+.2f}%)")

print("\n  SPY-beta translation (2026-08-11 trap): SVXY h=5 cell regressed on SPY")
h = 5
rs = vehicle_ret(px_post, [("SVXY", 1.0)], h, 1)
rm = vehicle_ret(px_post, [("SPY", 1.0)], h, 1)
ok = rs.notna() & rm.notna()
beta = np.polyfit(rm[ok].values, rs[ok].values, 1)[0]
d = pd.DatetimeIndex(px_post.index[ok.values]).intersection(trig_days)
d = declusters(d, 5, px_post.index[ok.values])
resid = (rs - beta * rm)
show([summarize(rs.loc[d].values, "SVXY raw (episodes)"),
      summarize(rm.loc[d].values, "SPY same days"),
      summarize(resid.loc[d].values, f"SVXY residual (beta={beta:.2f})"),
      summarize(resid[ok].values, "residual all days")],
     "D. beta-neutral residual, h=5")
