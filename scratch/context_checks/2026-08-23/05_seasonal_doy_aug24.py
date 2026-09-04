"""E:seasonal_doy for Aug 24: does it say anything the opex drill did not?

The engine anchors this cell on the trading day nearest asof's day-of-year in
each prior year, so h1 is that year's analogue of Monday. Late August sits
right on top of expiration week, so the first job is to measure the overlap
with drill 02 rather than publish the same fact twice.

Second job: NG=F midterm h5 came back 6 of 6 down at -5.27% (sign p 0.0156).
Natural gas rolls its front contract at the end of August, so that has to be
checked against a roll gap before it can be called a seasonal move.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
from pitch_lab import load_prices, load_events, fwd_ret, summarize, sign_test
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from seasonal_edge import seasonal_window_returns, _trading_doy

px = load_prices(["QQQ", "SPY", "IWM", "NG=F"])
ASOF = pd.Timestamp("2026-08-21")

print("########## which calendar day does the doy pick land on? ##########")
close = px["QQQ"]["Close"].dropna().loc[:ASOF]
doy = _trading_doy(close.index)
target = int(doy.loc[ASOF])
print(f"target trading-doy = {target} (2026-08-21)")
opex = set(load_events(["opex"])["date"])
picks = []
for y, g in close.groupby(close.index.year):
    d = doy.loc[g.index]
    j = (d - target).abs()
    if j.min() <= 2:
        day = j.idxmin()
        picks.append((y, day, day.strftime("%a"), day in opex))
for y, day, wd, isopex in picks:
    print(f"   {y}  {day.date()} {wd}  {'OPEX' if isopex else ''}")
n_opex = sum(1 for *_, o in picks if o)
print(f"   {n_opex} of {len(picks)} picks ARE the expiration bar")

print("\n########## the cell as the engine computed it ##########")
for t in ["QQQ", "SPY", "IWM"]:
    for h in (1, 5):
        for lab, filt in (("all", None), ("midterm", 2)):
            st = seasonal_window_returns(px[t], ASOF, h, cycle_phase_filter=filt)
            if not st or st.get("insufficient"):
                continue
            print(f"   {t:4s} h{h} {lab:8s} n={st['n']:2d} mean={100*st['mean']:+.3f}% "
                  f"med={100*st['median']:+.3f}% up-down {st['n_up']}-{st['n_down']} "
                  f"p={sign_test(max(st['n_up'], st['n_down']), st['n']):.4f}")

print("\n########## NG=F late-August, and whether it is the roll ##########")
ng = px["NG=F"]
for h in (1, 5):
    for lab, filt in (("all", None), ("midterm", 2)):
        st = seasonal_window_returns(ng, ASOF, h, cycle_phase_filter=filt)
        if not st or st.get("insufficient"):
            continue
        print(f"   h{h} {lab:8s} n={st['n']:2d} mean={100*st['mean']:+.3f}% "
              f"med={100*st['median']:+.3f}% up-down {st['n_up']}-{st['n_down']} "
              f"years={st['years']} rets={[round(100*r,1) for r in st['rets']]}")

c = ng["Close"].dropna().loc[:ASOF]
r1 = c.pct_change(fill_method=None)
print("\n   biggest single-session moves inside each midterm h5 window:")
d2 = _trading_doy(c.index)
tgt = int(d2.loc[d2.index[d2.index <= ASOF][-1]])
for y in [2002, 2006, 2010, 2014, 2018, 2022]:
    g = c[c.index.year == y]
    if g.empty:
        continue
    dd = d2.loc[g.index]
    j = (dd - tgt).abs()
    if j.min() > 2:
        continue
    start = j.idxmin()
    pos = c.index.get_loc(start)
    win = r1.iloc[pos + 1: pos + 6]
    tot = c.iloc[pos + 5] / c.iloc[pos] - 1.0
    print(f"   {y} from {start.date()}: total {100*tot:+6.2f}%  "
          f"worst day {100*win.min():+6.2f}% on {win.idxmin().date()}  "
          f"days {[round(100*x,1) for x in win.values]}")
