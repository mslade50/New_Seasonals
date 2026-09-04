"""C9 -- the decisive geometry check.

The paying cell in c9b is anchor k=-2 from the print with h=1, i.e. ENTER MOC
on the session IMMEDIATELY BEFORE the print and exit at the print close.

Today's pitch geometry under the book's own convention is: signal measured on
the 2026-09-01 close, entry MOC 2026-09-02, print 2026-09-04. That makes the
anchor k=-3 and the print-session exit h=2. This script proves the calendar
arithmetic and then prices the cell that is ACTUALLY live today.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa: F401,F403,E402
from pitch_lab import (close_panel, fwd_lag, summarize, sign_test, load_events,
                       rolling_on_valid, show, anchor_positions, bootstrap_p_le0)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 240)

px = close_panel(["^VIX", "^VIX3M", "SVXY", "UVXY", "SPY"])
cal = px["SPY"].dropna().index
vix = px["^VIX"]
rng21 = (rolling_on_valid(vix, lambda x: x.rolling(21).max())
         - rolling_on_valid(vix, lambda x: x.rolling(21).min()))
RNG = rolling_on_valid(rng21 / rolling_on_valid(vix, lambda x: x.rolling(21).mean()),
                       lambda x: x.rolling(252).rank(pct=True) * 100)
G15 = RNG <= 15.0

# ---------------------------------------------------------------------------
# calendar arithmetic, stated explicitly
# ---------------------------------------------------------------------------
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
last = cal[-1]
fut = [last + bd * i for i in range(1, 6)]
print("=" * 110)
print(f"last price bar (the SIGNAL close)        : {last.date()}")
print(f"next sessions                            : " +
      ", ".join(str(d.date()) for d in fut))
nfp_live = pd.Timestamp("2026-09-04")
n_td = sum(1 for d in fut if d <= nfp_live)
print(f"NFP print                                : {nfp_live.date()}  "
      f"= {n_td} trading sessions AFTER the signal close")
print(f"pitch entry (MOC, lag=1)                 : {fut[0].date()}  "
      f"= {n_td-1} sessions BEFORE the print")
print(f"-> anchor offset from the print k        : {-n_td}")
print(f"-> horizon that exits at the PRINT close : h = {n_td-1}")
print("=" * 110)
print("The paying cell found in c9b is k=-2, h=1 (enter the session immediately")
print("before the print). Today's live geometry is k=-3, h=2. Those are")
print("DIFFERENT trades, one session apart.")
print("=" * 110)


def cell(k, h, tkr, gate=G15):
    ev = load_events(["nfp"])["date"]
    p, _ = anchor_positions(cal, ev, k)
    a = pd.DatetimeIndex([cal[i] for i in p])
    a = a[gate.reindex(a).fillna(False).values]
    ss = px[tkr].dropna()
    f = fwd_lag(ss, h, lag=1)
    v = f.reindex(a).dropna()
    if len(v) == 0:
        return {"label": f"{tkr} k={k} h={h}", "n": 0}, v
    st = summarize(v.values, f"{tkr} k={k} h={h}")
    st["drift_pct"] = round(100 * f.dropna().mean(), 3)
    st["excess_pp"] = round(st["mean_pct"] - st["drift_pct"], 3)
    st["signp"] = round(sign_test(int((v.values > 0).sum()), len(v)), 4)
    st["bootP_le0"] = round(bootstrap_p_le0(v.values), 3)
    return st, v


print("\nA. TODAY'S LIVE CELL (k=-3): enter 2026-09-02 MOC")
for tkr in ("SVXY", "^VIX", "UVXY", "^VIX3M", "SPY"):
    rows = []
    for h in (1, 2, 3, 5):
        st, _ = cell(-3, h, tkr)
        rows.append(st)
    show(rows, f"{tkr}  (h=2 exits at the print close)")

print("\nB. TOMORROW'S CELL (k=-2, the one that pays): enter 2026-09-03 MOC")
for tkr in ("SVXY", "^VIX", "UVXY"):
    rows = []
    for h in (1, 2, 3):
        st, _ = cell(-2, h, tkr)
        rows.append(st)
    show(rows, tkr)

print("\nC. THE ONE-SESSION COST -- SVXY return over the session between the two")
print("   entries (i.e. the extra session today's entry is exposed to):")
st, v3 = cell(-3, 1, "SVXY")
print(f"   SVXY k=-3 h=1 (2 sessions before the print -> 1 session before): "
      f"n={st['n']} mean {st['mean_pct']:+.3f}%  hit {st['hit']:.1f}  "
      f"worst {st['worst_pct']:.2f}%")
st2, _ = cell(-2, 1, "SVXY")
print(f"   SVXY k=-2 h=1 (the print session itself):                       "
      f"n={st2['n']} mean {st2['mean_pct']:+.3f}%  hit {st2['hit']:.1f}")
st3, _ = cell(-3, 2, "SVXY")
print(f"   SVXY k=-3 h=2 (both sessions, = TODAY'S TRADE):                 "
      f"n={st3['n']} mean {st3['mean_pct']:+.3f}%  hit {st3['hit']:.1f}  "
      f"signp {st3['signp']}")

print("\nD. is the k=-3 h=2 cell distinguishable from SVXY doing nothing?")
for tkr in ("SVXY",):
    st, v = cell(-3, 2, tkr)
    print(f"   {tkr}: excess over own drift {st['excess_pp']:+.3f}pp, "
          f"sign p {st['signp']}, bootstrap P(mean<=0) {st['bootP_le0']}")
    print(f"   per-episode: " + ", ".join(
        f"{d.date()}:{100*r:+.2f}" for d, r in v.items()))

print("\nE. TOMORROW'S ARM -- what has to be true at the 2026-09-02 close")
print(f"   live rel-range percentile at the 2026-09-01 close: {RNG.iloc[-1]:.1f}")
print("   the gate needs RNG <= 15.0 on the 2026-09-02 close, with the print")
print("   on 2026-09-04 (k=-2 from that anchor).")
