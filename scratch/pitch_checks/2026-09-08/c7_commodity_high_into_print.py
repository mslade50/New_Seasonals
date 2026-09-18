"""C7 round 1 -- long the commodity complex at a 252d high into a back-to-back
inflation print (PPI +2td, CPI +3td from the entry close).

Live state 2026-09-04: DBC -0.19% off its 252d high, USO r5 87.7.
Entry MOC 2026-09-08 (lag=1 off the 09-04 signal bar).

Kill targets:
  - the print gate is decoration on a momentum state (gate attribution BOTH ways)
  - the state gate is decoration on a print calendar
  - USO roll decay eats the edge (registry: USO 10td unconditional drift +0.9bps)
  - placebo anchor ladder k=-5..+5
  - the live band [0, 0.25%) off the high quoted on its own (left-open rule)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

TICK = ["DBC", "USO", "XLE", "XOP", "SPY", "OIH"]
px = close_panel(TICK)
raw = load_prices(["DBC", "USO", "XLE", "XOP"])
IDX = px.index
POS = pd.Series(range(len(IDX)), index=IDX)


def dist_to_high(t: str, n: int = 252) -> pd.Series:
    """Distance below the trailing-n high, on the instrument's OWN series."""
    s = raw[t]["Close"]
    hi = rolling_on_valid(s, lambda x: x.rolling(n).max())
    return (1.0 - s / hi).reindex(IDX)


def shifted_event_dates(kinds, k: int) -> pd.DatetimeIndex:
    """Event calendar shifted k TRADING days on the price index (placebo)."""
    ev = load_events(list(kinds))["date"]
    pos, _ = anchor_positions(IDX, ev, offset=k)
    return pd.DatetimeIndex([IDX[p] for p in pos])


def flag_in_window(evdates: pd.DatetimeIndex, h: int, lag: int = 1) -> pd.Series:
    """True on signal date D when an event lands in (D+lag, D+lag+h]."""
    e = np.asarray(pd.DatetimeIndex(evdates).values, dtype="datetime64[ns]")
    out = np.zeros(len(IDX), dtype=bool)
    for i in range(len(IDX)):
        if i + lag + h >= len(IDX):
            continue
        lo = np.datetime64(IDX[i + lag])
        hi = np.datetime64(IDX[i + lag + h])
        out[i] = bool(((e > lo) & (e <= hi)).any())
    return pd.Series(out, index=IDX)


LAG, H = 1, 3
print("=" * 78)
print("C7  commodity complex at a 252d high into PPI+CPI   entry lag=1, h=3 td")
print("=" * 78)

d_dbc = dist_to_high("DBC")
print(f"\nlive DBC dist below 252d high (2026-09-04): {100*d_dbc.iloc[-1]:.3f}%   "
      f"(state file says -0.19%)")
print(f"live USO dist: {100*dist_to_high('USO').iloc[-1]:.3f}%   "
      f"XLE {100*dist_to_high('XLE').iloc[-1]:.3f}%   "
      f"XOP {100*dist_to_high('XOP').iloc[-1]:.3f}%")

pr_all = flag_in_window(shifted_event_dates(("ppi", "cpi"), 0), H, LAG)
pr_ppi = flag_in_window(shifted_event_dates(("ppi",), 0), H, LAG)
pr_cpi = flag_in_window(shifted_event_dates(("cpi",), 0), H, LAG)
both = pr_ppi & pr_cpi

TOL = 0.0025
state = d_dbc <= TOL
cell = state & pr_all
cell_both = state & both

print(f"\ncounts: DBC within {100*TOL:.2f}% of high = {int(state.sum())} days; "
      f"print-in-window = {int(pr_all.sum())}; cell = {int(cell.sum())}; "
      f"BOTH prints in window & state = {int(cell_both.sum())}")

# ---- the exact live cell -------------------------------------------------
battery(px, cell, [("DBC", 1.0)], H, "C7a  DBC within 0.25% of 252d high + print in 3td window",
        cost_bps=5.0, event_kinds=("cpi",),
        variants={
            "tol 0.10%": (d_dbc <= 0.0010) & pr_all,
            "tol 0.25% (LIVE)": cell,
            "tol 0.50%": (d_dbc <= 0.0050) & pr_all,
            "tol 1.00%": (d_dbc <= 0.0100) & pr_all,
            "tol 2.00%": (d_dbc <= 0.0200) & pr_all,
            "BOTH prints in window": cell_both,
        })

battery(px, cell, [("USO", 1.0)], H, "C7b  same state, USO vehicle (roll decay lives in CTRL-b)",
        cost_bps=4.0, event_kinds=("cpi",))

# ---- gate attribution BOTH ways -----------------------------------------
print("\n" + "=" * 78)
print("GATE ATTRIBUTION -- day level and episode level, both directions")
print("=" * 78)
for tkr, cost in (("DBC", 5.0), ("USO", 4.0)):
    ret = vehicle_ret(px, [(tkr, 1.0)], H, LAG)
    valid = ret.notna()

    def cut(m, lbl):
        d = IDX[m.values & valid.values]
        e = declusters(d, H, IDX)
        r_d = summarize(ret.loc[d].values, lbl + " [day]")
        r_e = summarize(ret.loc[e].values, lbl + " [epi]")
        return r_d, r_e

    rows = []
    for m, lbl in ((cell, "state AND print (the cell)"),
                   (state & ~pr_all, "state, NO print"),
                   (pr_all & ~state, "print, NO state"),
                   (pr_all, "print, any state"),
                   (state, "state, any calendar"),
                   (pd.Series(True, index=IDX), "all days")):
        a, b = cut(m, lbl)
        rows += [a, b]
    show(rows, f"{tkr}  h={H} lag=1")

# ---- placebo anchor ladder ----------------------------------------------
print("\n" + "=" * 78)
print("PLACEBO ANCHOR LADDER  k = -5..+5 trading days on the print calendar")
print("=" * 78)
for tkr in ("DBC", "USO"):
    ret = vehicle_ret(px, [(tkr, 1.0)], H, LAG)
    valid = ret.notna()
    rows = []
    for k in range(-5, 6):
        f = flag_in_window(shifted_event_dates(("ppi", "cpi"), k), H, LAG)
        m = state & f
        d = IDX[m.values & valid.values]
        e = declusters(d, H, IDX)
        r = summarize(ret.loc[e].values, f"k={k:+d}" + ("  <-- TRUE" if k == 0 else ""))
        rows.append(r)
    show(rows, f"{tkr} episode mean by placebo offset")
    means = [r.get("mean_pct", np.nan) for r in rows]
    true_m = means[5]
    rank = int(np.sum(np.asarray(means) >= true_m))
    print(f"  true anchor ranks {rank} of 11 by episode mean")
