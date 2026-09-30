"""A3 round 1: short EEM from the QE-5 close into the quarter-end close
(dollar-funding demand). Pre-specified sign SHORT. Cheapest kill first:
QE-5->QE EEM vs ordinary month-end run-ins (ME-5->ME) and all 5d windows."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kA_a2_qe_turn import month_ends, straddle, rec  # noqa: E402

if __name__ == "__main__":
    px = close_panel(["SPY", "EEM", "DX-Y.NYB"])
    spy = px["SPY"].dropna()
    c = px["EEM"].reindex(spy.index).dropna()
    me = month_ends(c.index)
    qe = me[me.month.isin([3, 6, 9, 12])]
    ome = me[~me.month.isin([3, 6, 9, 12])]
    sq = -straddle(c, qe, 5, 0)
    so = -straddle(c, ome, 5, 0)
    allw = -(c.shift(-5) / c - 1).dropna()
    rows = [rec(sq, "SHORT EEM QE-5->QE"), rec(so, "SHORT EEM ME-5->ME (ordinary)"),
            summarize(allw.values, "SHORT EEM all 5d windows")]
    for m in (3, 6, 9, 12):
        rows.append(rec(sq[sq.index.month == m], f"  QE month {m}"))
    rows += era_split(sq.index, sq.values, "2013-01-01")
    rows += era_split(sq.index, sq.values, "2018-01-01")
    mid = sq.index.year % 4 == 2
    rows.append(rec(sq[mid], "QE midterm yrs"))
    rows.append(rec(sq[~mid], "QE non-midterm"))
    show(rows, "A3 short EEM QE run-in")
    diff = sq.mean() - so.mean()
    se = np.sqrt(sq.var(ddof=1) / len(sq) + so.var(ddof=1) / len(so))
    print(f"QE minus ordinary-ME run-in (short) = {100*diff:+.3f}pp  welch t {diff/se:+.2f}")
    print(f"QE minus all-5d (short) = {100*(sq.mean()-allw.mean()):+.3f}pp")
    print("concentration:", cluster_note(sq.index, sq.values))

    # beta check: EEM residual vs SPY over the same windows
    s_spy = -straddle(spy, qe, 5, 0).reindex(sq.index)
    b = np.polyfit(-s_spy.values, -sq.values, 1)[0]
    resid = sq - b * s_spy
    rows = [rec(s_spy, "SHORT SPY QE-5->QE same anchors"),
            rec(resid, f"SHORT EEM residual vs beta {b:.2f} SPY")]
    # dollar leg: does DX rise into QE on these anchors?
    dx = px["DX-Y.NYB"].reindex(spy.index).ffill()
    rows.append(rec(straddle(dx, qe[qe >= "2003-06-01"], 5, 0), "LONG DX QE-5->QE (mechanism leg)"))
    rows.append(rec(straddle(dx, ome[ome >= "2003-06-01"], 5, 0), "LONG DX ME-5->ME (ordinary)"))
    show(rows, "beta + mechanism leg")
