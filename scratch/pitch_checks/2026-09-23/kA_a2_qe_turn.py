"""A2 round 1: short SPY across the quarter turn, QE-5 close -> QE+5 close
(buyback-blackout straddle). Pre-specified sign SHORT. Anchor = the QE-5 close
itself (today's MOC), so the window is C[QE+5]/C[QE-5]-1 on SPY's calendar.
Controls: ordinary month-end straddles (ME-5 -> ME+5, the other 8 months),
all 10-session windows, and per-quarter split (the mechanism predicts ALL FOUR)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd


def month_ends(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(idx, index=idx)
    me = s.groupby([idx.year, idx.month]).max()
    me = pd.DatetimeIndex(me.values)
    return me[me < idx[-1]]  # drop the in-progress month (Sep 2026 not complete)


def straddle(c: pd.Series, anchors: pd.DatetimeIndex, pre: int, post: int):
    idx = c.index
    pos = pd.Series(range(len(idx)), index=idx)
    out = {}
    for d in anchors:
        p = pos[d]
        a, b = p - pre, p + post
        if a < 0 or b >= len(idx):
            continue
        out[d] = c.iloc[b] / c.iloc[a] - 1.0
    return pd.Series(out)


def rec(v: pd.Series, label: str) -> dict:
    s = summarize(v.values, label)
    if s["n"]:
        w = int((v > 0).sum())
        s["rec"] = f"{w}-{len(v) - w}"
        s["sign_p"] = round(sign_test(w, len(v)), 4)
    return s


if __name__ == "__main__":
    c = close_panel(["SPY"])["SPY"].dropna()
    c = c[c.index >= "1994-01-01"]
    me = month_ends(c.index)
    qe = me[me.month.isin([3, 6, 9, 12])]
    ome = me[~me.month.isin([3, 6, 9, 12])]
    PRE, POST = 5, 5
    # SHORT return = -(long return)
    sq = -straddle(c, qe, PRE, POST)
    so = -straddle(c, ome, PRE, POST)
    allw = -(c.shift(-10) / c - 1).dropna()
    rows = [rec(sq, "SHORT SPY QE-5->QE+5 (quarter turns)"),
            rec(so, "SHORT SPY ME-5->ME+5 (ordinary month turns)"),
            summarize(allw.values, "SHORT SPY all 10d windows")]
    for m in (3, 6, 9, 12):
        rows.append(rec(sq[sq.index.month == m], f"  QE month {m}"))
    rows += era_split(sq.index, sq.values, "2013-01-01")
    rows += era_split(sq.index, sq.values, "2018-01-01")
    mid = sq.index.year % 4 == 2
    rows.append(rec(sq[mid], "QE midterm yrs"))
    rows.append(rec(sq[~mid], "QE non-midterm"))
    rows.append(rec(so[so.index.year % 4 == 2], "ordinary ME midterm yrs"))
    show(rows, "A2 short SPY quarter-turn straddle")
    diff = sq.mean() - so.mean()
    se = np.sqrt(sq.var(ddof=1) / len(sq) + so.var(ddof=1) / len(so))
    print(f"QE minus ordinary-ME straddle (short) = {100*diff:+.3f}pp  welch t {diff/se:+.2f}")
    print(f"QE minus all-10d (short) = {100*(sq.mean()-allw.mean()):+.3f}pp")
    print("concentration:", cluster_note(sq.index, sq.values))
    print(f"cost: edge {100*sq.mean():.3f}% = {1e4*sq.mean():.1f} bp vs ~3 bp RT")

    # decompose: run-in (QE-5->QE) vs turn (QE->QE+5), QE vs ordinary
    rows = []
    for lbl, a, pre, post in [("run-in QE-5->QE", qe, 5, 0), ("run-in ME-5->ME", ome, 5, 0),
                              ("after QE->QE+5", qe, 0, 5), ("after ME->ME+5", ome, 0, 5)]:
        rows.append(rec(-straddle(c, a, pre, post), f"SHORT {lbl}"))
    show(rows, "decomposition (short sign)")
    print("\nlast 12 QE straddles (short):")
    print((100 * sq.tail(12)).round(2).to_string())
