"""Follow-up to putcall_dial_study.py.

Q1: complacency signal (10d MA equity P/C, 252d pctile < 10) conjoined with
    SPY within 3% / 5% of 52w high (the first study used 2%).
Q2: fear tail as a LONG alpha signal: pctile > 85 crossed with SPY weakness
    - spy_rank: SPY close 252d rolling percentile < 15 (near lows)
    - r21_rank: SPY 21d return 252d rolling percentile < 15 (sharp selloff)
    Reported with the same signal_block stats (positive diff_mean = edge)
    plus episode-year clustering for the best cell.
"""
from __future__ import annotations

import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from build_signal_horizon_stats import signal_block  # noqa: E402
from scratch.putcall_dial_study import load_spy, rolling_pct_rank, fmt_block  # noqa: E402


def main() -> None:
    pc = pd.read_parquet(os.path.join(ROOT, "data", "cboe_putcall.parquet"))
    equity = pc["equity"].dropna().sort_index()
    spy = load_spy()
    cal = spy.index[(spy.index >= equity.index.min()) & (spy.index <= equity.index.max())]
    eq = equity.reindex(cal).ffill(limit=3)
    ma10 = eq.rolling(10, min_periods=10).mean()
    pct252 = rolling_pct_rank(ma10, 252)

    spy_al = spy.reindex(cal)
    hi52 = spy_al.rolling(252, min_periods=200).max()
    spy_rank = rolling_pct_rank(spy_al, 252)
    r21 = spy_al.pct_change(21)
    r21_rank = rolling_pct_rank(r21, 252)

    print("=== Q1: complacency + wider near-high bands ===")
    for band in (0.98, 0.97, 0.95):
        mask = (pct252 < 10) & (spy_al >= hi52 * band)
        fmt_block(f"pct252<10 & within {round((1-band)*100)}% of 52wh",
                  signal_block("x", mask.dropna(), spy))

    print("\n=== Q2: fear tail as long alpha ===")
    fear = pct252 > 85
    cells = {
        "pct252>85 alone": fear,
        "pct252>85 & spy_rank<15": fear & (spy_rank < 15),
        "pct252>85 & r21_rank<15": fear & (r21_rank < 15),
        "pct252>85 & spy_rank<25": fear & (spy_rank < 25),
        "pct252>85 & r21_rank<25": fear & (r21_rank < 25),
        "pct252>90 & r21_rank<15": (pct252 > 90) & (r21_rank < 15),
        "spy_rank<15 alone (control)": (spy_rank < 15).astype(bool),
        "r21_rank<15 alone (control)": (r21_rank < 15).astype(bool),
    }
    for name, mask in cells.items():
        fmt_block(name, signal_block("x", mask.dropna().astype(bool), spy))

    # episode-year clustering for the headline fear cell
    best = (fear & (r21_rank < 15)).astype(bool)
    starts = best & ~best.shift(1, fill_value=False)
    fwd21 = (spy.shift(-21) / spy - 1) * 100
    fwd63 = (spy.shift(-63) / spy - 1) * 100
    eps = pd.DataFrame({"fwd21": fwd21.reindex(best.index[starts]),
                        "fwd63": fwd63.reindex(best.index[starts])})
    eps["year"] = eps.index.year
    print("\npct252>85 & r21_rank<15 — episode starts by year:")
    print(eps.groupby("year").agg(n=("fwd21", "size"), fwd21=("fwd21", "mean"),
                                  fwd63=("fwd63", "mean")).round(2).to_string())


if __name__ == "__main__":
    main()
