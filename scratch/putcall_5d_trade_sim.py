"""Non-overlapping short-horizon trade sim for the equity P/C 10d-MA tails.

Trade rules: when the cell is ON and no position is open, enter at that
close, exit hold_days sessions later; eligible to re-enter immediately after
exit (captures persistent-episode re-fires without day-level overlap).
Stats per cell: N, avg/median return, win rate, t vs 0, t vs the
unconditional same-horizon mean (matched non-overlapping baseline), worst,
skew. Bear cells graded as SHORTS (alpha = negative raw return).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as sps

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from scratch.putcall_dial_study import load_spy, rolling_pct_rank  # noqa: E402


def trade_sim(mask: pd.Series, spy: pd.Series, hold: int) -> pd.Series:
    """Entry-close -> exit-close returns (%), non-overlapping."""
    px = spy.reindex(mask.index)
    m = mask.fillna(False).to_numpy()
    rets, dates = [], []
    i, n = 0, len(m)
    while i < n - hold:
        if m[i] and np.isfinite(px.iloc[i]) and np.isfinite(px.iloc[i + hold]):
            rets.append((px.iloc[i + hold] / px.iloc[i] - 1) * 100)
            dates.append(mask.index[i])
            i += hold
        else:
            i += 1
    return pd.Series(rets, index=pd.DatetimeIndex(dates))


def report(name: str, tr: pd.Series, base_mean: float, side: str) -> None:
    if len(tr) < 5:
        print(f"{name:38s} n<5, skipped")
        return
    mu = tr.mean()
    t0 = mu / (tr.std() / np.sqrt(len(tr)))
    tb = (mu - base_mean) / (tr.std() / np.sqrt(len(tr)))
    alpha = -mu if side == "short" else mu - base_mean
    print(f"{name:38s} n={len(tr):4d} avg={mu:6.2f} med={tr.median():6.2f} "
          f"win={100*(tr>0).mean():5.1f}% t0={t0:5.2f} tB={tb:5.2f} "
          f"worst={tr.min():6.2f} skew={tr.skew():5.2f} alpha={alpha:6.2f}")


def main() -> None:
    pc = pd.read_parquet(os.path.join(ROOT, "data", "cboe_putcall.parquet"))
    equity = pc["equity"].dropna().sort_index()
    spy = load_spy()
    cal = spy.index[(spy.index >= equity.index.min()) & (spy.index <= equity.index.max())]
    eq = equity.reindex(cal).ffill(limit=3)
    ma10 = eq.rolling(10, min_periods=10).mean()
    pct = rolling_pct_rank(ma10, 252)

    spy_al = spy.reindex(cal)
    hi52 = spy_al.rolling(252, min_periods=200).max()
    spy_rank = rolling_pct_rank(spy_al, 252)
    r21_rank = rolling_pct_rank(spy_al.pct_change(21), 252)

    bear = {
        "SHORT pct<10": pct < 10,
        "SHORT pct<5": pct < 5,
        "SHORT pct<10 & within2% 52wh": (pct < 10) & (spy_al >= hi52 * 0.98),
        "SHORT pct<5 & within2% 52wh": (pct < 5) & (spy_al >= hi52 * 0.98),
    }
    bull = {
        "LONG pct>85 & spy_rank<15": (pct > 85) & (spy_rank < 15),
        "LONG pct>85 & spy_rank<25": (pct > 85) & (spy_rank < 25),
        "LONG pct>85 & r21_rank<15": (pct > 85) & (r21_rank < 15),
        "LONG pct>90 & r21_rank<15": (pct > 90) & (r21_rank < 15),
        "CTRL spy_rank<25 alone": spy_rank < 25,
        "CTRL r21_rank<15 alone": r21_rank < 15,
    }

    for hold in (5, 10):
        # matched baseline: every hold-days non-overlapping return
        base = trade_sim(pd.Series(True, index=cal), spy, hold)
        print(f"\n===== hold {hold}d  (baseline avg {base.mean():.2f}%, "
              f"win {100*(base>0).mean():.1f}%, n={len(base)}) =====")
        for name, mask in bear.items():
            report(name, trade_sim(mask.astype(bool), spy, hold), base.mean(), "short")
        for name, mask in bull.items():
            report(name, trade_sim(mask.astype(bool), spy, hold), base.mean(), "long")

    # yearly detail for the two headline cells at 5d
    for label, mask in [("LONG pct>85 & spy_rank<25", (pct > 85) & (spy_rank < 25)),
                        ("SHORT pct<5 & within2% 52wh",
                         (pct < 5) & (spy_al >= hi52 * 0.98))]:
        tr = trade_sim(mask.astype(bool), spy, 5)
        d = pd.DataFrame({"ret": tr, "year": tr.index.year})
        print(f"\n{label} — 5d trades by year:")
        print(d.groupby("year")["ret"].agg(["count", "mean", "min"]).round(2).to_string())


if __name__ == "__main__":
    main()
