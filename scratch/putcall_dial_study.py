"""Equity put/call 10d-MA as a candidate risk-dial signal.

Pre-stated design (before any results were seen):
  PRIMARY: 10d MA of CBOE equity P/C, rolling 252d percentile < 10
           (complacency tail -> fragility ON). Percentile is PIT: rank of
           the current value within the trailing window including itself.
  Sensitivity grid (reported in full, no cherry-picking): pctile {5, 15},
           window 504d, conjunction with SPY within 2% of 52w high, and the
           opposite tail (pctile > 90) for completeness.
  Eval: exact signal_block methodology from scripts/build_signal_horizon_stats.py
           (SPY fwd 5/10/21/42/63d, diff_mean vs unconditional, Welch p,
           overlap-free episode mean/t).
  Marginal-value check: overlap with the incumbent composite
           (rd2_fragility_ts.parquet research series, 63d column >= 50) and
           forward returns on candidate-ON days the dial does NOT flag.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from build_signal_horizon_stats import signal_block, HORIZONS  # noqa: E402


def load_spy() -> pd.Series:
    mp = pd.read_parquet(os.path.join(ROOT, "data", "master_prices.parquet"),
                         columns=["ticker", "date", "Close"])
    spy = mp[mp["ticker"] == "SPY"].set_index("date")["Close"].sort_index()
    spy.index = pd.to_datetime(spy.index)
    if getattr(spy.index, "tz", None) is not None:
        spy.index = spy.index.tz_localize(None)
    return spy.dropna()


def rolling_pct_rank(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=window).apply(
        lambda w: (w <= w[-1]).mean() * 100.0, raw=True)


def fmt_block(name: str, block: dict | None) -> None:
    if block is None:
        print(f"{name}: no usable history")
        return
    print(f"\n{name}: n={block['n_events']} days, {block['n_episodes']} episodes, "
          f"{block['pct_active']}% active")
    print(f"  {'h':>4s} {'diff':>6s} {'sig':>6s} {'unc':>6s} {'hit%':>5s} "
          f"{'p':>6s} {'ep_mean':>8s} {'ep_t':>6s}")
    for h, v in block["horizons"].items():
        print(f"  {h:>4s} {v['diff_mean']:6.2f} {v['signal_mean']:6.2f} "
              f"{v['unconditional_mean']:6.2f} {v['hit_rate']:5.1f} "
              f"{v['p_value']:6.3f} "
              f"{v['episode_mean'] if v['episode_mean'] is not None else float('nan'):8.2f} "
              f"{v['episode_t'] if v['episode_t'] is not None else float('nan'):6.2f}")


def main() -> None:
    pc = pd.read_parquet(os.path.join(ROOT, "data", "cboe_putcall.parquet"))
    pc.index = pd.to_datetime(pc.index)
    equity = pc["equity"].dropna().sort_index()
    print(f"equity P/C: {len(equity)} rows {equity.index.min().date()} -> "
          f"{equity.index.max().date()}")
    spy = load_spy()

    # Align to SPY trading calendar; ffill scrape holes up to 3 sessions.
    cal = spy.index[(spy.index >= equity.index.min()) & (spy.index <= equity.index.max())]
    eq = equity.reindex(cal).ffill(limit=3)
    n_missing = int(eq.isna().sum())
    print(f"aligned to SPY calendar: {len(eq)} days, {n_missing} unfilled holes")

    ma10 = eq.rolling(10, min_periods=10).mean()
    spy_al = spy.reindex(cal)
    hi52 = spy_al.rolling(252, min_periods=200).max()
    near_high = spy_al >= hi52 * 0.98

    pct252 = rolling_pct_rank(ma10, 252)
    pct504 = rolling_pct_rank(ma10, 504)

    variants = {
        "PRIMARY pct252<10": pct252 < 10,
        "pct252<5": pct252 < 5,
        "pct252<15": pct252 < 15,
        "pct504<10": pct504 < 10,
        "pct252<10 & near 52wh": (pct252 < 10) & near_high,
        "opposite tail pct252>90": pct252 > 90,
    }

    for name, mask in variants.items():
        block = signal_block(name, mask.dropna(), spy)
        fmt_block(name, block)

    # Subsample match to the dial's own history window (2016+)
    print("\n=== 2016+ subsample (matches dial history depth) ===")
    m = (pct252 < 10).dropna()
    fmt_block("PRIMARY pct252<10, 2016+", signal_block(
        "sub", m[m.index >= "2016-01-01"], spy[spy.index >= "2015-01-01"]))

    # ---- marginal value vs incumbent composite ----
    frag_path = os.path.join(ROOT, "data", "rd2_fragility_ts.parquet")
    if os.path.exists(frag_path):
        frag = pd.read_parquet(frag_path)
        frag.index = pd.to_datetime(frag.index)
        if getattr(frag.index, "tz", None) is not None:
            frag.index = frag.index.tz_localize(None)
        dial = frag["63d"].dropna()
        cand = (pct252 < 10).dropna()
        both = pd.concat([cand.rename("cand"), (dial >= 50).rename("dial_hi"),
                          dial.rename("dial")], axis=1).dropna()
        both["cand"] = both["cand"].astype(bool)
        both["dial_hi"] = both["dial_hi"].astype(bool)
        print(f"\n=== overlap with incumbent dial (research recompute, "
              f"{both.index.min().date()} -> {both.index.max().date()}, "
              f"{len(both)} days) ===")
        on = both[both["cand"]]
        print(f"candidate ON: {len(on)} days | dial>=50 on those days: "
              f"{on['dial_hi'].mean()*100:.1f}% | mean dial {on['dial'].mean():.1f} "
              f"vs {both['dial'].mean():.1f} all-days")
        for h, days in HORIZONS.items():
            fwd = (spy.shift(-days) / spy - 1.0) * 100.0
            f = fwd.reindex(both.index)
            marg = both["cand"] & ~both["dial_hi"] & f.notna()
            base = f.notna()
            if marg.sum() >= 3:
                print(f"  {h}: cand-ON & dial<50 n={int(marg.sum())} "
                      f"fwd={f[marg].mean():6.2f} vs unc {f[base].mean():5.2f} "
                      f"(diff {f[marg].mean()-f[base].mean():6.2f})")
        # and correlation of the raw statistic with the dial
        ma10z = (ma10 - ma10.rolling(252).mean()) / ma10.rolling(252).std()
        j = pd.concat([(-ma10z).rename("neg_pc_z"), dial.rename("dial")],
                      axis=1).dropna()
        print(f"corr(-ma10 z, dial63) = {j['neg_pc_z'].corr(j['dial']):.2f}")
    else:
        print("rd2_fragility_ts.parquet not found — skipping overlap check")


if __name__ == "__main__":
    main()
