"""SPY forward returns by fragility-dial decile — monotonicity check.

Uses data/rd2_fragility.parquet 63d (5d-smoothed basis) and its 10d MA,
joined to SPY closes. Day-level (overlapping windows — fine for shape, not
for t-stats) plus a non-overlapping 21td-strided version as a sanity check.
Vintage caveat: rows before 2026-07-02 are recompute vintage.
"""
import os
import sys

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

frag = pd.read_parquet(os.path.join(_ROOT, "data", "rd2_fragility.parquet"))
s63 = frag["63d"].dropna().sort_index()
ma10 = s63.rolling(10, min_periods=1).mean()

mp = pd.read_parquet(os.path.join(_ROOT, "data", "master_prices.parquet"),
                     filters=[("ticker", "==", "SPY")])
spy = (mp.assign(date=pd.to_datetime(mp["date"]))
         .set_index("date")["Close"].sort_index())
spy = spy.reindex(s63.index).ffill()

fwd21 = spy.shift(-21) / spy - 1
fwd63 = spy.shift(-63) / spy - 1


def decile_table(score: pd.Series, label: str, stride: int = 1) -> None:
    df = pd.DataFrame({"score": score, "f21": fwd21, "f63": fwd63}).dropna()
    if stride > 1:
        df = df.iloc[::stride]
    df["decile"] = pd.qcut(df["score"], 10, labels=False, duplicates="drop")
    g = df.groupby("decile").agg(
        n=("score", "size"),
        score_lo=("score", "min"), score_hi=("score", "max"),
        f21_mean=("f21", "mean"), f21_med=("f21", "median"),
        f63_mean=("f63", "mean"), f63_med=("f63", "median"),
        f63_p10=("f63", lambda x: x.quantile(0.10)),
        pct_neg63=("f63", lambda x: (x < 0).mean()),
    )
    for c in ["f21_mean", "f21_med", "f63_mean", "f63_med", "f63_p10"]:
        g[c] = (g[c] * 100).round(2)
    g["pct_neg63"] = (g["pct_neg63"] * 100).round(0)
    g["score_lo"] = g["score_lo"].round(1)
    g["score_hi"] = g["score_hi"].round(1)
    print(f"\n== {label} (stride={stride}, N={len(df)}) ==")
    print(g.to_string())
    corr = df["score"].corr(df["f63"], method="spearman")
    print(f"spearman(score, fwd63) = {corr:.3f}")


pd.set_option("display.width", 160)
print(f"series: {s63.index.min().date()} -> {s63.index.max().date()}, "
      f"{len(s63)} rows (pre-2026-07-02 = recompute vintage)")
decile_table(ma10, "63d dial 10d-MA (the sizing statistic)")
decile_table(ma10, "63d dial 10d-MA, non-overlapping", stride=21)
decile_table(s63, "63d dial raw (5d-smoothed)")
