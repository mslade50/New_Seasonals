"""P5 is a TWO-SIDED trigger. Which side carries the number?

The engine fires `rank5_extreme` when the 5d return sits in the top OR bottom
5% of its trailing year and reports one pooled statistic: ^NDX n=606,
+0.236%, t=2.53, tagged solid and BH-passing. ^NDX is at rank 96.0 tonight,
so only the TOP side is live. If the pooled mean is a bottom-side rebound
effect then the cell says nothing about Monday, and the tag is on a number
that does not apply.

This is a definitional check, and the definition may turn out to be the
finding.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import fwd_ret, load_prices, sign_test, show, summarize  # noqa: E402

ASOF = pd.Timestamp("2026-08-07")
SUBJECTS = ["^NDX", "QQQ", "^IXIC", "SPY", "IWM"]
px = load_prices(SUBJECTS)

for ticker in SUBJECTS:
    close = px[ticker]["Close"].astype(float)
    close = close[close.index <= ASOF]
    rank = close.pct_change(5, fill_method=None).rolling(252).rank(pct=True) * 100
    f = fwd_ret(close, 1)
    valid = f.dropna().index

    top = pd.DatetimeIndex(rank.index[(rank >= 95).fillna(False).values])
    bot = pd.DatetimeIndex(rank.index[(rank <= 5).fillna(False).values])

    def cell(idx: pd.DatetimeIndex, label: str) -> dict:
        sel = idx.intersection(valid)
        vals = f.loc[sel].values
        row = summarize(vals, label)
        if row["n"]:
            up = int((vals > 0).sum())
            row["record"] = f"{up}-{row['n'] - up}"
            row["sign_p"] = round(sign_test(max(up, row["n"] - up), row["n"]), 4)
        return row

    rows = [cell(top.union(bot), "POOLED (what the engine reports)"),
            cell(top, "TOP 5% only  <- live tonight"),
            cell(bot, "BOTTOM 5% only"),
            cell(pd.DatetimeIndex(valid), "CTRL all days")]
    show(rows, f"{ticker}: 5d rank extreme, split by side "
               f"(now at rank {rank.iloc[-1]:.1f})")

    # Era check on the live side only.
    sel = top.intersection(valid)
    vals = f.loc[sel].values
    pre = np.asarray(sel) < np.datetime64(pd.Timestamp("2018-01-01"))
    show([summarize(vals[pre], "top-side pre-2018"),
          summarize(vals[~pre], "top-side 2018+")], f"  {ticker} era split")
