"""August Mondays: QQQ hit 66.1% (76-39, sign p 0.0004) on a bare calendar cell.

A two-thirds hit rate with a mean of only +0.146% is the signature of a cell
that wins small and often and loses large and rarely, so the questions are
whether the mean survives the losers, whether it is era-stable, and whether
the ^VIX +2.34% quoted alongside it is anything other than August 2011, 2015
and 2024.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from build_context_state import anchors_before  # noqa: E402
from pitch_lab import cluster_note, fwd_ret, load_prices, sign_test, show, summarize  # noqa: E402

ASOF = pd.Timestamp("2026-08-07")
SUBJECTS = ["QQQ", "SPY", "IWM", "^VIX", "TLT"]
px = load_prices(SUBJECTS)
ref = px["SPY"].index[px["SPY"].index <= ASOF]

target = (ref.weekday.to_numpy() == 0) & (ref.month.to_numpy() == 8)
anchors = anchors_before(ref, target)
print(f"anchors (session before an August Monday): {len(anchors)}")

# Control 1: Mondays in every OTHER month. Control 2: every August session.
aug_any = anchors_before(ref, ref.month.to_numpy() == 8)
mon_any = anchors_before(ref, ref.weekday.to_numpy() == 0)
mon_not_aug = mon_any.difference(anchors)

for ticker in SUBJECTS:
    close = px[ticker]["Close"].astype(float)
    f = fwd_ret(close, 1)
    valid = f.dropna().index

    def cell(idx: pd.DatetimeIndex, label: str) -> dict:
        sel = idx.intersection(valid)
        vals = f.loc[sel].values
        row = summarize(vals, label)
        if row["n"]:
            up = int((vals > 0).sum())
            row["record"] = f"{up}-{row['n'] - up}"
            row["sign_p"] = round(sign_test(max(up, row["n"] - up), row["n"]), 4)
        return row

    idx = anchors.intersection(valid)
    vals = f.loc[idx].values
    pre = np.asarray(idx) < np.datetime64(pd.Timestamp("2018-01-01"))
    rows = [cell(anchors, "August Mondays"),
            cell(mon_not_aug, "CTRL Mondays, other months"),
            cell(aug_any, "CTRL every August session"),
            cell(pd.DatetimeIndex(valid), "CTRL all days"),
            summarize(vals[pre], "August Mondays pre-2018"),
            summarize(vals[~pre], "August Mondays 2018+")]
    show(rows, f"{ticker}: the August Monday cell against its controls")
    print(f"  {cluster_note(pd.DatetimeIndex(idx), vals)}")
    worst = np.argsort(vals)[:3]
    print("  three worst: " + ", ".join(
        f"{pd.Timestamp(idx[i]).date()} {100 * vals[i]:+.2f}%" for i in worst))
