"""A1 round 2 (robustness of the round-1 kill): definition neighbours
(EMA span 3/5/10 and raw, index distance 0.5/1/2/3%), ex-2015-08-17, midterm
episode list, and gate attribution on episodes. Short SPY, lag=1."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kA_a1_nyse_div import load_state  # noqa: E402

if __name__ == "__main__":
    rows = []
    base = None
    for span in (3, 5, 10, "raw"):
        px, ema, raw, dist = load_state(5 if span == "raw" else span)
        sig = raw if span == "raw" else ema
        ok = sig.notna() & dist.notna() & (px.index >= "1996-01-01")
        for dcut in (0.005, 0.01, 0.02, 0.03):
            m = ok & (sig < 0) & (dist <= dcut)
            for h in (5, 10):
                r = vehicle_ret(px, [("SPY", -1.0)], h)
                d = px.index[m.values & r.notna().values]
                e = declusters(d, 21, px.index)
                v = r.loc[e].values
                w = int((v > 0).sum())
                s = summarize(v, f"span={span} dist<={dcut:.3f} h={h}")
                s["rec"] = f"{w}-{len(v)-w}"
                s["sign_p"] = round(sign_test(w, len(v)), 3)
                ex = r.loc[e.difference([pd.Timestamp("2015-08-17")])].values
                s["ex0817_pct"] = round(100 * np.nanmean(ex), 3)
                rows.append(s)
    show(rows, "definition neighbours, 21td-declustered episodes, SHORT SPY")

    px, ema, raw, dist = load_state(5)
    ok = ema.notna() & dist.notna() & (px.index >= "1996-01-01")
    a1 = ok & (ema < 0) & (dist <= 0.01)
    r = vehicle_ret(px, [("SPY", -1.0)], 10)
    d = px.index[a1.values & r.notna().values]
    e = declusters(d, 21, px.index)
    mid = e[e.year % 4 == 2]
    print("\nmidterm-year episodes h=10 (short):",
          {str(x.date()): round(100 * r.loc[x], 2) for x in mid})
    # all day-level midterm dates by year
    print("midterm day-level days by year:",
          pd.Series(d[d.year % 4 == 2].year).value_counts().sort_index().to_dict())
