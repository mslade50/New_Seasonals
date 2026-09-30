"""I1 round 1b: the most supportive series was ^HSI on its own calendar (17-9).
Does it survive its own placebo offset ladder and the other-quarter-end control?"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

REOPEN = {2000: "2000-10-09", 2001: "2001-10-08", 2002: "2002-10-08", 2003: "2003-10-08",
          2004: "2004-10-08", 2005: "2005-10-10", 2006: "2006-10-09", 2007: "2007-10-08",
          2008: "2008-10-06", 2009: "2009-10-09", 2010: "2010-10-08", 2011: "2011-10-10",
          2012: "2012-10-08", 2013: "2013-10-08", 2014: "2014-10-08", 2015: "2015-10-08",
          2016: "2016-10-10", 2017: "2017-10-09", 2018: "2018-10-08", 2019: "2019-10-08",
          2020: "2020-10-09", 2021: "2021-10-08", 2022: "2022-10-10", 2023: "2023-10-09",
          2024: "2024-10-08", 2025: "2025-10-09"}
HSI = load_prices(["^HSI"])["^HSI"]["Close"].dropna()
hc = HSI.index


def last_le(d):
    return int(hc.searchsorted(pd.Timestamp(d), side="right")) - 1


anc = {y: (last_le(f"{y}-09-30") - 3, int(hc.searchsorted(pd.Timestamp(r)))) for y, r in REOPEN.items()}
for era, yrs in (("all", list(anc)), ("ex2024", [y for y in anc if y != 2024]),
                 ("2015+ ex2024", [y for y in anc if y >= 2015 and y != 2024])):
    res = []
    for k in range(-10, 11):
        v = [HSI.iloc[x + k] / HSI.iloc[e + k] - 1 for y, (e, x) in anc.items() if y in yrs]
        res.append((k, 100 * np.mean(v), int((np.array(v) > 0).sum()), len(v)))
    m0 = [r for r in res if r[0] == 0][0]
    rk = 1 + sum(1 for r in res if r[1] > m0[1])
    print(f"HSI {era:13s} k=0 {m0[1]:+.2f}% ({m0[2]}-{m0[3]-m0[2]}) rank {rk} of 21 from TOP; ladder ex-0 "
          f"{np.mean([r[1] for r in res if r[0] != 0]):+.2f}% | " + " ".join(f"{k:+d}:{m:+.1f}" for k, m, _, _ in res))

rows = []
for mo in (3, 6, 9, 12):
    v = []
    for y in range(2000, 2026):
        a = last_le(pd.Timestamp(y, mo, 1) + pd.offsets.MonthEnd(0))
        if a + 6 < len(hc):
            v.append(HSI.iloc[a + 6] / HSI.iloc[a - 3] - 1)
    v = np.array(v)
    w = int((v > 0).sum())
    r = summarize(v, f"HSI QE month {mo:2d}: QE-3 -> QE+6")
    r["rec"] = f"{w}-{len(v)-w}"
    r["p"] = round(sign_test(w, len(v)), 4)
    rows.append(r)
d9 = (HSI.shift(-9) / HSI - 1).dropna()
rows.append({"label": "HSI own 9-session drift", "n": len(d9), "mean_pct": 100 * d9.mean(),
             "hit": 100 * (d9 > 0).mean()})
show(rows, "HSI same shape at every quarter end (2000-2025)")
