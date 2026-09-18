"""CHECK B addendum (2026-09-09) — VLO's trigger is one session OLD tonight.

VLO's cell (close at a 252d high AND z10 >= 1.5) fired on 2026-09-08 AND again
on 2026-09-09. Under 5-session declustering the EPISODE is 09-08, whose lag-1
MOC entry was tonight's close, which has already printed. An order placed at
tomorrow's close is therefore a DAY-2 entry relative to the statistic reported
in 04, and quoting the episode number for it would be a timing cheat.

XLE, CVX and XOP do NOT have this problem: their previous trigger was in August
or March, so tonight is a fresh first-day episode for each of them (verified in
the console output below).

This script measures the entry that is actually available on VLO: trigger days
that are the SECOND OR LATER day of a cluster, lag-1 MOC. First-day triggers
are shown beside them, and the all-trigger day-level number too, so the timing
cost is explicit rather than assumed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa: F401,F403

import numpy as np
import pandas as pd

ASOF = pd.Timestamp("2026-09-09")
HS = (1, 3, 5, 10)
NAMES = ["VLO", "XLE", "CVX", "XOP"]
px = load_prices(NAMES)


def cell(df):
    c = df["Close"].astype(float)
    z = c.pct_change(10) / (c.pct_change().rolling(21).std() * np.sqrt(10))
    dh = c / c.rolling(252).max() - 1.0
    return (dh >= 0.0) & (z >= 1.5)


def stat(v, label):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    s = summarize(v, label)
    if s["n"] == 0:
        return s
    up, dn = int((v > 0).sum()), int((v < 0).sum())
    s["record"] = f"{up}-{dn}"
    s["sign_p_up"] = round(sign_test(up, len(v)), 4) if len(v) <= 1500 else None
    s.pop("sd_pct", None)
    return s


for t in NAMES:
    df = px[t][px[t].index <= ASOF]
    c = df["Close"].astype(float)
    trig = df.index[cell(df).fillna(False).values]
    epi = declusters(trig, 5, df.index)
    day2 = pd.DatetimeIndex([d for d in trig if d not in set(epi)])
    print("\n" + "=" * 96)
    print(f"{t}: {len(trig)} trigger days = {len(epi)} first-day episodes + "
          f"{len(day2)} day-2+ repeats")
    print(f"  tonight a trigger: {ASOF in set(trig)}   tonight a FIRST-day "
          f"episode: {ASOF in set(epi)}   previous trigger: "
          f"{trig[-2].date() if len(trig) > 1 else 'n/a'}")
    rows = []
    for h in HS:
        s = (c.shift(-(1 + h)) / c.shift(-1) - 1.0)
        valid = s.dropna().index
        rows.append(stat(s.reindex(pd.DatetimeIndex(epi).intersection(valid)).values,
                         f"first-day episodes h={h}"))
        rows.append(stat(s.reindex(pd.DatetimeIndex(day2).intersection(valid)).values,
                         f"  day-2+ repeats h={h}"))
        rows.append(stat(s.reindex(pd.DatetimeIndex(trig).intersection(valid)).values,
                         f"  all trigger days h={h}"))
        rows.append(stat(s.loc[valid].values, f"  CTRL all days h={h}"))
    show(rows, f"{t} lag-1 MOC by position in the cluster")
print("\nDONE.")
