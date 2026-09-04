"""Tonight's context brief carries the TLT full-August cell (N=24, +1.98%,
66.7%, t 2.39) with an era-break note. It's Aug 13: half the month is gone,
so the postable trade is REST-of-August, entered MOO tomorrow. Check the
tradeable slice per year (open of the first session after Aug 13 -> last
August close), era split, and a control (same-length window in every other
month). If the back half of August is where the edge lives and it's not one
era's artifact, an idea can ship; otherwise the stat runs alone or dies."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
from pitch_lab import era_split, show, sign_test, summarize
from pitch_lab import load_prices

px = load_prices(["TLT"])["TLT"]
close = px["Close"].dropna()
opn = px["Open"].reindex(close.index)
idx = close.index

rest_vals, rest_years = [], []
full_vals = []
for yr in range(int(idx[0].year), int(idx[-1].year) + 1):
    aug = idx[(idx.year == yr) & (idx.month == 8)]
    if len(aug) < 10:
        continue
    # full month: prior July close -> last August close
    pos0 = idx.searchsorted(aug[0])
    if pos0 > 0:
        full_vals.append(close.iloc[idx.searchsorted(aug[-1])] / close.iloc[pos0 - 1] - 1.0)
    # rest of month: first session strictly after Aug 13, entered at its OPEN
    back = aug[aug.day > 13]
    if len(back) < 5:
        continue
    e_open = opn.loc[back[0]]
    x_close = close.loc[back[-1]]
    if np.isnan(e_open) or np.isnan(x_close):
        continue
    rest_vals.append(x_close / e_open - 1.0)
    rest_years.append(pd.Timestamp(f"{yr}-08-14"))

rest = np.array(rest_vals)
full = np.array(full_vals)

# control: the same "day>13 open -> month-end close" window in every other month
ctrl = []
for yr in range(int(idx[0].year), int(idx[-1].year) + 1):
    for mo in range(1, 13):
        if mo == 8:
            continue
        mdays = idx[(idx.year == yr) & (idx.month == mo)]
        back = mdays[mdays.day > 13]
        if len(back) < 5:
            continue
        e, x = opn.loc[back[0]], close.loc[back[-1]]
        if np.isnan(e) or np.isnan(x):
            continue
        ctrl.append(x / e - 1.0)
ctrl = np.array(ctrl)

rows = []
for label, v in [("Aug 14->EOM (tradeable)", rest),
                 ("full August (brief's cell)", full),
                 ("other months, day14->EOM", ctrl)]:
    r = summarize(v, f"TLT {label}")
    wins = int((v > 0).sum())
    r["sign_p"] = round(sign_test(wins, len(v)), 4)
    r["record"] = f"{wins}/{len(v)}"
    rows.append(r)
show(rows, "TLT rest-of-August vs controls")

show(era_split(pd.DatetimeIndex(rest_years), rest), "rest-of-August era split")
print("per-year (yr, ret%):",
      [(d.year, round(100 * v, 2)) for d, v in zip(rest_years, rest)])
