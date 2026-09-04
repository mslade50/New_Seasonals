"""Last night's context brief had an anecdote cell: EWZ down-streak running
while SPY sits at a 52-week high, N=13, next-session mean -3.36%?? (looks
like a longer horizon), hit 23.1. Candidate SHORT idea. Reconstruct it
honestly: EWZ down >= 4 of the last 5 sessions AND 5d return <= -4% AND SPY
within 0.5% of its 252d closing high. Declustered 10td. Short leg entered
MOO t+1, held to close t+5 (report the SHORT's return = -(long return)).
First: does the trigger even fire on tonight's bar? If not, no idea."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
from pitch_lab import declusters, era_split, show, sign_test, summarize
from pitch_lab import load_prices

H = 5
pxs = load_prices(["EWZ", "SPY"])
ewz, spy = pxs["EWZ"], pxs["SPY"]
close = ewz["Close"].dropna()
opn = ewz["Open"].reindex(close.index)
spy_c = spy["Close"].reindex(close.index).ffill()
idx = close.index

r1 = close.pct_change()
down4of5 = (r1 < 0).rolling(5).sum() >= 4
r5 = close.pct_change(5)
spy_high = spy_c / spy_c.rolling(252).max() >= 0.995
mask = down4of5 & (r5 <= -0.04) & spy_high

today = idx[-1]
print(f"tonight ({today.date()}): down4of5={bool(down4of5.iloc[-1])} "
      f"r5={100*r5.iloc[-1]:.2f}% spy_at_high={bool(spy_high.iloc[-1])} "
      f"-> trigger={bool(mask.iloc[-1])}")

trig = declusters(idx[mask.fillna(False)], 10, idx)
short_oc, kept = [], []
for d in trig:
    pos = idx.searchsorted(d)
    if pos + H >= len(idx):
        continue
    e = opn.iloc[pos + 1]
    if np.isnan(e):
        continue
    short_oc.append(-(close.iloc[pos + H] / e - 1.0))
    kept.append(d)
short_oc = np.array(short_oc)
kept = pd.DatetimeIndex(kept)

# control: same EWZ washout WITHOUT the SPY-at-high condition
mask_ctrl = down4of5 & (r5 <= -0.04) & ~spy_high
ctrl_short = []
for d in declusters(idx[mask_ctrl.fillna(False)], 10, idx):
    pos = idx.searchsorted(d)
    if pos + H >= len(idx):
        continue
    e = opn.iloc[pos + 1]
    if np.isnan(e):
        continue
    ctrl_short.append(-(close.iloc[pos + H] / e - 1.0))
ctrl_short = np.array(ctrl_short)

rows = []
for label, v in [("SHORT, washout + SPY 52wh", short_oc),
                 ("SHORT, washout w/o SPY high", ctrl_short)]:
    r = summarize(v, f"EWZ {label}")
    wins = int((v > 0).sum())
    r["sign_p"] = round(sign_test(wins, len(v)), 4)
    r["record"] = f"{wins}/{len(v)}"
    rows.append(r)
show(rows, f"EWZ streak short, declustered N={len(kept)}")
if len(kept):
    show(era_split(kept, short_oc), "era split (short leg)")
    print("trigger dates:", [str(d.date()) for d in kept])
