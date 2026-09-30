"""Does the yen-cross cluster mean anything when the tape is CALM?

06 found 4+ crosses at a 21d bottom-5% is followed by SPY +2.54% over 21 sessions
(20-6, sign p 0.005) and the VIX -13%. But the episode years are 2007, 2008, 2022 heavy:
yen crosses collapse in risk-off, so the cell is largely a panic-washout detector and the
bounce is the washout unwinding. Friday's tape is the opposite of a panic: VIX 14.81,
18% BELOW its 200d SMA, SPY 2.1% off its 52-week high.

Split the episodes by the vol state at the trigger.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (close_panel, fwd_ret, declusters, local_control,  # noqa
                       summarize, show, sign_test, cluster_note)

CROSSES = ["EURJPY=X", "GBPJPY=X", "CHFJPY=X", "NZDJPY=X", "CADJPY=X", "AUDJPY=X"]
px = close_panel(CROSSES + ["JPY=X", "SPY", "^VIX"]).dropna(subset=CROSSES + ["^VIX", "SPY"])
rank21 = lambda s: ((s / s.shift(21) - 1.0).rolling(252).rank(pct=True) * 100)
low = sum((rank21(px[c]) <= 5).astype(int) for c in CROSSES)

vix = px["^VIX"]
vix_vs_200 = vix / vix.rolling(200).mean() - 1.0
spy_dd = px["SPY"] / px["SPY"].rolling(252).max() - 1.0
print(f"LIVE: {int(low.iloc[-1])} crosses at 21d rank <= 5, "
      f"VIX {float(vix.iloc[-1]):.2f} = {100*float(vix_vs_200.iloc[-1]):+.1f}% vs its 200d, "
      f"SPY {100*float(spy_dd.iloc[-1]):+.2f}% from its 52w high")

mask = low >= 4
trig = px.index[mask.reindex(px.index).fillna(False)]
dec = declusters(trig, 21, px.index)
calm = vix_vs_200.reindex(dec) <= -0.05
stress = ~calm
print(f"\nepisodes: {len(dec)} total, calm (VIX >=5% under its 200d) {int(calm.sum())}, "
      f"stressed {int(stress.sum())}")
print("  calm episode dates:", [str(d.date()) for d in dec[calm.values]])
print("  stressed dates:    ", [str(d.date()) for d in dec[stress.values]])

ctrl = local_control(px.index, trig, 126)
for h in (1, 5, 21):
    rows = []
    for lbl, idx in (("CALM", dec[calm.values]), ("STRESSED", dec[stress.values]),
                     ("ALL", dec)):
        for name in ("SPY", "^VIX", "EURJPY=X"):
            f = fwd_ret(px[name], h)
            v = f.reindex(idx).dropna().values
            if len(v) < 3:
                continue
            r = summarize(v, f"{lbl} {name} h{h}")
            up = int((v > 0).sum())
            r["rec"] = f"{up}-{len(v)-up}"
            r["sign_p"] = round(sign_test(max(up, len(v) - up), len(v)), 4)
            cv = f.reindex(ctrl).dropna().values
            r["ctrl_local"] = round(100 * cv.mean(), 3) if len(cv) else np.nan
            rows.append(r)
    show(rows, f"h={h}")

print("\nVERDICT INPUT: compare the CALM SPY rows against the STRESSED ones.")
print("If the whole effect sits in the stressed bucket the cell does not describe Friday.")
for h in (5, 21):
    v = fwd_ret(px["SPY"], h).reindex(dec[calm.values]).dropna()
    if len(v):
        print(f"  calm SPY h{h}: {cluster_note(v.index, v.values)}")
