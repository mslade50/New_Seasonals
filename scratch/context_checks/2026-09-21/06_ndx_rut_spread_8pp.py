"""21d NDX-minus-RUT spread widened from 6.4pp (Friday, held Sunday) to 8.25pp today,
with the Nasdaq at +4.3% and the Russell at -3.9%. Re-cut at today's level: spread >= 8pp
with NDX 21d > 0 AND RUT 21d < 0 (both legs pointing opposite ways, the live shape).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import close_panel, fwd_ret, declusters, local_control, summarize, show, sign_test, cluster_note, era_split  # noqa

px = close_panel(["^NDX", "^RUT", "^GSPC", "^DJI"]).dropna(subset=["^NDX", "^RUT", "^GSPC"])
px = px[px.index >= "1999-01-01"]
r21 = lambda s: s / s.shift(21) - 1.0
n21, u21 = r21(px["^NDX"]), r21(px["^RUT"])
sp = n21 - u21
rank = sp.rolling(252).rank(pct=True) * 100
full_rank = (sp.rank(pct=True) * 100)
print(f"LIVE spread {100*sp.iloc[-1]:+.2f}pp  trailing-yr rank {rank.iloc[-1]:.1f}  full-history pctile {full_rank.iloc[-1]:.1f}")
print(f"last time spread >= live level: {sp[sp >= sp.iloc[-1]].index[-2].date() if (sp >= sp.iloc[-1]).sum() > 1 else 'never'}")

mask = (sp >= 0.08) & (n21 > 0) & (u21 < 0)
trig = px.index[mask.fillna(False).values]
trig = trig[trig < px.index[-1]]
dec = declusters(trig, 21, px.index)
print(f"raw days {len(trig)}, declustered episodes {len(dec)}")
print("episodes:", [str(d.date()) for d in dec])
ctrl = local_control(px.index, trig, 126)
for h in (5, 21):
    rows = []
    for name in ("^GSPC", "^NDX", "^RUT"):
        f = fwd_ret(px[name], h)
        v = f.reindex(dec).dropna().values
        r = summarize(v, f"{name} h{h}")
        up = int((v > 0).sum())
        r["rec"] = f"{up}-{len(v)-up}"
        r["ctl"] = round(100 * f.reindex(ctrl).dropna().mean(), 3)
        rows.append(r)
    s = fwd_ret(px["^NDX"], h) - fwd_ret(px["^RUT"], h)
    v = s.reindex(dec).dropna().values
    r = summarize(v, f"NDX-RUT h{h}")
    up = int((v > 0).sum())
    r["rec"] = f"{up}-{len(v)-up}"
    r["ctl"] = round(100 * s.reindex(ctrl).dropna().mean(), 3)
    rows.append(r)
    show(rows, f"h{h}")
s21 = (fwd_ret(px["^NDX"], 21) - fwd_ret(px["^RUT"], 21)).reindex(dec).dropna()
print("spread h21 concentration:", cluster_note(s21.index, s21.values))
show(era_split(s21.index, s21.values), "era spread h21")
det = pd.DataFrame({"sp": 100 * sp.reindex(dec), "gspc_h21": 100 * fwd_ret(px["^GSPC"], 21).reindex(dec),
                    "sp_h21": 100 * s21.reindex(dec)})
print(det.round(2).to_string())
