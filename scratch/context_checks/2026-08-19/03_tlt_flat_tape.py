"""TLT gained 1.67% (about 2.1 ATR) while SPY moved +0.21% and the 10y-5y curve
flattened 2sd. The engine dropped TLT from the 2-ATR trigger on the per-trigger
cap, so recompute it, then cross it with the flat-equity condition."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    close_panel, cluster_note, declusters, era_split, fwd_ret, local_control,
    load_prices, sign_test, summarize,
)

TK = ["TLT", "SPY", "^TNX", "^FVX", "IEF"]
px = close_panel(TK).dropna(subset=["TLT", "SPY"])
tlt, spy = px["TLT"], px["SPY"]
rt, rs = tlt.pct_change(), spy.pct_change()

raw = load_prices(["TLT"])["TLT"]
tr = pd.concat([
    raw["High"] - raw["Low"],
    (raw["High"] - raw["Close"].shift()).abs(),
    (raw["Low"] - raw["Close"].shift()).abs()], axis=1).max(axis=1)
atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
move_atr = ((raw["Close"] - raw["Close"].shift()) / atr.shift()).reindex(px.index)

print("TLT", tlt.index[0].date(), "->", tlt.index[-1].date(), "n", len(tlt))
print("today  TLT", round(float(rt.iloc[-1]) * 100, 2), "% =",
      round(float(move_atr.iloc[-1]), 2), "ATR | SPY",
      round(float(rs.iloc[-1]) * 100, 2), "%")

up2 = (move_atr >= 2).fillna(False)
flat = (rs.abs() < 0.005).fillna(False)
cells = {
    "TLT >= 2 ATR up": up2,
    "TLT >= 2 ATR up AND |SPY| < 0.5%": up2 & flat,
    "TLT >= 2 ATR up AND SPY down": up2 & (rs < 0).fillna(False),
}
for lab, mask in cells.items():
    idx = pd.DatetimeIndex([d for d in px.index[mask] if d < px.index[-1]])
    trig = declusters(idx, 5, px.index)
    print(f"\n=== {lab}  (raw {len(idx)}, declustered {len(trig)}) ===")
    for h in (1, 5, 21):
        f = fwd_ret(tlt, h).reindex(trig).dropna()
        if len(f) == 0:
            continue
        s = summarize(f.values)
        nup = int((f > 0).sum())
        allc = summarize(fwd_ret(tlt, h).dropna().values)
        loc = summarize(fwd_ret(tlt, h).reindex(
            local_control(px.index, trig, 126)).dropna().values)
        print(f"  TLT h{h:<3} n={s['n']:<3} mean={s['mean_pct']:+.3f}%  "
              f"{nup}-{len(f)-nup} up  t={s['t']:+.2f}  "
              f"sign_p={sign_test(nup, len(f)):.4f}  | all {allc['mean_pct']:+.3f}%  "
              f"local {loc['mean_pct']:+.3f}%")
    fs = fwd_ret(spy, 5).reindex(trig).dropna()
    if len(fs):
        s = summarize(fs.values)
        nup = int((fs > 0).sum())
        allc = summarize(fwd_ret(spy, 5).dropna().values)
        print(f"  SPY h5   n={s['n']:<3} mean={s['mean_pct']:+.3f}%  "
              f"{nup}-{len(fs)-nup} up  t={s['t']:+.2f}  | all {allc['mean_pct']:+.3f}%")
    f5 = fwd_ret(tlt, 5).reindex(trig).dropna()
    if len(f5) >= 6:
        print("  era h5:", [(e["label"], e["n"], round(e["mean_pct"], 3))
                            for e in era_split(f5.index, f5.values)])
        print("  concentration h5:", cluster_note(f5.index, f5.values))
    if len(trig) <= 20:
        print("  episodes:", [d.date().isoformat() for d in trig])

# the curve leg: 10y-5y daily change vs its own trailing 1y sd
print("\n=== 10y-5y flattening, and the same day with TLT 2 ATR up ===")
cur = (px["^TNX"] - px["^FVX"]).dropna()
dc = cur.diff()
z = dc / dc.rolling(252).std()
print("today curve", round(float(cur.iloc[-1]), 3), "change",
      round(float(dc.iloc[-1]), 3), "= z", round(float(z.iloc[-1]), 2))
flt = (z <= -2).fillna(False)
for lab, mask in (("curve flattens 2sd", flt),
                  ("curve flattens 2sd AND TLT >= 2 ATR up",
                   flt & up2.reindex(flt.index).fillna(False))):
    idx = pd.DatetimeIndex([d for d in flt.index[mask] if d < px.index[-1]])
    trig = declusters(idx, 5, px.index)
    print(f"\n--- {lab}  (raw {len(idx)}, declustered {len(trig)}) ---")
    for h in (1, 5, 21):
        f = fwd_ret(tlt, h).reindex(trig).dropna()
        if len(f) == 0:
            continue
        s = summarize(f.values)
        nup = int((f > 0).sum())
        print(f"  TLT h{h:<3} n={s['n']:<3} mean={s['mean_pct']:+.3f}%  "
              f"{nup}-{len(f)-nup} up  t={s['t']:+.2f}  sign_p={sign_test(nup, len(f)):.4f}")
    fc = fwd_ret(cur, 5).reindex(trig).dropna()
    if len(fc):
        vals = (cur.shift(-5) - cur).reindex(trig).dropna()
        nflat = int((vals < 0).sum())
        print(f"  curve itself, next 5d: mean {float(vals.mean()):+.4f}  "
              f"{nflat} of {len(vals)} flattened further")
    if len(trig) <= 25:
        print("  episodes:", [d.date().isoformat() for d in trig])
