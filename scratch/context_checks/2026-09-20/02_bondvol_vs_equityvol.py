"""Bond vol rising hard while equity vol sleeps.

Friday's state: ^MOVE +13.2% over 21d (21d rank 82) and +5.79% on the session,
while ^VIX closed 14.81, 52% below its 52w high and 18.2% under its own 200d SMA.
Cell: MOVE 21d change in the top quintile AND VIX below its 200d SMA.
Forward: SPY, TLT, ^VIX, ^MOVE.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (close_panel, fwd_ret, declusters, local_control,  # noqa
                       summarize, show, sign_test, cluster_note, era_split)

T = ["^MOVE", "^VIX", "SPY", "TLT", "^GSPC", "IEF"]
px = close_panel(T).dropna(subset=["^MOVE", "^VIX", "SPY"])
print("coverage:", px.index.min().date(), "->", px.index.max().date(), "n", len(px))

move, vix, spy = px["^MOVE"], px["^VIX"], px["SPY"]
move21 = move / move.shift(21) - 1.0
vix_sma200 = vix.rolling(200).mean()
vix_vs_sma = vix / vix_sma200 - 1.0
move21_rank = move21.rolling(252).rank(pct=True) * 100

live_move21 = float(move21.iloc[-1] * 100)
live_rank = float(move21_rank.iloc[-1])
live_vix_vs = float(vix_vs_sma.iloc[-1] * 100)
print(f"LIVE 2026-09-18: MOVE 21d {live_move21:+.2f}% (rank {live_rank:.0f}), "
      f"VIX {vix.iloc[-1]:.2f} = {live_vix_vs:+.1f}% vs its 200d")

# cell: bond vol up a lot over 21d, equity vol below its own 200d
mask = (move21 >= 0.10) & (vix_vs_sma <= -0.10)
mask = mask & move21.notna() & vix_vs_sma.notna()
trig = px.index[mask.reindex(px.index).fillna(False)]
trig = trig[trig <= px.index[-1]]
print(f"\nraw trigger days: {len(trig)}")
dec = declusters(trig, 21, px.index)
print(f"declustered (21td): {len(dec)} episodes, {dec.min().date()} -> {dec.max().date()}")
print("  years:", dict(pd.Series(1, index=dec).groupby(dec.year).sum()))

ctrl = local_control(px.index, trig, 126)
for h in (1, 5, 10, 21):
    rows = []
    for name in ("SPY", "TLT", "^VIX", "^MOVE"):
        s = px[name]
        f = fwd_ret(s, h)
        v = f.reindex(dec).dropna().values
        r = summarize(v, f"{name} h{h}")
        up = int((v > 0).sum())
        r["rec"] = f"{up}-{len(v)-up}"
        r["sign_p"] = round(sign_test(max(up, len(v) - up), len(v)), 4)
        cv = f.reindex(ctrl).dropna().values
        r["ctrl_local"] = round(100 * cv.mean(), 3) if len(cv) else np.nan
        av = f.dropna().values
        r["ctrl_all"] = round(100 * av.mean(), 3)
        rows.append(r)
    show(rows, f"h={h} (declustered episodes)")

print("\n--- SPY detail at the horizons that matter ---")
for h in (5, 10, 21):
    v = fwd_ret(px["SPY"], h).reindex(dec).dropna()
    print(f"h{h}: {cluster_note(v.index, v.values)}")
    for e in era_split(v.index, v.values):
        print("   ", {k: (round(x, 3) if isinstance(x, float) else x) for k, x in e.items()
                      if k in ("label", "n", "mean_pct", "hit", "t")})

print("\n--- how often does bond vol lead equity vol higher? VIX h21 from each episode ---")
v21 = fwd_ret(px["^VIX"], 21).reindex(dec).dropna()
print(pd.DataFrame({"date": [str(d.date()) for d in v21.index],
                    "vix_h21_pct": (100 * v21.values).round(1)}).to_string(index=False))
