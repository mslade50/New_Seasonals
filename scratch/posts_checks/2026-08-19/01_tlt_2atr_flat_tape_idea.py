"""Idea check: long TLT after a >=2 ATR up close on a flat equity session.

Tonight's state: TLT +1.67% (~2.1 Wilder-ATR) while SPY moved +0.21%.
The context brief's lag-0 cell ran 11-2 at +1.56% (h5). A POST idea has to
survive the pitch doctrine instead: lag-1 entry (we can only buy tomorrow),
5 td declustering, local control, era split, concentration. Also freeze the
Wilder-14 ATR and ref_close for the idea spec.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    close_panel, cluster_note, declusters, era_split, fwd_lag, fwd_ret,
    load_prices, local_control, sign_test, summarize, wilder_atr,
)

px = close_panel(["TLT", "SPY"]).dropna()
tlt, spy = px["TLT"], px["SPY"]
rt, rs = tlt.pct_change(), spy.pct_change()

raw = load_prices(["TLT"])["TLT"]
atr = pd.Series(wilder_atr(raw["High"], raw["Low"], raw["Close"]), index=raw.index)
move_atr = ((raw["Close"] - raw["Close"].shift()) / atr.shift()).reindex(px.index)

print("TLT panel", px.index[0].date(), "->", px.index[-1].date(), "n", len(px))
print("tonight: TLT %+.2f%% = %.2f ATR | SPY %+.2f%% | Wilder-14 ATR %.4f | ref_close %.2f"
      % (rt.iloc[-1] * 100, move_atr.iloc[-1], rs.iloc[-1] * 100,
         atr.iloc[-1], raw["Close"].iloc[-1]))

up2 = (move_atr >= 2).fillna(False)
flat = (rs.abs() < 0.005).fillna(False)
mask = up2 & flat
idx = pd.DatetimeIndex([d for d in px.index[mask] if d < px.index[-1]])
trig = declusters(idx, 5, px.index)
print(f"\ntrigger: TLT >= 2 ATR up AND |SPY| < 0.5%  raw {len(idx)}, declustered {len(trig)}")
print("episodes:", [d.date().isoformat() for d in trig])

for h in (1, 3, 5, 10):
    f = fwd_lag(tlt, h).reindex(trig).dropna()          # entry NEXT close, the honest one
    f0 = fwd_ret(tlt, h).reindex(trig).dropna()         # lag-0 contrast (the brief's basis)
    if not len(f):
        continue
    s = summarize(f.values)
    nup = int((f > 0).sum())
    allc = summarize(fwd_lag(tlt, h).dropna().values)
    loc = summarize(fwd_lag(tlt, h).reindex(local_control(px.index, trig, 126)).dropna().values)
    print(f"  lag1 h{h:<2} n={s['n']:<3} mean={s['mean_pct']:+.3f}%  {nup}-{len(f)-nup} up  "
          f"t={s['t']:+.2f}  sign_p={sign_test(nup, len(f)):.4f}  "
          f"| all {allc['mean_pct']:+.3f}%  local {loc['mean_pct']:+.3f}%  "
          f"| lag0 {summarize(f0.values)['mean_pct']:+.3f}%")

f5 = fwd_lag(tlt, 5).reindex(trig).dropna()
print("\nera h5 lag1:", [(e["label"], e["n"], round(e["mean_pct"], 3))
                         for e in era_split(f5.index, f5.values)])
print("concentration h5 lag1:", cluster_note(f5.index, f5.values))

# the kill that took out every recent TLT seasonal: does the cell survive
# the state TLT is actually in (within a few % of a 52-week low)?
lo252 = raw["Close"].rolling(252).min().reindex(px.index)
near_low = (tlt / lo252 - 1) < 0.04
sub = trig[near_low.reindex(trig).fillna(False)]
f5s = fwd_lag(tlt, 5).reindex(sub).dropna()
print(f"\nsubset near a 52w low (<4% above): n={len(f5s)}", end="")
if len(f5s):
    nup = int((f5s > 0).sum())
    print(f"  mean={summarize(f5s.values)['mean_pct']:+.3f}%  {nup}-{len(f5s)-nup} up"
          f"  sign_p={sign_test(nup, len(f5s)):.4f}")
    print("  dates:", [d.date().isoformat() for d in f5s.index])
else:
    print()
