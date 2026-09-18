"""^FCHI 21d return in the bottom 5% of its year while ^GDAXI is mid-pack.

The engine cell BH-passed (n=400, hit 57.2%, sign p 0.0022) but reports
era_stable = False, so it publishes with the era split stated or not at all.
The live state is a France-specific one: ^FCHI 21d rank 4.8 against ^GDAXI
34.9, so test the SPREAD version too.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (close_panel, fwd_ret, summarize, sign_test, era_split,  # noqa
                       cluster_note, pct_rank, declusters, local_control)

px = close_panel(["^FCHI", "^GDAXI", "^GSPC"])
fchi, dax = px["^FCHI"].dropna(), px["^GDAXI"].dropna()
idx = fchi.index
rf, rd = pct_rank(fchi, 21, 252), pct_rank(dax, 21, 252)
print(f"^FCHI 21d rank {rf.iloc[-1]:.1f} (21d {100*fchi.pct_change(21).iloc[-1]:+.2f}%), "
      f"^GDAXI 21d rank {rd.iloc[-1]:.1f} ({100*dax.pct_change(21).iloc[-1]:+.2f}%)")


def line(label, dates, s, h):
    d = pd.DatetimeIndex([x for x in dates if x in s.index])
    r = fwd_ret(s, h).reindex(d).dropna()
    if len(r) < 3:
        print(f"  {label:46} h{h:<3} n={len(r)} thin"); return None
    v = r.values; up = int((v > 0).sum()); st = summarize(v, label)
    print(f"  {label:46} h{h:<3} n={len(v):5d} mean={st['mean_pct']:+7.3f}% "
          f"med={st['median_pct']:+7.3f}% {up}-{len(v)-up} hit={st['hit']:5.1f}% "
          f"t={st['t']:+5.2f} signp={sign_test(up, len(v)):.4f}")
    return r


for nm, m in [("^FCHI 21d rank <= 5 (engine cell)", rf <= 5),
              ("  ... while ^GDAXI rank > 25 (tonight)", (rf <= 5) & (rd > 25))]:
    trig = pd.DatetimeIndex([d for d in m.index[m.fillna(False)] if d < idx[-1]])
    dec = declusters(trig, 10, idx)
    print(f"\n{nm}: {len(trig)} sessions, {len(dec)} declustered")
    if len(dec) >= 3:
        print(f"  years {sorted(set(d.year for d in dec))}")
    for h in (1, 5, 21):
        line(nm.strip(), dec, fchi, h)
    r = fwd_ret(fchi, 5).reindex(pd.DatetimeIndex([d for d in dec if d in fchi.index])).dropna()
    if len(r) >= 6:
        print("  era:", [f"{e['label']} n={e['n']} mean={e['mean_pct']:+.2f}% hit={e['hit']:.1f}%"
                         for e in era_split(r.index, r.values)])
        print("  concentration:", cluster_note(r.index, r.values, 2))
    ctrl = local_control(idx, dec, 126)
    for h in (1, 5):
        line("  local +/-126td control", ctrl, fchi, h)
