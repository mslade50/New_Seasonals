"""Reply ammunition: crude (USO) is 32.6% above its 200d SMA tonight, +90% over
a year. How often has USO been this stretched, and what followed? First day
of each episode only (declustered 21 td), forward 10 and 21 sessions, against
the all-days base and a local control. Description, not a trade.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    cluster_note, declusters, era_split, fwd_lag, load_prices, local_control,
    sign_test, summarize,
)

warnings.filterwarnings("ignore")
ASOF = pd.Timestamp("2026-09-03")
d = load_prices(["USO"])["USO"]
c = d["Close"].dropna()
sma = c.rolling(200).mean()
ext = 100 * (c / sma - 1)
print(f"USO {c.iloc[-1]:.2f} on {c.index[-1].date()}  {ext.iloc[-1]:+.1f}% vs 200d SMA  "
      f"252d {100*(c.iloc[-1]/c.iloc[-253]-1):+.1f}%  first bar {c.index[0].date()}")
valid = ext.dropna()
print(f"days with ext >= 30%: {int((valid >= 30).sum())} of {len(valid)}  "
      f"(pctile of tonight: {100*(valid <= ext.iloc[-1]).mean():.1f})")
print("by year:", valid[valid >= 30].groupby(valid[valid >= 30].index.year).size().to_dict())

for thr in (30, 25):
    trig = valid.index[(valid >= thr).values]
    trig = trig[trig < ASOF]
    epi = declusters(trig, 21, c.index)
    print(f"\n=== ext >= {thr}%: {len(trig)} days, {len(epi)} episodes (first day, dc21) ===")
    print("   ", [e.date().isoformat() for e in epi])
    for h in (5, 10, 21):
        f = fwd_lag(c, h, 0)
        v = f.reindex(epi).dropna()
        st = summarize(v.values)
        nup = int((v > 0).sum())
        loc = f.reindex(local_control(c.index, v.index, 126)).dropna()
        print(f"  h{h:<3} n={st['n']:<3} mean={st['mean_pct']:+.2f}%  med={st['median_pct']:+.2f}%  "
              f"{nup}-{len(v)-nup}  sp={sign_test(nup, len(v)):.3f}  | all {100*f.dropna().mean():+.2f}% "
              f"hit {100*(f.dropna()>0).mean():.1f}%  local {100*loc.mean():+.2f}%  | worst {st['worst_pct']:+.1f}%")
        if h == 21:
            print("     era:", [(e["label"], e["n"], round(e.get("mean_pct", np.nan), 2), round(e.get("hit", np.nan), 1))
                               for e in era_split(v.index, v.values)])
            print("     concentration:", cluster_note(v.index, v.values))
            print("     all:", [(dd.date().isoformat(), round(100 * x, 1)) for dd, x in v.items()])
