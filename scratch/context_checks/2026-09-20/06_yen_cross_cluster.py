"""Five yen crosses in the 21-day bottom 5% at once, while USD/JPY sits in its top decile.

Friday: CHFJPY 21d rank 0.8, NZDJPY 2.0, EURJPY 2.8, CADJPY 3.2, GBPJPY 3.6,
AUDJPY 12.3, but JPY=X (USD/JPY) 5d rank 94.8 and 21d rank 22.6. The yen is bid
against everything except the dollar. The engine scored each cross alone; the
simultaneity is the cell.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (close_panel, fwd_ret, declusters, local_control,  # noqa
                       summarize, show, sign_test, cluster_note, era_split)

CROSSES = ["EURJPY=X", "GBPJPY=X", "CHFJPY=X", "NZDJPY=X", "CADJPY=X", "AUDJPY=X"]
T = CROSSES + ["JPY=X", "SPY", "^GSPC", "DX-Y.NYB", "^VIX", "EEM"]
px = close_panel(T).dropna(subset=CROSSES + ["JPY=X"])
print("coverage:", px.index.min().date(), "->", px.index.max().date(), "n", len(px))


def rank21(s):
    return ((s / s.shift(21) - 1.0).rolling(252).rank(pct=True) * 100)


rk = {c: rank21(px[c]) for c in CROSSES}
low = sum((rk[c] <= 5).astype(int) for c in CROSSES)
print("live ranks:", {c: round(float(rk[c].iloc[-1]), 1) for c in CROSSES})
print("live count of crosses at 21d rank <= 5:", int(low.iloc[-1]))
print("live USD/JPY 21d rank:", round(float(rank21(px['JPY=X']).iloc[-1]), 1))

for k in (4, 5):
    mask = low >= k
    trig = px.index[mask.reindex(px.index).fillna(False)]
    dec = declusters(trig, 21, px.index)
    print(f"\n### {k}+ of 6 yen crosses at 21d rank <= 5: {len(trig)} days, {len(dec)} episodes")
    print("  by year:", dict(pd.Series(1, index=dec).groupby(dec.year).sum()))
    if len(dec) < 5:
        print("  too thin, skipping forwards")
        continue
    ctrl = local_control(px.index, trig, 126)
    for h in (1, 5, 21):
        rows = []
        for name in ("EURJPY=X", "JPY=X", "SPY", "^VIX", "DX-Y.NYB", "EEM"):
            if name not in px:
                continue
            f = fwd_ret(px[name], h)
            v = f.reindex(dec).dropna().values
            if not len(v):
                continue
            r = summarize(v, f"{name} h{h}")
            up = int((v > 0).sum())
            r["rec"] = f"{up}-{len(v)-up}"
            r["sign_p"] = round(sign_test(max(up, len(v) - up), len(v)), 4)
            cv = f.reindex(ctrl).dropna().values
            r["ctrl_local"] = round(100 * cv.mean(), 3) if len(cv) else np.nan
            rows.append(r)
        show(rows, f"{k}+ crosses, h={h}")
    for name in ("EURJPY=X", "SPY"):
        v = fwd_ret(px[name], 5).reindex(dec).dropna()
        print(f"  {name} h5 {cluster_note(v.index, v.values)}")
        for e in era_split(v.index, v.values):
            print("     ", {kk: (round(x, 3) if isinstance(x, float) else x) for kk, x in e.items()
                            if kk in ("label", "n", "mean_pct", "hit", "t")})

print("\n### the live twist: crosses broadly weak WHILE USD/JPY is strong (yen bid vs all but USD)")
usd_strong = rank21(px["JPY=X"]) >= 50
m2 = (low >= 4)
print("  with USD/JPY 21d rank >= 50:", int((m2 & usd_strong).sum()), "days")
print("  with USD/JPY 21d rank <  50:", int((m2 & ~usd_strong).sum()), "days")
print("  NOTE live USD/JPY 21d rank is 22.6, so the live state is the SECOND bucket:")
print("  the yen is up against the crosses over 21d AND up against the dollar over 21d,")
print("  it is only the last FIVE sessions where the dollar has taken it back.")
