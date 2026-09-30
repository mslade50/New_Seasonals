"""X1 round 1c: is the one live-looking neighbour (+1.50% & vol up, h=2) a watchlist
entry or noise? It was found by MY 8-variant x 3-horizon neighbour sweep (24 cells),
so it owes a search charge. Concentration, era, drop-best-2, dose ladder at h=2."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

P = load_prices(["USDMXN=X", "^MOVE", "^VIX"])
fx = P["USDMXN=X"]["Close"].dropna()
fx = fx[fx > 0]
cal = fx.index
mxn = 1.0 / fx
u1 = fx.pct_change()


def vchg(t):
    s = P[t]["Close"].dropna()
    return (s / s.shift(1) - 1).reindex(cal)


volup = (vchg("^MOVE") > 0) | (vchg("^VIX") > 0)
rows = []
for h in (1, 2, 3, 5):
    r = fwd_lag(mxn, h)
    base = r.dropna()
    ph = float((base > 0).mean())
    for thr in (0.0125, 0.0135, 0.0145, 0.015, 0.0175, 0.02):
        d = cal[((u1 >= thr) & volup).values & r.notna().values]
        e = declusters(d, 10, cal)
        v = r.loc[e].values
        w = int((v > 0).sum())
        srt = np.sort(v)
        row = summarize(v, f"h={h} >= +{100*thr:.2f}% & vol up")
        row.update({"edge_pct": round(100 * (v.mean() - base.mean()), 3), "rec": f"{w}-{len(v)-w}",
                    "p_vs_base": round(sign_test(w, len(v), ph), 4),
                    "drop_best2_pct": round(100 * srt[:-2].mean(), 3)})
        m18 = e >= "2018-01-01"
        row["pre18"] = round(100 * v[~m18].mean(), 3)
        row["post18"] = round(100 * v[m18].mean(), 3) if m18.any() else np.nan
        row["n18"] = int(m18.sum())
        rows.append(row)
        if h == 2 and thr == 0.015:
            print("h=2 +1.50%:", cluster_note(e, v))
show(rows, "dose ladder, long MXN spot, episodes gap 10")
print(f"\nlive USDMXN 1d move 2026-09-24: {100*u1.iloc[-1]:+.3f}%")
