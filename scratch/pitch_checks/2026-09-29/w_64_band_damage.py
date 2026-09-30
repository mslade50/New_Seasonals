"""W64 pre-read: the damage-half band cell the arm names, UNCHARGED (no band-walk charge).

Conventions copied from 2026-09-24/kD_v1_svxy_movespike_b.py: full-history MOVE quantiles, 2018-03+ sample,
SVXY residual against beta-SPY at lag 1, declustered at gap h.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

POST = pd.Timestamp("2018-03-01")
IDX = load_prices(["SPY"])["SPY"].index
px = close_panel(["SVXY", "SPY", "^VIX", "^MOVE"]).reindex(IDX)
rs = px["SPY"].pct_change()
rv = rolling_on_valid(px["^VIX"], lambda x: x.pct_change())
rm = rolling_on_valid(px["^MOVE"], lambda x: x.pct_change())
post = pd.Series(IDX >= POST, index=IDX)
q = {p: rm.dropna().quantile(p) for p in (0.90, 0.97)}


def hb(h: int) -> float:
    a, b = fwd_lag(px["SVXY"], h, 1), fwd_lag(px["SPY"], h, 1)
    m = post & a.notna() & b.notna()
    return np.polyfit(b[m].values, a[m].values, 1)[0]


band = (rm >= q[0.90]) & (rm < q[0.97])
for lbl, m in [("band & SPY<=-0.75%", band & (rs <= -0.0075)),
               ("band & SPY<=-0.75% & VIX up", band & (rs <= -0.0075) & (rv > 0)),
               ("band & SPY>-0.75% (no-damage)", band & (rs > -0.0075))]:
    for h in (1, 2, 3):
        r = vehicle_ret(px, [("SVXY", 1.0), ("SPY", -hb(h))], h, 1)
        ok = r.notna() & post
        d = IDX[(m.reindex(IDX, fill_value=False) & ok).values]
        e = declusters(d, h, IDX)
        v = r.loc[e].values
        w = int((v > 0).sum())
        print(f"{lbl:<32} h={h}: {100 * v.mean():+.3f}% on {w}-{len(v) - w} (sign p {sign_test(w, len(v)):.3f}) against +0.35%")
print(f"q90 {100 * q[0.90]:+.3f}% q97 {100 * q[0.97]:+.3f}% | today MOVE {100 * rm.dropna().iloc[-1]:+.3f}% SPY {100 * rs.iloc[-1]:+.3f}%")
