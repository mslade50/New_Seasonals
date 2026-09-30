"""C2 round 2 on what the joint gate actually keys on: the ^TNX-at-252-high leg
(the DX-thrust leg is wrong-signed alone). Short DX k=-4 -> NFP (h4) / NFP+1 (h5).
Is it NFP-specific (vs any day, vs non-NFP month turns under the same gate)?
Does the yield crowd resolve too (TLT in the same windows)? Episodes, era, midterm,
drop-best-2, threshold / entry-offset neighbours."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

V = ["DX-Y.NYB", "UUP", "TLT", "GLD", "SPY"]
raw = load_prices(V + ["^TNX"])
IDX = raw["SPY"]["Close"].index
PX = pd.DataFrame({t: raw[t]["Close"].reindex(IDX).ffill(limit=2) for t in V})
dx = raw["DX-Y.NYB"]["Close"].dropna()
DXR = pct_rank(dx, 21, 252).reindex(IDX).ffill(limit=2).to_numpy()
tnx = raw["^TNX"]["Close"].dropna()
TNR = (tnx / tnx.rolling(252).max()).reindex(IDX).ffill(limit=2).to_numpy()
nfp = load_events(["nfp"])["date"]
pos, _ = anchor_positions(IDX, nfp[nfp <= IDX[-1]], 0)
pos = np.array(pos)
pos = pos[(pos - 12 >= 0) & (pos + 1 < len(IDX))]
tdom = pd.Series(1, index=IDX).groupby([IDX.year, IDX.month]).cumsum().to_numpy()


def w(t, ends, k, hx):
    c = PX[t].to_numpy()
    return c[ends + hx] / c[ends + k] - 1.0


def rec(v):
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return "n/a"
    n = int((v > 0).sum())
    return f"{100*v.mean():+.3f} {n}-{len(v)-n} p{sign_test(n, len(v)):.3f}"


G = TNR[pos - 5] >= 0.99
print("TNX-leg episodes (signal k=-5), short DX / short UUP / long TLT / long GLD, h4 (NFP close) and h5:")
for p in pos[G]:
    print(f"  NFP {IDX[p].date()} TNX/hi {TNR[p-5]:.4f} DXr21 {DXR[p-5]:5.1f}  "
          f"-DX {-100*w('DX-Y.NYB', np.array([p]), -4, 0)[0]:+.2f}/{-100*w('DX-Y.NYB', np.array([p]), -4, 1)[0]:+.2f}  "
          f"-UUP {-100*w('UUP', np.array([p]), -4, 0)[0]:+.2f}/{-100*w('UUP', np.array([p]), -4, 1)[0]:+.2f}  "
          f"TLT {100*w('TLT', np.array([p]), -4, 0)[0]:+.2f}/{100*w('TLT', np.array([p]), -4, 1)[0]:+.2f}  "
          f"GLD {100*w('GLD', np.array([p]), -4, 0)[0]:+.2f}")

v4 = -w("DX-Y.NYB", pos[G], -4, 0)
D = IDX[pos[G]]
print(f"\nshort DX h4 TNX leg: {rec(v4)}  | {cluster_note(D, v4)}")
srt = np.sort(v4)[::-1]
print(f"  drop-best-2: {100*srt[2:].mean():+.3f}%  | pre-2018 {rec(v4[D < '2018-01-01'])}  2018+ {rec(v4[D >= '2018-01-01'])}")
print(f"  midterm {rec(v4[np.array([d.year % 4 == 2 for d in D])])}")
print(f"  TLT same windows h4: {rec(w('TLT', pos[G], -4, 0))} (the yield crowd does it resolve?)")

print("\nNeighbours (short DX, h4 | h5), NFP anchors:")
for lab, thr in (("TNX at 252 high", 0.99999), ("within 0.5%", 0.995), ("within 1% (cell)", 0.99),
                 ("within 2%", 0.98), ("within 3%", 0.97), ("within 5%", 0.95), ("NOT within 1%", None)):
    g = (TNR[pos - 5] >= thr) if thr else (TNR[pos - 5] < 0.99)
    print(f"  {lab:18s} {rec(-w('DX-Y.NYB', pos[g], -4, 0))} | {rec(-w('DX-Y.NYB', pos[g], -4, 1))}"
          f"   UUP h4 {rec(-w('UUP', pos[g], -4, 0))}")
print("\nEntry-offset ladder under the TNX leg (gate read at entry-1), short DX exit NFP close:")
for k in (-8, -6, -5, -4, -3, -2, -1):
    g = TNR[pos + k - 1] >= 0.99
    print(f"  k={k:+d} {rec(-w('DX-Y.NYB', pos[g], k, 0))}   ungated {rec(-w('DX-Y.NYB', pos, k, 0))}")

# any-day and non-NFP month-turn windows under the same TNX gate (declustered 5td)
c = PX["DX-Y.NYB"].to_numpy()
isn = np.zeros(len(IDX) + 2, bool)
isn[pos] = True
out = []
for n in (4, 5):
    sig = np.arange(260, len(IDX) - n - 1)
    fr = -(c[sig + 1 + n] / c[sig + 1] - 1)
    g = TNR[sig] >= 0.99
    e = declusters(IDX[sig[g]], n, IDX)
    ser = pd.Series(fr, index=IDX[sig])
    out.append(summarize(ser.loc[e].to_numpy(), f"h{n} any day TNX gate, declustered"))
    out.append(summarize(fr, f"h{n} all days"))
    ends = np.array([p for p in range(10, len(IDX) - 2) if tdom[p] <= 8 and not isn[p - 3:p + 2].any()])
    ends = ends[ends + (n - 4) < len(IDX)]
    vc = -(c[ends + (n - 4)] / c[ends - 4] - 1)
    gc = TNR[ends - 5] >= 0.99
    ee = declusters(IDX[ends[gc]], n, IDX)
    sc = pd.Series(vc[gc], index=IDX[ends[gc]])
    out.append(summarize(sc.loc[ee].to_numpy(), f"h{n} non-NFP tdom<=8 turn, TNX gate, decl."))
    out.append(summarize(vc, f"h{n} non-NFP tdom<=8 turn, all"))
show(out, "short DX: is the TNX leg NFP-specific?")
