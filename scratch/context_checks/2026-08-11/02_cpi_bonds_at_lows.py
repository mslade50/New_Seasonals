"""CPI prints that arrive with the long end already sitting on a 52-week low.

TLT closed 0.33% above its trailing-252 low tonight, IEF 0.77%, LQD 0.62%.
The unconditional CPI/TLT cell is nothing (+0.057%, t=0.99). Does the
conditioned one have anything, and is it the low or the print doing the work?
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

BAND = 0.015          # within 1.5% of the trailing-252 low
px = load_prices(["TLT", "IEF", "^TNX", "SPY"])
ev = load_events(["cpi"])
cpi = pd.DatetimeIndex(sorted(pd.to_datetime(ev["date"]).unique()))


def anchors_for(close: pd.Series) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    d = close.index
    a, p = [], []
    for x in cpi:
        pos = d.searchsorted(x)
        if pos <= 0 or pos >= len(d) or d[pos] != x:
            continue
        a.append(d[pos - 1])
        p.append(d[pos])
    return pd.DatetimeIndex(a), pd.DatetimeIndex(p)


for tk in ("TLT", "IEF"):
    close = px[tk]["Close"].dropna()
    lo252 = close.rolling(252, min_periods=252).min()
    dist = close / lo252 - 1.0                       # 0 = printing the low
    anc, prn = anchors_for(close)
    ok = dist.reindex(anc).dropna()
    near = pd.DatetimeIndex([d for d in anc if d in ok.index and ok[d] <= BAND])
    far = pd.DatetimeIndex([d for d in anc if d in ok.index and ok[d] > BAND])

    r = close.pct_change()
    nxt = r.shift(-1)                                # the print session's move

    v = nxt.reindex(near).dropna()
    s = summarize(v.values, f"{tk} print | near low")
    print(f"\n=== {tk}: CPI anchors within {BAND*100:.1f}% of a 52w low ===")
    print(f"  n={s['n']}  print-day mean {s['mean_pct']:+.3f}%  hit {s['hit']:.1f}%  t={s['t']:+.2f}  "
          f"record {(v>0).sum()}-{(v<0).sum()}  sign p {sign_test(int((v>0).sum()), int(len(v))):.4f}")
    vf = nxt.reindex(far).dropna()
    sf = summarize(vf.values, "far")
    print(f"  other CPI anchors:  n={sf['n']}  {sf['mean_pct']:+.3f}%  hit {sf['hit']:.1f}%  t={sf['t']:+.2f}")
    print(f"  edge over other CPI prints: {s['mean_pct']-sf['mean_pct']:+.3f}%")

    # control: near-the-low sessions that are NOT CPI anchors
    all_near = pd.DatetimeIndex([d for d in dist.index if pd.notna(dist.get(d, np.nan)) and dist[d] <= BAND])
    ctrl = all_near.difference(anc)
    vc = nxt.reindex(ctrl).dropna()
    sc = summarize(vc.values, "near low, no print")
    print(f"  near a low with NO print next: n={sc['n']}  {sc['mean_pct']:+.3f}%  hit {sc['hit']:.1f}%  t={sc['t']:+.2f}")
    print(f"  edge over that: {s['mean_pct']-sc['mean_pct']:+.3f}%")

    if len(v):
        print("  era:", [(e['label'], e['n'], round(e['mean_pct'], 3), round(e['hit'], 1)) for e in era_split(v.index, v.values)])
        print("  concentration:", cluster_note(v.index, v.values))
        print("  episodes:", [(str(d.date()), round(x*100, 2)) for d, x in v.items()][-14:])
        print("  years:", sorted(set(v.index.year)))

    # h5 as well
    h5 = (close.shift(-5) / close - 1.0)
    v5 = h5.reindex(near).dropna()
    s5 = summarize(v5.values, "h5")
    print(f"  h5 from the anchor: n={s5['n']} mean {s5['mean_pct']:+.3f}% hit {s5['hit']:.1f}% t={s5['t']:+.2f}")

# yields on the same TLT-conditioned anchors
print("\n=== ^TNX on the same TLT-near-low CPI anchors ===")
close = px["TLT"]["Close"].dropna()
lo252 = close.rolling(252, min_periods=252).min()
dist = close / lo252 - 1.0
anc, _ = anchors_for(close)
ok = dist.reindex(anc).dropna()
near = pd.DatetimeIndex([d for d in anc if d in ok.index and ok[d] <= BAND])
tnx = px["^TNX"]["Close"].dropna()
tr = tnx.pct_change().shift(-1)
vt = tr.reindex(near).dropna()
st = summarize(vt.values, "tnx")
print(f"  n={st['n']}  yield change on the print {st['mean_pct']:+.3f}%  "
      f"down {(vt<0).sum()}-{(vt>0).sum()} up  t={st['t']:+.2f}  "
      f"sign p {sign_test(int((vt<0).sum()), int(len(vt))):.4f}")
