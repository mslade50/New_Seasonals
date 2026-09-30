"""C6 round 2b: does the live regime (GLD under its 200d, ^TNX within 2% of its 252 max)
hold up in the WIDER QE-flush set? The at-close gate (6 GLD-era QEs) and the QE-1 gate
(5 QEs, only 2013 shared) are unioned so the regime split has more than one episode.
The union is a grid I walked (two gate timings), charged as such.
Also: every GLD-era QE with a 5d return <= -2% at QE or QE-1, split by the live legs.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

P = load_prices(["GLD", "^TNX", "DX-Y.NYB"])
g = P["GLD"]["Close"].dropna()
tnx = P["^TNX"]["Close"].dropna()
idx = g.index
LAST = idx[-1]
s_ = pd.Series(idx, index=idx)
ME = pd.DatetimeIndex(s_.groupby([idx.year, idx.month]).max().values)
ME = ME[(ME < LAST) & (ME >= "2004-12-01")]
QE = ME[ME.month.isin([3, 6, 9, 12])]
qm1 = pd.DatetimeIndex([idx[idx.get_loc(q) - 1] for q in QE])
r5 = g / g.shift(5) - 1.0
tnx_hi = (tnx >= 0.98 * rolling_on_valid(tnx, lambda x: x.rolling(252).max())).reindex(idx).fillna(False).astype(bool)
tnx_hi_m1 = tnx_hi.shift(1).fillna(False).astype(bool)
below200 = g < g.rolling(200).mean()
off = 1 - g / g.rolling(252).max()


def table(thr: float, title: str):
    at = r5.reindex(QE).values <= thr
    m1 = r5.reindex(qm1).values <= thr
    sel = QE[at | m1]
    rows = []
    for q in sel:
        r = {"QE": q.date(), "gate": ("close" if r5.loc[q] <= thr else "") + ("+" if (r5.loc[q] <= thr and r5.shift(1).loc[q] <= thr) else "") + ("QE-1" if r5.shift(1).loc[q] <= thr else ""),
             "r5_QE": 100 * r5.loc[q], "below200": bool(below200.loc[q]), "TNXhi": bool(tnx_hi.loc[q] or tnx_hi_m1.loc[q]),
             "off_hi": 100 * off.loc[q]}
        for h in (1, 3, 5, 7, 10):
            p = idx.get_loc(q)
            r[f"h{h}"] = 100 * (g.iloc[p + h] / g.iloc[p] - 1) if p + h < len(idx) else np.nan
        rows.append(r)
    df = pd.DataFrame(rows)
    print(f"\n=== {title} (n={len(df)}) ===")
    print(df.round(2).to_string(index=False))
    for lbl, m in (("ALL", np.ones(len(df), bool)), ("above 200d", ~df.below200.values),
                   ("below 200d (live)", df.below200.values), ("TNX not at high", ~df.TNXhi.values),
                   ("TNX at high (live)", df.TNXhi.values),
                   ("below200 OR TNXhi", (df.below200 | df.TNXhi).values)):
        sub = df[m]
        out = []
        for h in (3, 5, 7):
            v = sub[f"h{h}"].dropna()
            w = int((v > 0).sum())
            out.append(f"h{h} {v.mean():+.2f}% {w}-{len(v)-w} p{sign_test(w, len(v)) if len(v) else np.nan:.3f}")
        print(f"  {lbl:20s} n={len(sub):2d}  " + " | ".join(out))
    return df


table(-0.03, "QE with GLD 5d <= -3% at the QE close OR the QE-1 close (union)")
table(-0.02, "neighbour: 5d <= -2% at QE or QE-1")
