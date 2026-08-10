"""E1b -- the confound the refunding label cannot survive: MONTH-OF-YEAR.

Round 1 falsified the SHAPE (the predicted tdom 4-8 concession block is
POSITIVE, +1.42 bps/day TLT and +1.95 with 5/5 positive days on IEF; the
refunding-minus-nonrefunding difference peaks at tdom 14 and 17, i.e. AFTER
the auctions clear) and showed the closer-to-actual-auction anchor INVERTS
(calendar dom 8 -> dom 13: -0.249pp excess, welch t -1.17).

This script asks the remaining question.  "Refunding month" is a LABEL on the
set {Feb, May, Aug, Nov}.  Bond month-of-year seasonality is a real and
famous artifact, and the registry already kills famous calendar cells.  If a
FAKE refunding label -- {Mar, Jun, Sep, Dec} or {Jan, Apr, Jul, Oct} -- pays
as well or better at the same tdom, the word "refunding" is doing no work.

Also tested here: the per-MONTH decomposition, so it is visible whether one
of the four refunding months carries the whole cell.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

ENTRY_TDOM, H = 6, 5
px = close_panel(["TLT", "IEF"])

for TKR in ["TLT", "IEF"]:
    s = px[TKR].dropna()
    idx, c = s.index, s.values
    ym = pd.Series(idx.year * 100 + idx.month, index=idx)
    tdom = ym.groupby(ym.values).cumcount().values + 1
    mon = np.array([d.month for d in idx])
    yr = np.array([d.year for d in idx])
    r = np.full(len(c), np.nan)
    r[:len(c) - H] = c[H:] / c[:-H] - 1.0
    base = (tdom == ENTRY_TDOM) & ~np.isnan(r)

    print("\n" + "=" * 96)
    print(f"{TKR}: entry tdom {ENTRY_TDOM}, h={H}.  REAL vs FAKE refunding labels")
    print("=" * 96)
    labels = {
        "REAL refunding  (Feb May Aug Nov)": (2, 5, 8, 11),
        "FAKE label A    (Mar Jun Sep Dec)": (3, 6, 9, 12),
        "FAKE label B    (Jan Apr Jul Oct)": (1, 4, 7, 10),
    }
    rows = []
    for lbl, ms in labels.items():
        m = base & np.isin(mon, ms)
        o = base & ~np.isin(mon, ms)
        a, b = r[m], r[o]
        w = int((a > 0).sum())
        rows.append({"label": lbl, "N": len(a),
                     "mean_pct": round(100 * a.mean(), 4),
                     "other_months_pct": round(100 * b.mean(), 4),
                     "EXCESS_bps": round(1e4 * (a.mean() - b.mean()), 1),
                     "hit": round(100 * w / len(a), 1),
                     "t": round(a.mean() / (a.std(ddof=1) / np.sqrt(len(a))), 2)})
    print(pd.DataFrame(rows).to_string(index=False))

    print(f"\n  per-MONTH breakdown of the tdom-{ENTRY_TDOM} h={H} entry "
          f"({TKR}), all 12 months:")
    rows = []
    for mo in range(1, 13):
        m = base & (mon == mo)
        v = r[m]
        rows.append({"month": mo, "refunding": mo in (2, 5, 8, 11), "N": len(v),
                     "mean_pct": round(100 * v.mean(), 4),
                     "hit": round(100 * (v > 0).mean(), 1),
                     "t": round(v.mean() / (v.std(ddof=1) / np.sqrt(len(v))), 2)})
    mt = pd.DataFrame(rows)
    print(mt.to_string(index=False))
    ref = mt[mt.refunding].sort_values("mean_pct", ascending=False)
    print(f"  refunding months ranked: "
          f"{[(int(a), b) for a, b in zip(ref.month, ref.mean_pct)]}")
    print(f"  -> drop the single best refunding month "
          f"({int(ref.month.iloc[0])}): remaining three average "
          f"{ref.mean_pct.iloc[1:].mean():+.4f}% vs non-refunding "
          f"{mt[~mt.refunding].mean_pct.mean():+.4f}%")

    # AUGUST specifically -- the month actually being traded today
    aug = base & (mon == 8)
    v = r[aug]
    w = int((v > 0).sum())
    print(f"\n  *** AUGUST ONLY (the month today's trade is in): N={len(v)} "
          f"{100*v.mean():+.4f}%  hit {100*w/len(v):.1f}%  sign p "
          f"{sign_test(w, len(v)):.4f}  worst {100*v.min():+.2f}%")
    ay = yr[aug]
    print(f"      August by era: pre-2018 {100*v[ay<2018].mean():+.4f}% "
          f"(N={int((ay<2018).sum())})  |  2018+ {100*v[ay>=2018].mean():+.4f}% "
          f"(N={int((ay>=2018).sum())})")
    print(f"      August midterm years {100*v[(ay%4)==2].mean():+.4f}% "
          f"(N={int(((ay%4)==2).sum())})  |  non-midterm "
          f"{100*v[(ay%4)!=2].mean():+.4f}% (N={int(((ay%4)!=2).sum())})")
    print(f"      concentration: {cluster_note(idx[aug], v)}")
