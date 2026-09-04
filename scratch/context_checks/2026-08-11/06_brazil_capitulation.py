"""Brazil: two independent triggers named the same market tonight.

^BVSP fired P7b (5+ consecutive down closes, n=143, +0.406%, 85-58, t=1.87)
and P5 (5d return in the bottom 5% of its year, n=314, +0.174%). EWZ fired the
second as well (-3.44% today, 5d -5.85%, 5d rank 2.8).

The engine measured each trigger alone. The cross is the cell: a losing streak
that is ALSO a bottom-5% week. And the tradeable US-listed expression is EWZ,
so measure both.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

px = load_prices(["^BVSP", "EWZ", "EEM"])

for tk in ("^BVSP", "EWZ"):
    c = px[tk]["Close"].dropna()
    d = c.index
    r = c.pct_change()
    r5 = c.pct_change(5)
    rank5 = r5.rolling(252, min_periods=252).rank(pct=True) * 100

    down = (r < 0).astype(int)
    streak = down * 0
    run = 0
    vals = []
    for x in down.values:
        run = run + 1 if x else 0
        vals.append(run)
    streak = pd.Series(vals, index=d)

    st5 = pd.DatetimeIndex([x for x in d if streak[x] >= 5])
    bot5 = pd.DatetimeIndex([x for x in d if pd.notna(rank5.get(x, np.nan)) and rank5[x] <= 5])
    both = st5.intersection(bot5)
    print(f"\n=== {tk} ===  history {d[0].date()} .. {d[-1].date()}")
    print(f"  5+ down closes: {len(st5)} days | 5d rank <=5: {len(bot5)} days | BOTH: {len(both)} days")

    for name, sel in [("streak only", st5), ("bottom-5% wk only", bot5), ("BOTH", both)]:
        dec = declusters(sel, 5, d)
        if len(dec) < 5:
            print(f"  {name:<18} n={len(dec)} too small")
            continue
        row = []
        for h in (1, 3, 5, 10, 21):
            fw = (c.shift(-h) / c - 1.0)
            v = fw.reindex(dec).dropna()
            s = summarize(v.values, f"h{h}")
            row.append((h, s['n'], s['mean_pct'], s['hit'], s['t'],
                        int((v > 0).sum()), int((v < 0).sum()),
                        sign_test(int((v > 0).sum()), int(len(v)))))
        print(f"  {name:<18} episodes={len(dec)}")
        for h, n, m, hit, t, u, dn, p in row:
            fw = (c.shift(-h) / c - 1.0)
            ctrl = local_control(d, dec, 126)
            vc = fw.reindex(ctrl).dropna()
            sc = summarize(vc.values, "ctrl")
            print(f"      h{h:<2} n={n:<4} mean {m:+7.3f}%  hit {hit:5.1f}%  t={t:+5.2f}  "
                  f"{u}-{dn}  sign p {p:.4f}  | local ctrl {sc['mean_pct']:+.3f}%  edge {m-sc['mean_pct']:+.3f}%")

    dec = declusters(both, 5, d)
    if len(dec) >= 5:
        fw = (c.shift(-5) / c - 1.0)
        v = fw.reindex(dec).dropna()
        print(f"  BOTH h5 era: {[(e['label'], e['n'], round(e['mean_pct'],2), round(e['hit'],1)) for e in era_split(v.index, v.values)]}")
        print(f"  BOTH h5 concentration: {cluster_note(v.index, v.values)}")
        print(f"  BOTH episodes ({len(dec)}): {[str(x.date()) for x in dec]}")
        fw1 = (c.shift(-1) / c - 1.0)
        v1 = fw1.reindex(dec).dropna()
        print(f"  BOTH h1 era: {[(e['label'], e['n'], round(e['mean_pct'],2), round(e['hit'],1)) for e in era_split(v1.index, v1.values)]}")
