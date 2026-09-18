"""c4 completion: the ^VIX SPY-residual Sept cell is strongest at h=2 (19-7),
and the post-break hedged short SVXY reads +0.587% at h=2 (5-3). Round 1 ran
the placebo ladder at h=3 only. Ladder at h=1/h=2 for the residual, the real
post-break vehicle and the 14-year spliced vehicle, each on its own beta.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pitch_lab import *  # noqa
from kA_common import *  # noqa
import numpy as np
import pandas as pd

px = build_panel()
cal = px.index
opex = pd.DatetimeIndex(sorted(set(load_events(["opex"])["date"]) & set(cal)))
opex = opex[opex < pd.Timestamp("2026-09-18")]
sep = opex[opex.month == 9]
post = pd.Series(cal >= POST, index=cal)
pre = pd.Series((cal < BREAK) & (cal >= pd.Timestamp("2011-10-10")), index=cal)


def at(dates, off):
    p = cal.get_indexer(pd.DatetimeIndex(dates)) + off
    p = p[(p >= 0) & (p < len(cal))]
    return cal[p]


for h in (1, 2):
    rv = vehicle_ret(px, [("^VIX", 1.0)], h, 0)
    rs = vehicle_ret(px, [("SPY", 1.0)], h, 0)
    ok = rv.notna() & rs.notna()
    bv = np.polyfit(rs[ok].values, rv[ok].values, 1)[0]
    res = rv - bv * rs
    b = hedge_beta(px, "SVXY", h, 0, post)
    hs = vehicle_ret(px, [("SVXY", -1.0), ("SPY", b)], h, 0)
    bp = hedge_beta(px, "SVS", h, 0, pre)
    hp = vehicle_ret(px, [("SVS", -1.0), ("SPY", bp)], h, 0)
    splice = pd.Series(np.where(cal >= POST, hs.values, hp.values), index=cal)
    for lbl, ser, dates in [("^VIX residual 2000+", res, sep),
                            ("hedged short SVXY post-break", hs, sep[sep >= POST]),
                            ("hedged short spliced 2011+ (own-era betas)", splice,
                             sep[sep >= pd.Timestamp("2011-10-10")])]:
        rows = []
        for k in range(-5, 6):
            v = ser.reindex(at(dates, k)).dropna().values
            w = int((v > 0).sum())
            rows.append({"k": k, "n": len(v), "mean_pct": 100 * v.mean(),
                         "median_pct": 100 * np.median(v), "rec": f"{w}-{len(v)-w}"})
        df = pd.DataFrame(rows)
        df["rank"] = df["mean_pct"].rank(ascending=False).astype(int)
        print(f"\n-- h={h} {lbl} --")
        print(df.round(3).to_string(index=False))
        print(f"   TRUE k=0 ranks {int(df.loc[df.k == 0, 'rank'].iloc[0])} of {len(df)}")
    v = splice.reindex(sep[sep >= pd.Timestamp("2011-10-10")]).dropna()
    print(f"   spliced 2011+ h={h} k=0 by year: " + ", ".join(f"{d.year}:{100*x:+.2f}" for d, x in v.items()))
    print("   ", signed_concentration(v.index, v.values))
