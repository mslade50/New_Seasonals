"""C3 round 2: decluster + concentration, definition neighbours (1d +4/+5/+6%
x volume 2x/3x/4x), volume-definition variants, era/regime split, gate
attribution, and the day-t+1 reversal that the lag=0 row exposed."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kC_c3_ung_volthrust import build, eps  # noqa: E402

if __name__ == "__main__":
    d, px, r1, vr = build()
    idx = px.index
    v = d["Volume"].astype(float)
    vr_ex = v / v.shift(1).rolling(63).mean()
    vr_med = v / v.rolling(63).median()
    legs = [("UNG", 1.0)]
    rets = {h: vehicle_ret(px, legs, h) for h in (1, 2, 3, 5)}

    # 1. neighbour grid
    rows = []
    for th in (0.04, 0.05, 0.06):
        for vm in (2.0, 3.0, 4.0):
            m = (r1 >= th) & (vr >= vm)
            for h in (2, 3):
                e, x = eps(rets[h], m, h, idx)
                post = pd.DatetimeIndex(e) >= "2011-01-01"
                r = summarize(x, f"1d>={th:.0%} vol>={vm:.0f}x h={h}")
                r["mean_2011plus"] = 100 * np.nanmean(x[post]) if post.any() else np.nan
                r["n_2011plus"] = int(post.sum())
                rows.append(r)
    show(rows, "1. neighbour grid (long UNG, episodes)")

    # 2. volume definition variants at the pitched rung
    rows = []
    for lbl, vv in [("vol/63d mean incl today", vr), ("vol/63d mean ex today", vr_ex),
                    ("vol/63d median", vr_med)]:
        m = (r1 >= 0.05) & (vv >= 3)
        for h in (2, 3):
            e, x = eps(rets[h], m, h, idx)
            rows.append(summarize(x, f"{lbl} h={h}"))
    show(rows, "2. volume-definition variants (1d>=5%, >=3x)")

    trig = (r1 >= 0.05) & (vr >= 3)
    # 3. decluster + concentration at h=3
    for gap in (3, 10, 21):
        e, x = eps(rets[3], trig, gap, idx)
        x = rets[3].loc[e].values
        w = int((x > 0).sum())
        print(f"3. h=3 decluster {gap}td: N={len(e)} mean {100*np.nanmean(x):+.3f}% "
              f"median {100*np.nanmedian(x):+.3f}% record {w}-{len(x)-w} "
              f"sign p {sign_test(w, len(x)):.4f} boot P<=0 {bootstrap_p_le0(x):.3f}")
    e3, x3 = eps(rets[3], trig, 3, idx)
    print("   ", cluster_note(e3, x3))
    yrs = pd.Series(x3, index=pd.DatetimeIndex(e3).year).groupby(level=0).agg(["sum", "count"])
    yrs["sum"] = (100 * yrs["sum"]).round(2)
    print("   year sums (pp) / counts:", yrs.to_dict("index"))
    yy = pd.DatetimeIndex(e3).year
    show([summarize(x3[~yy.isin([2009])], "ex-2009"),
          summarize(x3[~yy.isin([2021])], "ex-2021"),
          summarize(x3[~yy.isin([2009, 2018])], "ex-2009 & ex-2018 (top-2 years)"),
          summarize(x3[yy >= 2011], "2011+"),
          summarize(x3[yy >= 2018], "2018+")], "3b. concentration / era (h=3 episodes)")

    # 4. gate attribution: volume leg vs parent at h=2/3 already in round 1;
    #    add the volume-only parent (vol>=3x, any 1d up) to see which leg carries it
    rows = []
    for h in (2, 3):
        for lbl, m in [("vol>=3x & 1d>=5%", trig), ("vol>=3x & 0<1d<5%", (vr >= 3) & (r1 > 0) & (r1 < 0.05)),
                       ("vol>=3x & 1d<=0", (vr >= 3) & (r1 <= 0)), ("1d>=5% & vol<3x", (r1 >= 0.05) & (vr < 3))]:
            e, x = eps(rets[h], m, h, idx)
            rows.append(summarize(x, f"h={h} {lbl}"))
    show(rows, "4. which leg carries it")

    # 5. the day t+1 move (entry day) and the lag-1 outcome
    c = px["UNG"]
    d1 = c.shift(-1) / c - 1.0
    e, x = eps(rets[3], trig, 3, idx)
    dd = d1.loc[e].values
    dn = dd < 0
    show([summarize(dd, "day t+1 return (signal -> entry close)"),
          summarize(x[dn], f"h=3 lag1 | t+1 DOWN (N={int(dn.sum())})"),
          summarize(x[~dn], f"h=3 lag1 | t+1 UP (N={int((~dn).sum())})")],
         "5. entry-day reversal and what follows")
    # t+1 reversal in the matched control for contrast
    em, xm = eps(rets[3], (r1 >= 0.05) & (vr < 3), 3, idx)
    print(f"   matched (no-vol) day t+1 mean {100*np.nanmean(d1.loc[em].values):+.3f}%")
    # 6. month split and live-season context
    mon = pd.DatetimeIndex(e3).month
    print("6. h=3 by quarter:", pd.Series(100 * x3, index=((mon - 1) // 3 + 1)).groupby(level=0)
          .agg(["mean", "count"]).round(2).to_dict("index"))
