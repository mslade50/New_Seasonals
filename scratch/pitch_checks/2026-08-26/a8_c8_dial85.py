"""a8 / C8: the fragility dial's 10d MA of the 63d column at or above 85, as
a DIRECTIONAL analogue for SPY, h=1..10, direction from the data.

Two things are confronted before any statistic is quoted:
  1. THE BOOK'S OWN NEGATIVE. CLAUDE.md records the dial-conditioned
     book-wide throttle as dead (aggregate PIT t=-0.23, taper -11.4R,
     rest-of-book at dial>=50 p=.47 clustered). The registry adds "the
     fragility dial as a DIRECTIONAL signal (level or rate of change)" as a
     strictly stronger claim that fails at a lower bar.
  2. THE VINTAGE RULE. data/rd2_fragility.parquet is append-only PIT only
     since 2026-07-02; everything earlier is a recompute vintage that drifted
     up to ~7 points. data/rd2_fragility_ts.parquet is the raw-basis research
     recompute. Both are run and both are stated.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (close_panel, declusters, show, sign_test,  # noqa
                       summarize, vehicle_ret)

ROOT = Path(__file__).resolve().parents[3]
PIT_CUT = pd.Timestamp("2026-07-02")


def dial(path: str) -> pd.Series:
    f = pd.read_parquet(ROOT / "data" / path)
    f.index = pd.to_datetime(f.index)
    return f["63d"].rolling(10, min_periods=10).mean().dropna()


def main() -> None:
    px = close_panel(["SPY", "TLT"])
    for name in ("rd2_fragility.parquet", "rd2_fragility_ts.parquet"):
        ma = dial(name)
        print(f"\n############ VINTAGE {name} ############")
        print(f"  span {ma.index.min().date()}..{ma.index.max().date()}  "
              f"n={len(ma)}  last={ma.iloc[-1]:.1f}  max={ma.max():.1f} "
              f"({ma.idxmax().date()})")
        for cut in (85, 80, 75, 70, 65):
            d = ma.index[ma >= cut]
            d = pd.DatetimeIndex(d).intersection(px.index)
            ep = declusters(d, 10, px.index)
            yrs = pd.Series(1, index=d).groupby(d.year).sum().to_dict() if len(d) else {}
            print(f"  >= {cut}: days={len(d):4d} ({100*len(d)/len(ma):.1f}% of series) "
                  f"episodes={len(ep):3d}  years={yrs}")
            if len(ep):
                print(f"      episodes: {', '.join(str(x.date()) for x in ep)}")
        if name.startswith("rd2_fragility."):
            d85 = pd.DatetimeIndex(ma.index[ma >= 85])
            n_pit = int((d85 >= PIT_CUT).sum())
            print(f"  VINTAGE SPLIT of the >=85 cell: {n_pit} of {len(d85)} days "
                  f"({100*n_pit/max(len(d85),1):.1f}%) sit on the post-2026-07-02 "
                  f"append-only PIT vintage; the rest are recompute vintage.")

    ma = dial("rd2_fragility.parquet")
    ma = ma.reindex(px.index).dropna()
    px2 = px.loc[ma.index.min():]

    print("\n### 1. horizon profile, dial >= 85 (sizing parquet) ###")
    for cut in (85, 80, 70):
        m = (ma >= cut).reindex(px2.index, fill_value=False)
        d = px2.index[m.values]
        rows = []
        for h in (1, 2, 3, 5, 7, 10):
            r = vehicle_ret(px2, [("SPY", 1.0)], h, 1)
            dd = pd.DatetimeIndex(d).intersection(r.dropna().index)
            ep = declusters(dd, 10, px2.index)
            s = summarize(r.reindex(ep).values, f"h={h} dial>={cut} EPISODES")
            s["day_n"] = len(dd)
            s["day_mean"] = round(100 * r.reindex(dd).mean(), 3)
            allm = 100 * r.dropna().mean()
            s["ctl"] = round(allm, 3)
            s["edge"] = round(s.get("mean_pct", np.nan) - allm, 3)
            rows.append(s)
        show(rows, f"dial >= {cut}")

    print("\n### 2. the level ladder: is 85 special or is it just 'the dial'? ###")
    rows = []
    for lo, hi in [(0, 20), (20, 35), (35, 50), (50, 65), (65, 80), (80, 200)]:
        m = ((ma >= lo) & (ma < hi)).reindex(px2.index, fill_value=False)
        d = px2.index[m.values]
        for h in (3, 10):
            r = vehicle_ret(px2, [("SPY", 1.0)], h, 1)
            dd = pd.DatetimeIndex(d).intersection(r.dropna().index)
            ep = declusters(dd, 10, px2.index)
            s = summarize(r.reindex(ep).values, f"[{lo},{hi}) h={h}")
            s["day_n"] = len(dd)
            rows.append(s)
    show(rows, "monotone in the dial? (episodes)")

    print("\n### 3. the >=85 cell, day by day, both directions, all vehicles ###")
    m = (ma >= 85).reindex(px2.index, fill_value=False)
    d = px2.index[m.values]
    print("  trigger days:", ", ".join(str(x.date()) for x in d))
    ep = declusters(pd.DatetimeIndex(d), 10, px2.index)
    print("  episodes:", ", ".join(str(x.date()) for x in ep))
    rows = []
    for h in (1, 2, 3, 5, 10):
        for tk, w, lbl in [("SPY", 1.0, "LONG SPY"), ("SPY", -1.0, "SHORT SPY"),
                           ("TLT", 1.0, "LONG TLT")]:
            r = vehicle_ret(px2, [(tk, w)], h, 1)
            dd = pd.DatetimeIndex(d).intersection(r.dropna().index)
            if len(dd) == 0:
                continue
            s = summarize(r.reindex(dd).values, f"{lbl} h={h} DAY-LEVEL")
            allm = 100 * r.dropna().mean()
            s["ctl"] = round(allm, 3)
            s["edge"] = round(s["mean_pct"] - allm, 3)
            w_ = int((r.reindex(dd) > 0).sum())
            s["sign_p_own"] = round(sign_test(w_, len(dd),
                                              float((r.dropna() > 0).mean())), 4)
            rows.append(s)
    show(rows, "the whole >=85 sample (day level: N is what it is)")

    print("\n### 4. concentration: how many DISTINCT episodes carry it? ###")
    r10 = vehicle_ret(px2, [("SPY", 1.0)], 10, 1)
    dd = pd.DatetimeIndex(d).intersection(r10.dropna().index)
    g = pd.Series(r10.reindex(dd).values, index=dd)
    byep = g.groupby(pd.Series(dd).dt.to_period("M").values).agg(["count", "mean"])
    byep["mean"] = (100 * byep["mean"]).round(3)
    print(byep.to_string())

    print("\n### 5. cross-vintage agreement on the >=85 state ###")
    ts = dial("rd2_fragility_ts.parquet")
    both = ma.index.intersection(ts.index)
    a = (ma.loc[both] >= 85)
    b = (ts.loc[both] >= 85)
    print(f"  overlapping days {len(both)}: sizing>=85 {int(a.sum())}, "
          f"research>=85 {int(b.sum())}, BOTH {int((a & b).sum())}")
    dif = (ma.loc[both] - ts.loc[both])
    print(f"  ma10(63d) sizing minus research: mean {dif.mean():+.2f} "
          f"sd {dif.std():.2f} max |{dif.abs().max():.2f}|")
    hi = ma.loc[both][ma.loc[both] >= 80]
    if len(hi):
        print("  on sizing-vintage days >=80, the research vintage reads: "
              f"{ts.loc[hi.index].min():.1f}..{ts.loc[hi.index].max():.1f} "
              f"(mean {ts.loc[hi.index].mean():.1f})")


if __name__ == "__main__":
    main()
