"""a8b / C8 round 2. a8 established that the >=85 cell has 22 days in three
declustered blocks, two of which (2021-12-20, 2022-01-04) are one contiguous
market episode and the third (2026-08-18) is TODAY'S OWN, unresolved.

This script asks the only questions left:
  A. How many observations of the >=85 cell have a COMPLETE forward return
     that is not the December-2021 top? (answer decides the candidate)
  B. Drop-episode / drop-year on the loosest form that still has a spread of
     episodes (>=70), because if THAT is what carries the sign then C8 is the
     plain dial level, i.e. the book's own dead throttle wearing a hat.
  C. Vintage: does the >=70 short survive on the research recompute?
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (close_panel, declusters, show, sign_test,  # noqa
                       summarize, vehicle_ret)

ROOT = Path(__file__).resolve().parents[3]


def dial(path):
    f = pd.read_parquet(ROOT / "data" / path)
    f.index = pd.to_datetime(f.index)
    return f["63d"].rolling(10, min_periods=10).mean().dropna()


def main() -> None:
    px = close_panel(["SPY"])
    ma = dial("rd2_fragility.parquet").reindex(px.index).dropna()
    pxs = px.loc[ma.index.min():]

    print("### A. the >=85 cell: complete observations by market episode ###")
    d = pd.DatetimeIndex(ma.index[ma >= 85])
    for h in (1, 3, 5, 10):
        r = vehicle_ret(pxs, [("SPY", -1.0)], h, 1)   # SHORT, the sign a8 found
        dd = d.intersection(r.dropna().index)
        old = dd[dd < pd.Timestamp("2023-01-01")]
        new = dd[dd >= pd.Timestamp("2023-01-01")]
        print(f"  h={h}: {len(dd)} complete of {len(d)} trigger days -> "
              f"{len(old)} from the 2021-12/2022-01 top, {len(new)} from "
              f"anywhere else")
        if len(new):
            print("      other-episode days:", ", ".join(str(x.date()) for x in new),
                  f" mean {100*r.reindex(new).mean():+.3f}%")
    print("  => the ENTIRE resolved sample of the pitched cell is ONE market "
          "episode (Dec-2021 top). 2026 is the second, and it is the one being "
          "traded, so it cannot also be its own evidence.")

    print("\n### B. >=70 (the loosest form with a spread of episodes) ###")
    d70 = pd.DatetimeIndex(ma.index[ma >= 70])
    for h in (3, 5, 10):
        r = vehicle_ret(pxs, [("SPY", -1.0)], h, 1)
        dd = d70.intersection(r.dropna().index)
        ep = declusters(dd, 10, pxs.index)
        v = r.reindex(ep).values
        allm = 100 * r.dropna().mean()
        print(f"\n  h={h} SHORT SPY, {len(ep)} episodes: mean "
              f"{100*np.nanmean(v):+.3f}% vs short drift {allm:+.3f}% "
              f"(edge {100*np.nanmean(v)-allm:+.3f}pp)")
        by = pd.Series(v, index=ep.year).groupby(level=0).agg(["count", "mean"])
        by["mean"] = (100 * by["mean"]).round(3)
        print(by.to_string())
        for y in sorted(set(ep.year)):
            keep = ep.year != y
            print(f"    drop {y}: {100*np.nanmean(v[keep]):+.3f}% "
                  f"(n={int(keep.sum())})")
        order = np.argsort(-np.nan_to_num(v, nan=-9e9))
        for k in (1, 2, 3):
            keep = np.setdiff1d(np.arange(len(v)), order[:k])
            print(f"    drop-best-{k} {[str(ep[i].date()) for i in order[:k]]}: "
                  f"{100*np.nanmean(v[keep]):+.3f}%")
        # ex the currently-live 2026 block, which is not out of sample
        keep = ep < pd.Timestamp("2026-01-01")
        print(f"    ex-2026 (the live episode): {100*np.nanmean(v[keep]):+.3f}% "
              f"(n={int(keep.sum())})")

    print("\n### C. same >=70 short on the RESEARCH recompute vintage ###")
    ts = dial("rd2_fragility_ts.parquet").reindex(px.index).dropna()
    d70t = pd.DatetimeIndex(ts.index[ts >= 70])
    for h in (3, 5, 10):
        r = vehicle_ret(pxs, [("SPY", -1.0)], h, 1)
        dd = d70t.intersection(r.dropna().index)
        ep = declusters(dd, 10, pxs.index)
        s = summarize(r.reindex(ep).values, f"h={h} research vintage >=70")
        s["sizing_n_ep"] = len(declusters(d70.intersection(r.dropna().index),
                                          10, pxs.index))
        show([s], "")
    both = ma.index.intersection(ts.index)
    a, b = (ma.loc[both] >= 70), (ts.loc[both] >= 70)
    print(f"  state agreement on >=70 over {len(both)} shared days: "
          f"sizing {int(a.sum())}, research {int(b.sum())}, both {int((a&b).sum())} "
          f"-> jaccard {int((a&b).sum())/max(int((a|b).sum()),1):.3f}")


if __name__ == "__main__":
    main()
