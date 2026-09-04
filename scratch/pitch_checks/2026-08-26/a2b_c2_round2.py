"""a2b / C2 round 2: close the last escape hatch, the RELATIVE form
(long IWM against short SPY) at h=10, which was the only expression in a2
with a positive edge. Two legs means a 12 bp round trip and a 60 bp bar.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (close_panel, declusters, show, sign_test,  # noqa
                       summarize, vehicle_ret, bootstrap_p_le0)

PC = Path(__file__).resolve().parents[3] / "data" / "cboe_putcall.parquet"


def pit_pctile(s, ma=10, window=252):
    m = s.dropna().rolling(ma, min_periods=ma).mean()
    return m.rolling(window, min_periods=window).apply(
        lambda w: (w <= w[-1]).mean() * 100.0, raw=True)


def main() -> None:
    pc = pd.read_parquet(PC)
    px = close_panel(["IWM", "SPY"])
    etp = pit_pctile(pc["etp"]).reindex(px.index).ffill(limit=3)
    px = px.loc[etp.dropna().index.min():]
    etp = etp.loc[px.index]
    m = (etp <= 10).fillna(False)

    legs = [("IWM", 1.0), ("SPY", -1.0)]
    for h in (7, 10):
        r = vehicle_ret(px, legs, h, 1)
        d = px.index[m.values & r.notna().values]
        ep = declusters(d, 10, px.index)
        v = r.loc[ep].values
        allm = 100 * r.dropna().mean()
        print(f"\n### IWM - SPY, h={h}, {len(ep)} episodes ###")
        print(f"  mean {100*v.mean():+.3f}%  control {allm:+.3f}%  "
              f"edge {100*v.mean()-allm:+.3f}pp  hit {100*(v>0).mean():.1f}%")
        print(f"  cost: 2 legs x 6 bps = 12 bps -> "
              f"{100*v.mean()*100/12:.1f}x (need >=5x)")
        print(f"  bootstrap P(mean<=0) = {bootstrap_p_le0(v):.3f}  "
              f"record {(v>0).sum()}-{(v<=0).sum()}, sign p "
              f"{sign_test(int((v>0).sum()), len(v)):.4f}")
        order = np.argsort(-v)
        for k in (1, 2, 3):
            keep = np.setdiff1d(np.arange(len(v)), order[:k])
            print(f"  drop-best-{k} {[str(ep[i].date()) for i in order[:k]]}: "
                  f"{100*v[keep].mean():+.3f}%  -> "
                  f"{100*v[keep].mean()*100/12:.1f}x cost")
        by = pd.Series(v, index=ep.year).groupby(level=0).agg(["count", "mean"])
        by["mean"] = (100 * by["mean"]).round(3)
        print(by.to_string())
        mt = ep.year % 4 == 2
        show([summarize(v[mt], f"h={h} MIDTERM"),
              summarize(v[~mt], f"h={h} non-midterm")], f"midterm split h={h}")

    print("\n### horizon sign profile, all three expressions (episodes) ###")
    rows = []
    for h in range(1, 11):
        row = {"h": h}
        for lbl, lg in [("IWM", [("IWM", 1.0)]), ("IWM-SPY", legs)]:
            r = vehicle_ret(px, lg, h, 1)
            d = px.index[m.values & r.notna().values]
            ep = declusters(d, 10, px.index)
            row[f"{lbl}_pct"] = round(100 * r.loc[ep].mean(), 3)
            row[f"{lbl}_n"] = len(ep)
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))
    s = pd.DataFrame(rows)
    print(f"  IWM sign changes across h=1..10: "
          f"{int((np.sign(s['IWM_pct']).diff().abs() > 0).sum())}; "
          f"IWM-SPY: {int((np.sign(s['IWM-SPY_pct']).diff().abs() > 0).sum())}")


if __name__ == "__main__":
    main()
