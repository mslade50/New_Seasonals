"""a1b / C1 round 2: the SHORT side, definition neighbours, midterm split,
drop-best. Round 1 (a1) showed the LONG has a negative edge against its own
same-era control at all six horizons and that the equity-mid-range gate is
anti-selective. Round 2 asks whether the other sign is a trade.
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
    px = close_panel(["SPY"])

    def mask_for(ma, win, cut=10, eq_lo=25, eq_hi=75, gate=True):
        i = pit_pctile(pc["index"], ma, win).reindex(px.index).ffill(limit=3)
        e = pit_pctile(pc["equity"], ma, win).reindex(px.index).ffill(limit=3)
        m = (i <= cut)
        if gate:
            m = m & (e >= eq_lo) & (e <= eq_hi)
        return m.fillna(False), i

    base, idxp = mask_for(10, 252)
    first = idxp.dropna().index.min()
    pxe = px.loc[first:]
    base = base.loc[first:]

    print("### A. SHORT SPY, episode level, all horizons ###")
    rows = []
    for h in (1, 2, 3, 5, 7, 10):
        r = vehicle_ret(pxe, [("SPY", -1.0)], h, 1)
        d = pxe.index[base.values & r.notna().values]
        ep = declusters(d, 10, pxe.index)
        s = summarize(r.loc[ep].values, f"SHORT h={h}")
        allr = r.dropna()
        s["ctl_short_drift"] = round(100 * allr.mean(), 3)
        s["edge_pct"] = round(s["mean_pct"] - 100 * allr.mean(), 3)
        s["x_cost"] = round(s["mean_pct"] * 100 / 6.0, 2)
        # sign test against SPY's OWN down-rate over the same era
        base_down = float((allr > 0).mean())
        w = int((r.loc[ep] > 0).sum())
        s["sign_p_vs_own"] = round(sign_test(w, len(ep), base_down), 4)
        rows.append(s)
    show(rows, "short side (cost bar 30 bps = 5x one leg)")

    print("\n### B. drop-best / concentration on the SHORT at h=10 ###")
    r = vehicle_ret(pxe, [("SPY", -1.0)], 10, 1)
    d = pxe.index[base.values & r.notna().values]
    ep = declusters(d, 10, pxe.index)
    v = r.loc[ep].values
    order = np.argsort(-v)
    print(f"  N={len(v)} mean={100*v.mean():+.3f}%  "
          f"record {(v>0).sum()}-{(v<=0).sum()}")
    for k in (1, 2, 3):
        keep = np.setdiff1d(np.arange(len(v)), order[:k])
        print(f"  drop-best-{k} ({[str(ep[i].date()) for i in order[:k]]}): "
              f"{100*v[keep].mean():+.3f}%")
    by_yr = pd.Series(v, index=ep.year).groupby(level=0).agg(["count", "mean"])
    by_yr["mean"] = (100 * by_yr["mean"]).round(3)
    print("  by year:\n", by_yr.to_string())

    print("\n### C. definition neighbours (episode mean, LONG, h=5 and h=10) ###")
    out = []
    for ma in (5, 10, 21):
        for win in (126, 252, 504):
            m, ip = mask_for(ma, win)
            f = ip.dropna().index.min()
            pe = px.loc[f:]
            me = m.loc[f:]
            row = {"ma": ma, "win": win}
            for h in (5, 10):
                rr = vehicle_ret(pe, [("SPY", 1.0)], h, 1)
                dd = pe.index[me.values & rr.notna().values]
                ee = declusters(dd, 10, pe.index)
                allm = 100 * rr.dropna().mean()
                row[f"h{h}_mean"] = round(100 * rr.loc[ee].mean(), 3)
                row[f"h{h}_edge"] = round(100 * rr.loc[ee].mean() - allm, 3)
                row[f"h{h}_n"] = len(ee)
            out.append(row)
    print(pd.DataFrame(out).to_string(index=False))

    print("\n### D. midterm split (2022, 2026) vs the rest, LONG ###")
    for h in (5, 10):
        rr = vehicle_ret(pxe, [("SPY", 1.0)], h, 1)
        dd = pxe.index[base.values & rr.notna().values]
        ee = declusters(dd, 10, pxe.index)
        mt = ee.year % 4 == 2
        show([summarize(rr.loc[ee[mt]].values, f"h={h} MIDTERM (2022/2026)"),
              summarize(rr.loc[ee[~mt]].values, f"h={h} non-midterm")],
             f"midterm split h={h}")

    print("\n### E. coverage: what era does this cell live in at all? ###")
    dd = pxe.index[base.values]
    print("  days by year:", pd.Series(1, index=dd).groupby(dd.year).sum().to_dict())
    print(f"  series first usable day {first.date()} -> the pre-2018 era split "
          "is STRUCTURALLY UNAVAILABLE; every observation is post-COVID.")


if __name__ == "__main__":
    main()
