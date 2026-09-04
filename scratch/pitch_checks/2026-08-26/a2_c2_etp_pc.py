"""a2 / C2: CBOE ETP (ETF-options) put/call 10d MA at a trailing-252d
percentile <= 10. Vehicle IWM, with SPY as the control leg.

a0 already answered the duplicate question: index<=10 and etp<=10 share only
38 of 289 / 140 days (jaccard 0.097, raw-ratio corr 0.116), so C2 is NOT C1
in a costume and has to be killed on its own numbers.

Round 1 + the parts of round 2 that are cheap:
  0. horizon scan on IWM long, IWM short, and IWM-minus-SPY.
  1. battery on whichever sign the table picks.
  2. BETA-NEUTRAL RESIDUAL: regress the IWM cell return on SPY over the same
     window. A small-cap claim that is SPY beta is not a small-cap claim.
  3. threshold ladder + definition neighbours (MA, lookback).
  4. midterm split, concentration.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (battery, close_panel, declusters, horizon_scan,  # noqa
                       show, sign_test, summarize, vehicle_ret)

PC = Path(__file__).resolve().parents[3] / "data" / "cboe_putcall.parquet"


def pit_pctile(s, ma=10, window=252):
    m = s.dropna().rolling(ma, min_periods=ma).mean()
    return m.rolling(window, min_periods=window).apply(
        lambda w: (w <= w[-1]).mean() * 100.0, raw=True)


def resid(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """OLS y ~ a + b x; return (alpha_pct, beta)."""
    m = ~(np.isnan(y) | np.isnan(x))
    y, x = y[m], x[m]
    if len(y) < 5:
        return np.nan, np.nan
    b = np.polyfit(x, y, 1)
    return 100 * float(b[1]), float(b[0])


def main() -> None:
    pc = pd.read_parquet(PC)
    px = close_panel(["IWM", "SPY"])
    etp = pit_pctile(pc["etp"]).reindex(px.index).ffill(limit=3)
    first = etp.dropna().index.min()
    px = px.loc[first:]
    etp = etp.loc[first:]
    m = (etp <= 10).fillna(False)
    d = px.index[m.values]
    print(f"trigger days {len(d)}  span {d.min().date()}..{d.max().date()}  "
          f"by year {pd.Series(1, index=d).groupby(d.year).sum().to_dict()}")

    print("\n### 0. horizon scan, three expressions ###")
    for lbl, legs in [("LONG IWM", [("IWM", 1.0)]),
                      ("SHORT IWM", [("IWM", -1.0)]),
                      ("IWM - SPY", [("IWM", 1.0), ("SPY", -1.0)])]:
        show(horizon_scan(px, d, legs, hs=(1, 2, 3, 5, 7, 10), min_gap=10), lbl)

    rows = []
    for h in (1, 2, 3, 5, 7, 10):
        for lbl, legs in [("IWM", [("IWM", 1.0)]),
                          ("IWM-SPY", [("IWM", 1.0), ("SPY", -1.0)])]:
            r = vehicle_ret(px, legs, h, 1)
            rows.append(summarize(r.dropna().values, f"all days {lbl} h={h}"))
    show(rows, "0b. unconditional over the SAME era")

    print("\n### 2. beta-neutral residual of the IWM cell against SPY ###")
    out = []
    for h in (1, 2, 3, 5, 10):
        ri = vehicle_ret(px, [("IWM", 1.0)], h, 1)
        rs = vehicle_ret(px, [("SPY", 1.0)], h, 1)
        dd = px.index[m.values & ri.notna().values & rs.notna().values]
        ep = declusters(dd, 10, px.index)
        a, b = resid(ri.loc[ep].values, rs.loc[ep].values)
        # full-sample beta, then the cell's alpha against it
        af, bf = resid(ri.dropna().values, rs.reindex(ri.dropna().index).values)
        cell_res = 100 * (ri.loc[ep] - bf * rs.loc[ep]).mean()
        all_res = 100 * (ri - bf * rs).dropna().mean()
        out.append({"h": h, "n_ep": len(ep),
                    "IWM_pct": round(100 * ri.loc[ep].mean(), 3),
                    "SPY_pct": round(100 * rs.loc[ep].mean(), 3),
                    "full_beta": round(bf, 3),
                    "cell_resid_pct": round(cell_res, 3),
                    "alldays_resid_pct": round(all_res, 3),
                    "resid_edge_pp": round(cell_res - all_res, 3)})
    print(pd.DataFrame(out).to_string(index=False))

    print("\n### 3. threshold ladder on the etp percentile (episodes) ###")
    for h in (3, 10):
        r = vehicle_ret(px, [("IWM", 1.0)], h, 1)
        rr = []
        for cut in (2, 5, 10, 15, 20, 30, 50):
            mm = (etp <= cut).fillna(False)
            dd = px.index[mm.values & r.notna().values]
            ep = declusters(dd, 10, px.index)
            s = summarize(r.loc[ep].values, f"h={h} etp<={cut}")
            s["n_days"] = len(dd)
            rr.append(s)
        show(rr, f"ladder h={h}")

    print("\n### 3b. definition neighbours (MA length x lookback), LONG IWM ###")
    out = []
    for ma in (5, 10, 21):
        for win in (126, 252, 504):
            e = pit_pctile(pc["etp"], ma, win).reindex(px.index).ffill(limit=3)
            mm = (e <= 10).fillna(False)
            row = {"ma": ma, "win": win}
            for h in (3, 10):
                r = vehicle_ret(px, [("IWM", 1.0)], h, 1)
                dd = px.index[mm.values & r.notna().values]
                ep = declusters(dd, 10, px.index)
                allm = 100 * r.dropna().mean()
                row[f"h{h}_mean"] = round(100 * r.loc[ep].mean(), 3)
                row[f"h{h}_edge"] = round(100 * r.loc[ep].mean() - allm, 3)
                row[f"h{h}_n"] = len(ep)
            out.append(row)
    print(pd.DataFrame(out).to_string(index=False))

    print("\n### 4. midterm split + concentration, LONG IWM ###")
    for h in (3, 10):
        r = vehicle_ret(px, [("IWM", 1.0)], h, 1)
        dd = px.index[m.values & r.notna().values]
        ep = declusters(dd, 10, px.index)
        v = r.loc[ep].values
        mt = ep.year % 4 == 2
        show([summarize(v[mt], f"h={h} MIDTERM"), summarize(v[~mt], f"h={h} non-mid")],
             f"midterm h={h}")
        order = np.argsort(-v)
        print(f"  h={h} N={len(v)} mean {100*v.mean():+.3f}%")
        for k in (1, 2, 3):
            keep = np.setdiff1d(np.arange(len(v)), order[:k])
            print(f"   drop-best-{k} {[str(ep[i].date()) for i in order[:k]]}: "
                  f"{100*v[keep].mean():+.3f}%")
        by = pd.Series(v, index=ep.year).groupby(level=0).agg(["count", "mean"])
        by["mean"] = (100 * by["mean"]).round(3)
        print(by.to_string())

    print("\n### 1. FULL BATTERY ###")
    variants = {f"etp<={c}": (etp <= c).fillna(False) for c in (5, 15, 20, 30)}
    for h in (3, 10):
        battery(px, m, [("IWM", 1.0)], h, "C2 ETP P/C <=10 pctile, LONG IWM",
                cost_bps=6.0, variants=variants, min_gap=10)


if __name__ == "__main__":
    main()
