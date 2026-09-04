"""a6 / C6: HYG prints a fresh 52-week high (within 0.05% of its trailing-252
max) while SPY is at least 1% BELOW its own. Vehicle SPY, long.

The registry hands this one two shapes to beat before anything else:
  - "An equity dip with credit refusing to confirm it" (2026-08-24): the
    credit gate was worth -0.022pp and the reference class over 14 vehicles
    put HYG 7 of 14. Duration wearing a credit label, three times.
  - "Jackson Hole on credit" (2026-08-24): HYG's excess sits BELOW SPY's, so
    credit SUBTRACTS.
So GATE ATTRIBUTION and the REFERENCE CLASS run first, not last. If "SPY 1%
off its high" alone pays the same thing, C6 is "buy a 1.5% dip" with a
credit label.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (battery, close_panel, declusters, horizon_scan,  # noqa
                       rolling_on_valid, show, sign_test, summarize,
                       vehicle_ret, bootstrap_p_le0)

REF = ["HYG", "LQD", "IEF", "TLT", "XLU", "XLK", "XLP", "XLV", "XLF", "EFA",
       "EEM", "GLD", "QQQ", "IWM"]


def off_high(px: pd.DataFrame, t: str, n: int = 252) -> pd.Series:
    hi = rolling_on_valid(px[t], lambda x: x.rolling(n).max())
    return px[t] / hi - 1.0


def main() -> None:
    tick = sorted(set(REF + ["SPY"]))
    px = close_panel(tick)
    oh = {t: off_high(px, t) for t in tick}
    print("live state 2026-08-25: " + "  ".join(
        f"{t} {100*oh[t].iloc[-1]:+.2f}%" for t in ["HYG", "SPY", "LQD", "IEF"]))

    hyg_hi = (oh["HYG"] >= -0.0005)
    spy_off = (oh["SPY"] <= -0.01)
    m_joint = (hyg_hi & spy_off).fillna(False)
    m_spy = spy_off.fillna(False)
    m_hyg = hyg_hi.fillna(False)
    m_comp = (spy_off & ~hyg_hi).fillna(False)

    span0 = px.index[m_joint.values].min()
    pxs = px.loc[span0:]
    for k in (m_joint, m_spy, m_hyg, m_comp):
        pass
    d_joint = pxs.index[m_joint.loc[pxs.index].values]
    print(f"\njoint days {len(d_joint)}  span {d_joint.min().date()}.."
          f"{d_joint.max().date()}  episodes(gap10)="
          f"{len(declusters(d_joint, 10, pxs.index))}")
    print("  P(HYG at high | SPY >=1% off) = "
          f"{m_joint.loc[pxs.index].sum() / max(m_spy.loc[pxs.index].sum(),1):.3f}"
          f"   (SPY-off days in span = {int(m_spy.loc[pxs.index].sum())})")
    print("  years:", pd.Series(1, index=d_joint).groupby(d_joint.year).sum().to_dict())

    print("\n### 0. horizon scan, LONG SPY ###")
    show(horizon_scan(pxs, d_joint, [("SPY", 1.0)], hs=(1, 2, 3, 5, 7, 10),
                      min_gap=10), "JOINT cell")

    print("\n### 2. GATE ATTRIBUTION: does the credit leg add anything? ###")
    rows = []
    for h in (1, 2, 3, 5, 10):
        r = vehicle_ret(pxs, [("SPY", 1.0)], h, 1)
        allm = 100 * r.dropna().mean()
        cells = {}
        for lbl, m in [("SPY>=1% off (parent)", m_spy),
                       ("joint (+HYG at high)", m_joint),
                       ("complement (SPY off, HYG not)", m_comp),
                       ("HYG at high alone", m_hyg)]:
            dd = pxs.index[m.loc[pxs.index].values & r.loc[pxs.index].notna().values]
            ep = declusters(dd, 10, pxs.index)
            cells[lbl] = (len(ep), 100 * r.loc[ep].mean())
        rows.append({"h": h, "all_days": round(allm, 3),
                     **{f"{k}": f"{v[1]:+.3f} (n={v[0]})" for k, v in cells.items()},
                     "gate_value_pp": round(cells["joint (+HYG at high)"][1]
                                            - cells["SPY>=1% off (parent)"][1], 3)})
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n### 2b. gate threshold walk: how far off its high may HYG be? ###")
    for h in (3, 10):
        r = vehicle_ret(pxs, [("SPY", 1.0)], h, 1)
        out = []
        for tol in (0.0005, 0.002, 0.005, 0.01, 0.02, 0.05, 1.0):
            m = (spy_off & (oh["HYG"] >= -tol)).fillna(False)
            dd = pxs.index[m.loc[pxs.index].values & r.loc[pxs.index].notna().values]
            ep = declusters(dd, 10, pxs.index)
            s = summarize(r.loc[ep].values, f"h={h} HYG within {100*tol:.2f}%")
            s["n_days"] = len(dd)
            out.append(s)
        show(out, f"toward-inert walk h={h} (last row = NO credit gate at all)")

    print("\n### 3. REFERENCE CLASS: 'X at a 52w high while SPY is 1% off' ###")
    for h in (3, 10):
        r = vehicle_ret(pxs, [("SPY", 1.0)], h, 1)
        allm = 100 * r.dropna().mean()
        base_m = m_spy.loc[pxs.index]
        base_d = pxs.index[base_m.values & r.loc[pxs.index].notna().values]
        base_ep = declusters(base_d, 10, pxs.index)
        base_mean = 100 * r.loc[base_ep].mean()
        out = []
        for t in REF:
            m = (spy_off & (oh[t] >= -0.0005)).fillna(False)
            dd = pxs.index[m.loc[pxs.index].values & r.loc[pxs.index].notna().values]
            if len(dd) == 0:
                out.append({"vehicle": t, "n_ep": 0}); continue
            ep = declusters(dd, 10, pxs.index)
            out.append({"vehicle": t, "n_ep": len(ep),
                        "mean_pct": round(100 * r.loc[ep].mean(), 3),
                        "gate_pp": round(100 * r.loc[ep].mean() - base_mean, 3),
                        "hit": round(100 * float((r.loc[ep] > 0).mean()), 1)})
        df = pd.DataFrame(out).sort_values("gate_pp", ascending=False)
        print(f"\n  h={h}: parent 'SPY >=1% off high' = {base_mean:+.3f}% "
              f"(n={len(base_ep)}), all days {allm:+.3f}%")
        print(df.to_string(index=False))
        r_hyg = df[df.vehicle == "HYG"]
        pos = int((df["gate_pp"] > 0).sum())
        print(f"  HYG rank {list(df.vehicle).index('HYG')+1} of {len(df)};"
              f" {pos} of {len(df)} vehicles have a positive gate value;"
              f" mean gate value {df['gate_pp'].mean():+.3f}pp")

    print("\n### 4. dial conditioning: what is the MAX dial this cell has seen? ###")
    fr = pd.read_parquet(Path(__file__).resolve().parents[3] / "data"
                         / "rd2_fragility.parquet")
    ma = fr["63d"].rolling(10, min_periods=10).mean()
    ma.index = pd.to_datetime(ma.index)
    dj = pd.DatetimeIndex(d_joint).intersection(ma.dropna().index)
    print(f"  cell days with a dial reading: {len(dj)} of {len(d_joint)}")
    if len(dj):
        v = ma.loc[dj]
        print(f"  max ma10(63d) on a cell day = {v.max():.1f} "
              f"(on {v.idxmax().date()});  today = {ma.iloc[-1]:.1f}")
        print(f"  cell days >= 85: {int((v >= 85).sum())};  >= 70: "
              f"{int((v >= 70).sum())};  >= 50: {int((v >= 50).sum())}")
        for h in (3, 10):
            r = vehicle_ret(pxs, [("SPY", 1.0)], h, 1)
            hi = dj[v >= 50]
            lo = dj[v < 50]
            show([summarize(r.reindex(hi).values, f"h={h} dial>=50 (N={len(hi)})"),
                  summarize(r.reindex(lo).values, f"h={h} dial<50 (N={len(lo)})")],
                 f"dial split h={h} (day level, cell days with a reading)")

    print("\n### 5. era + midterm split, concentration ###")
    for h in (3, 10):
        r = vehicle_ret(pxs, [("SPY", 1.0)], h, 1)
        dd = pxs.index[m_joint.loc[pxs.index].values & r.loc[pxs.index].notna().values]
        ep = declusters(dd, 10, pxs.index)
        v = r.loc[ep].values
        pre = ep < pd.Timestamp("2018-01-01")
        mt = ep.year % 4 == 2
        show([summarize(v[pre], f"h={h} pre-2018"), summarize(v[~pre], f"h={h} 2018+"),
              summarize(v[mt], f"h={h} MIDTERM"), summarize(v[~mt], f"h={h} non-mid")],
             f"splits h={h}")
        order = np.argsort(-v)
        print(f"  h={h} N={len(v)} mean {100*v.mean():+.3f}%  "
              f"boot P(mean<=0)={bootstrap_p_le0(v):.3f}")
        for k in (1, 2, 3):
            keep = np.setdiff1d(np.arange(len(v)), order[:k])
            print(f"   drop-best-{k} {[str(ep[i].date()) for i in order[:k]]}: "
                  f"{100*v[keep].mean():+.3f}%")
        by = pd.Series(v, index=ep.year).groupby(level=0).agg(["count", "mean"])
        by["mean"] = (100 * by["mean"]).round(3)
        print(by.to_string())

    print("\n### 1. FULL BATTERY ###")
    variants = {
        "SPY>=0.5% off": (hyg_hi & (oh["SPY"] <= -0.005)).fillna(False),
        "SPY>=2% off": (hyg_hi & (oh["SPY"] <= -0.02)).fillna(False),
        "SPY>=3% off": (hyg_hi & (oh["SPY"] <= -0.03)).fillna(False),
        "HYG within 0.5%": (spy_off & (oh["HYG"] >= -0.005)).fillna(False),
        "SPY off, no credit gate": m_spy,
    }
    for h in (3, 10):
        battery(pxs, m_joint.loc[pxs.index], [("SPY", 1.0)], h,
                "C6 HYG fresh 52w high while SPY >=1% off, LONG SPY",
                cost_bps=6.0, variants=variants, min_gap=10)


if __name__ == "__main__":
    main()
