"""Q1: bare post-closure drift, 17 proxies x 5 horizons x 2 entry forms.

Cell = first NYSE session after a >= 4 calendar-day closure (k=0).
Controls: (a) the instrument's own all-days drift at the same horizon and
entry form, (b) the ordinary 3-calendar-day weekend cell, (c) the local
+/-126td neighbourhood ex-trigger (pitch_lab.local_control).

GRID SIZE: 17 proxies x 5 horizons x 2 entry forms = 170 primary cells.
Nothing here is charged for that; every headline number below is a RAW
per-cell statistic and must be read as a screen hit, not a finding.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import _closure_common as C
from pitch_lab import local_control, summarize


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return np.nan
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return (a.mean() - b.mean()) / se if se > 0 else np.nan


def main() -> None:
    px = C.load_panel()
    cal = C.nyse_calendar(px)
    ct = C.closure_table(cal)
    hol = pd.DatetimeIndex(ct.loc[ct["kind"] == "holiday", "anchor"])
    wkd = pd.DatetimeIndex(ct.loc[ct["kind"] == "weekend", "anchor"])
    hol_sched = pd.DatetimeIndex(
        ct.loc[(ct["kind"] == "holiday") & (~ct["unscheduled"]), "anchor"])

    print("=" * 100)
    print("Q1  POST-CLOSURE DRIFT  |  anchor k=0 = first NYSE session after the gap")
    print(f"NYSE calendar: {len(cal)} sessions {cal[0].date()} .. {cal[-1].date()}")
    print(f"holiday closures (gap>=4): {len(hol)}   ordinary weekends (gap==3): {len(wkd)}")
    print(f"scheduled-holiday subset (ex 2001-09-17, 2012-10-31): {len(hol_sched)}")
    print("MOO(h): Open[k0] -> Close[k0+h-1].   MOC(h): Close[k0] -> Close[k0+h].")
    print("Both hold exactly h sessions; both are lag=1 (decision precedes k0).")
    print("=" * 100)

    rows = []
    for cls, tkr in C.PROXIES:
        df = px[tkr]
        a_hol = C.anchors_on(df, hol)
        a_wkd = C.anchors_on(df, wkd)
        a_sch = C.anchors_on(df, hol_sched)
        loc_dates = local_control(pd.DatetimeIndex(df.index), a_hol, win=126)
        for h in C.HORIZONS:
            for form, ser in (("MOO", C.moo_series(df, h)),
                              ("MOC", C.moc_series(df, h))):
                v_h, d_h = C.cell(ser, a_hol)
                v_w, _ = C.cell(ser, a_wkd)
                v_s, _ = C.cell(ser, a_sch)
                all_v = ser.dropna().values.astype(float)
                v_l, _ = C.cell(ser, loc_dates)
                r = summarize(v_h)
                if not r.get("n"):
                    continue
                cost = C.COST_RT_BPS.get(tkr, np.nan)
                edge_bps = r["mean_pct"] * 100.0
                rows.append({
                    "class": cls, "proxy": tkr, "h": h, "form": form,
                    "n": r["n"], "mean_pct": r["mean_pct"],
                    "med_pct": r["median_pct"], "hit": r["hit"], "t": r["t"],
                    "sd_pct": r["sd_pct"],
                    "ctrl_all_pct": 100 * all_v.mean(),
                    "ctrl_wknd_pct": 100 * v_w.mean() if len(v_w) else np.nan,
                    "n_wknd": len(v_w),
                    "ctrl_loc_pct": 100 * v_l.mean() if len(v_l) else np.nan,
                    "edge_vs_all": r["mean_pct"] - 100 * all_v.mean(),
                    "edge_vs_wknd": (r["mean_pct"] - 100 * v_w.mean()
                                     if len(v_w) else np.nan),
                    "t_vs_wknd": welch_t(v_h, v_w),
                    "sched_mean_pct": 100 * v_s.mean() if len(v_s) else np.nan,
                    "n_sched": len(v_s),
                    "cost_rt_bps": cost,
                    "x_cost": abs(edge_bps) / cost if cost == cost and cost else np.nan,
                })
    out = pd.DataFrame(rows)
    out.to_csv(C.HERE / "01_drift_grid.csv", index=False)

    print(f"\nCELLS TESTED IN THIS SCRIPT: {len(out)}")

    for cls, tkr in C.PROXIES:
        sub = out[out["proxy"] == tkr]
        if sub.empty:
            continue
        print(f"\n--- {cls:14s} {tkr:9s}  cost RT {C.COST_RT_BPS.get(tkr)} bps ---")
        show = sub[["h", "form", "n", "mean_pct", "med_pct", "hit", "t",
                    "ctrl_all_pct", "ctrl_wknd_pct", "n_wknd", "ctrl_loc_pct",
                    "edge_vs_all", "edge_vs_wknd", "t_vs_wknd",
                    "sched_mean_pct", "x_cost"]].copy()
        for c in show.columns:
            if show[c].dtype.kind == "f":
                show[c] = show[c].round(3)
        print(show.to_string(index=False))

    print("\n" + "=" * 100)
    print("RANKED by |edge vs the 3-day-weekend control| (RAW, uncharged for a 170-cell grid)")
    rk = out.reindex(out["edge_vs_wknd"].abs().sort_values(ascending=False).index)
    cols = ["class", "proxy", "h", "form", "n", "mean_pct", "hit", "t",
            "ctrl_wknd_pct", "edge_vs_wknd", "t_vs_wknd", "x_cost"]
    sh = rk[cols].head(25).copy()
    for c in sh.columns:
        if sh[c].dtype.kind == "f":
            sh[c] = sh[c].round(3)
    print(sh.to_string(index=False))

    print("\nRANKED by |t| of the holiday cell itself (RAW)")
    rk2 = out.reindex(out["t"].abs().sort_values(ascending=False).index)
    sh2 = rk2[cols].head(20).copy()
    for c in sh2.columns:
        if sh2[c].dtype.kind == "f":
            sh2[c] = sh2[c].round(3)
    print(sh2.to_string(index=False))

    print("\nBonferroni reference for a 170-cell grid: |t| ~ 3.5 buys p~0.05 family-wise.")
    print(f"cells with |t| >= 3.5: {int((out['t'].abs() >= 3.5).sum())}   "
          f"|t| >= 2.5: {int((out['t'].abs() >= 2.5).sum())}   "
          f"|t| >= 2.0: {int((out['t'].abs() >= 2.0).sum())} of {len(out)}")
    trad = out[out["x_cost"].notna()]
    print(f"cells clearing the 3x-cost slot bar: "
          f"{int((trad['x_cost'] >= 3).sum())} of {len(trad)} tradeable cells")


if __name__ == "__main__":
    main()
