"""Q2: the closure gap itself. Open[k0] / Close[k0-1] - 1.

Is the gap over a >= 4 calendar-day NYSE closure systematically signed, and
does it differ from an ordinary 3-day weekend gap?

STATED PLAINLY: neither entry form in this survey captures this leg. A MOO
order fills AT Open[k0] and a MOC order fills at Close[k0]; both are placed
after the closure and therefore after the gap has printed. The gap is a
CONDITIONER (knowable at 9:30 before a MOC entry), not a tradeable leg,
unless the position is put on before the closure -- which today's calendar
does not allow (the decision is being made on the holiday itself).

GRID: 17 proxies x 1 statistic x {holiday, weekend, all-days} = 17 comparisons.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import _closure_common as C
from pitch_lab import summarize


def welch_t(a, b):
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
    ld = pd.DatetimeIndex(ct.loc[ct["labor_day"], "anchor"])

    print("=" * 108)
    print("Q2  THE CLOSURE GAP   Open[k0] / Close[k0-1] - 1")
    print("NOT capturable by either entry form today; reported as a state variable.")
    print("DX-Y.NYB CAVEAT: the dollar index trades some NYSE holidays, so its")
    print("'prior bar' can be inside the NYSE closure -- its gap is not always 4 days.")
    print("=" * 108)

    rows = []
    for cls, tkr in C.PROXIES:
        df = px[tkr]
        g = C.gap_series(df)
        a_h = C.anchors_on(df, hol)
        a_w = C.anchors_on(df, wkd)
        a_l = C.anchors_on(df, ld)
        v_h, _ = C.cell(g, a_h)
        v_w, _ = C.cell(g, a_w)
        v_l, _ = C.cell(g, a_l)
        all_v = g.dropna().values.astype(float)
        rh, rw, rl = summarize(v_h), summarize(v_w), summarize(v_l)
        rows.append({
            "class": cls, "proxy": tkr,
            "n_hol": rh["n"], "hol_mean": rh["mean_pct"], "hol_med": rh["median_pct"],
            "hol_hit": rh["hit"], "hol_t": rh["t"],
            "n_wknd": rw["n"], "wknd_mean": rw["mean_pct"], "wknd_hit": rw["hit"],
            "allday_mean": 100 * all_v.mean(),
            "hol_minus_wknd": rh["mean_pct"] - rw["mean_pct"],
            "t_hol_vs_wknd": welch_t(v_h, v_w),
            "hol_sd": rh["sd_pct"], "wknd_sd": rw["sd_pct"],
            "sd_ratio": rh["sd_pct"] / rw["sd_pct"] if rw["sd_pct"] else np.nan,
            "n_LD": rl["n"], "LD_mean": rl["mean_pct"], "LD_hit": rl["hit"],
        })
    out = pd.DataFrame(rows)
    out.to_csv(C.HERE / "01_gap_grid.csv", index=False)

    disp = out.copy()
    for c in disp.columns:
        if disp[c].dtype.kind == "f":
            disp[c] = disp[c].round(3)
    print("\n-- signed gap, holiday vs ordinary weekend vs all overnight sessions --")
    print(disp[["class", "proxy", "n_hol", "hol_mean", "hol_med", "hol_hit", "hol_t",
                "n_wknd", "wknd_mean", "wknd_hit", "allday_mean",
                "hol_minus_wknd", "t_hol_vs_wknd"]].to_string(index=False))

    print("\n-- gap DISPERSION: does a 4-day closure widen the open? --")
    print(disp[["class", "proxy", "hol_sd", "wknd_sd", "sd_ratio"]].to_string(index=False))

    print("\n-- Labor Day gaps only --")
    print(disp[["class", "proxy", "n_LD", "LD_mean", "LD_hit"]].to_string(index=False))

    print(f"\ncells: {len(out)} proxies x 1 gap statistic. "
          f"|t_hol_vs_wknd| >= 2: {int((out['t_hol_vs_wknd'].abs() >= 2).sum())}, "
          f">= 2.5: {int((out['t_hol_vs_wknd'].abs() >= 2.5).sum())}")


if __name__ == "__main__":
    main()
