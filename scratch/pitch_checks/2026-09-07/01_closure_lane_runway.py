"""Q4: the runway conditioner. Post-closure cells split by sessions to the
next scheduled macro print.

runway(k0) = trading sessions from the anchor session k0 to the NEXT
scheduled print strictly after it, over subjects {nfp, cpi, ppi,
fomc_decision} from data/macro_events.csv.

TODAY: anchor is Tue 2026-09-08. PPI is Thu 2026-09-10 (+2), CPI Fri
2026-09-11 (+3). So runway = 2 -> the SHORT half. The parked cell that says
post-closure long vol needs a CLEAR calendar (runway >= 4) is therefore NOT
the live cell; the short half is, and nobody in this repo has measured it.

GRID: 17 proxies x 5 horizons x 2 entry forms x 2 runway halves = 340 cells.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import _closure_common as C
from pitch_lab import summarize, sign_test


def welch_t(a, b):
    if len(a) < 2 or len(b) < 2:
        return np.nan
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return (a.mean() - b.mean()) / se if se > 0 else np.nan


def main() -> None:
    px = C.load_panel()
    cal = C.nyse_calendar(px)
    ct = C.closure_table(cal)
    rw = C.runway_map(cal)

    hol = ct[ct["kind"] == "holiday"].copy()
    hol["runway"] = rw.reindex(pd.DatetimeIndex(hol["anchor"])).values
    wknd = ct[ct["kind"] == "weekend"].copy()
    wknd["runway"] = rw.reindex(pd.DatetimeIndex(wknd["anchor"])).values

    short_h = pd.DatetimeIndex(hol.loc[hol["runway"] <= 3, "anchor"])
    long_h = pd.DatetimeIndex(hol.loc[hol["runway"] >= 4, "anchor"])
    short_w = pd.DatetimeIndex(wknd.loc[wknd["runway"] <= 3, "anchor"])
    long_w = pd.DatetimeIndex(wknd.loc[wknd["runway"] >= 4, "anchor"])
    ld = ct.loc[ct["labor_day"]].copy()
    ld["runway"] = rw.reindex(pd.DatetimeIndex(ld["anchor"])).values
    ld_short = pd.DatetimeIndex(ld.loc[ld["runway"] <= 3, "anchor"])

    print("=" * 112)
    print("Q4  RUNWAY SPLIT   sessions from k0 to the next nfp/cpi/ppi/fomc print")
    print("TODAY'S CELL: anchor 2026-09-08, PPI +2, CPI +3  ->  runway = 2  ->  SHORT half")
    print(f"holiday closures: runway<=3 N={len(short_h)}   runway>=4 N={len(long_h)}")
    print(f"weekend controls: runway<=3 N={len(short_w)}   runway>=4 N={len(long_w)}")
    print(f"Labor Day anchors with runway<=3: {len(ld_short)} of {len(ld)}  "
          f"({', '.join(str(d.year) for d in ld_short)})")
    print("runway distribution over holiday closures:",
          hol["runway"].value_counts().sort_index().to_dict())
    print("=" * 112)

    rows = []
    for cls, tkr in C.PROXIES:
        df = px[tkr]
        a_sh = C.anchors_on(df, short_h)
        a_lg = C.anchors_on(df, long_h)
        a_sw = C.anchors_on(df, short_w)
        a_lw = C.anchors_on(df, long_w)
        a_lds = C.anchors_on(df, ld_short)
        for h in C.HORIZONS:
            for form, ser in (("MOO", C.moo_series(df, h)),
                              ("MOC", C.moc_series(df, h))):
                v_s, _ = C.cell(ser, a_sh)
                v_l, _ = C.cell(ser, a_lg)
                v_sw, _ = C.cell(ser, a_sw)
                v_lw, _ = C.cell(ser, a_lw)
                v_ld, _ = C.cell(ser, a_lds)
                if len(v_s) == 0 and len(v_l) == 0:
                    continue
                rs, rl = summarize(v_s), summarize(v_l)
                cost = C.COST_RT_BPS.get(tkr, np.nan)
                wins = int((v_s > 0).sum())
                rows.append({
                    "class": cls, "proxy": tkr, "h": h, "form": form,
                    "n_short": rs.get("n", 0),
                    "short_mean": rs.get("mean_pct", np.nan),
                    "short_med": rs.get("median_pct", np.nan),
                    "short_hit": rs.get("hit", np.nan),
                    "short_t": rs.get("t", np.nan),
                    "short_signp": sign_test(wins, len(v_s)) if len(v_s) else np.nan,
                    "n_long": rl.get("n", 0),
                    "long_mean": rl.get("mean_pct", np.nan),
                    "long_hit": rl.get("hit", np.nan),
                    "long_t": rl.get("t", np.nan),
                    "short_minus_long": (rs.get("mean_pct", np.nan)
                                         - rl.get("mean_pct", np.nan)),
                    "t_short_vs_long": welch_t(v_s, v_l),
                    "wknd_short_mean": 100 * v_sw.mean() if len(v_sw) else np.nan,
                    "wknd_long_mean": 100 * v_lw.mean() if len(v_lw) else np.nan,
                    "short_vs_wkndshort": (rs.get("mean_pct", np.nan)
                                           - 100 * v_sw.mean() if len(v_sw) else np.nan),
                    "t_shortvswkndshort": welch_t(v_s, v_sw),
                    "n_LDshort": len(v_ld),
                    "LDshort_mean": 100 * v_ld.mean() if len(v_ld) else np.nan,
                    "x_cost_short": (abs(rs.get("mean_pct", np.nan)) * 100 / cost
                                     if cost == cost and cost else np.nan),
                })
    out = pd.DataFrame(rows)
    out.to_csv(C.HERE / "01_runway_grid.csv", index=False)
    print(f"\nCELLS TESTED IN THIS SCRIPT: {2 * len(out)} "
          f"({len(out)} short/long pairs)")

    for cls, tkr in C.PROXIES:
        sub = out[out["proxy"] == tkr]
        if sub.empty:
            continue
        print(f"\n--- {cls:14s} {tkr:9s}  (SHORT runway = today's cell) ---")
        sh = sub[["h", "form", "n_short", "short_mean", "short_med", "short_hit",
                  "short_t", "short_signp", "n_long", "long_mean", "long_hit",
                  "long_t", "short_minus_long", "t_short_vs_long",
                  "wknd_short_mean", "short_vs_wkndshort", "t_shortvswkndshort",
                  "x_cost_short"]].copy()
        for c in sh.columns:
            if sh[c].dtype.kind == "f":
                sh[c] = sh[c].round(3)
        print(sh.to_string(index=False))

    print("\n" + "=" * 112)
    print("SHORT-RUNWAY half ranked by |t| (RAW, uncharged for a 340-cell grid)")
    rk = out.reindex(out["short_t"].abs().sort_values(ascending=False).index)
    cols = ["class", "proxy", "h", "form", "n_short", "short_mean", "short_hit",
            "short_t", "short_signp", "long_mean", "long_t", "short_minus_long",
            "t_short_vs_long", "short_vs_wkndshort", "x_cost_short"]
    sh = rk[cols].head(22).copy()
    for c in sh.columns:
        if sh[c].dtype.kind == "f":
            sh[c] = sh[c].round(3)
    print(sh.to_string(index=False))

    print("\nRUNWAY CONTRAST ranked by |t_short_vs_long| -- does the calendar matter at all?")
    rk2 = out.reindex(out["t_short_vs_long"].abs().sort_values(ascending=False).index)
    sh2 = rk2[cols].head(18).copy()
    for c in sh2.columns:
        if sh2[c].dtype.kind == "f":
            sh2[c] = sh2[c].round(3)
    print(sh2.to_string(index=False))

    print("\nVOL LANE, the parked cell's own coordinates (long vol post-closure):")
    vol = out[out["proxy"].isin(["^VIX", "SVXY"])]
    sh3 = vol[cols].copy()
    for c in sh3.columns:
        if sh3[c].dtype.kind == "f":
            sh3[c] = sh3[c].round(3)
    print(sh3.to_string(index=False))

    print(f"\nshort-half cells |t| >= 3.0: {int((out['short_t'].abs() >= 3).sum())}, "
          f">= 2.5: {int((out['short_t'].abs() >= 2.5).sum())}, "
          f">= 2.0: {int((out['short_t'].abs() >= 2.0).sum())} of {len(out)}")
    print(f"runway-contrast cells |t_short_vs_long| >= 2.5: "
          f"{int((out['t_short_vs_long'].abs() >= 2.5).sum())}, >= 2.0: "
          f"{int((out['t_short_vs_long'].abs() >= 2.0).sum())} of {len(out)}")


if __name__ == "__main__":
    main()
