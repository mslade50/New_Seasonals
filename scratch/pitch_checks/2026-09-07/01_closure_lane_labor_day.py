"""Q3: the LABOR DAY closure specifically, separated from the other holidays.

Labor Day anchor = the Tuesday after the first Monday of September that
follows a >= 4 calendar-day NYSE closure. 26 of them in the 2000-2025 cache
(2026's is the session this survey is for, so it carries no forward data).

N ~ 26 per proxy, less for late-inception vehicles. The honest statistic at
this N is pitch_lab.sign_test on the record, NOT a t-stat, per the house
doctrine. t is printed only as context and must not be quoted as the test.

GRID: 17 proxies x 5 horizons x 2 entry forms = 170 cells, all at N<=26.
Read every one of them as a screen hit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import _closure_common as C
from pitch_lab import summarize, sign_test


def main() -> None:
    px = C.load_panel()
    cal = C.nyse_calendar(px)
    ct = C.closure_table(cal)
    ld = pd.DatetimeIndex(ct.loc[ct["labor_day"], "anchor"])
    other = pd.DatetimeIndex(
        ct.loc[(ct["kind"] == "holiday") & (~ct["labor_day"]), "anchor"])

    print("=" * 110)
    print("Q3  LABOR DAY ONLY   (the September closure, split out from the rest)")
    print(f"Labor Day anchors in cache: {len(ld)}  "
          f"{ld[0].date()} .. {ld[-1].date()}")
    print(f"other holiday closures (the contrast cell): {len(other)}")
    print("sign p = pitch_lab.sign_test(wins, n) one-sided vs a coin. N is small; "
          "small N is not a kill here, it is a stated limit.")
    print("=" * 110)

    rows = []
    for cls, tkr in C.PROXIES:
        df = px[tkr]
        a_ld = C.anchors_on(df, ld)
        a_ot = C.anchors_on(df, other)
        for h in C.HORIZONS:
            for form, ser in (("MOO", C.moo_series(df, h)),
                              ("MOC", C.moc_series(df, h))):
                v, d = C.cell(ser, a_ld)
                v_o, _ = C.cell(ser, a_ot)
                allv = ser.dropna().values.astype(float)
                r = summarize(v)
                if not r.get("n"):
                    continue
                wins = int((v > 0).sum())
                n = len(v)
                base = float((allv > 0).mean())
                cost = C.COST_RT_BPS.get(tkr, np.nan)
                rows.append({
                    "class": cls, "proxy": tkr, "h": h, "form": form,
                    "n": n, "record": f"{wins}-{n - wins}",
                    "mean_pct": r["mean_pct"], "med_pct": r["median_pct"],
                    "hit": r["hit"],
                    "sign_p_up": sign_test(wins, n),
                    "sign_p_dn": sign_test(n - wins, n),
                    "sign_p_vs_base": sign_test(wins, n, base),
                    "base_hit": 100 * base,
                    "other_hol_mean": 100 * v_o.mean() if len(v_o) else np.nan,
                    "allday_mean": 100 * allv.mean(),
                    "worst_pct": r["worst_pct"], "best_pct": r["best_pct"],
                    "t_context": r["t"],
                    "x_cost": (abs(r["mean_pct"]) * 100 / cost
                               if cost == cost and cost else np.nan),
                })
    out = pd.DataFrame(rows)
    out.to_csv(C.HERE / "01_laborday_grid.csv", index=False)
    print(f"\nCELLS TESTED IN THIS SCRIPT: {len(out)}")

    for cls, tkr in C.PROXIES:
        sub = out[out["proxy"] == tkr]
        if sub.empty:
            continue
        print(f"\n--- {cls:14s} {tkr:9s} ---")
        sh = sub[["h", "form", "n", "record", "mean_pct", "med_pct", "hit",
                  "sign_p_up", "sign_p_dn", "sign_p_vs_base", "base_hit",
                  "other_hol_mean", "allday_mean", "worst_pct", "best_pct",
                  "t_context", "x_cost"]].copy()
        for c in sh.columns:
            if sh[c].dtype.kind == "f":
                sh[c] = sh[c].round(3)
        print(sh.to_string(index=False))

    print("\n" + "=" * 110)
    print("BEST RECORDS (min(sign_p_up, sign_p_dn) <= 0.10), RAW, uncharged for 170 cells")
    out["best_p"] = out[["sign_p_up", "sign_p_dn"]].min(axis=1)
    rk = out[out["best_p"] <= 0.10].sort_values("best_p")
    sh = rk[["class", "proxy", "h", "form", "n", "record", "mean_pct", "med_pct",
             "hit", "best_p", "sign_p_vs_base", "base_hit", "other_hol_mean",
             "worst_pct", "x_cost"]].copy()
    for c in sh.columns:
        if sh[c].dtype.kind == "f":
            sh[c] = sh[c].round(4)
    print(sh.to_string(index=False))
    print(f"\ncells with sign p <= 0.10 in either direction: {len(rk)} of {len(out)}. "
          f"Expected by chance at ~0.20 two-sided coverage: ~{0.20*len(out):.0f}.")

    # per-year detail on the two most-watched proxies
    for tkr in ("SPY", "GLD"):
        df = px[tkr]
        a_ld = C.anchors_on(df, ld)
        s = C.moc_series(df, 3)
        v, d = C.cell(s, a_ld)
        print(f"\n{tkr} Labor Day MOC h=3 year by year:")
        print("  " + "  ".join(f"{x.year}:{100*y:+.2f}" for x, y in zip(d, v)))


if __name__ == "__main__":
    main()
