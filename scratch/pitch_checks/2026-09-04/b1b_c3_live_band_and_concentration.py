"""C3 round 2. Three probes that decide it.

1. BAND decomposition of the surprise conditioner. The <= -50k cut is a
   left-open bucket; today's live reading is -103k. Split it into
   (-100, -50] and <= -100 and ask which half carries the number, then ask
   which half TODAY is in.
2. CONCENTRATION by year. The battery reports best year 2021 at +7.5pp
   against a +6.43pp total, i.e. more than all of it. Recompute the cell
   with each year dropped.
3. THE LIVE HOLD's own calendar. The gated cell's whole edge sits in the
   cpi-in-hold sub-cell. Establish, by date, whether today's h=3 hold
   contains a CPI print.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

REL = Path(__file__).resolve().parents[3] / "data" / "macro_release_history.parquet"
TODAY = pd.Timestamp("2026-09-04")
LIVE = -103.0


def nfp_table() -> pd.DataFrame:
    df = pd.read_parquet(REL)
    n = df[df["event_name"] == "Non Farm Payrolls"].copy()
    n = n.dropna(subset=["surprise"]).sort_values("release_date")
    n = n.drop_duplicates(subset=["release_date"], keep="last")
    n = n[["release_date", "surprise"]].reset_index(drop=True)
    n["prior_surprise"] = n["surprise"].shift(1)
    n["midterm"] = n["release_date"].dt.year % 4 == 2
    return n.dropna(subset=["prior_surprise"])


def main() -> None:
    n = nfp_table()
    px = close_panel(["TLT", "IEF"]).dropna()
    pos, kept = anchor_positions(px.index, pd.DatetimeIndex(n["release_date"]),
                                 offset=-1)
    sub = n[np.isin(np.asarray(n["release_date"]), np.asarray(kept))].copy()
    sub["signal"] = pd.DatetimeIndex(px.index[pos])

    def cell(dates, tkr, h=3):
        return vehicle_ret(px, [(tkr, 1.0)], h, 1).reindex(dates).dropna()

    # ---------- 1. BAND DECOMPOSITION ----------
    print("=" * 78)
    print("1. BAND DECOMPOSITION of the prior-surprise conditioner (h=3, long)")
    print("=" * 78)
    print(f"   TODAY's live prior surprise = {LIVE:+.0f}k -> lands in the <= -100k band\n")
    bands = [(-1e9, -100, "<= -100k   *** THE LIVE BAND ***"),
             (-100, -75, "(-100, -75]"),
             (-75, -50, "( -75, -50]"),
             (-50, -25, "( -50, -25]"),
             (-25, 25, "( -25, +25]"),
             (25, 1e9, "> +25k")]
    for tkr in ("TLT", "IEF"):
        print(f"  {tkr}")
        for lo, hi, lbl in bands:
            m = (sub["prior_surprise"] > lo) & (sub["prior_surprise"] <= hi)
            v = cell(pd.DatetimeIndex(sub.loc[m, "signal"]), tkr)
            if not len(v):
                print(f"    {lbl:<34} N=0")
                continue
            w = int((v > 0).sum())
            print(f"    {lbl:<34} N={len(v):>3}  mean {100*v.mean():>+7.3f}%  "
                  f"med {100*v.median():>+7.3f}%  record {w}-{len(v)-w}  "
                  f"sign p={sign_test(w, len(v)):.4f}")
        # the two halves of the shipped <= -50k cut
        a = cell(pd.DatetimeIndex(sub.loc[(sub["prior_surprise"] > -100)
                                          & (sub["prior_surprise"] <= -50),
                                          "signal"]), tkr)
        b = cell(pd.DatetimeIndex(sub.loc[sub["prior_surprise"] <= -100,
                                          "signal"]), tkr)
        tot = cell(pd.DatetimeIndex(sub.loc[sub["prior_surprise"] <= -50,
                                            "signal"]), tkr)
        print(f"    -> shipped cut <= -50k: N={len(tot)} mean {100*tot.mean():+.3f}% "
              f"= moderate half ({len(a)} obs, {100*a.mean():+.3f}%) "
              f"+ LIVE half ({len(b)} obs, {100*b.mean():+.3f}%)")
        print(f"    -> total pp: moderate {100*a.sum():+.2f}pp, "
              f"live-band {100*b.sum():+.2f}pp, all {100*tot.sum():+.2f}pp\n")

    # ---------- 1b. midterm crossed with the LIVE band ----------
    print("1b. THE ACTUAL LIVE CELL: prior <= -100k AND midterm year")
    for tkr in ("TLT", "IEF"):
        m = (sub["prior_surprise"] <= -100) & sub["midterm"]
        v = cell(pd.DatetimeIndex(sub.loc[m, "signal"]), tkr)
        w = int((v > 0).sum())
        print(f"   {tkr}: N={len(v)}  mean {100*v.mean():+.3f}%  record {w}-{len(v)-w}  "
              f"dates {[str(d.date()) for d in sub.loc[m,'release_date']]}")

    # ---------- 2. YEAR CONCENTRATION ----------
    print("\n" + "=" * 78)
    print("2. DROP-ONE-YEAR on the shipped cell (TLT, h=3, prior <= -50k)")
    print("=" * 78)
    g = pd.DatetimeIndex(sub.loc[sub["prior_surprise"] <= -50, "signal"])
    v = cell(g, "TLT")
    yrs = pd.DatetimeIndex(v.index).year
    print(f"   full cell N={len(v)}  mean {100*v.mean():+.3f}%  "
          f"total {100*v.sum():+.2f}pp")
    by = pd.Series(100 * v.values).groupby(yrs.values).agg(["count", "sum", "mean"])
    print("\n   contribution by year (pp):")
    print(by.round(3).to_string())
    print("\n   leave-one-year-out:")
    rows = []
    for y in sorted(set(yrs)):
        vv = v[yrs != y]
        rows.append((y, len(vv), 100 * vv.mean(), 100 * vv.sum(),
                     100 * (vv > 0).mean()))
    for y, nn, mn, tt, hh in rows:
        flag = "  <-- SIGN FLIPS" if mn < 0 else ""
        print(f"     drop {y}: N={nn:>3}  mean {mn:>+7.3f}%  total {tt:>+7.2f}pp  "
              f"hit {hh:>5.1f}%{flag}")

    # ---------- 3. TODAY'S OWN HOLD CALENDAR ----------
    print("\n" + "=" * 78)
    print("3. TODAY'S HOLD: does a CPI print land inside it?")
    print("=" * 78)
    ev = load_events(["cpi", "ppi"])
    fut = ev[(ev["date"] >= TODAY) & (ev["date"] <= TODAY + pd.Timedelta(days=30))]
    print(fut.to_string(index=False))
    print("\n   Entry MOC 2026-09-04. 2026-09-07 is Labor Day (closed).")
    print("   h=3 sessions: 09-08, 09-09, 09-10 -> exit MOC 2026-09-10.")
    print("   PPI 2026-09-10 is inside; CPI 2026-09-11 is NOT.")
    print("   -> today lands in the cpi-OUT sub-cell.")
    for tkr in ("TLT", "IEF"):
        fl = event_in_window(g, px.index, 3, 1, ("cpi",))
        vv = cell(g, tkr).values
        fl = fl[: len(vv)]
        print(f"   {tkr} cpi-IN  N={int(fl.sum()):>2} mean {100*vv[fl].mean():+.3f}% "
              f"hit {100*(vv[fl]>0).mean():.1f}%   |   "
              f"cpi-OUT N={int((~fl).sum()):>2} mean {100*vv[~fl].mean():+.3f}% "
              f"hit {100*(vv[~fl]>0).mean():.1f}%  <-- TODAY")


if __name__ == "__main__":
    main()
