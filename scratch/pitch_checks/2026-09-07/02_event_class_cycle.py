"""Stage B1: cycle-year (midterm vs rest) split of the six named cells.

2026 is a MIDTERM year (2026 % 4 == 2) and the repo's own Event Sleeve is
built on a midterm INVERSION in exactly this pre-FOMC window (T1 long SPY in
non-midterm years, T2 short SPY in midterms). So the cycle split is not an
optional slice here, it is the first thing an adversarial checker will ask
for.

Part 0 is a CALIBRATION CHECK: re-run the sleeve's exact T1/T2 window
(MOC at decision-4 td -> MOO on the decision-day open) and confirm this
pipeline reproduces the house result before trusting anything it says about
the cells nobody has mapped.

Parts 1-2 are the new cross-asset ground: pre-FOMC TLT / GLD / UUP / SVXY,
and the run-up INTO September quad witching on IWM (the opposite side of the
sleeve's T3, which is SHORT IWM from Sep opex to the Sep last session).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _eventgrid import event_dates, midterm_mask, runway, split_report  # noqa

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    anchor_positions, cluster_note, load_prices, pct_rank, sign_test,
    summarize,
)

pd.set_option("display.width", 200)

TICKERS = ["SPY", "IWM", "TLT", "GLD", "UUP", "SVXY", "IEF", "EFA"]


def sleeve_replication(px: dict) -> None:
    """T1/T2: MOC at fomc-4 td -> MOO on the decision-day open."""
    print("=" * 100)
    print("PART 0 -- CALIBRATION: Event Sleeve T1/T2 window reproduced")
    print("  prereg: T1 long SPY 25% NAV, MOC 4 sessions before decision -> "
          "MOO decision-day open, NON-midterm years only")
    print("          T2 short SPY 10%, same window, MIDTERM years, only when "
          "SPY 21d-return rank (252d, lag-1) < 50")
    print("=" * 100)
    spy = px["SPY"]
    idx = spy.index
    ev = event_dates("fomc_decision")
    pos, kept = anchor_positions(idx, ev, offset=0)
    rank21 = pct_rank(spy["Close"], 21, 252).shift(1)
    recs = []
    for p, d in zip(pos, kept):
        e_in = p - 4
        if e_in < 0 or p >= len(idx):
            continue
        recs.append({
            "entry": idx[e_in], "decision": d,
            "ret": float(spy["Open"].iloc[p] / spy["Close"].iloc[e_in] - 1.0),
            "rank21": float(rank21.iloc[e_in]),
        })
    r = pd.DataFrame(recs).set_index("entry")
    mm = midterm_mask(r.index)
    rows = []
    for lbl, m in (("ALL fomc", np.ones(len(r), bool)),
                   ("T1 cell: NON-midterm", ~mm),
                   ("T2 cell: MIDTERM (all)", mm),
                   ("T2 cell: MIDTERM & rank21<50", mm & (r["rank21"] < 50).values),
                   ("MIDTERM & rank21>=50 (excluded)",
                    mm & (r["rank21"] >= 50).values)):
        v = r["ret"].values[m]
        v = v[~np.isnan(v)]
        if len(v) == 0:
            rows.append({"cell": lbl, "n": 0})
            continue
        w = int((v > 0).sum())
        s = summarize(v, lbl)
        rows.append({"cell": lbl, "n": len(v),
                     "mean_pct": round(s["mean_pct"], 3),
                     "med_pct": round(s["median_pct"], 3),
                     "rec": f"{w}-{len(v)-w}", "hit": round(s["hit"], 1),
                     "p_coin_long": round(sign_test(w, len(v)), 4),
                     "p_coin_short": round(sign_test(len(v) - w, len(v)), 4),
                     "worst_pct": round(s["worst_pct"], 2),
                     "best_pct": round(s["best_pct"], 2)})
    print(pd.DataFrame(rows).to_string(index=False))
    print("\n  VERDICT: the sign flip between the T1 and T2 cells is the "
          "house result. If it is not visible above, this pipeline is wrong "
          "and nothing else in this folder should be believed.")
    print(f"  2026 is midterm -> the LIVE sleeve cell is T2 (SHORT SPY), "
          f"conditional on SPY 21d rank < 50 measured lag-1 at the entry.")


def named_cells(px: dict) -> None:
    print("\n" + "=" * 100)
    print("PART 1 -- pre-FOMC by cycle year. entry MOC at fomc-6 td "
          "(= the 2026-09-08 close), exits both ways.")
    print("  'pre'  exits MOC 2026-09-15, the session before the decision "
          "(h=5, no decision risk)")
    print("  'thru' exits MOC 2026-09-16, the decision close (h=6)")
    print("=" * 100)
    ev = event_dates("fomc_decision")
    for tkr in ["SPY", "TLT", "GLD", "UUP", "SVXY"]:
        c = px[tkr]["Close"]
        for mode, h in (("pre", 5), ("thru", 6)):
            r = runway(c, ev, 6, mode)
            split_report(f"fomc_decision x {tkr}  [{mode}]", r, c, h)
            mm = midterm_mask(r.index)
            if mm.sum() > 2:
                print("    midterm concentration:",
                      cluster_note(pd.DatetimeIndex(r.index[mm]),
                                   r["ret"].values[mm]))


def quad_iwm(px: dict) -> None:
    print("\n" + "=" * 100)
    print("PART 2 -- the RUN-UP into quad witching on IWM (the opposite side "
          "of the sleeve's T3, which shorts IWM FROM Sep opex).")
    print("  entry MOC at quad-8 td (= 2026-09-08 close); 'pre' exits the "
          "session before quad, 'thru' exits on the quad session.")
    print("=" * 100)
    c = px["IWM"]["Close"]
    for kind in ("quad_witching", "opex"):
        ev = event_dates(kind)
        for mode, h in (("pre", 7), ("thru", 8)):
            r = runway(c, ev, 8, mode)
            split_report(f"{kind} x IWM  [{mode}]  ALL MONTHS", r, c, h)
            sep = r[pd.DatetimeIndex(r["event"]).month == 9]
            if len(sep):
                v = sep["ret"].values
                w = int((v > 0).sum())
                s = summarize(v, "sep only")
                print(f"    SEPTEMBER ONLY: n={len(v)} mean {s['mean_pct']:+.3f}%"
                      f" med {s['median_pct']:+.3f}% rec {w}-{len(v)-w} "
                      f"p_coin={sign_test(w, len(v)):.4f} "
                      f"worst {s['worst_pct']:+.2f}% best {s['best_pct']:+.2f}%")
                smm = midterm_mask(sep.index)
                if smm.any():
                    vv = v[smm]
                    ww = int((vv > 0).sum())
                    print(f"    SEPTEMBER x MIDTERM: n={len(vv)} mean "
                          f"{100*vv.mean():+.3f}% rec {ww}-{len(vv)-ww} "
                          f"p_coin={sign_test(ww, len(vv)):.4f}  years "
                          f"{sorted(pd.DatetimeIndex(sep.index[smm]).year)}")


def flagged_cells_cycle(px: dict) -> None:
    print("\n" + "=" * 100)
    print("PART 3 -- cycle split of the cells the grid screen flagged "
          "(uncharged; this is a slice of a screen hit, not a finding)")
    print("=" * 100)
    for kind, tkr, k, mode, h in (
            ("vix_expiry", "SVXY", 6, "pre", 5),
            ("vix_expiry", "UUP", 6, "thru", 6),
            ("vix_expiry", "SLV", 6, "thru", 6),
            ("ppi", "USO", 2, "pre", 1)):
        if tkr not in px:
            px.update({t: v for t, v in load_prices([tkr]).items()})
        c = px[tkr]["Close"]
        r = runway(c, event_dates(kind), k, mode)
        split_report(f"{kind} x {tkr} [{mode}]", r, c, h)
        sep = r[pd.DatetimeIndex(r["event"]).month == 9]
        if len(sep):
            v = sep["ret"].values
            w = int((v > 0).sum())
            print(f"    SEPTEMBER ONLY: n={len(v)} mean {100*v.mean():+.3f}% "
                  f"med {100*np.median(v):+.3f}% rec {w}-{len(v)-w} "
                  f"p_coin={sign_test(w, len(v)):.4f}")


def main() -> None:
    px = load_prices(TICKERS + ["SLV", "USO"])
    sleeve_replication(px)
    named_cells(px)
    quad_iwm(px)
    flagged_cells_cycle(px)
    print("\nCELL COUNT for this script: 5 proxies x 2 exit modes x 3 cycle "
          "subsets (10 cells x 3) + 2 opex kinds x 2 modes x 3 subsets on IWM "
          "+ 4 flagged-cell splits x 3 = 78 sub-cells, all sliced OUT of the "
          "192-cell grid. Nothing here is independently charged; the slices "
          "make the grid's multiplicity WORSE, not better.")


if __name__ == "__main__":
    main()
