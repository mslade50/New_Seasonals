"""Stage B1: the full LIVE-EVENT x ASSET-CLASS cross, 2026-09-07.

Every scheduled event inside the [-5, +15] td window, crossed with one or two
proxies per asset class, measured as the PRE-EVENT RUNWAY a position opened
on the next session (2026-09-08) would actually get. Convention, controls and
the no-lag justification live in _eventgrid.py's docstring.

This is a SEARCH. The cell count is printed at the top and again at the
bottom; anything the screen flags is UNCHARGED for multiplicity and is a
candidate for stage C, never a finding.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _eventgrid import (  # noqa: E402
    CLASS_OF, CLASSES, LIVE_EVENTS, MIN_EDGE_BPS, NON_TRADEABLE, PROXIES,
    cell_stats, event_dates, is_pulse, load_closes,
)

pd.set_option("display.width", 220)


def main() -> None:
    n_events, n_proxies, n_modes = len(LIVE_EVENTS), len(PROXIES), 2
    total = n_events * n_proxies * n_modes
    print("=" * 100)
    print("STAGE B1 EVENT x CLASS CROSS -- asof 2026-09-07, next session "
          "2026-09-08, freshest bar 2026-09-04")
    print(f"GRID: {n_events} live events x {n_proxies} proxies x {n_modes} "
          f"exit modes = {total} cells. Every number below is UNCHARGED for "
          f"that search.")
    print("modes: pre  = enter MOC at event-k, exit MOC the session BEFORE "
          "the event (h=k-1, no event risk)")
    print("       thru = same entry, exit MOC ON the event session (h=k, "
          "carries the print)")
    print(f"cost gate: ~5 bps round trip, 3x = {MIN_EDGE_BPS:.0f} bps mean "
          "before a cell can pay a slot")
    print("=" * 100)

    closes = load_closes()
    evd = {k: event_dates(k) for k, _, _ in LIVE_EVENTS}

    all_rows = []
    for kind, dstr, k in LIVE_EVENTS:
        for mode in ("pre", "thru"):
            h = k - 1 if mode == "pre" else k
            rows = []
            for tkr in PROXIES:
                c = cell_stats(closes[tkr], evd[kind], k, mode)
                rec = {
                    "class": CLASS_OF[tkr], "proxy": tkr,
                    "n": c["n"], "h": h,
                    "mean_pct": round(c.get("mean_pct", float("nan")), 3),
                    "med_pct": round(c.get("med_pct", float("nan")), 3),
                    "hit": round(c.get("hit", float("nan")), 1),
                    "rec": c.get("rec", ""),
                    "p_coin": round(c.get("p_coin", float("nan")), 4),
                    "p_dir": round(c.get("p_dir", float("nan")), 4),
                    "ctrlA_pct": round(c.get("ctrl_a_pct", float("nan")), 3),
                    "ctrlC_pct": round(c.get("ctrl_c_pct", float("nan")), 3),
                    "edge_bps": round(c.get("edge_bps", float("nan")), 1),
                    "from": str(c.get("first", "")),
                }
                rec["FLAG"] = "*" if is_pulse(c) else ""
                if tkr in NON_TRADEABLE:
                    rec["FLAG"] += "(ref)"
                rows.append(rec)
                all_rows.append({"event": kind, "mode": mode, **rec})
            print(f"\n########## {kind}  {dstr}  (k={k} td from 2026-09-08) "
                  f"| mode={mode}, hold h={h} sessions ##########")
            print(pd.DataFrame(rows).to_string(index=False))

    df = pd.DataFrame(all_rows)
    df.to_csv(Path(__file__).with_name("02_event_class_grid.csv"), index=False)

    print("\n" + "=" * 100)
    print("SCREEN FLAGS (n>=8, |edge over own drift|>=15bps, mean and median "
          "same sign, directional sign p vs own base rate <=0.10)")
    print("=" * 100)
    fl = df[df["FLAG"].str.startswith("*")].sort_values(
        "p_dir").reset_index(drop=True)
    if fl.empty:
        print("  none")
    else:
        print(fl.to_string(index=False))
    print(f"\n{len(fl)} flags out of {total} cells tested. Expected pure-noise "
          f"flags at the 0.10 one-sided gate on {total} correlated cells: "
          f"order {0.10*total:.0f}. THE FLAGS ARE UNCHARGED.")

    print("\n" + "=" * 100)
    print("EMPTY CELLS -- the classes with nothing, with the number that "
          "makes them empty (|mean| under the 15 bps cost gate OR sign p "
          "worse than 0.35 on every event)")
    print("=" * 100)
    for cls, tks in CLASSES:
        sub = df[df["proxy"].isin(tks) & ~df["proxy"].isin(NON_TRADEABLE)]
        if sub.empty:
            sub = df[df["proxy"].isin(tks)]
        verdict = "PULSE" if (sub["FLAG"].str.startswith("*")).any() else "EMPTY"
        bp = sub.loc[sub["p_dir"].idxmin()]           # most significant cell
        be = sub.loc[sub["edge_bps"].abs().idxmax()]  # biggest edge cell
        print(f"  {cls:14s} {verdict}")
        print(f"      lowest sign p over its {len(sub)} cells: {bp['event']}/"
              f"{bp['mode']}/{bp['proxy']} p_dir={bp['p_dir']:.3f}, mean "
              f"{bp['mean_pct']:+.3f}% vs own drift {bp['ctrlA_pct']:+.3f}% "
              f"(edge {bp['edge_bps']:+.0f} bps), N={bp['n']}, rec {bp['rec']}")
        print(f"      biggest edge: {be['event']}/{be['mode']}/{be['proxy']} "
              f"{be['edge_bps']:+.0f} bps but median {be['med_pct']:+.3f}%, "
              f"rec {be['rec']}, p_dir={be['p_dir']:.3f}")

    print("\nwrote 02_event_class_grid.csv")


if __name__ == "__main__":
    main()
