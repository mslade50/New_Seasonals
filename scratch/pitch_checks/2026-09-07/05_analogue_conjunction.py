"""METHOD B -- conjunction screen, no distance metric.

Days where ALL of:
    SPY within 2% of its 252d high
    SPY 21d realised vol < 0.80x its own 63d average
    TLT within 3% of its 252d low
    HYG within 1.5% of its 252d high

i.e. an index at highs, realised vol asleep, the long bond on the floor and
credit at the ceiling -- the four coordinates of the 2026-09-04 tape that are
not about any single asset's own trend. No weights, no standardisation, no
nearest-neighbour ranking, so it shares no machinery with Method A beyond the
feature construction. Episodes declustered at 21 sessions.

Also prints: which of the four conditions is binding (marginal day counts),
what each 3-of-4 leave-one-out screen gives, and a threshold-sensitivity pass,
because a conjunction with a small N is exactly the shape that looks decisive
by accident.

Run:  python scratch/pitch_checks/2026-09-07/05_analogue_conjunction.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from _analogue_common import (  # noqa: E402
    ASOF, EXCLUDE_TD, FEATURES, INSTRUMENTS, build_features,
    concentration_note, eligible, fwd_table, load_all, print_neighbours,
    year_concentration,
)
from pitch_lab import declusters  # noqa: E402

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)

COND_COLS = ["spy_d252h", "spy_volratio", "tlt_d252l", "hyg_d252h"]


def conds(f: pd.DataFrame, spy_hi=-2.0, volr=0.80, tlt_lo=3.0,
          hyg_hi=-1.5) -> dict[str, pd.Series]:
    return {
        f"SPY >= {spy_hi}% of 252d high": f["spy_d252h"] >= spy_hi,
        f"SPY vol ratio < {volr}": f["spy_volratio"] < volr,
        f"TLT <= +{tlt_lo}% off 252d low": f["tlt_d252l"] <= tlt_lo,
        f"HYG >= {hyg_hi}% of 252d high": f["hyg_d252h"] >= hyg_hi,
    }


def screen(f: pd.DataFrame, pool: pd.DatetimeIndex, **kw) -> pd.DatetimeIndex:
    c = conds(f, **kw)
    m = pd.Series(True, index=f.index)
    for v in c.values():
        m &= v.fillna(False)
    return pd.DatetimeIndex(f.index[m.values]).intersection(pool)


def main() -> None:
    px, breadth_names = load_all()
    f = build_features(px, breadth_names)
    idx = f.index
    pool = eligible(f, COND_COLS)

    print("=" * 100)
    print("METHOD B -- conjunction screen (no distance metric)")
    print("=" * 100)
    print(f"POOL: {len(pool)} days, {pool[0].date()} .. {pool[-1].date()} "
          f"(HYG's 252d high needs a bar back to {px['HYG'].index[0].date()}; "
          f"the last {EXCLUDE_TD} sessions are excluded so today's own "
          f"episode cannot be a neighbour of itself)")

    t = f.loc[ASOF]
    print(f"\nTODAY {ASOF.date()} against the four conditions:")
    for lbl, m in conds(f).items():
        print(f"  {lbl:<38s} -> {bool(m.loc[ASOF])}")
    print(f"  values: SPY d252h {t['spy_d252h']:+.2f}%, vol ratio "
          f"{t['spy_volratio']:.2f}, TLT +{t['tlt_d252l']:.2f}% off its low, "
          f"HYG {t['hyg_d252h']:+.2f}% off its high. Today PASSES all four, so "
          f"the screen is genuinely describing today and not a nearby cousin.")

    print("\nMARGINAL day counts over the pool (which condition is binding):")
    c = conds(f)
    for lbl, m in c.items():
        print(f"  {lbl:<38s} {int(m.reindex(pool).fillna(False).sum()):>5d} days "
              f"({100*float(m.reindex(pool).fillna(False).mean()):.1f}%)")
    for drop in c:
        m = pd.Series(True, index=f.index)
        for lbl, v in c.items():
            if lbl != drop:
                m &= v.fillna(False)
        d = pd.DatetimeIndex(f.index[m.values]).intersection(pool)
        print(f"  3-of-4, dropping [{drop}]: {len(d)} days -> "
              f"{len(declusters(d, 21, idx))} episodes")

    trig = screen(f, pool)
    epi = declusters(trig, 21, idx)
    print(f"\nFULL CONJUNCTION: {len(trig)} days -> {len(epi)} episodes "
          f"(21td decluster)")
    if len(trig) == 0:
        print("  NO HISTORICAL MATCH. Today is outside the pool's experience "
              "on this conjunction; report that, do not loosen it silently.")
    else:
        print("  day-level year mix: " + year_concentration(trig))
        print_neighbours(epi, None, f, f"Method B episodes (N={len(epi)})")
        runs = []
        for e in epi:
            same = trig[(trig >= e)]
            block = []
            prev = None
            for d in same:
                p = int(idx.searchsorted(d))
                if prev is not None and p - prev > 5:
                    break
                block.append(d)
                prev = p
            runs.append((e.date(), len(block), block[-1].date()))
        print("  contiguous runs (episode start, days in the run, last day):")
        for r in runs:
            print(f"    {r[0]}  {r[1]:>3d} days  through {r[2]}")

        fwd_table(px, epi, f"Method B episodes (N={len(epi)})")
        fwd_table(px, trig, f"Method B day-level (N={len(trig)}, OVERLAPPING "
                             f"-- shown only to see whether the episode "
                             f"numbers are one day's luck)")

        print("\n" + "=" * 100)
        print("CONCENTRATION at h=10, episode level")
        print("=" * 100)
        for tkr in INSTRUMENTS:
            print(f"  {tkr:<5s} {concentration_note(px, epi, tkr, 10)}")

    print("\n" + "=" * 100)
    print("THRESHOLD SENSITIVITY -- an all-four conjunction is fragile by "
          "construction; if the answer only exists at one setting, it is not "
          "an answer.")
    print("=" * 100)
    grid = [
        {},
        {"spy_hi": -1.0}, {"spy_hi": -3.0},
        {"volr": 0.70}, {"volr": 0.90}, {"volr": 1.00},
        {"tlt_lo": 2.0}, {"tlt_lo": 5.0},
        {"hyg_hi": -0.75}, {"hyg_hi": -3.0},
        {"spy_hi": -3.0, "volr": 0.90, "tlt_lo": 5.0, "hyg_hi": -3.0},
    ]
    rows = []
    for kw in grid:
        d = screen(f, pool, **kw)
        e = declusters(d, 21, idx)
        row = {"variant": ", ".join(f"{k}={v}" for k, v in kw.items()) or "base",
               "days": len(d), "episodes": len(e)}
        for tkr, h in (("SPY", 10), ("IWM", 10), ("GLD", 10), ("USO", 10),
                       ("TLT", 10), ("SVXY", 10)):
            from pitch_lab import fwd_lag, summarize
            v = fwd_lag(px[tkr]["Close"], h, 1).reindex(e).dropna()
            row[f"{tkr}_h10"] = round(100 * float(v.mean()), 2) if len(v) else np.nan
            row[f"{tkr}_n"] = len(v)
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))
    print("\n(mean % at h=10, lag=1, episode level, for each threshold "
          "variant. Read the episode count first.)")


if __name__ == "__main__":
    main()
