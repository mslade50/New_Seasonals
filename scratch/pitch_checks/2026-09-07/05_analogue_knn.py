"""METHOD A -- nearest-neighbour tapes to the 2026-09-04 close.

Seven features, standardised over the whole eligible pool, Euclidean distance,
k = 20 and k = 50 nearest days. The k nearest days are then DECLUSTERED with
pitch_lab.declusters at a 21-session gap, exactly as the axis brief asks, so
one regime cannot supply ten neighbours; the surviving count is reported next
to the raw k every time. A second, GREEDY selection (walk the distance
ordering, accept a day only if it is >= 21 sessions from every accepted day,
stop at k) is printed as a robustness pass, because "the 20 nearest days,
declustered" and "the 20 nearest distinct episodes" are different sets and
neither is obviously the right one.

Read the year-concentration line before the return table. A k-NN set is a
SELECTION: if the neighbours are one regime, the forward returns are that
regime's next fortnight dressed up as a sample.

Run:  python scratch/pitch_checks/2026-09-07/05_analogue_knn.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from _analogue_common import (  # noqa: E402
    ASOF, EXCLUDE_TD, FEATURES, FEATURE_LABELS, INSTRUMENTS,
    align, build_features, breadth_universe, concentration_note, eligible,
    fwd_table, load_all, print_neighbours, year_concentration,
)
from pitch_lab import declusters, pct_rank  # noqa: E402

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)


def knn(f: pd.DataFrame, cols: list[str], pool: pd.DatetimeIndex,
        asof: pd.Timestamp) -> pd.Series:
    """Euclidean distance in z-space from the as-of vector, over the pool."""
    sub = f.loc[pool, cols]
    mu, sd = sub.mean(), sub.std(ddof=0)
    z = (sub - mu) / sd
    z_today = (f.loc[asof, cols] - mu) / sd
    d = np.sqrt(((z - z_today) ** 2).sum(axis=1))
    return d.sort_values()


def greedy_episodes(dist: pd.Series, k: int, all_dates: pd.DatetimeIndex,
                    gap: int = 21) -> pd.DatetimeIndex:
    pos = pd.Series(range(len(all_dates)), index=all_dates)
    picked: list[pd.Timestamp] = []
    picked_pos: list[int] = []
    for d in dist.index:
        p = pos.get(d)
        if p is None:
            continue
        if all(abs(p - q) >= gap for q in picked_pos):
            picked.append(d)
            picked_pos.append(p)
        if len(picked) == k:
            break
    return pd.DatetimeIndex(sorted(picked))


def main() -> None:
    px, breadth_names = load_all()
    f = build_features(px, breadth_names)
    idx = f.index

    # ---------------- today's state, verified from the cache ---------------
    print("=" * 100)
    print(f"STATE VECTOR VERIFICATION  as-of close {ASOF.date()}  "
          f"(last bar in cache for SPY: {px['SPY'].index[-1].date()})")
    print("=" * 100)
    t = f.loc[ASOF]
    claims = {
        "spy_d252h": -0.99, "spy_dsma200": 8.41, "spy_rank21": 31.3,
        "breadth200": 66.5, "tnx_d252h": -0.25,
    }
    for c in ["spy_d252h", "spy_dsma200", "spy_rank21", "spy_rv21_ann",
              "spy_volratio", "breadth200", "tnx_d252h", "tnx_dsma200",
              "tlt_d252l", "hyg_d252h", "credit_div"]:
        claim = claims.get(c)
        tag = "" if claim is None else f"   [brief said {claim:+.2f}, delta {t[c]-claim:+.2f}]"
        print(f"  {c:<14s} {FEATURE_LABELS.get(c, ''):<46s} {t[c]:8.2f}{tag}")
    print(f"  breadth universe = {len(breadth_names)} fixed names "
          f"(LIQUID_PLUS_COMMODITIES with a bar on/before 2007-01-01); the "
          f"brief's 66.5% came off today's full 218-name tape.")

    spy = px["SPY"]["Close"]
    r63 = pct_rank(spy, 63, 252)
    print(f"  cross-checks: SPY 63d rank {r63.loc[ASOF]:.1f} (brief 47.6), "
          f"^VIX close {px['^VIX']['Close'].loc[ASOF]:.2f} (brief 15.0), "
          f"^VIX3M {px['^VIX3M']['Close'].loc[ASOF]:.2f}, "
          f"^SKEW {px['^SKEW']['Close'].loc[ASOF]:.2f}, "
          f"^TNX {px['^TNX']['Close'].loc[ASOF]:.3f}")
    print("  NOTE the brief's 'vol_vs_63d 0.69 = realised vol below its 63d "
          "average' is a misread of build_pitch_state._metrics_for: that field "
          "is VOLUME over its 63d average volume. Today's true 21d realised "
          f"vol ratio is {t['spy_volratio']:.2f} "
          f"({t['spy_rv21_ann']:.1f}% annualised). Method A uses the true "
          "realised-vol ratio as the brief's feature list specifies.")

    # ---------------- pool ----------------
    pool = eligible(f, FEATURES)
    print(f"\nPOOL: {len(pool)} eligible days, {pool[0].date()} .. "
          f"{pool[-1].date()}. Pool starts in 2008 because the credit feature "
          f"needs HYG's 252d high and HYG's first bar is "
          f"{px['HYG'].index[0].date()}; the last {EXCLUDE_TD} sessions before "
          f"{ASOF.date()} are excluded so today's own episode cannot be its "
          f"own neighbour.")

    dist = knn(f, FEATURES, pool, ASOF)
    print(f"\ndistance quantiles over the pool: min {dist.iloc[0]:.3f}  "
          f"p1 {dist.quantile(0.01):.3f}  p5 {dist.quantile(0.05):.3f}  "
          f"median {dist.median():.3f}")

    sets: dict[str, pd.DatetimeIndex] = {}
    for k in (20, 50):
        raw = pd.DatetimeIndex(sorted(dist.index[:k]))
        dec = declusters(raw, 21, idx)
        print(f"\n{'-'*100}\nk={k} nearest days -> {len(dec)} survive "
              f"pitch_lab.declusters(gap=21td). Raw span "
              f"{raw[0].date()} .. {raw[-1].date()}.")
        print("  raw k year mix:      " + year_concentration(raw))
        print_neighbours(dec, dist, f, f"Method A, k={k} declustered (N={len(dec)})")
        sets[f"A_k{k}_declustered"] = dec

        greedy = greedy_episodes(dist, k, idx, gap=21)
        print_neighbours(greedy, dist, f,
                         f"Method A, k={k} GREEDY distinct episodes (N={len(greedy)})")
        sets[f"A_k{k}_greedy"] = greedy

    # ---------------- forward returns ----------------
    for name, dates in sets.items():
        fwd_table(px, dates, f"{name} (N={len(dates)})")

    # ---------------- concentration of the headline cells ----------------
    print("\n" + "=" * 100)
    print("CONCENTRATION: how much of each h=10 total sits in the top-2 episodes")
    print("=" * 100)
    for name in ("A_k20_declustered", "A_k50_greedy"):
        print(f"\n{name}:")
        for tkr in INSTRUMENTS:
            print(f"  {tkr:<5s} {concentration_note(px, sets[name], tkr, 10)}")

    # ---------------- sensitivity: drop the credit feature ----------------
    print("\n" + "=" * 100)
    print("SENSITIVITY: same k-NN WITHOUT credit_div (6 features). This is the "
          "only way to reach pre-2008 tapes; if the neighbour set moves a long "
          "way, the 2008 pool start is doing the work.")
    print("=" * 100)
    cols6 = [c for c in FEATURES if c != "credit_div"]
    pool6 = eligible(f, cols6)
    dist6 = knn(f, cols6, pool6, ASOF)
    print(f"POOL: {len(pool6)} days, {pool6[0].date()} .. {pool6[-1].date()}")
    for k in (20, 50):
        raw6 = pd.DatetimeIndex(sorted(dist6.index[:k]))
        dec6 = declusters(raw6, 21, idx)
        print(f"\nk={k} (6-feature) -> {len(dec6)} declustered")
        print("  " + year_concentration(dec6))
        print("  dates: " + ", ".join(str(d.date()) for d in dec6))
        overlap = len(set(dec6) & set(sets[f"A_k{k}_declustered"]))
        print(f"  overlap with the 7-feature k={k} declustered set: "
              f"{overlap} of {len(dec6)}")
        fwd_table(px, dec6, f"A6_k{k}_declustered no-credit (N={len(dec6)})")


if __name__ == "__main__":
    main()
