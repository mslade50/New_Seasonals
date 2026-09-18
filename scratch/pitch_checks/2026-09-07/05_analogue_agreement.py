"""WHERE THE TWO METHODS AGREE -- the only output that earns a check.

Recomputes the neighbour sets from 05_analogue_knn.py (Method A: k=20/50,
declustered and greedy) and 05_analogue_conjunction.py (Method B: episodes and
day-level) by IMPORTING those modules, so no number here is a re-derivation,
then lays the forward returns out side by side and flags the (instrument,
horizon) cells where every construction points the same way.

Two sign tests are reported per cell and they are not the same question:
  MEAN  -- did the instrument go up after these tapes?
  EDGE  -- did it go up MORE than it goes up on an average day?
GLD, SVXY and USO all carry large positive unconditional drifts in this
sample, so a positive mean with a negative edge is a common and honest result.

Run:  python scratch/pitch_checks/2026-09-07/05_analogue_agreement.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from _analogue_common import (  # noqa: E402
    ASOF, FEATURES, INSTRUMENTS, build_features, eligible, load_all,
)
from pitch_lab import declusters, fwd_lag, sign_test, summarize  # noqa: E402

import importlib

knn_mod = importlib.import_module("05_analogue_knn")
conj_mod = importlib.import_module("05_analogue_conjunction")

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_rows", 300)

HS = (1, 3, 5, 10)


def build_sets(px, f, idx):
    pool = eligible(f, FEATURES)
    dist = knn_mod.knn(f, FEATURES, pool, ASOF)
    sets = {
        "A_k20_dec": declusters(pd.DatetimeIndex(sorted(dist.index[:20])), 21, idx),
        "A_k20_greedy": knn_mod.greedy_episodes(dist, 20, idx, 21),
        "A_k50_dec": declusters(pd.DatetimeIndex(sorted(dist.index[:50])), 21, idx),
        "A_k50_greedy": knn_mod.greedy_episodes(dist, 50, idx, 21),
    }
    pool_b = eligible(f, conj_mod.COND_COLS)
    trig = conj_mod.screen(f, pool_b)
    sets["B_episodes"] = declusters(trig, 21, idx)
    sets["B_daylevel"] = trig
    return sets


def main() -> None:
    px, breadth_names = load_all()
    f = build_features(px, breadth_names)
    idx = f.index
    sets = build_sets(px, f, idx)

    print("=" * 110)
    print("NEIGHBOUR SETS UNDER TEST")
    print("=" * 110)
    for k, v in sets.items():
        print(f"  {k:<14s} N={len(v):>3d}   {v[0].date()} .. {v[-1].date()}")
    a_only = [k for k in sets if k.startswith("A_")]
    b_only = [k for k in sets if k.startswith("B_")]
    ov = len(set(sets["A_k50_greedy"]) & set(sets["B_daylevel"]))
    print(f"\n  date overlap between the widest A set (k50 greedy, N=50) and "
          f"every Method B day (N={len(sets['B_daylevel'])}): {ov} dates. "
          f"The two methods are close to INDEPENDENT selections of history, "
          f"which is what makes an agreement worth anything.")

    rows = []
    for tkr in INSTRUMENTS:
        s = px[tkr]["Close"]
        for h in HS:
            r = fwd_lag(s, h, lag=1)
            base = r.dropna()
            base = base[base.index <= ASOF]
            drift, bhit = float(base.mean()), float((base.values > 0).mean())
            row = {"ticker": tkr, "h": h, "drift_pct": round(100 * drift, 3)}
            for name, dates in sets.items():
                v = r.reindex(pd.DatetimeIndex(dates)).dropna()
                if len(v) == 0:
                    row[name] = np.nan
                    continue
                row[name] = round(100 * float(v.mean()) - 100 * drift, 3)
                row[name + "_mean"] = round(100 * float(v.mean()), 3)
                w = int((v.values > 0).sum())
                row[name + "_rec"] = f"{w}-{len(v) - w}"
                row[name + "_p"] = round(sign_test(w, len(v), bhit), 3)
            rows.append(row)
    df = pd.DataFrame(rows)

    print("\n" + "=" * 110)
    print("EDGE vs the instrument's OWN all-days drift, percentage points, "
          "lag=1 MOC entry")
    print("=" * 110)
    print(df[["ticker", "h", "drift_pct"] + list(sets)].to_string(index=False))

    print("\n" + "=" * 110)
    print("RAW MEAN forward return, percent")
    print("=" * 110)
    print(df[["ticker", "h"] + [k + "_mean" for k in sets]].to_string(index=False))

    print("\n" + "=" * 110)
    print("AGREEMENT -- cells where every construction has the same sign")
    print("=" * 110)
    for basis, cols in (("EDGE", list(sets)), ("MEAN", [k + "_mean" for k in sets])):
        for scope, use in (("all 6 sets", cols),
                           ("4 A-sets + B episodes", cols[:5])):
            hits = []
            for _, r in df.iterrows():
                v = np.array([r[c] for c in use], dtype=float)
                if np.isnan(v).any():
                    continue
                if (v > 0).all() or (v < 0).all():
                    hits.append((r["ticker"], int(r["h"]),
                                 "+" if v[0] > 0 else "-",
                                 round(float(np.mean(v)), 3),
                                 round(float(np.min(np.abs(v))), 3)))
            print(f"\n  {basis}, {scope}: {len(hits)} unanimous cells")
            for t, h, sgn, avg, weakest in sorted(hits, key=lambda x: -abs(x[3])):
                sub = df[(df.ticker == t) & (df.h == h)].iloc[0]
                recs = " ".join(f"{k.replace('_','')[:6]}:{sub[k+'_rec']}"
                                for k in sets)
                print(f"    {t:<5s} h={h:<3d} {sgn}  avg {basis.lower()} "
                      f"{avg:+.3f}pp, weakest |cell| {weakest:.3f}pp | {recs}")

    print("\n" + "=" * 110)
    print("HIT-RATE view for the unanimous cells (conditional hit vs the "
          "instrument's own all-days up-rate)")
    print("=" * 110)
    for tkr in INSTRUMENTS:
        s = px[tkr]["Close"]
        for h in HS:
            r = fwd_lag(s, h, lag=1)
            base = r.dropna()
            base = base[base.index <= ASOF]
            bhit = 100 * float((base.values > 0).mean())
            cells = []
            for name, dates in sets.items():
                v = r.reindex(pd.DatetimeIndex(dates)).dropna()
                if len(v) == 0:
                    cells.append(np.nan)
                    continue
                cells.append(100 * float((v.values > 0).mean()))
            arr = np.array(cells, dtype=float)
            if np.isnan(arr).any():
                continue
            if (arr > bhit).all() or (arr < bhit).all():
                print(f"  {tkr:<5s} h={h:<3d} base {bhit:.1f}%  cells "
                      + " ".join(f"{c:.0f}" for c in arr)
                      + ("  ALL ABOVE" if arr[0] > bhit else "  ALL BELOW"))


if __name__ == "__main__":
    main()
