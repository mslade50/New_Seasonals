"""C12b -- why the watchlist's ITA numbers do not reproduce.

c12 got h=5 N=42 +0.629% and h=10 N=28 +1.443%; the watchlist entry quotes
h=5 N=43 +0.543% and h=10 N=29 +1.223%. One extra episode in each, with a
LOWER mean. Hypothesis: _survey_lib.align() does union -> ffill -> reindex on
the FORWARD-RETURN series, and a forward return is NaN for the last lag+h
rows by construction. ffill carries the last resolvable forward return into
those tail rows, which (a) manufactures a trigger day that has no future and
(b) books a stale, duplicated return for it. ITA is IN the cell today, so the
tail rows are exactly trigger rows.

If that is the cause, replaying the align() form reproduces 43 / +0.543 and
29 / +1.223 exactly.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

warnings.filterwarnings("ignore")


def roll_max(s, n=252):
    return rolling_on_valid(s, lambda x: x.rolling(n).max())


def align(s, idx):
    if s.dtype == bool:
        s = s.astype(float)
    return s.reindex(idx.union(s.index)).ffill().reindex(idx)


PX = load_prices(["ITA", "SPY"])
IDX = PX["SPY"].index
C = {t: PX[t]["Close"] for t in PX}
near_hi = C["SPY"] >= 0.98 * roll_max(C["SPY"], 252)
rank = pct_rank(C["ITA"], 21)

for tag, use_align in (("CLEAN reindex (c12)", False), ("_survey_lib align+ffill", True)):
    print(f"\n=== {tag} ===")
    for h in (5, 10):
        if use_align:
            f = align(fwd_lag(C["ITA"], h, 1), IDX)
            m = align(rank <= 10, IDX).fillna(0).astype(bool) & \
                align(near_hi, IDX).fillna(0).astype(bool)
        else:
            f = fwd_lag(C["ITA"], h, 1).reindex(IDX)
            m = ((rank <= 10).reindex(IDX).fillna(False)
                 & near_hi.reindex(IDX).fillna(False))
        valid = f.dropna().index
        trig = pd.DatetimeIndex(IDX[m.values]).intersection(valid)
        epi = declusters(trig, h, valid)
        ep = f.loc[epi].values
        drift = float(f.loc[valid].mean())
        w = int((ep > 0).sum())
        print(f"  h={h:2d}  N_days {len(trig):3d}  N_epi {len(epi):3d}  "
              f"mean {100*ep.mean():+.3f}%  edge {100*(ep.mean()-drift):+.3f}pp  "
              f"rec {w}-{len(epi)-w}  bootP {bootstrap_p_le0(ep):.3f}  "
              f"last epi {epi[-1].date()}  last valid fwd {valid[-1].date()}")
        tail = [d for d in epi if d > valid[-1] - pd.Timedelta(days=0)]
        if use_align:
            phantom = [d for d in epi if d not in
                       pd.DatetimeIndex(IDX[m.values]).intersection(
                           fwd_lag(C["ITA"], h, 1).reindex(IDX).dropna().index)]
            print(f"      phantom tail episodes (no real forward return): "
                  f"{[str(d.date()) for d in phantom]}")
            if phantom:
                print(f"      their booked returns: "
                      f"{[round(100*float(f.loc[d]), 3) for d in phantom]}%  "
                      f"(= the last resolvable value, ffilled)")
