"""S7 -- the overlap attack on the three price-state survivors.

Every cell in this survey came out of a grid, and the two strongest ones sit on
PERSISTENT states: the fragility dial can stay in its top decile for months, and
TLT can sit within 2% of a 252d low for a whole quarter. Declustering at h td
(pitch_lab.battery's default) does NOT make those episodes independent -- a dial
top-decile run in 2021 contributed 22 of 71 "episodes" at h=5. This script pushes
the decluster gap out to regime scale and reports what survives, plus a
per-regime (contiguous-run) accounting where each RUN of trigger days counts once.

Survivors under attack:
  S6b  long SPY / short IWM at the 10d-MA-63d dial top decile   (h=5, h=10)
  S2b  long TLT when TLT is within 2% of its 252d low AND HYG is within 1% of
       its 252d high                                            (h=5, h=10)
  S4   long ITA (or ITA minus SPY) at rank21<=10 with SPY within 2% of its high
                                                                (h=5, h=10)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _survey_lib import (  # noqa: E402
    align, cluster_note, declusters, fwd_lag, load_prices, np, pct_rank, pd,
    roll_max, roll_min, show, sign_test, summarize, bootstrap_p_le0,
)

ROOT = Path(__file__).resolve().parents[3]
FRAG = pd.read_parquet(ROOT / "data" / "rd2_fragility.parquet")
PX = load_prices(["SPY", "IWM", "TLT", "HYG", "ITA"])
IDX = PX["SPY"].index
C = {t: PX[t]["Close"] for t in PX}

ma = FRAG["63d"].rolling(10).mean()
m_dial = align((ma >= ma.expanding(252).quantile(0.90)).fillna(False), IDX).fillna(0).astype(bool)

HYG_IDX = PX["HYG"].index
m_tlt = (align(C["TLT"] <= 1.02 * roll_min(C["TLT"], 252), IDX).fillna(0).astype(bool)
         & align(C["HYG"] >= 0.99 * roll_max(C["HYG"], 252), IDX).fillna(0).astype(bool))
m_ita = (align(pct_rank(C["ITA"], 21) <= 10, IDX).fillna(0).astype(bool)
         & align(C["SPY"] >= 0.98 * roll_max(C["SPY"], 252), IDX).fillna(0).astype(bool))


def spy_iwm(h):
    return align(fwd_lag(C["SPY"], h, 1), IDX) - align(fwd_lag(C["IWM"], h, 1), IDX)


def tlt(h):
    return align(fwd_lag(C["TLT"], h, 1), IDX)


def ita(h):
    return align(fwd_lag(C["ITA"], h, 1), IDX)


def ita_rel(h):
    return align(fwd_lag(C["ITA"], h, 1), IDX) - align(fwd_lag(C["SPY"], h, 1), IDX)


SURV = [
    ("S6b long SPY / short IWM | dial top decile", spy_iwm, m_dial, 5.0),
    ("S2b long TLT | TLT@252-low & HYG@252-high", tlt, m_tlt, 4.0),
    ("S4  long ITA | rank21<=10 & SPY near high", ita, m_ita, 5.0),
    ("S4  ITA minus SPY | same mask", ita_rel, m_ita, 9.0),
]

print("=" * 78)
print("S7  OVERLAP ATTACK: does each survivor hold when episodes are made")
print("    independent at REGIME scale rather than at h td?")
print("=" * 78)

for label, vf, mask, cost_bps in SURV:
    trig_all = IDX[mask.values]
    print("\n" + "=" * 78)
    print(f"{label}")
    print(f"  trigger days {len(trig_all)}   {trig_all[0].date()} .. {trig_all[-1].date()}"
          f"   years {sorted(set(trig_all.year))}")
    # contiguous runs: a new run starts when the previous trading day was not a trigger
    pos = pd.Series(range(len(IDX)), index=IDX)
    p = pos.loc[trig_all].values
    run_starts = [trig_all[0]] + [trig_all[i] for i in range(1, len(trig_all))
                                  if p[i] - p[i - 1] > 1]
    print(f"  contiguous RUNS of the state: {len(run_starts)}"
          f"   (mean run length {len(trig_all) / len(run_starts):.1f} sessions)")
    for h in (5, 10):
        r = vf(h)
        valid = r.dropna().index
        t = pd.DatetimeIndex(trig_all).intersection(valid)
        rows = []
        for gap in (h, 10, 21, 63, 126):
            epi = declusters(t, gap, valid)
            ep = r.loc[epi].values
            w = int((ep > 0).sum())
            rows.append({"h": h, "gap_td": gap, "n": len(epi),
                         "mean_pct": round(100 * ep.mean(), 3),
                         "median_pct": round(100 * np.median(ep), 3),
                         "hit": round(100 * (ep > 0).mean(), 1),
                         "worst_pct": round(100 * ep.min(), 2),
                         "rec": f"{w}-{len(epi) - w}",
                         "sign_p": round(sign_test(w, len(epi)), 4),
                         "bootP<=0": round(bootstrap_p_le0(ep), 3),
                         "x_cost": round(100 * ep.mean() * 100 / cost_bps, 1)})
        # RUN-LEVEL: one observation per contiguous run, taken at the run's FIRST day
        rs = pd.DatetimeIndex([d for d in run_starts]).intersection(valid)
        ep = r.loc[rs].values
        w = int((ep > 0).sum())
        rows.append({"h": h, "gap_td": "RUN-1st", "n": len(rs),
                     "mean_pct": round(100 * ep.mean(), 3),
                     "median_pct": round(100 * np.median(ep), 3),
                     "hit": round(100 * (ep > 0).mean(), 1),
                     "worst_pct": round(100 * ep.min(), 2),
                     "rec": f"{w}-{len(rs) - w}",
                     "sign_p": round(sign_test(w, len(rs)), 4),
                     "bootP<=0": round(bootstrap_p_le0(ep), 3),
                     "x_cost": round(100 * ep.mean() * 100 / cost_bps, 1)})
        # YEAR-LEVEL: average the episode returns inside each calendar year, then
        # treat each YEAR as one observation. Kills within-year overlap entirely.
        epi = declusters(t, h, valid)
        ser = pd.Series(r.loc[epi].values, index=epi)
        by_year = ser.groupby(ser.index.year).mean()
        w = int((by_year > 0).sum())
        rows.append({"h": h, "gap_td": "YEAR-mean", "n": len(by_year),
                     "mean_pct": round(100 * by_year.mean(), 3),
                     "median_pct": round(100 * by_year.median(), 3),
                     "hit": round(100 * (by_year > 0).mean(), 1),
                     "worst_pct": round(100 * by_year.min(), 2),
                     "rec": f"{w}-{len(by_year) - w}",
                     "sign_p": round(sign_test(w, len(by_year)), 4),
                     "bootP<=0": round(bootstrap_p_le0(by_year.values), 3),
                     "x_cost": round(100 * by_year.mean() * 100 / cost_bps, 1)})
        print(pd.DataFrame(rows).to_string(index=False))
        print(f"    per-year means (h={h}): "
              + ", ".join(f"{y}:{100 * v:+.2f}%" for y, v in by_year.items()))

print("\n" + "=" * 78)
print("READING: gap=h is pitch_lab's default and is the LOOSEST of these. RUN-1st")
print("and YEAR-mean are the strictest -- each contiguous stretch of the state, or")
print("each calendar year, counts once. A cell that only works at gap=h is an")
print("overlap artefact; a cell that survives RUN-1st and YEAR-mean is not.")
