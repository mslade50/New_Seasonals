"""C7 round 1, and the brief says run the killer FIRST: the placebo anchor
ladder k=-8..+8 around the payrolls print, for "the energy complex closed AT a
52-week high" -> long energy into the print.

Registry: 2026-09-02 killed the same construction on a CPI anchor. Its placebo
ladder was four-for-four as a killer (live config +0.782pp, k=-7 +3.365pp), and
the parent at-a-high cell is dead underneath (90 episodes +0.332% vs local
control +0.224%, Welch t +0.78, top-2 episodes 60% of total).

Live geometry: today's close is the k=-1 entry off a k=-2 anchor (2026-09-02),
i.e. anchor offset -2 with lag=1. If the true anchor does not rank at or near
1 of 17 on the ladder, this dies here.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import (anchor_positions, close_panel, declusters, load_events,
                       load_prices, local_control, rolling_on_valid, show,
                       summarize, vehicle_ret, sign_test, cluster_note)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 240)

VEHICLES = ["XLE", "XOP", "DBC", "USO", "COP", "CVX", "VLO", "OIH"]
LIVE_K = -2

print("=" * 78)
print("C7  energy AT a 52-week high into payrolls -- PLACEBO ANCHOR LADDER FIRST")
print("=" * 78)

raw = load_prices(VEHICLES)
nfp = load_events(["nfp"])["date"]
print("NFP prints in calendar:", len(nfp), nfp.iloc[0].date(), "->", nfp.iloc[-1].date())

# at-a-52w-high mask per vehicle, on its OWN calendar (rolling_on_valid rule)
at_high = {}
for t in VEHICLES:
    s = raw[t]["Close"].dropna()
    hi = s.rolling(252).max()
    at_high[t] = (s >= hi - 1e-9)
    print(f"  {t:5s} history {s.index[0].date()} .. {s.index[-1].date()}  "
          f"at-252d-high days {int(at_high[t].sum())}  live={bool(at_high[t].iloc[-1])}")


def ladder(tkr: str, h: int, lag: int = 1) -> pd.DataFrame:
    s = raw[tkr]["Close"].dropna()
    cal = s.index
    px1 = pd.DataFrame({tkr: s})
    rows = []
    for k in range(-8, 9):
        pos, kept = anchor_positions(cal, nfp, k)
        anch = pd.DatetimeIndex([cal[i] for i in pos])
        m = at_high[tkr].reindex(anch).fillna(False).values
        d = anch[m]
        ret = vehicle_ret(px1, [(tkr, 1.0)], h, lag)
        d = d[ret.reindex(d).notna().values]
        epi = declusters(d, max(h, 5), cal)
        r = summarize(ret.reindex(epi).values, f"k={k:+d}")
        base = ret.dropna()
        r["ctl_all_pct"] = round(100 * base.mean(), 3)
        r["excess_pp"] = round(r["mean_pct"] - 100 * base.mean(), 3) if r["n"] else np.nan
        r["k"] = k
        r["live"] = "<== LIVE" if k == LIVE_K else ""
        rows.append(r)
    return pd.DataFrame(rows)


for h in (3, 5):
    print("\n" + "#" * 78)
    print(f"# PLACEBO LADDER, h={h} td, entry lag=1 (so k=-2 -> entry on the k=-1 close)")
    print("#" * 78)
    best_rank = {}
    for t in VEHICLES:
        df = ladder(t, h)
        ok = df[df["n"] >= 3].copy()
        if ok.empty:
            print(f"\n{t}: no ladder rung with n>=3")
            continue
        ok = ok.sort_values("excess_pp", ascending=False).reset_index(drop=True)
        live_row = ok[ok["k"] == LIVE_K]
        rk = int(live_row.index[0]) + 1 if len(live_row) else np.nan
        best_rank[t] = (rk, len(ok))
        print(f"\n--- {t}  (live k={LIVE_K} ranks {rk} of {len(ok)} rungs with n>=3) ---")
        show(ok[["label", "n", "mean_pct", "median_pct", "hit", "t",
                 "worst_pct", "ctl_all_pct", "excess_pp", "live"]].to_dict("records"))
    print(f"\nSUMMARY h={h}: live-anchor rank of {len(best_rank)} vehicles")
    for t, (rk, tot) in best_rank.items():
        print(f"   {t:5s}  rank {rk} of {tot}"
              + ("   <-- ranks 1" if rk == 1 else ""))

# --- the parent, ungated by any event, for completeness --------------------
print("\n" + "=" * 78)
print("PARENT (no event anchor): at a 52-week high -> long, does the base pay?")
print("=" * 78)
for h in (3, 5):
    rows = []
    for t in VEHICLES:
        s = raw[t]["Close"].dropna()
        px1 = pd.DataFrame({t: s})
        ret = vehicle_ret(px1, [(t, 1.0)], h, 1)
        d = s.index[at_high[t].values & ret.notna().values]
        epi = declusters(d, max(h, 5), s.index)
        r = summarize(ret.reindex(epi).values, t)
        loc = local_control(s.index[ret.notna().values], d)
        r["local_ctl_pct"] = round(100 * ret.reindex(loc).mean(), 3)
        r["excess_vs_local_pp"] = round(r["mean_pct"] - 100 * ret.reindex(loc).mean(), 3) if r["n"] else np.nan
        rows.append(r)
    show(rows, f"at-52w-high parent, h={h}")

# --- and the NFP-anchored live cell measured against its own parent --------
print("\n" + "=" * 78)
print("LIVE CELL vs its own at-a-high parent (does the NFP anchor add anything?)")
print("=" * 78)
for h in (3, 5):
    rows = []
    for t in VEHICLES:
        s = raw[t]["Close"].dropna()
        cal = s.index
        px1 = pd.DataFrame({t: s})
        ret = vehicle_ret(px1, [(t, 1.0)], h, 1)
        pos, _ = anchor_positions(cal, nfp, LIVE_K)
        anch = pd.DatetimeIndex([cal[i] for i in pos])
        d_ev = anch[at_high[t].reindex(anch).fillna(False).values]
        d_ev = d_ev[ret.reindex(d_ev).notna().values]
        d_par = cal[at_high[t].values & ret.notna().values]
        d_no = pd.DatetimeIndex(sorted(set(d_par) - set(d_ev)))
        e_ev = declusters(d_ev, max(h, 5), cal)
        e_no = declusters(d_no, max(h, 5), cal)
        a = summarize(ret.reindex(e_ev).values, f"{t} at-high AND k={LIVE_K} NFP")
        b = summarize(ret.reindex(e_no).values, f"{t} at-high, NOT an NFP anchor")
        rows += [a, b]
    show(rows, f"NFP-anchor attribution, h={h}")

print("\nDONE C7")
