"""C7 round 1: long gold from the day after a crash into the payrolls close.

Signal: a FIRST complex break (GLD, SLV, GDX each <= -2%, none in prior 5) or a
GLD <= -3.5% day, on a session k = 2..5 sessions before an NFP. Entry MOC next
close (lag 1), exit at the NFP close, so h = k - 1 (today k=4, h=3).

Must beat (i) the SAME trigger at the same h without the NFP (C1's unconditional
crash result) and (ii) gold's ungated NFP run-in (entry NFP-3 close, exit NFP close,
registry +0.290% on 150-111).
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
BAR = pd.Timestamp("2026-09-28")
TK = ["GLD", "SLV", "GDX", "GC=F"]
raw = load_prices(TK)
idx = raw["GLD"].loc[:BAR].index
px = pd.DataFrame({t: raw[t]["Close"].reindex(idx).ffill(limit=3) for t in TK})
r1 = px.pct_change(fill_method=None)
faith = (r1["GLD"] <= -0.02) & (r1["SLV"] <= -0.02) & (r1["GDX"] <= -0.02)
first = faith & ~faith.shift(1).rolling(5).max().fillna(0).astype(bool)
crash = r1["GLD"] <= -0.035
crash1st = crash & ~crash.shift(1).rolling(21, min_periods=1).max().fillna(0).astype(bool)

nfp = load_events(["nfp"])["date"]
qpos, qd = anchor_positions(idx, nfp, 0)
# for each session p, the sessions to the NEXT nfp (k>=1)
k_to = pd.Series(np.nan, index=idx)
qs = np.array(sorted(qpos))
for p in range(len(idx)):
    j = qs[qs > p]
    if len(j):
        k_to.iloc[p] = j[0] - p
print(f"TODAY {BAR.date()}: faithful={bool(faith.loc[BAR])} first={bool(first.loc[BAR])} "
      f"crash={bool(crash.loc[BAR])} sessions-to-NFP k={k_to.loc[BAR]} (NFP 2026-10-02 in "
      f"calendar: {pd.Timestamp('2026-10-02') in set(pd.DatetimeIndex(nfp))})")


def into_nfp(mask, ks, veh="GLD"):
    """Return entry close p+1 -> NFP close p+k for masked sessions with k in ks."""
    c = px[veh].values
    rows = []
    for d in idx[mask.fillna(False).values]:
        p = idx.get_loc(d)
        k = k_to.iloc[p]
        if np.isnan(k) or int(k) not in ks or p + int(k) >= len(idx):
            continue
        k = int(k)
        rows.append((d, k, c[p + k] / c[p + 1] - 1.0))
    return pd.DataFrame(rows, columns=["date", "k", "ret"])


def uncond(mask, h, veh="GLD"):
    r = fwd_lag(px[veh], h, 1)
    d = idx[mask.fillna(False).values & r.notna().values]
    e = declusters(d, h, idx)
    return r.loc[e].values


def rec(v):
    v = np.asarray(v)
    w = int((v > 0).sum())
    return f"{w}-{len(v)-w}", round(sign_test(w, len(v)), 4)


for veh in ("GLD", "GC=F"):
    print(f"\n################ vehicle {veh} ################")
    drift = {h: fwd_lag(px[veh], h, 1).mean() for h in (1, 2, 3, 4)}
    # ungated NFP run-in: signal k sessions before (entry k-1 sessions before), exit NFP close
    rows = []
    for k in (2, 3, 4, 5):
        v = into_nfp(pd.Series(True, index=idx), (k,), veh)["ret"].values
        o = summarize(v, f"ALL NFPs, signal k={k} (h={k-1})")
        o["rec"], o["p"] = rec(v)
        o["drift_h"] = round(100 * drift[k - 1], 3)
        rows.append(o)
    show(rows, f"{veh}: ungated NFP run-in (every NFP)")

    for tlbl, m in (("FIRST complex break", first), ("ALL faithful break", faith),
                    ("GLD <= -3.5%", crash), ("GLD <= -3.5% first-in-21d", crash1st)):
        rows = []
        for ks, lab in (((4,), "k=4 (today, h=3)"), ((2, 3, 4), "k=2..4"),
                        ((2, 3, 4, 5), "k=2..5")):
            df = into_nfp(m, ks, veh)
            v = df["ret"].values
            o = summarize(v, f"{tlbl} into NFP, {lab}")
            if len(v):
                o["rec"], o["p"] = rec(v)
                # matched controls: same trigger unconditional at each episode's h; all-NFP run-in at same k
                un = np.mean([np.nanmean(uncond(m, int(k) - 1, veh)) for k in df["k"]])
                nf = np.mean([np.nanmean(into_nfp(pd.Series(True, index=idx), (int(k),), veh)["ret"])
                              for k in df["k"]])
                o["trig_uncond_pct"] = round(100 * un, 3)
                o["nfp_runin_pct"] = round(100 * nf, 3)
                o["vs_trig_pp"] = round(o["mean_pct"] - 100 * un, 3)
                o["vs_runin_pp"] = round(o["mean_pct"] - 100 * nf, 3)
            rows.append(o)
        show(rows, f"{veh}: {tlbl} x NFP")
        df = into_nfp(m, (2, 3, 4, 5), veh)
        if len(df):
            print("  episodes:", ", ".join(f"{d.date()}(k{k}:{100*r:+.2f})" for d, k, r in df.values))
        v3 = uncond(m, 3, veh)
        print(f"  {tlbl} UNCONDITIONAL h=3 (C1-style): n={len(v3)} mean {100*np.nanmean(v3):+.3f}% "
              f"rec {rec(v3[~np.isnan(v3)])}")

# placebo: first complex break with k in 2..5 vs k in 7..10 (same trigger, NFP not in the hold)
print("\n=== placebo: FIRST break, NFP distance buckets, GLD, h=3 fixed ===")
r3 = fwd_lag(px["GLD"], 3, 1)
for lo, hi in ((2, 5), (6, 10), (11, 16), (17, 30)):
    d = idx[first.fillna(False).values & r3.notna().values & k_to.between(lo, hi).values]
    v = r3.loc[d].values
    print(f"k in [{lo},{hi}]: n={len(v)} mean {100*v.mean():+.3f}% rec {rec(v)}")
