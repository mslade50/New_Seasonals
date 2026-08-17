"""C7 round 1 - XRT / big-box cohort into the retail earnings cluster.

Cell: the big-box complex has been SOLD into its own print week (XRT 5d rank
18.7 while its 63d rank is 85.7; TJX 5d rank 2.0, ROST 5.6, HD 7.1) and the
cluster (HD 08-18, LOW/TGT/TJX 08-19, WMT/ROST 08-20) lands inside a 3-5 td
hold from a MOC entry today.

Two forms:
  (a) XRT the wrapper, washed 5d inside an intact 63d uptrend, cluster ahead.
  (b) the equal-weight basket of the 3 most-washed actual reporters.

Mandatory kills wired in:
  - GATE ATTRIBUTION: the same washout with NO cluster ahead. If the cluster
    adds nothing the trade is a plain dip-buy (= the book's LT Trend ST OS /
    Weak Close family) and is a rip-off.
  - ALPHABETICAL PLACEBO on form (b) (registry 2026-08-14).
  - print-day gap distribution for the single names.

Conventions: lag=1 (signal close D=2026-08-14, entry MOC D+1=2026-08-17),
fractions in / percent out, episodes declustered.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from earnings_filter import load_earnings_dates_map  # noqa: E402

COHORT = ["HD", "LOW", "TGT", "TJX", "ROST", "WMT"]
TICKERS = ["XRT", "XLY", "SPY"] + COHORT

px = close_panel(TICKERS)
px = px.dropna(subset=["XRT"])
idx = px.index
print(f"panel {idx[0].date()} .. {idx[-1].date()}  N={len(idx)}")

emap = load_earnings_dates_map()
for t in COHORT:
    arr = emap.get(t)
    print(f"  {t}: {len(arr) if arr is not None else 0} earnings rows, "
          f"{pd.Timestamp(arr[0]).date() if len(arr) else '-'} .. "
          f"{pd.Timestamp(arr[-1]).date() if len(arr) else '-'}")

# ---------------------------------------------------------------------------
# cluster indicator: how many cohort names print strictly after the entry close
# (D+lag) and on/before the exit close (D+lag+h)
# ---------------------------------------------------------------------------
pos = pd.Series(range(len(idx)), index=idx)
edates = {t: pd.DatetimeIndex(emap.get(t, np.array([], dtype="datetime64[D]")))
          for t in COHORT}


def prints_in_window(h: int, lag: int = 1) -> pd.DataFrame:
    """rows = signal date, cols = cohort ticker, True if it prints in the hold."""
    out = {}
    for t in COHORT:
        ed = edates[t]
        flags = np.zeros(len(idx), dtype=bool)
        if len(ed):
            # map each earnings date to the first session >= it
            p = idx.searchsorted(ed)
            p = p[(p >= 0) & (p < len(idx))]
            for q in np.unique(p):
                lo = q - (lag + h)      # signal dates whose window contains q
                hi = q - (lag + 1)
                lo = max(lo, 0)
                if hi >= 0:
                    flags[lo:hi + 1] = True
        out[t] = pd.Series(flags, index=idx)
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# washout gate on XRT
# ---------------------------------------------------------------------------
r5 = pct_rank(px["XRT"], 5)
r63 = pct_rank(px["XRT"], 63)
ret5 = px["XRT"].pct_change(5)
print(f"\nlive XRT: 5d rank {r5.iloc[-1]:.1f} (ret {100*ret5.iloc[-1]:+.2f}%), "
      f"63d rank {r63.iloc[-1]:.1f}  [state says 18.7 / 85.7]")

H = 3
piw = prints_in_window(H)
n_prints = piw.sum(axis=1)
print(f"live cohort prints inside a h={H} hold from a 08-17 entry: "
      f"{int(n_prints.iloc[-1])} of 6 -> "
      f"{[t for t in COHORT if piw[t].iloc[-1]]}")

washout = (r5 < 25) & (r63 > 75)
cluster = n_prints >= 3

print(f"\ntrigger counts (days): washout {int(washout.sum())}, "
      f"cluster {int(cluster.sum())}, both {int((washout & cluster).sum())}")

# ---------------------------------------------------------------------------
# FORM (a): XRT long
# ---------------------------------------------------------------------------
for h in (3, 5):
    piw_h = prints_in_window(h)
    cl_h = piw_h.sum(axis=1) >= 3
    m = washout & cl_h
    battery(px, m, [("XRT", 1.0)], h,
            f"C7a XRT washed 5d (<25 rank) + 63d>75 + >=3 cohort prints in hold",
            cost_bps=4.0,
            variants={
                "5d<25 only (NO cluster gate)": washout,
                "cluster only (NO washout)": cl_h,
                "washout AND NO cluster": washout & ~cl_h,
                "5d<15 + cluster": (r5 < 15) & (r63 > 75) & cl_h,
                "5d<35 + cluster": (r5 < 35) & (r63 > 75) & cl_h,
                "5d<25 + 63d>50 + cluster": (r5 < 25) & (r63 > 50) & cl_h,
                "5d<25 + NO 63d gate + cluster": (r5 < 25) & cl_h,
            },
            min_gap=10, event_kinds=("cpi",))

# ---------------------------------------------------------------------------
# GATE ATTRIBUTION, stated head on
# ---------------------------------------------------------------------------
print("\n\n===== GATE ATTRIBUTION: is the cluster decoration? =====")
rows = []
for h in (3, 5):
    piw_h = prints_in_window(h)
    cl_h = piw_h.sum(axis=1) >= 3
    ret = fwd_lag(px["XRT"], h, 1)
    valid = ret.notna()
    for lbl, m in [("washout + cluster", washout & cl_h),
                   ("washout, NO cluster", washout & ~cl_h),
                   ("cluster, NO washout", cl_h & ~washout),
                   ("neither (all other days)", ~washout & ~cl_h)]:
        d = idx[m.values & valid.values]
        e = declusters(d, 10, idx)
        r = summarize(ret.loc[e].values, f"h={h} {lbl}")
        r["n_days"] = len(d)
        rows.append(r)
show(rows, "XRT h=3/5 by cell")

# ---------------------------------------------------------------------------
# FORM (b): the washed reporters basket + ALPHABETICAL PLACEBO
# ---------------------------------------------------------------------------
print("\n\n===== FORM (b): 3-name basket of actual reporters =====")
K = 3
ret5_all = {t: px[t].pct_change(5) for t in COHORT}

for h in (3, 5):
    piw_h = prints_in_window(h)
    fwd = {t: fwd_lag(px[t], h, 1) for t in COHORT}
    recs = []
    for i, d in enumerate(idx):
        elig = [t for t in COHORT if piw_h[t].iloc[i]]
        if len(elig) < K:
            continue
        r5v = {t: ret5_all[t].iloc[i] for t in elig}
        if any(pd.isna(v) for v in r5v.values()):
            continue
        washed = sorted(elig, key=lambda t: r5v[t])[:K]
        alpha = sorted(elig)[:K]
        rnd = sorted(elig, key=lambda t: (hash((t, i)) % 1000))[:K]
        allw = elig
        f = {t: fwd[t].iloc[i] for t in elig}
        if any(pd.isna(v) for v in f.values()):
            continue
        recs.append({
            "date": d,
            "washed": float(np.mean([f[t] for t in washed])),
            "alpha": float(np.mean([f[t] for t in alpha])),
            "rand": float(np.mean([f[t] for t in rnd])),
            "all": float(np.mean([f[t] for t in allw])),
            "xrt": fwd_lag(px["XRT"], h, 1).iloc[i],
            "wash_gate": bool(washout.iloc[i]),
            "names": "/".join(washed),
        })
    B = pd.DataFrame(recs).set_index("date")
    if B.empty:
        print(f"h={h}: no eligible cluster days")
        continue
    epi = declusters(B.index, 10, idx)
    Be = B.loc[epi]
    show([summarize(Be["washed"].values, f"h={h} 3 MOST WASHED"),
          summarize(Be["alpha"].values, f"h={h} 3 ALPHABETICAL (placebo)"),
          summarize(Be["rand"].values, f"h={h} 3 pseudo-random (placebo)"),
          summarize(Be["all"].values, f"h={h} all eligible reporters"),
          summarize(Be["xrt"].values, f"h={h} XRT wrapper, same days")],
         f"basket selection rules, cluster days, episodes N={len(epi)}")
    # and the same restricted to the washed-XRT state that is live today
    Bw = Be[Be["wash_gate"]]
    if len(Bw):
        show([summarize(Bw["washed"].values, f"h={h} WASHED (XRT gate on)"),
              summarize(Bw["alpha"].values, f"h={h} ALPHA (XRT gate on)"),
              summarize(Bw["xrt"].values, f"h={h} XRT (gate on)")],
             f"restricted to the live XRT washout state, N={len(Bw)}")
    print(f"  h={h} washed-vs-alpha diff = "
          f"{100*(Be['washed'].mean()-Be['alpha'].mean()):+.3f}pp   "
          f"washed record {(Be['washed']>0).sum()}-{(Be['washed']<=0).sum()}, "
          f"sign p={sign_test(int((Be['washed']>0).sum()), len(Be)):.4f}")
    print("  last 8 episodes:",
          ", ".join(f"{d.date()}[{Be.loc[d,'names']}] "
                    f"{100*Be.loc[d,'washed']:+.2f}%" for d in epi[-8:]))

# ---------------------------------------------------------------------------
# 6. tomorrow-specific tail risk: print-day move distribution
# ---------------------------------------------------------------------------
print("\n\n===== print-day single-name move distribution (close-to-close) =====")
rows = []
for t in COHORT:
    ed = edates[t]
    d1 = px[t].pct_change()
    p = idx.searchsorted(ed)
    p = p[(p > 0) & (p < len(idx))]
    v = d1.iloc[p].dropna().values
    r = summarize(v, t)
    r["p05_pct"] = round(100 * float(np.percentile(v, 5)), 2)
    r["p95_pct"] = round(100 * float(np.percentile(v, 95)), 2)
    r["abs_mean_pct"] = round(100 * float(np.abs(v).mean()), 2)
    rows.append(r)
show(rows, "print-day moves (all history in the calendar)")
