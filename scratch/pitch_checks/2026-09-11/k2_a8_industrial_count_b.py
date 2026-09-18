"""A8 round 2 - the candidate's STRONGEST available form.

Round 1 killed the pitch_lab-convention cell (live count 3, whose exact bucket
pays -0.242% against an all-days +0.211%). This script gives the candidate the
benefit of the doubt twice over:

 (1) recompute the count under the TAPE convention
     (build_pitch_state._metrics_for: 10d return / (21d sd * sqrt(10))), which
     is where the handover's MMM -3.55 / ITW -2.60 numbers came from, and test
     the count the tape actually shows;
 (2) score whatever rung is live under that convention, with the same set
     membership / complement / reference-class charges.

If the object dies on BOTH conventions there is no convention argument left.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

ASOF = pd.Timestamp("2026-09-10")
Z_THR = -2.0
H = 5

COMPLEX = {
    "XLI": ["MMM", "ITW", "GD", "PH", "DOV", "NSC", "EMR", "HON", "CAT",
            "UNP", "ROK"],
    "XLE": ["XOM", "CVX", "COP", "SLB", "EOG", "PSX", "MPC", "VLO", "OXY",
            "HAL", "WMB"],
    "XLK": ["AAPL", "MSFT", "NVDA", "AVGO", "CSCO", "ORCL", "CRM", "ACN",
            "ADBE", "TXN", "INTC"],
    "XLF": ["JPM", "BAC", "WFC", "C", "GS", "MS", "USB", "PNC", "AXP",
            "SCHW", "BLK"],
    "XLV": ["JNJ", "PFE", "MRK", "CI", "LLY", "UNH", "TMO", "ABT", "BMY",
            "AMGN", "CVS"],
    "XLP": ["PG", "KO", "PEP", "WMT", "COST", "MO", "PM", "MDLZ", "CL",
            "KMB", "GIS"],
    "XLY": ["AMZN", "HD", "MCD", "NKE", "SBUX", "LOW", "TJX", "YUM", "TGT",
            "F", "GM"],
    "XLB": ["LIN", "APD", "SHW", "ECL", "NEM", "FCX", "IP", "DD", "PPG",
            "NUE", "VMC"],
    "XLU": ["NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL", "ED",
            "PEG", "WEC"],
}
DEFENDED = "XLI"
need = sorted({t for v in COMPLEX.values() for t in v} | set(COMPLEX))
px = load_prices(need)


def z10_tape(s: pd.Series) -> pd.Series:
    """build_pitch_state._metrics_for convention: 10d return over 21d daily
    sd scaled to 10 days. A DIFFERENT object from pitch_lab.zscore."""
    v = s.dropna()
    r10 = v.pct_change(10)
    vol21 = v.pct_change().rolling(21).std()
    return (r10 / (vol21 * np.sqrt(10))).reindex(s.index)


ZT = {t: z10_tape(px[t]["Close"]) for t in px}
ZL = {t: zscore(px[t]["Close"], 10) for t in px}


def count_series(sector, Z):
    members = [t for t in COMPLEX[sector] if t in px]
    idx = px[sector].index
    hit = pd.DataFrame({t: (Z[t] <= Z_THR).reindex(idx).fillna(False).astype(bool)
                        for t in members})
    defined = pd.DataFrame({t: Z[t].reindex(idx).notna() for t in members})
    return hit.sum(axis=1).where(defined.all(axis=1)), members


print("=" * 78)
print("1. TODAY'S COUNT under BOTH conventions")
ct, members = count_series(DEFENDED, ZT)
cl, _ = count_series(DEFENDED, ZL)
print("  member z10 (tape / pitch_lab):")
for t in members:
    print(f"    {t:<5} tape={float(ZT[t].loc[:ASOF].iloc[-1]):+.2f}   "
          f"lab={float(ZL[t].loc[:ASOF].iloc[-1]):+.2f}")
LIVE_T = int(ct.loc[:ASOF].iloc[-1])
LIVE_L = int(cl.loc[:ASOF].iloc[-1])
print(f"  COUNT tape convention = {LIVE_T}    pitch_lab convention = {LIVE_L}")
print(f"  XLI own z10: tape {float(ZT[DEFENDED].loc[:ASOF].iloc[-1]):+.2f}  "
      f"lab {float(ZL[DEFENDED].loc[:ASOF].iloc[-1]):+.2f}")
agree = ((ct >= LIVE_T) == (cl >= LIVE_T)).mean()
print(f"  the two conventions agree on 'count >= {LIVE_T}' on "
      f"{100*agree:.1f}% of days")

# ---------------------------------------------------------------------------
xli = px[DEFENDED]["Close"]
print("\n" + "=" * 78)
print(f"2. DOSE RESPONSE under the TAPE convention, long XLI h={H}")
fw = fwd_lag(xli, H, 1)
val = fw.dropna().index
base = fw.loc[val]
print(f"  all days N={len(base)} mean={100*base.mean():+.3f}%")
rows = []
for k in range(0, 12):
    v = base[(ct.reindex(val) == k).fillna(False).values]
    if len(v) < 3:
        continue
    rows.append(summarize(v.values, f"count=={k}"))
show(rows, "exact count (tape convention)")
rows = []
for k in range(1, 10):
    m = (ct.reindex(val) >= k).fillna(False)
    v = base[m.values]
    if len(v) < 3:
        continue
    r = summarize(v.values, f"count>={k}")
    ep = declusters(v.index, H, val)
    r["epi_n"] = len(ep)
    r["epi_mean_pct"] = round(100 * base.loc[ep].mean(), 3)
    r["excess_vs_alldays"] = round(100 * (base.loc[ep].mean() - base.mean()), 3)
    rows.append(r)
show(rows, "cumulative count (tape convention)")

# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print(f"3. SET MEMBERSHIP + COMPLEMENT at the live tape rung (>= {LIVE_T})")
own = (ZT[DEFENDED].reindex(val) <= Z_THR).fillna(False)
m = (ct.reindex(val) >= LIVE_T).fillna(False)
print(f"  XLI own tape z10 <= -2 base rate {100*own.mean():.1f}%; "
      f"P(own | count>={LIVE_T}) = {float(own[m.values].mean()):.3f}; "
      f"count-ON w/o XLI = {int((m & ~own).sum())} days")
rows = []
for lbl, mm in [("XLI own tape z<=-2 ALONE", own),
                (f"count>={LIVE_T} ALONE", m),
                (f"own AND count>={LIVE_T}", own & m),
                (f"own AND count<{LIVE_T} (COMPLEMENT)", own & ~m),
                (f"count>={LIVE_T} AND own FALSE", m & ~own)]:
    v = base[mm.values]
    if len(v) < 3:
        rows.append({"label": lbl, "n": len(v)})
        continue
    r = summarize(v.values, lbl)
    ep = declusters(v.index, H, val)
    r["epi_n"] = len(ep)
    r["epi_mean_pct"] = round(100 * base.loc[ep].mean(), 3)
    rows.append(r)
show(rows, "count vs the single-instrument cell (tape convention)")

# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print(f"4. REFERENCE CLASS at the live rung, tape convention, charged max-of-K")
res = []
for sector in COMPLEX:
    c, _ = count_series(sector, ZT)
    s = px[sector]["Close"]
    f = fwd_lag(s, H, 1)
    v_ = f.dropna().index
    mm = (c.reindex(v_) >= LIVE_T).fillna(False)
    sub = f.loc[v_][mm.values]
    if len(sub) < 3:
        res.append((sector, len(sub), 0, np.nan, np.nan))
        continue
    ep = declusters(sub.index, H, v_)
    e = f.loc[ep].values
    res.append((sector, len(sub), len(ep), 100 * e.mean(),
                100 * (e.mean() - f.loc[v_].mean())))
rd = pd.DataFrame(res, columns=["sector", "n_days", "n_epi", "epi_mean_pct",
                                "excess_pct"])
print(rd.round(3).to_string(index=False))
ok = rd.dropna(subset=["excess_pct"])
d = float(ok.set_index("sector").loc[DEFENDED, "excess_pct"])
rank = int((ok["excess_pct"] > d).sum()) + 1
print(f"  {DEFENDED} excess {d:+.3f}% ranks {rank} of {len(ok)}")

# charged max-of-K permutation against the DEFENDED cell
print("\n  charged max-of-K permutation (K = the 9 sectors x the 4 rungs "
      "3,4,5,6 walked in round 1 = 36 cells), scored on the DEFENDED excess")
rng = np.random.default_rng(11)
cells = []
for sector in COMPLEX:
    c, _ = count_series(sector, ZT)
    s = px[sector]["Close"]
    f = fwd_lag(s, H, 1)
    v_ = f.dropna().index
    for K in (3, 4, 5, 6):
        mm = (c.reindex(v_) >= K).fillna(False)
        if mm.sum() < 5:
            continue
        cells.append((f.loc[v_].values, mm.values))
nulls = []
for _ in range(2000):
    best = -np.inf
    for arr, mm in cells:
        n = int(mm.sum())
        idx = rng.integers(0, len(arr), size=n)
        best = max(best, arr[idx].mean() - arr.mean())
    nulls.append(best)
nulls = np.asarray(nulls)
print(f"  null-max median excess {100*np.median(nulls):+.3f}%; "
      f"P(null max >= defended {d:+.3f}%) = {float((nulls >= d/100).mean()):.4f}")
