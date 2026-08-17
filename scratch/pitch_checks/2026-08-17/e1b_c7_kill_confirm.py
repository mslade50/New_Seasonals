"""C7 KILL CONFIRMATION - today's XRT / TJX / ROST / HD state IS the cell the
registry already killed three times. Quotes LIVE numbers, not precedent alone.

Registry entries being confirmed against today's tape:
  1. 2026-08-10 "The August big-box retail earnings cluster" - anchor worth
     1.7 bps against 18 bps of cost.
  2. 2026-08-14 "The pre-print washout ... the two names the idea was about are
     negative in their own cell (TJX -0.543%, ROST -0.425%)".
  3. 2026-08-14 "An 'intact trend' gate on a breadth washout is an INVERTER,
     not a filter" - the first thing to try on any washout-inside-an-uptrend
     construction. Run head on for XRT here: if the split comes out the OTHER
     way on XRT than it did on insurance, that is a real finding.

lag=1, episodes declustered at 10 td.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from earnings_filter import load_earnings_dates_map  # noqa: E402

COHORT = ["HD", "LOW", "TGT", "TJX", "ROST", "WMT"]
px = close_panel(["XRT", "XLY", "SPY"] + COHORT).dropna(subset=["XRT"])
idx = px.index

r5 = pct_rank(px["XRT"], 5)
r63 = pct_rank(px["XRT"], 63)
print(f"LIVE XRT 5d rank {r5.iloc[-1]:.1f}, 63d rank {r63.iloc[-1]:.1f}  "
      f"(5d ret {100*px['XRT'].pct_change(5).iloc[-1]:+.2f}%, "
      f"63d ret {100*px['XRT'].pct_change(63).iloc[-1]:+.2f}%)")

# ---------------------------------------------------------------------------
# TEST 1 - the intact-trend gate as an INVERTER, on XRT
# ---------------------------------------------------------------------------
print("\n===== TEST 1: does the intact-63d gate invert the XRT washout? =====")
wash = r5 < 25
rows = []
for h in (3, 5, 10):
    ret = fwd_lag(px["XRT"], h, 1)
    v = ret.notna()
    ctl = 100 * ret[v].mean()
    for lbl, m in [("washout ALONE (5d rank<25)", wash),
                   ("  + INTACT 63d (>75)  <-- LIVE", wash & (r63 > 75)),
                   ("  + NOT intact (63d<=75)", wash & (r63 <= 75)),
                   ("  + BROKEN 63d (<25)", wash & (r63 < 25))]:
        d = idx[m.values & v.values]
        e = declusters(d, 10, idx)
        r = summarize(ret.loc[e].values, f"h={h} {lbl}")
        r["ctl_all_pct"] = round(ctl, 3)
        r["edge_pp"] = round(r["mean_pct"] - ctl, 3) if r["n"] else np.nan
        rows.append(r)
show(rows, "XRT 5d washout, split by the state of the 63d trend")

# ---------------------------------------------------------------------------
# TEST 2 - the two names the idea is about, in their own pre-print cell
# ---------------------------------------------------------------------------
print("\n===== TEST 2: TJX / ROST / HD in their own pre-print washout cell =====")
emap = load_earnings_dates_map()
rows = []
for t in ["TJX", "ROST", "HD", "LOW", "TGT", "WMT"]:
    ed = pd.DatetimeIndex(emap.get(t, []))
    p = idx.searchsorted(ed)
    p = p[(p > 0) & (p < len(idx))]
    # signal day = 2 sessions before the print (entry MOC next close, print inside)
    sig = np.zeros(len(idx), dtype=bool)
    sig[np.clip(p - 2, 0, len(idx) - 1)] = True
    sig = pd.Series(sig, index=idx)
    r5t = pct_rank(px[t], 5)
    washed = sig & (r5t < 25)
    for h, lbl in [(3, "h=3")]:
        ret = fwd_lag(px[t], h, 1)
        v = ret.notna()
        d = idx[washed.values & v.values]
        e = declusters(d, 10, idx)
        r = summarize(ret.loc[e].values, f"{t} pre-print WASHED {lbl}")
        allpp = idx[sig.values & v.values]
        r["all_prints_pct"] = round(100 * ret.loc[declusters(allpp, 10, idx)].mean(), 3)
        r["ctl_all_pct"] = round(100 * ret[v].mean(), 3)
        rows.append(r)
show(rows, "single-name pre-print washout, h=3, episodes")

# ---------------------------------------------------------------------------
# TEST 3 - the cluster anchor placebo ladder (registry: 7-for-7)
# ---------------------------------------------------------------------------
print("\n===== TEST 3: cluster-anchor placebo ladder on the XRT wrapper =====")
# anchor = the session 2 td before the FIRST cohort print of a cluster
first_prints = []
allp = sorted(set(np.concatenate(
    [idx.searchsorted(pd.DatetimeIndex(emap.get(t, []))) for t in COHORT])))
allp = [q for q in allp if 0 < q < len(idx)]
cl, last = [], -99
for q in allp:
    if q - last > 15:
        cl.append(q)
    last = q
print(f"  {len(cl)} big-box clusters detected since {idx[0].date()}")

h = 3
ret = fwd_lag(px["XRT"], h, 1)
lad = []
for shift in range(-13, 6):
    pts = [q - 2 + shift for q in cl if 0 <= q - 2 + shift < len(idx)]
    d = idx[sorted(set(pts))]
    e = declusters(d, 10, idx)
    r = summarize(ret.loc[e].dropna().values, f"anchor shift {shift:+d}"
                                              f"{'  <-- TRUE' if shift == 0 else ''}")
    lad.append(r)
show(lad, f"XRT h={h} by anchor offset (0 = the real cluster anchor)")
true = [r for r in lad if r["label"].startswith("anchor shift +0")]
vals = [r["mean_pct"] for r in lad if r["n"]]
t0 = [r["mean_pct"] for r in lad if "TRUE" in r["label"]]
if t0:
    rank = 1 + sum(1 for v in vals if v > t0[0])
    print(f"  TRUE anchor ranks {rank} of {len(vals)} offsets; "
          f"true {t0[0]:+.3f}% vs placebo mean {np.mean(vals):+.3f}% "
          f"-> true-minus-placebo {t0[0]-np.mean(vals):+.3f}pp")
