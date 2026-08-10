"""C4 round 1 -- GDX into CPI on a miner thrust.

The registry kills GLD into CPI ("underperforms its own h=2 drift"; "conditioning
on gold already rallying selects the crash tail"). Question: does the kill
transfer to GDX, which is a levered, different instrument?

Anchor: the session k td BEFORE a scheduled CPI print. Today = CPI+2 td.
Entry MOC D+1 (so D = CPI-3 gives a hold that straddles the print; today D is
CPI-2 and the entry close is CPI-1). Priced at k = 1..4 and h = 1..4 so the
"into the print" window is explicit.
Conditioner: GDX rank5 >= 90 (today 100.0).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

LAG = 1
P = close_panel(["GDX", "GLD"]).dropna(subset=["GDX"])
ASOF = P.index[-1]
g = P["GDX"]
idx = g.index
pos = pd.Series(range(len(idx)), index=idx)
rk5 = pct_rank(g, 5)
print(f"sample {idx.min().date()} .. {ASOF.date()}  n={len(idx)}")
print(f"TODAY: GDX rank5={rk5.loc[ASOF]:.1f}")

cpi = load_events(["cpi"])["date"]
cpi = pd.DatetimeIndex([d for d in cpi if d in pos.index or True])

# anchor = the k-th trading day BEFORE each CPI print
def anchors(k: int) -> pd.DatetimeIndex:
    out = []
    for d in cpi:
        # position of the first session on/after the print
        p = int(np.searchsorted(idx.values, np.datetime64(d)))
        if p - k < 0 or p >= len(idx):
            continue
        out.append(idx[p - k])
    return pd.DatetimeIndex(sorted(set(out)))


print(f"\nCPI prints in sample window: {int(((cpi >= idx[0]) & (cpi <= idx[-1])).sum())}")

# --------------------------------------------- the plain pre-CPI fingerprint
print("\n### 1. plain pre-CPI GDX (no thrust conditioner) vs GDX's own drift ###")
for k in (1, 2, 3, 4):
    a = anchors(k)
    rows = []
    for h in (1, 2, 3, 5):
        fw = fwd_lag(g, h, LAG)
        ok = fw.notna()
        t = pd.DatetimeIndex(a).intersection(idx[ok.values])
        e = declusters(t, h, idx[ok.values])
        r = summarize(fw.loc[e].values, f"CPI-{k} anchor, h={h}")
        r["ctrl_all_pct"] = round(100 * fw[ok].mean(), 3)
        r["edge_pct"] = round(r["mean_pct"] - 100 * fw[ok].mean(), 3)
        rows.append(r)
    show(rows, f"anchor = CPI - {k} td (entry MOC CPI-{k-1})")

# --------------------------------------- 2. with the thrust conditioner ON
print("\n### 2. thrust-conditioned: GDX rank5 >= 90 at the anchor (today = 100) ###")
for k in (2, 3):
    a = anchors(k)
    for thr in (80.0, 90.0):
        rows = []
        for h in (1, 2, 3, 5):
            fw = fwd_lag(g, h, LAG)
            ok = fw.notna()
            m = (rk5 >= thr).fillna(False)
            t = pd.DatetimeIndex(a).intersection(idx[m.values & ok.values])
            if len(t) == 0:
                rows.append({"label": f"k={k} rk>={thr} h={h}", "n": 0})
                continue
            e = declusters(t, h, idx[ok.values])
            v = fw.loc[e].values
            r = summarize(v, f"CPI-{k}, rank5>={thr:.0f}, h={h}")
            r["ctrl_allCPI_pct"] = round(
                100 * fw.loc[pd.DatetimeIndex(a).intersection(idx[ok.values])].mean(), 3)
            r["ctrl_alldays_pct"] = round(100 * fw[ok].mean(), 3)
            wins = int((v > 0).sum())
            r["sign_p"] = round(sign_test(wins, len(v)), 4)
            rows.append(r)
        show(rows, f"CPI-{k} anchor x GDX rank5 >= {thr:.0f}")

# --------------------------------------- 3. does the GLD kill transfer?
print("\n### 3. side by side with GLD, the instrument the registry already killed ###")
gl = P["GLD"]
for k in (2, 3):
    a = anchors(k)
    rows = []
    for tk, ser in (("GDX", g), ("GLD", gl)):
        for h in (2, 3):
            fw = fwd_lag(ser, h, LAG)
            ok = fw.notna()
            t = pd.DatetimeIndex(a).intersection(idx[ok.values])
            e = declusters(t, h, idx[ok.values])
            r = summarize(fw.loc[e].values, f"{tk} CPI-{k} h={h}")
            r["own_drift_pct"] = round(100 * fw[ok].mean(), 3)
            r["edge_pct"] = round(r["mean_pct"] - 100 * fw[ok].mean(), 3)
            rows.append(r)
    show(rows, f"transfer test, anchor CPI-{k}")

# --------------------------------------- 4. the crash tail on a rallying metal
print("\n### 4. crash tail check: worst outcomes when metals were already rallying ###")
k, h, thr = 2, 3, 80.0
a = anchors(k)
fw = fwd_lag(g, h, LAG)
ok = fw.notna()
m = (rk5 >= thr).fillna(False)
t = pd.DatetimeIndex(a).intersection(idx[m.values & ok.values])
e = declusters(t, h, idx[ok.values])
v = fw.loc[e].values
srt = np.argsort(v)
print(f"  N={len(v)}  mean={100*v.mean():+.3f}%  worst 4:",
      [(str(pd.Timestamp(e[i]).date()), round(100 * v[i], 2)) for i in srt[:4]])
print(f"  best 4:", [(str(pd.Timestamp(e[i]).date()), round(100 * v[i], 2)) for i in srt[-4:]])
mt = np.array([d.year % 4 == 2 for d in e])
show([summarize(v[mt], f"MIDTERM (N={int(mt.sum())})"), summarize(v[~mt], "non-midterm")],
     "midterm split")
show(era_split(e, v), "era split")
print("  concentration:", cluster_note(e, v))
