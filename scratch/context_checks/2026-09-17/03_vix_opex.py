"""VIX into monthly opex when it arrives already crushed.

`E:opex|^VIX|k1` is one of three BH survivors in a 1217-cell sweep: n=320,
opex-session mean -1.01%, 107-210 down, t -2.63, era-stable. But today the VIX
already fell 12.82% to 15.44. The cell's whole claim is that opex week bleeds
vol; arriving pre-bled is the condition that could kill it. Split on the
anchor session's own VIX move, then check the level dependence.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    close_panel, load_events, summarize, show, sign_test, cluster_note,
)

px = close_panel(["^VIX", "SPY", "^VIX3M"])
idx = px.index
vix = px["^VIX"].dropna()
r1 = vix / vix.shift(1) - 1.0
fwd1 = vix.shift(-1) / vix - 1.0

opex = pd.DatetimeIndex(load_events(["opex"])["date"])
pos = pd.Series(range(len(idx)), index=idx)
# anchor = the session immediately before an opex session
anchors = []
for d in opex:
    p = pos.get(d)
    if p is None or p == 0:
        continue
    anchors.append(idx[p - 1])
anchors = pd.DatetimeIndex(anchors).intersection(fwd1.dropna().index)
print(f"opex anchors: {len(anchors)}  {anchors[0].date()} .. {anchors[-1].date()}")

base = fwd1.dropna()
rows = [summarize(fwd1.loc[anchors].values, "all opex sessions"),
        summarize(base.values, "CTL all sessions")]
show(rows, "VIX change on the opex session")

print("\n=== split by the ANCHOR session's own VIX move (today: -12.82%) ===")
anc_move = r1.reindex(anchors)
cuts = [("anchor VIX <= -8%", anc_move <= -0.08),
        ("anchor VIX -8% to -3%", (anc_move > -0.08) & (anc_move <= -0.03)),
        ("anchor VIX -3% to 0", (anc_move > -0.03) & (anc_move <= 0)),
        ("anchor VIX up", anc_move > 0)]
rows = []
for lab, m in cuts:
    d = anchors[m.values]
    r = summarize(fwd1.loc[d].values, lab)
    if r["n"]:
        k = int((fwd1.loc[d] > 0).sum())
        r["record"] = f"{k}-{r['n'] - k} up"
        r["sign_p_down"] = round(sign_test(r["n"] - k, r["n"]), 4)
    rows.append(r)
show(rows, "opex-session VIX change by anchor move")

hard = anchors[(anc_move <= -0.08).values]
print(f"  anchor VIX <= -8% dates ({len(hard)}): "
      f"{[str(d.date()) for d in hard][-12:]}")
v = fwd1.loc[hard]
k = int((v > 0).sum())
print(f"  record {k}-{len(v) - k} up, mean {100 * v.mean():+.2f}%, "
      f"median {100 * v.median():+.2f}%, sign p(up) = {sign_test(k, len(v)):.4f}")
for lab, m in [("pre-2018", hard < pd.Timestamp("2018-01-01")),
               ("2018+", hard >= pd.Timestamp("2018-01-01"))]:
    vv = fwd1.loc[hard[m]]
    kk = int((vv > 0).sum())
    print(f"    {lab:<10} n={len(vv):<3} mean={100 * vv.mean():+.3f}%  {kk}-{len(vv) - kk} up")
print(f"  concentration: {cluster_note(hard, v.values, k=2)}")

print("\n=== the same split, but SPY on the opex session ===")
spy = px["SPY"].dropna()
sf = spy.shift(-1) / spy - 1.0
rows = []
for lab, m in cuts:
    d = anchors[m.values].intersection(sf.dropna().index)
    r = summarize(sf.loc[d].values, lab)
    if r["n"]:
        k = int((sf.loc[d] > 0).sum())
        r["record"] = f"{k}-{r['n'] - k} up"
    rows.append(r)
rows.append(summarize(sf.dropna().values, "CTL all sessions"))
show(rows, "SPY opex-session return by anchor VIX move")

print("\n=== level dependence: VIX below 17 into opex ===")
lvl = vix.reindex(anchors)
rows = []
for lab, m in [("VIX < 15", lvl < 15), ("VIX 15-17", (lvl >= 15) & (lvl < 17)),
               ("VIX 17-22", (lvl >= 17) & (lvl < 22)), ("VIX >= 22", lvl >= 22)]:
    d = anchors[m.values]
    r = summarize(fwd1.loc[d].values, lab)
    if r["n"]:
        k = int((fwd1.loc[d] > 0).sum())
        r["record"] = f"{k}-{r['n'] - k} up"
    rows.append(r)
show(rows, "opex-session VIX change by starting level")

print("\n=== term structure: VIX3M/VIX today is 18.55/15.44 = 1.201 ===")
ts = (px["^VIX3M"] / px["^VIX"]).dropna()
tsa = ts.reindex(anchors).dropna()
rows = []
for lab, m in [("VIX3M/VIX >= 1.18", tsa >= 1.18), ("1.10-1.18", (tsa >= 1.10) & (tsa < 1.18)),
               ("< 1.10", tsa < 1.10)]:
    d = tsa.index[m.values]
    r = summarize(fwd1.loc[d].values, lab)
    if r["n"]:
        k = int((fwd1.loc[d] > 0).sum())
        r["record"] = f"{k}-{r['n'] - k} up"
    rows.append(r)
show(rows, "opex-session VIX change by term structure")
