"""C2 round 2: is the silver/gold pair after a first complex break anything other than
watchlist 28's outright silver short (registry 2776 already filed its GLD-beta residual)?

(a) collision: episode correlation of the pair with the W28 outright short, and the
    share of the pair's return carried by the silver leg;
(b) mechanism/live-state test: split first breaks by the SIGNAL-DAY silver residual
    (SLV move minus PIT beta x GLD move). Today silver FELL LESS than its beta implied
    (+1.50pp); a "high-beta member underperforms the anchor" story owes this split;
(c) entry-day split (registry 2662): silver's move on the entry session;
(d) the W28 split charge: first vs all vs follow at h=1..3, pre-2018 vs 2018+ vs cost;
(e) NFP-in-hold at h=2/3.
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
px = close_panel(["GLD", "SLV", "GDX"]).dropna().loc[:BAR]
idx = px.index
r1 = px.pct_change(fill_method=None)
faith = (r1["GLD"] <= -0.02) & (r1["SLV"] <= -0.02) & (r1["GDX"] <= -0.02)
first = faith & ~faith.shift(1).rolling(5).max().fillna(0).astype(bool)
beta = r1["SLV"].rolling(252).cov(r1["GLD"]) / r1["GLD"].rolling(252).var()
resid_d = r1["SLV"] - beta * r1["GLD"]
entry_slv = r1["SLV"].shift(-1)  # silver's move on the entry session (D+1), known at entry close


def pair(h, lag=1):
    return -fwd_lag(px["SLV"], h, lag) + beta * fwd_lag(px["GLD"], h, lag)


def outright(h, lag=1):
    return -fwd_lag(px["SLV"], h, lag)


def stats(v, label, base):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    w = int((v > 0).sum())
    o = summarize(v, label)
    if len(v):
        o["rec"] = f"{w}-{len(v)-w}"
        o["edge_pp"] = round(o["mean_pct"] - 100 * np.nanmean(base), 3)
        o["p_coin"] = round(sign_test(w, len(v)), 4)
    return o


print(f"TODAY residual {100*resid_d.loc[BAR]:+.2f}pp (PIT beta {beta.loc[BAR]:.3f}); "
      f"pct rank of today's residual among first breaks: "
      f"{100*(resid_d[first] < resid_d.loc[BAR]).mean():.0f}")

# (a) collision with W28's outright
for h in (1, 2, 3):
    p, o = pair(h), outright(h)
    ok = p.notna() & o.notna() & beta.notna()
    e = declusters(idx[first.values & ok.values], h, idx)
    c = np.corrcoef(p.loc[e], o.loc[e])[0, 1]
    gl = (beta * fwd_lag(px["GLD"], h)).loc[e]
    print(f"(a) h={h}: n={len(e)} pair {100*p.loc[e].mean():+.3f}%  W28 outright {100*o.loc[e].mean():+.3f}%  "
          f"corr {c:+.2f}  silver-leg share of pair {100*o.loc[e].mean()/p.loc[e].mean():.0f}%  "
          f"beta-GLD leg {100*gl.mean():+.3f}%  (gold leg up in {100*(gl>0).mean():.0f}% of eps)")

# (b) signal-day residual split
for h in (1, 2, 3):
    p = pair(h)
    ok = p.notna() & beta.notna()
    base = p[ok]
    e = declusters(idx[first.values & ok.values], h, idx)
    rd = resid_d.loc[e]
    q = rd.quantile([1 / 3, 2 / 3]).values
    rows = [stats(p.loc[e].values, "all first breaks", base),
            stats(p.loc[e[rd.values > 0]].values, "silver fell LESS than beta (resid>0, TODAY)", base),
            stats(p.loc[e[rd.values <= 0]].values, "silver fell MORE than beta (resid<=0)", base),
            stats(p.loc[e[rd.values <= q[0]]].values, f"resid tercile 1 (<= {100*q[0]:+.2f}pp)", base),
            stats(p.loc[e[(rd.values > q[0]) & (rd.values <= q[1])]].values, "resid tercile 2", base),
            stats(p.loc[e[rd.values > q[1]]].values, f"resid tercile 3 (> {100*q[1]:+.2f}pp)", base),
            stats(p.loc[e[rd.values >= 0.01]].values, "resid >= +1.0pp", base)]
    show(rows, f"(b) pair split by signal-day silver residual, h={h} lag 1")
    oo = outright(h)
    show([stats(oo.loc[e[rd.values > 0]].values, "W28 outright, resid>0", oo.dropna()),
          stats(oo.loc[e[rd.values <= 0]].values, "W28 outright, resid<=0", oo.dropna())],
         f"   same split on the W28 outright short, h={h}")

# (c) entry-day split (silver on D+1)
for h in (1, 3):
    p = pair(h)
    ok = p.notna() & beta.notna()
    e = declusters(idx[first.values & ok.values], h, idx)
    es = entry_slv.loc[e]
    show([stats(p.loc[e[es.values >= 0.01]].values, "entry-day SLV >= +1%", p[ok]),
          stats(p.loc[e[(es.values > -0.01) & (es.values < 0.01)]].values, "entry-day SLV within 1%", p[ok]),
          stats(p.loc[e[es.values <= -0.01]].values, "entry-day SLV <= -1%", p[ok])],
         f"(c) entry-day split, pair h={h}")

# (d) split charge + era vs cost (6 bp round trip for the pair)
rows = []
for h in (1, 2, 3):
    p = pair(h)
    ok = p.notna() & beta.notna()
    for lbl, m in (("FIRST", first), ("ALL", faith)):
        e = declusters(idx[m.values & ok.values], h, idx)
        for era, sel in (("pre-2018", e < "2018-01-01"), ("2018+", e >= "2018-01-01"),
                         ("2018-2024", (e >= "2018-01-01") & (e < "2025-01-01")),
                         ("2025-26", e >= "2025-01-01")):
            o = stats(p.loc[e[sel]].values, f"{lbl} h={h} {era}", p[ok])
            if o["n"]:
                o["x_cost"] = round(o["mean_pct"] * 100 / 6.0, 1)
            rows.append(o)
show(rows, "(d) era slices vs 6 bp pair round trip")

# (e) NFP in hold
for h in (2, 3):
    p = pair(h)
    ok = p.notna() & beta.notna()
    e = declusters(idx[first.values & ok.values], h, idx)
    fl = event_in_window(e, idx, h, 1, ("nfp",))
    show([stats(p.loc[e[fl]].values, f"NFP in hold h={h}", p[ok]),
          stats(p.loc[e[~fl]].values, f"no NFP h={h}", p[ok])], f"(e) NFP split h={h}")
