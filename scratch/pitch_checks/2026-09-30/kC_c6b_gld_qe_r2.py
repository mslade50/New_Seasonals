"""C6 round 2: long GLD from the QE close after a QE-week flush (5d <= -3%).

(1) concentration (cluster_note, drop-best-2); (2) placebo offset ladder: the same flush
gate at QE+k, k=-6..+6 (is the QUARTER-END close special?); (3) live-regime split: the
live state is ^TNX within 2% of its 252 max, GLD under its 200d, DX 21d rank >= 85, GLD
>= 15% under its 252 high -- measured in the QE cell and in the broad flush family
(declustered non-QE flush episodes); (4) per-session decomposition of the 7 episodes;
(5) the tradeable QE-1-gated form.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

P = load_prices(["GLD", "GC=F", "^TNX", "DX-Y.NYB"])
g = P["GLD"]["Close"].dropna()
gc = P["GC=F"]["Close"].dropna()
tnx = P["^TNX"]["Close"].dropna()
dx = P["DX-Y.NYB"]["Close"].dropna()
idx = g.index
LAST = idx[-1]
s_ = pd.Series(idx, index=idx)
ME = pd.DatetimeIndex(s_.groupby([idx.year, idx.month]).max().values)
ME = ME[(ME < LAST) & (ME >= "2004-12-01")]
QE = ME[ME.month.isin([3, 6, 9, 12])]
qpos = np.array([idx.get_loc(q) for q in QE])

r5 = g / g.shift(5) - 1.0
flush = r5 <= -0.03
tnx_hi = (tnx >= 0.98 * rolling_on_valid(tnx, lambda x: x.rolling(252).max())).reindex(idx).fillna(False).astype(bool)
below200 = (g < g.rolling(200).mean())
dx21 = pct_rank(dx, 21).reindex(idx).ffill()
dxhot = dx21 >= 85
deep = g <= 0.85 * g.rolling(252).max()
live = pd.DataFrame({"TNXhi": tnx_hi, "below200": below200, "DXr21>=85": dxhot, ">=15%offHi": deep}).iloc[-1]
print("live state 09-29:", live.to_dict())


def fwd(s, anchors, h):
    return (s.shift(-h) / s - 1.0).reindex(pd.DatetimeIndex(anchors)).dropna()


def rec(v):
    w = int((v > 0).sum())
    return f"{w}-{len(v)-w}", sign_test(w, len(v)) if len(v) else np.nan


# the 7 episodes: GC=F 2002-06-28 + 6 GLD
gq = QE[flush.reindex(QE).fillna(False).values]
gc_ep = pd.Timestamp("2002-06-28")

# (1) concentration
print("\n(1) concentration (GLD-era 6 + GC=F 2002):")
for h in (3, 5, 7, 10):
    v = pd.concat([fwd(gc, [gc_ep], h), fwd(g, gq, h)])
    srt = np.sort(v.values)[::-1]
    print(f"  h={h}: mean {100*v.mean():+.3f}%  {cluster_note(v.index, v.values)}; drop-best-2 mean "
          f"{100*srt[2:].mean():+.3f}% ({int((srt[2:]>0).sum())}-{int((srt[2:]<=0).sum())})")

# (2) placebo offset ladder (GLD era only)
print("\n(2) placebo offset ladder: flush gate (5d <= -3%) at QE+k, long GLD from that close")
rows = []
for k in range(-6, 7):
    a = idx[np.clip(qpos + k, 0, len(idx) - 1)]
    a = a[flush.reindex(a).fillna(False).values]
    for h in (5, 7):
        v = fwd(g, a, h)
        r, p = rec(v)
        rows.append({"k": k, "h": h, "n": len(v), "mean_pct": 100 * v.mean(), "rec": r, "sign_p": p})
lad = pd.DataFrame(rows)
print(lad.pivot(index="k", columns="h", values=["n", "mean_pct", "rec"]).to_string())
for h in (5, 7):
    x = lad[lad.h == h].set_index("k")["mean_pct"]
    print(f"  h={h}: QE+0 {x.loc[0]:+.3f}%, rank {int(x.rank(ascending=False).loc[0])} of 13; "
          f"mean of k != 0 rungs {x.drop(0).mean():+.3f}%")

# (3) live-regime split, QE cell and the broad flush family (declustered, non-QE)
print("\n(3) regime split")
nonqe = idx[flush.values & ~idx.isin(QE)]
rows = []
for h in (3, 5):
    ep = declusters(nonqe, h, idx)
    base_v = fwd(g, ep, h)
    for lbl, st in (("ALL", pd.Series(True, index=idx)), ("TNX at high", tnx_hi), ("below 200d", below200),
                    ("DX r21>=85", dxhot), (">=15% off high", deep),
                    ("TNXhi & below200", tnx_hi & below200),
                    ("LIVE: all four", tnx_hi & below200 & dxhot & deep)):
        stv = st.reindex(idx).fillna(False).astype(bool)
        q = gq[stv.reindex(gq).values]
        vq = fwd(g, q, h)
        e2 = ep[stv.reindex(ep).values]
        ve = fwd(g, e2, h)
        rq, pq = rec(vq)
        re_, pe = rec(ve)
        rows.append({"h": h, "state": lbl, "QE_n": len(vq), "QE_mean": 100 * vq.mean() if len(vq) else np.nan,
                     "QE_rec": rq, "flush_nonQE_n": len(ve), "flush_nonQE_mean": 100 * ve.mean() if len(ve) else np.nan,
                     "flush_rec": re_, "flush_sign_p": pe})
show(rows, "QE-gated cell vs declustered non-QE flush episodes, by live-state leg (long GLD)")
print("  QE episodes' state:")
for q in gq:
    print(f"   {q.date()} TNXhi={bool(tnx_hi.loc[q])} below200={bool(below200.loc[q])} "
          f"DXr21={dx21.loc[q]:.0f} off-hi={100*(1-g.loc[q]/g.rolling(252).max().loc[q]):.1f}%")

# (4) per-session decomposition
print("\n(4) per-session returns of the 7 episodes (long), sessions 1..10:")
paths = episode_paths(pd.DataFrame({"GLD": g}), gq, [("GLD", 1.0)], 10, lag=0)
sess = paths.diff(axis=1)
sess[1] = paths[1]
sess = sess[sorted(sess.columns)]
print((100 * sess).round(2).to_string())
print("  mean by session:", (100 * sess.mean()).round(3).to_dict())
print(f"  share of the h=5 mean from session 1: {sess[1].mean()/paths[5].mean():.2f}")

# (5) tradeable QE-1 gate
print("\n(5) gate read at the QE-1 close (known before the MOC), entry QE close:")
qm1 = pd.DatetimeIndex([idx[p - 1] for p in qpos])
m1 = QE[flush.reindex(qm1).fillna(False).values]
for h in (1, 3, 5, 7, 10):
    v = fwd(g, m1, h)
    r, p = rec(v)
    print(f"  h={h}: n {len(v)} mean {100*v.mean():+.3f}% {r} p {p:.3f}  dates {[str(d.date()) for d in m1]}"
          if h == 1 else f"  h={h}: n {len(v)} mean {100*v.mean():+.3f}% {r} p {p:.3f}")
# union of at-close and QE-1 gates
uni = QE[(flush.reindex(qm1).fillna(False).values) | (flush.reindex(QE).fillna(False).values)]
for h in (3, 5, 7):
    v = fwd(g, uni, h)
    r, p = rec(v)
    print(f"  UNION (either gate) h={h}: n {len(v)} mean {100*v.mean():+.3f}% {r} p {p:.3f}")
