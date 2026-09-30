"""kA c1 round 2 - W23 armed: XLU r21 <= 5 AND TLT r21 < 25, long XLU.

Pays the watchlist debts and runs the adverse slices:
  A. clustering / decluster-gap sensitivity / first-day vs later days
  B. definition neighbours (lookbacks, IEF for TLT, ^TNX rank for TLT, raw returns)
  C. era + regime: midterm, SPY vs 200d, midterm x above-200d (TODAY's slice, W35)
  D. gate attribution: filter_vs_reanchor(parent = XLU washout, child = joint)
  E. debt 2: ten-sector reference class on the SAME form + rate-sensitive family
  F. does the TLT gate just re-anchor to "^TNX at a 252 high"?
  G. debt 1: in-sample episodes after the arm was written, and the 2026-08-10 instance
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
from pitch_lab import _valid_pct_change
import numpy as np
import pandas as pd
from scipy.stats import chi2

SECT = ["XLU", "XLK", "XLI", "XLP", "XLV", "XLE", "XLF", "XLY", "XLB", "XLRE"]
FAM = ["XLU", "XLRE", "XLP", "ITB", "XHB", "VNQ"]
TK = sorted(set(SECT + FAM + ["TLT", "IEF", "SPY", "^TNX"]))
px = close_panel(TK)
px = px[px.index >= "2002-07-30"]
idx = px.index
GAP = 21

rk = {(t, n): pct_rank(px[t], n) for t in TK for n in (10, 15, 21, 42)}
wash = rk[("XLU", 21)] <= 5
tlt_hit = rk[("TLT", 21)] < 25
joint = (wash & tlt_hit).reindex(idx, fill_value=False)
sma200 = rolling_on_valid(px["SPY"], lambda x: x.rolling(200).mean())
above200 = (px["SPY"] > sma200).reindex(idx, fill_value=False)
midterm = pd.Series(idx.year % 4 == 2, index=idx)
tnx_hi = rolling_on_valid(px["^TNX"], lambda x: x.rolling(252).max())
tnx_gap_bp = 100 * (tnx_hi - px["^TNX"])   # yield points x100 = bp below the 252 max


def ep(mask, h, legs=(("XLU", 1.0),), gap=GAP, lbl=""):
    ret = vehicle_ret(px, list(legs), h)
    m = mask.reindex(idx, fill_value=False)
    d = idx[m.values].intersection(ret.dropna().index)
    if len(d) == 0:
        return {"label": lbl, "n": 0}, np.array([]), pd.DatetimeIndex([])
    e = declusters(d, gap, idx)
    v = ret.loc[e].values
    r = summarize(v, lbl)
    r["n_days"] = len(d)
    r["base_pct"] = 100 * ret.dropna().mean()
    r["edge_pp"] = r["mean_pct"] - r["base_pct"]
    r["sign_p"] = sign_test(int((v > 0).sum()), len(v))
    return r, v, e


# ------------------------------------------------------------------ A
print("=" * 78, "\nA. decluster sensitivity and first-day vs later days (h=5)\n" + "=" * 78)
rows = []
for g in (5, 10, 21, 42, 63):
    r, v, e = ep(joint, 5, gap=g, lbl=f"gap {g}")
    rows.append(r)
show(rows, "joint, h=5, by decluster gap")
ret5 = vehicle_ret(px, [("XLU", 1.0)], 5)
d = idx[joint.values].intersection(ret5.dropna().index)
e = declusters(d, GAP, idx)
later = d.difference(e)
show([summarize(ret5.loc[e].values, "first day of each 21td cluster"),
      summarize(ret5.loc[later].values, "later days in cluster")], "joint h=5 first vs later")
# random-day-in-cluster placebo: pick a random day of each cluster
pos = pd.Series(range(len(idx)), index=idx)
clusters, cur = [], []
for dd in d:
    if cur and pos[dd] - pos[cur[0]] >= GAP:
        clusters.append(cur)
        cur = []
    cur.append(dd)
if cur:
    clusters.append(cur)
rng = np.random.default_rng(3)
means = [np.nanmean([ret5.loc[c[rng.integers(len(c))]] for c in clusters]) for _ in range(2000)]
print(f"  random-day-per-cluster h=5 mean {100*np.mean(means):+.3f}% "
      f"(5-95: {100*np.percentile(means,5):+.3f} .. {100*np.percentile(means,95):+.3f}); "
      f"first-day {100*ret5.loc[e].mean():+.3f}%; clusters {len(clusters)}; "
      f"days/cluster median {np.median([len(c) for c in clusters]):.0f}")

# ------------------------------------------------------------------ B
print("\n" + "=" * 78, "\nB. definition neighbours (episode level, gap 21)\n" + "=" * 78)
for h in (3, 5):
    rows = []
    specs = {
        "PRE-REG XLU r21<=5 & TLT r21<25": joint,
        "XLU r10<=5 & TLT r10<25": (rk[("XLU", 10)] <= 5) & (rk[("TLT", 10)] < 25),
        "XLU r15<=5 & TLT r15<25": (rk[("XLU", 15)] <= 5) & (rk[("TLT", 15)] < 25),
        "XLU r42<=5 & TLT r42<25": (rk[("XLU", 42)] <= 5) & (rk[("TLT", 42)] < 25),
        "XLU r21<=5 & IEF r21<25": wash & (rk[("IEF", 21)] < 25),
        "XLU r21<=5 & TNX r21>75": wash & (rk[("^TNX", 21)] > 75),
        "XLU r21<=5 & TLT r21<25 & TLT r21 ret<0": joint & (_valid_pct_change(px["TLT"], 21) < 0),
        "XLU r21<=5 & TLT r21 in [25,75] (pitched 08-25)": wash & (rk[("TLT", 21)] >= 25) & (rk[("TLT", 21)] <= 75),
        "XLU r21<=5 & TLT r21>75": wash & (rk[("TLT", 21)] > 75),
    }
    for lbl, m in specs.items():
        rows.append(ep(m, h, lbl=lbl)[0])
    show(rows, f"h={h}")

# ------------------------------------------------------------------ C
print("\n" + "=" * 78, "\nC. era and regime splits\n" + "=" * 78)
for h in (3, 5):
    rows = []
    for lbl, m in [("all", joint),
                   ("midterm", joint & midterm), ("non-midterm", joint & ~midterm),
                   ("SPY > 200d", joint & above200), ("SPY < 200d", joint & ~above200),
                   ("midterm & SPY > 200d  <-- TODAY", joint & midterm & above200),
                   ("non-mid & SPY > 200d", joint & ~midterm & above200),
                   ("pre-2018", joint & pd.Series(idx < "2018-01-01", index=idx)),
                   ("2018+", joint & pd.Series(idx >= "2018-01-01", index=idx)),
                   ("ex-2008/2009", joint & pd.Series(~idx.year.isin([2008, 2009]), index=idx))]:
        rows.append(ep(m, h, lbl=lbl)[0])
    show(rows, f"h={h}")
r, v, e = ep(joint & midterm & above200, 5, lbl="x")
print("  midterm & >200d episodes h=5:", [(str(a.date()), round(100 * b, 2)) for a, b in zip(e, v)])
r, v, e = ep(joint & above200, 5, lbl="x")
print("  all >200d episodes h=5:", [(str(a.date()), round(100 * b, 2)) for a, b in zip(e, v)])

# ------------------------------------------------------------------ D
print("\n" + "=" * 78, "\nD. gate attribution\n" + "=" * 78)
for h in (3, 5):
    ret = vehicle_ret(px, [("XLU", 1.0)], h)
    p_ep = declusters(idx[wash.reindex(idx, fill_value=False).values].intersection(ret.dropna().index), GAP, idx)
    c_ep = declusters(idx[joint.values].intersection(ret.dropna().index), GAP, idx)
    pm = pd.Series(idx.isin(p_ep), index=idx)
    cm = pd.Series(idx.isin(c_ep), index=idx)
    out = filter_vs_reanchor(ret, pm, cm, idx, window_td=21, label=f"h={h} episode anchors")
    if out["n_matched"]:
        rn = reanchor_null(ret, [a for a, _, _ in out["pairs"]], out["shifts"], idx,
                           float(ret.reindex(pd.DatetimeIndex([b for _, b, _ in out["pairs"]])).mean()))
        print(f"  reanchor_null p {rn['p']:.3f} (null mean {rn['null_mean_pct']:+.3f}%, "
              f"p95 {rn['null_p95_pct']:+.3f}%)  shifts {out['shifts']}")
    # day-level: joint vs complement
    dj = idx[joint.values].intersection(ret.dropna().index)
    dc = idx[(wash & ~tlt_hit).reindex(idx, fill_value=False).values].intersection(ret.dropna().index)
    show([summarize(ret.loc[dj].values, "day-level joint"),
          summarize(ret.loc[dc].values, "day-level washout & TLT>=25 (discarded)")])

# ------------------------------------------------------------------ E
print("\n" + "=" * 78, "\nE. DEBT 2: ten-sector reference class, same form (sector r21<=5 & TLT r21<25)\n" + "=" * 78)
for h in (3, 5):
    rows, means, ses, names = [], [], [], []
    for t in SECT:
        m = (rk[(t, 21)] <= 5) & tlt_hit
        r, v, e = ep(m, h, legs=((t, 1.0),), lbl=t)
        if r.get("n", 0) >= 3:
            means.append(r["edge_pp"])
            ses.append(r["sd_pct"] / np.sqrt(r["n"]))
            names.append(t)
        rows.append(r)
    show(rows, f"h={h}")
    means, ses = np.array(means), np.array(ses)
    w = 1 / ses ** 2
    pooled = (w * means).sum() / w.sum()
    Q = float((w * (means - pooled) ** 2).sum())
    dfree = len(means) - 1
    I2 = max(0.0, (Q - dfree) / Q) if Q > 0 else 0.0
    tt = means / ses
    xi = names.index("XLU")
    print(f"  Cochran Q {Q:.2f} on {dfree} df, p {1-chi2.cdf(Q, dfree):.3f}, I2 {100*I2:.0f}%; "
          f"fixed-effect common excess {pooled:+.3f}pp (t {pooled/np.sqrt(1/w.sum()):+.2f}); "
          f"XLU edge {means[xi]:+.3f}pp rank {1+int((means > means[xi]).sum())} of {len(means)}, "
          f"t {tt[xi]:+.2f} rank-by-t {1+int((tt > tt[xi]).sum())} of {len(tt)}; "
          f"ex-XLU mean edge {np.mean(np.delete(means, xi)):+.3f}pp")
print("\n  rate-sensitive family, own r21<=5 & TLT r21<25:")
for h in (3, 5):
    rows = []
    for t in FAM:
        m = (rk[(t, 21)] <= 5) & tlt_hit
        rows.append(ep(m, h, legs=((t, 1.0),), lbl=t)[0])
    show(rows, f"family h={h}")

# ------------------------------------------------------------------ F
print("\n" + "=" * 78, "\nF. TLT-hit vs '^TNX at/near its 252 high' (re-anchor to a yield high?)\n" + "=" * 78)
near_hi = tnx_gap_bp <= 5
for h in (3, 5):
    rows = []
    for lbl, m in [("joint (pre-reg)", joint),
                   ("washout & TNX within 5bp of 252 max", wash & near_hi),
                   ("washout & TNX within 15bp of 252 max", wash & (tnx_gap_bp <= 15)),
                   ("joint & TNX within 5bp of max  <-- TODAY", joint & near_hi),
                   ("joint & TNX > 15bp below max", joint & (tnx_gap_bp > 15)),
                   ("washout & TLT<25 & NOT near TNX high", wash & tlt_hit & ~near_hi)]:
        rows.append(ep(m, h, lbl=lbl)[0])
    show(rows, f"h={h}")
jd = idx[joint.values]
print(f"  joint days with TNX within 5bp of its 252 max: {int(near_hi.reindex(jd).sum())} of {len(jd)}; "
      f"today gap {tnx_gap_bp.iloc[-1]:.1f} bp")
r, v, e = ep(joint & near_hi, 5, lbl="x")
print("  joint & TNX-near-high episodes h=5:", [(str(a.date()), round(100 * b, 2)) for a, b in zip(e, v)])

# ------------------------------------------------------------------ G
print("\n" + "=" * 78, "\nG. DEBT 1: forward sample since the arm (2026-08-25)\n" + "=" * 78)
r, v, e = ep(joint, 5, lbl="x")
for a, b in zip(e, v):
    if a >= pd.Timestamp("2025-01-01"):
        print(f"  episode {a.date()}  h=5 {100*b:+.2f}%  (in-sample: the arm was written 2026-08-25)")
post = idx[joint.values & (idx > "2026-08-25")]
print(f"  joint days after 2026-08-25: {[str(x.date()) for x in post]}  -> forward sample N=0")
