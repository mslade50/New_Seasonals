"""Shared helpers for today's k2_* checker scripts (cell stats + reference-class null)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)


def near_high(s: pd.Series, tol: float = 0.02, n: int = 252) -> pd.Series:
    v = s.dropna()
    m = v.rolling(n, min_periods=n).max()
    return ((v / m - 1) >= -tol).reindex(s.index, fill_value=False)


def cell(px: pd.DataFrame, mask: pd.Series, legs, h: int, lag: int = 1,
         min_gap: int | None = None, since=None) -> dict:
    """Episode-level excess vs the vehicle's own all-days drift (same span)."""
    ret = vehicle_ret(px, legs, h, lag)
    valid = ret.notna()
    if since is not None:
        valid &= px.index >= pd.Timestamp(since)
    sig = px.index[mask.reindex(px.index, fill_value=False).values & valid.values]
    if len(sig) == 0:
        return {"n": 0}
    epi = declusters(sig, min_gap or h, px.index)
    ep = ret.loc[epi].values
    span = (px.index >= sig[0]) & valid.values
    drift = ret[span].mean()
    ex = ep - drift
    w = int((ep > 0).sum())
    t = ex.mean() / (ep.std(ddof=1) / np.sqrt(len(ep))) if len(ep) > 2 else np.nan
    return {"n": len(ep), "n_days": len(sig), "mean_pct": 100 * ep.mean(),
            "drift_pct": 100 * drift, "excess_pp": 100 * ex.mean(), "t": t,
            "rec": f"{w}-{len(ep) - w}", "sign_p": sign_test(w, len(ep)),
            "worst_pct": 100 * ep.min(), "epi": epi, "ep": ep, "ex": 100 * ex}


def fmt(c: dict, label: str) -> str:
    if not c.get("n"):
        return f"{label:44s} n=0"
    return (f"{label:44s} n={c['n']:3d} days={c['n_days']:4d} mean {c['mean_pct']:+.3f}% "
            f"drift {c['drift_pct']:+.3f}% ex {c['excess_pp']:+.3f}pp t {c['t']:+.2f} "
            f"rec {c['rec']} p {c['sign_p']:.4f} worst {c['worst_pct']:+.2f}%")


def conc(c: dict) -> str:
    if not c.get("n"):
        return ""
    ex = np.sort(c["ex"])[::-1]
    tot = ex.sum()
    top2 = ex[:2].sum() / tot if tot > 0 else np.nan
    dropbest = ex[1:].mean() if len(ex) > 1 else np.nan
    return f"top-2 share of excess {100*top2:.0f}%  drop-best mean ex {dropbest:+.3f}pp"


def null_maxk(book: dict, focus: str, label: str, nb: int = 10000) -> float:
    names = [c for c in book if len(book[c]) > 1]
    obs = float(np.mean(book[focus]))
    means = {c: float(np.mean(book[c])) for c in names}
    cm = float(np.mean(list(means.values())))
    cen = {c: np.asarray(book[c]) - means[c] + cm for c in names}
    ns = {c: len(book[c]) for c in names}
    mx = np.empty(nb)
    for i in range(nb):
        mx[i] = max(rng.choice(cen[c], size=ns[c], replace=True).mean() for c in names)
    p = float((mx >= obs).mean())
    rank = 1 + sum(1 for c in names if means[c] > obs)
    ses = {c: np.std(book[c], ddof=1) / np.sqrt(len(book[c])) for c in names}
    wts = {c: 1 / ses[c] ** 2 for c in names}
    fe = sum(wts[c] * means[c] for c in names) / sum(wts.values())
    Q = sum(wts[c] * (means[c] - fe) ** 2 for c in names)
    df = len(names) - 1
    I2 = max(0.0, 100 * (Q - df) / Q) if Q > 0 else 0.0
    print(f"--- max-of-K null on {label} (K={len(names)}) ---")
    print("  per-name excess: " + ", ".join(f"{c} {means[c]:+.2f}({ns[c]})" for c in
                                          sorted(names, key=lambda k: -means[k])))
    print(f"  positive {sum(1 for c in names if means[c] > 0)}/{len(names)}; "
          f"{focus} {obs:+.3f}pp rank {rank}/{len(names)}; null max median "
          f"{np.median(mx):+.3f} 95th {np.percentile(mx, 95):+.3f}; P = {p:.4f}")
    print(f"  fixed-effect common excess {fe:+.3f}pp; Cochran Q {Q:.2f} on {df} df; "
          f"I2 {I2:.1f}%")
    return p
