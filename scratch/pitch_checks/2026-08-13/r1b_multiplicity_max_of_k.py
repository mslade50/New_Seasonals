"""r1b - price the cross-sectional multiplicity honestly.

r1 established the state is NOT generic (pooled excess -0.231pp, 11/27
positive). That leaves the other horn: IHI is the 2nd of 27 tickers tested,
so is +1.211pp distinguishable from the top of 27 noisy estimates around a
common mean?

NOTE r1's max-of-K line was WRONG: it compared a FRACTION-unit resampled mean
against a PERCENT observed excess (the registry's own double-scale trap, 100x).
Recomputed here in percent, three ways:

 A. HETEROGENEITY. Is the cross-sectional sd of the 27 excess estimates any
    larger than their own sampling SE? (Cochran Q, I^2.) If not, the whole
    -2.96..+1.33 spread is noise and there is nothing ticker-specific to find.
 B. MAX-STATISTIC PERMUTATION. Under "the trigger carries no information for
    any ticker", redraw each ticker's episode dates at random from its own
    valid span (same count, same 5td declustering), recompute excess over own
    drift, take the max over the 27. P(max >= IHI's observed).
 C. Same permutation restricted to IHI alone (single-ticker p, no multiplicity)
    so the two can be read side by side.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

H = 5
TK = ["XLV", "XLK", "XLE", "XLF", "XLI", "XLB", "XLP", "XLU", "XLY",
      "SMH", "XBI", "KRE", "IHI", "VNQ", "XOP", "OIH", "GDX", "XME",
      "ITA", "ITB", "IYR", "IYT", "XRT", "XHB", "IBB", "GDXJ", "COPX"]
px_map = load_prices(TK)
TK = [t for t in TK if t in px_map]

st = {}
for t in TK:
    c = px_map[t]["Close"].dropna()
    r21 = pct_rank(c, 21)
    dd = c / c.rolling(252).max() - 1.0
    m = ((r21 >= 99) & (dd <= -0.10)).fillna(False)
    ret = fwd_lag(c, H)
    valid = ret.notna()
    trig = c.index[m.values & valid.values]
    epi = declusters(trig, 5, c.index)
    epi = epi[ret.reindex(epi).notna().values]
    span_mask = (c.index >= trig[0]) & (c.index <= trig[-1]) & valid.values
    span_idx = c.index[span_mask]
    v = ret.loc[epi].values
    st[t] = {"epi": epi, "v": v, "ret": ret, "span_idx": span_idx,
             "ctrl": ret[span_mask].values,
             "excess": 100 * (v.mean() - ret[span_mask].values.mean()),
             "se": 100 * v.std(ddof=1) / np.sqrt(len(v)), "n": len(v)}

ex = np.array([st[t]["excess"] for t in TK])
se = np.array([st[t]["se"] for t in TK])
obs = st["IHI"]["excess"]

print("=== A. HETEROGENEITY: is any of the cross-sectional spread real? ===")
w = 1 / se**2
mu = float((w * ex).sum() / w.sum())
Q = float((w * (ex - mu) ** 2).sum())
k = len(ex)
I2 = max(0.0, (Q - (k - 1)) / Q) * 100 if Q > 0 else 0.0
try:
    from scipy import stats as sps
    pQ = float(sps.chi2.sf(Q, k - 1))
except Exception:  # noqa: BLE001
    pQ = float("nan")
print(f"  K={k} tickers.  observed cross-sectional sd of excess = {ex.std(ddof=1):.3f}pp")
print(f"  mean per-ticker sampling SE                          = {se.mean():.3f}pp")
print(f"  ratio (observed spread / sampling noise)              = "
      f"{ex.std(ddof=1)/se.mean():.2f}   (1.0 = pure noise)")
print(f"  fixed-effect pooled excess mu = {mu:+.3f}pp;  Cochran Q = {Q:.2f} on "
      f"{k-1} df, p = {pQ:.3f};  I^2 = {I2:.1f}%")
print(f"  IHI excess {obs:+.3f}pp = {(obs - mu)/st['IHI']['se']:+.2f} of its OWN SE "
      f"above the common mean")
print(f"  IHI z vs common mean, Sidak-corrected over K={k}: "
      f"p_single -> p_family")
try:
    z = (obs - mu) / st["IHI"]["se"]
    p1 = float(sps.norm.sf(z))
    print(f"    z={z:+.2f}  p_single={p1:.4f}  "
          f"p_family=1-(1-p)^{k} = {1-(1-p1)**k:.4f}")
except Exception:  # noqa: BLE001
    pass

print("\n=== B. MAX-STATISTIC PERMUTATION (random dates, same count/decluster) ===")
rng = np.random.default_rng(2026)
NB = 3000
maxes = np.zeros(NB)
ihi_only = np.zeros(NB)
for b in range(NB):
    best = -1e9
    for t in TK:
        s = st[t]
        idx = s["span_idx"]
        n = s["n"]
        # random declustered draw of n dates from the valid span
        picks = []
        tries = 0
        posmap = np.arange(len(idx))
        while len(picks) < n and tries < 200:
            cand = rng.choice(posmap, size=n * 3, replace=False) if len(posmap) > n * 3 else posmap.copy()
            picks = []
            last = -10**9
            for p in np.sort(cand):
                if p - last >= 5:
                    picks.append(p)
                    last = p
                if len(picks) == n:
                    break
            tries += 1
        if len(picks) < n:
            picks = list(rng.choice(posmap, size=n, replace=False))
        vals = s["ret"].reindex(idx[np.array(picks)]).values
        e = 100 * (np.nanmean(vals) - s["ctrl"].mean())
        if t == "IHI":
            ihi_only[b] = e
        best = max(best, e)
    maxes[b] = best
print(f"  null MAX over {k} tickers: mean {maxes.mean():+.3f}pp, "
      f"p50 {np.percentile(maxes,50):+.3f}, p90 {np.percentile(maxes,90):+.3f}, "
      f"p95 {np.percentile(maxes,95):+.3f}, p99 {np.percentile(maxes,99):+.3f}")
print(f"  P(null max >= IHI's {obs:+.3f}pp) = {(maxes >= obs).mean():.4f}   "
      f"<-- FAMILY-WISE p for the cross-section")
print(f"  null IHI-ALONE: mean {ihi_only.mean():+.3f}pp, p95 "
      f"{np.percentile(ihi_only,95):+.3f};  P(>= obs) = "
      f"{(ihi_only >= obs).mean():.4f}   <-- single-ticker p, no multiplicity")

print("\n=== C. how many of the 27 clear their own single-ticker permutation? ===")
# reuse the ihi_only machinery per ticker (cheaper: normal approx on its own SE)
rows = []
for t in TK:
    s = st[t]
    try:
        p = float(sps.norm.sf(s["excess"] / s["se"]))
    except Exception:  # noqa: BLE001
        p = np.nan
    rows.append({"ticker": t, "n": s["n"], "excess_pp": round(s["excess"], 3),
                 "se_pp": round(s["se"], 3), "z": round(s["excess"] / s["se"], 2),
                 "p_1sided": round(p, 4)})
d = pd.DataFrame(rows).sort_values("z", ascending=False)
print(d.to_string(index=False))
n05 = int((d["p_1sided"] < 0.05).sum())
print(f"  tickers with p<0.05 one-sided: {n05} of {k}  "
      f"(expected under a global null: {0.05*k:.1f})")
print(f"  Sidak on IHI's own p: 1-(1-{d.loc[d.ticker=='IHI','p_1sided'].iloc[0]:.4f})^{k} "
      f"= {1-(1-d.loc[d.ticker=='IHI','p_1sided'].iloc[0])**k:.4f}")
