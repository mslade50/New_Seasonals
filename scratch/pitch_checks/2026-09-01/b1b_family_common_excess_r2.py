"""B1 round 2 -- the ONE thing that could survive round 1.

Round 1 found a HOMOGENEOUS family (Cochran Q 9.26 on 14 df, p 0.8138,
I-squared 0.0%, cross-sectional sd / mean sampling SE = 1.03) with a
fixed-effect COMMON excess of -0.274pp at z -2.51. No individual class
survives the max-of-15 charge. So the honest object, if there is one, is
"everything is weak into a midterm-year FOMC" -- one common excess wearing
fifteen labels.

That z of -2.51 is NOT usable as stated: inverse-variance weighting assumes
the fifteen classes are independent and they are not (SPY/IWM/XLK/EFA/EEM are
one equity factor). This charges the COMMON excess three ways:

  1. a correlation-preserving permutation null on the family average
  2. the placebo offset ladder ON THE FAMILY AVERAGE (does the common excess
     isolate at the anchor, or is it a slow ramp?)
  3. the era / episode histogram of the family average, plus drop-best-k

If (2) shows a ramp rather than a spike, there is no anchor here and the
whole B1 cross-asset family closes.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

ASOF = pd.Timestamp("2026-08-31")
H = 10
CLASSES = {"us_large": "SPY", "us_small": "IWM", "rates": "TLT",
           "rates_belly": "IEF", "credit": "HYG", "gold": "GLD",
           "miners": "GDX", "metals": "SLV", "energy": "USO",
           "energy_eq": "XLE", "dollar": "UUP", "intl_dev": "EFA",
           "intl_em": "EEM", "vol_inv": "SVXY", "tech": "XLK"}
px = load_prices(sorted(set(CLASSES.values())))
S = {k: px[v]["Close"].dropna()[lambda s: s.index <= ASOF] for k, v in CLASSES.items()}
FR = {k: (S[k].shift(-H) / S[k] - 1.0) for k in S}

ev = load_events(["fomc_decision"])
FOM = pd.DatetimeIndex(sorted(ev["date"].unique()))
FOM = FOM[FOM <= ASOF]
MID = pd.DatetimeIndex([d for d in FOM if d.year % 4 == 2])


def edges(anchors, k):
    """Per-class (edge vs own drift over the anchor span, n)."""
    out = {}
    for c in S:
        s, fr = S[c], FR[c]
        pos, kept = anchor_positions(s.index, anchors, offset=k)
        if len(pos) < 8:
            continue
        vv = fr.iloc[pos].dropna()
        if len(vv) < 8:
            continue
        base = fr.dropna()
        base = base[(base.index >= kept[0]) & (base.index <= kept[-1])]
        out[c] = (float(vv.mean() - base.mean()), len(vv))
    return out


print("=" * 78)
print("1. CORRELATION-PRESERVING PERMUTATION NULL ON THE FAMILY AVERAGE")
print("=" * 78)
obs = edges(MID, -10)
fam_obs = float(np.mean([e for e, _ in obs.values()]))
print("  observed simple-average excess across %d classes = %+.3fpp" % (len(obs), fam_obs))

rng = np.random.default_rng(11)
cal = S["us_large"].index[S["us_large"].index >= pd.Timestamp("2002-08-01")][:-(H + 15)]
NB = 4000
draws = np.zeros(NB)
for b in range(NB):
    dts = pd.DatetimeIndex(rng.choice(cal, size=len(MID), replace=False))
    e = edges(dts, -10)
    draws[b] = np.mean([x for x, _ in e.values()]) if e else np.nan
draws = draws[~np.isnan(draws)]
print("  permutation null on %d random date sets of size %d:" % (len(draws), len(MID)))
print("    null mean %+.3fpp, sd %.3fpp, p05 %+.3fpp, p95 %+.3fpp"
      % (100 * draws.mean(), 100 * draws.std(ddof=1),
         100 * np.percentile(draws, 5), 100 * np.percentile(draws, 95)))
print("    P(perm average <= observed) = %.4f   [two-sided P(|perm| >= |obs|) = %.4f]"
      % ((draws <= fam_obs).mean(), (np.abs(draws) >= abs(fam_obs)).mean()))
print("    the naive inverse-variance z was -2.51; the correlation-aware")
print("    permutation number above is the honest one.")

print("\n" + "=" * 78)
print("2. PLACEBO OFFSET LADDER ON THE FAMILY AVERAGE")
print("   k = entry session relative to the decision; k=-10 is the true anchor")
print("   (the only k whose 10-td hold exits ON the decision close).")
print("=" * 78)
rows = []
for k in range(-20, 1):
    e = edges(MID, k)
    fa = np.mean([x for x, _ in e.values()])
    eq = np.mean([e[c][0] for c in ("us_large", "us_small", "tech", "intl_dev", "intl_em")
                  if c in e])
    rows.append({"k": k, "exit_at": f"dec{k + H:+d}", "family_avg_pp": round(100 * fa, 3),
                 "equity5_avg_pp": round(100 * eq, 3), "n_classes": len(e),
                 "TRUE": "<== anchor" if k == -10 else ""})
show(rows, "family-average excess by entry offset, MIDTERM")
col = pd.DataFrame(rows).set_index("k")["family_avg_pp"]
print("  true anchor |value| ranks %d of %d; most negative k = %d at %+.3fpp"
      % (int((col.abs() >= abs(col.loc[-10])).sum()), len(col),
         int(col.idxmin()), col.min()))
print("  shape: %s" % ("MONOTONE RAMP (no anchor)"
                       if (col.loc[-20:-10].is_monotonic_decreasing
                           or col.loc[-16:-6].is_monotonic_decreasing)
                       else "not monotone -- inspect the table"))

print("\n" + "=" * 78)
print("3. ERA HISTOGRAM AND CONCENTRATION OF THE FAMILY AVERAGE")
print("   (per-decision cross-class average return, midterm only)")
print("=" * 78)
per = {}
for c in S:
    s, fr = S[c], FR[c]
    pos, kept = anchor_positions(s.index, MID, offset=-10)
    vv = fr.iloc[pos]
    vv.index = kept
    per[c] = vv
P = pd.DataFrame(per)
avg = P.mean(axis=1).dropna()
print("  per-decision cross-class average return (n=%d decisions, %d classes each"
      " on average)" % (len(avg), int(P.notna().sum(axis=1).mean())))
by = avg.groupby(avg.index.year)
print(pd.DataFrame({"n": by.size(), "mean_pct": 100 * by.mean(),
                    "sum_pct": 100 * by.sum()}).round(3).to_string())
short = -avg.values   # the traded side of a "everything is weak" view
o = np.argsort(-short)
w = int((short > 0).sum())
print("\n  SHORT-the-family (the traded side): mean %+.3f%%, record %d-%d, sign p %.4f"
      % (100 * short.mean(), w, len(short) - w, sign_test(w, len(short))))
print("    total %+.2fpp | best2 %+.2fpp (%s) = %.0f%% of total"
      % (100 * short.sum(), 100 * short[o[:2]].sum(),
         [str(avg.index[i].date()) for i in o[:2]],
         100 * short[o[:2]].sum() / short.sum()))
print("    mean %+.3f%% | drop-best-2 %+.3f%% | drop-best-3 %+.3f%%"
      % (100 * short.mean(), 100 * short[o[2:]].mean(), 100 * short[o[3:]].mean()))
show(era_split(avg.index, short), "era split of the short-the-family side")
