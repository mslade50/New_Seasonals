"""Supplementary checks for the proxy-falsification study."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
nb = pd.read_parquet(ROOT / "scratch" / "ultracode_research" / "proxy_joined.parquet")
PROXIES = ["vix", "vix_ma10", "dist200", "rv21_pct", "dd252", "days_since_5dd"]
frag_hi = nb.frag >= 50
frac_hi = frag_hi.mean()

DIRS = {p: np.sign(stats.spearmanr(nb.frag, nb[p])[0]) for p in PROXIES}
masks = {}
for p in PROXIES:
    v = nb[p] * DIRS[p]
    masks[p] = v >= v.quantile(1 - frac_hi)


def clustered_t(hi_df, lo_df):
    hm = hi_df.groupby("ym")["R_Multiple"].mean()
    lm = lo_df.groupby("ym")["R_Multiple"].mean()
    tt = stats.ttest_ind(hm, lm, equal_var=False)
    return tt.statistic, tt.pvalue, len(hm), len(lm)


# 1. "any proxy high" union as a standalone simple rule
any_hi = np.logical_or.reduce([masks[p].values for p in PROXIES])
any_hi = pd.Series(any_hi, index=nb.index)
tstat, p, mh, ml = clustered_t(nb[any_hi], nb[~any_hi])
print(f"UNION rule (any of 6 proxies in its top 21.3% tail): N_hi={any_hi.sum()}")
print(f"  avgR hi {nb.loc[any_hi,'R_Multiple'].mean():+.3f} vs lo "
      f"{nb.loc[~any_hi,'R_Multiple'].mean():+.3f}  t={tstat:+.2f} p={p:.3f}")

# pairwise OR rules (vix_ma10 | dist200 etc.), matched roughly to 21% coverage
print("\npairwise OR rules (matched-N tails):")
import itertools
for a, b in itertools.combinations(PROXIES, 2):
    m = masks[a] | masks[b]
    tstat, p, _, _ = clustered_t(nb[m], nb[~m])
    print(f"  {a:>14} | {b:<14}: N={m.sum():4d}  "
          f"hi {nb.loc[m,'R_Multiple'].mean():+.3f} lo "
          f"{nb.loc[~m,'R_Multiple'].mean():+.3f}  t={tstat:+.2f} p={p:.3f}")

# 2. LOYO: frag effect within rv21_pct-LO stratum (the big one)
print("\nLOYO: frag>=50 vs <50 WITHIN rv21_pct-LOW stratum:")
sub_all = nb[~masks["rv21_pct"]]
for drop in [None] + sorted(sub_all.yr.unique()):
    sub = sub_all if drop is None else sub_all[sub_all.yr != drop]
    fh = sub.frag >= 50
    if fh.sum() < 5:
        continue
    tstat, p, mh, ml = clustered_t(sub[fh], sub[~fh])
    print(f"  drop {str(drop):>4}: hi {sub.loc[fh,'R_Multiple'].mean():+.3f} "
          f"(N={fh.sum():3d},{mh}mo) lo {sub.loc[~fh,'R_Multiple'].mean():+.3f} "
          f"(N={(~fh).sum():3d})  t={tstat:+.2f} p={p:.3f}")

# 3. worst-cell composition: frag>=50 & rvHI
worst = nb[frag_hi & masks["rv21_pct"]]
print("\nworst cell (frag>=50 & rv21pct-HI) by year:")
print(worst.groupby("yr")["R_Multiple"].agg(["size", "mean"]).round(3).to_string())
print("\nby strategy:")
print(worst.groupby("Strategy")["R_Multiple"].agg(["size", "mean"]).round(3)
      .sort_values("size", ascending=False).to_string())

# 4. rank-corr of frag with R within strata (threshold-free check)
print("\nSpearman(frag, R) within strata:")
for lab, m in [("all", pd.Series(True, index=nb.index)),
               ("rvLO", ~masks["rv21_pct"]), ("rvHI", masks["rv21_pct"]),
               ("anyProxyLO", ~any_hi), ("anyProxyHI", any_hi)]:
    sub = nb[m]
    rho, p = stats.spearmanr(sub.frag, sub.R_Multiple)
    print(f"  {lab:>10}: rho={rho:+.3f} p={p:.3f} N={len(sub)}")

# 5. simple-rule best-shot: optimize VIX-MA10 threshold in-sample (upper bound
# on what a tuned simple rule could do) — deliberately overfit to show ceiling
print("\nin-sample-tuned VIX_MA10 threshold sweep (overfit ceiling):")
for thr in [13, 14, 15, 16, 18, 20, 22, 25]:
    m = nb.vix_ma10 >= thr
    if m.sum() < 30 or (~m).sum() < 30:
        continue
    tstat, p, _, _ = clustered_t(nb[m], nb[~m])
    print(f"  vix_ma10>={thr}: N={m.sum():4d}  hi {nb.loc[m,'R_Multiple'].mean():+.3f} "
          f"lo {nb.loc[~m,'R_Multiple'].mean():+.3f}  t={tstat:+.2f} p={p:.3f}")
print("\nin-sample-tuned dist200 sweep (below-SMA rule):")
for thr in [0.0, -0.02, -0.05]:
    m = nb.dist200 <= thr
    if m.sum() < 30:
        continue
    tstat, p, _, _ = clustered_t(nb[m], nb[~m])
    print(f"  dist200<={thr:+.2f}: N={m.sum():4d}  hi {nb.loc[m,'R_Multiple'].mean():+.3f} "
          f"lo {nb.loc[~m,'R_Multiple'].mean():+.3f}  t={tstat:+.2f} p={p:.3f}")
