"""Episode-clustered check on the high-fragility claim (claim 4)."""
import numpy as np
import pandas as pd
from scipy import stats as sps

UNIV = ["SPY", "QQQ", "IWM", "EFA", "EEM", "FXI", "VNQ", "TLT", "IEF", "LQD", "HYG",
        "GLD", "SLV", "DBC", "USO", "UUP"]

# reuse the saved series from my own run? No - recompute via the verify script's logic
# is heavy; instead load their saved parquet ONLY to cross-check identity vs my series.
import importlib.util
spec = importlib.util.spec_from_file_location(
    "vtf", "scratch/ultracode_research/verify_trend-following.py")

# Simpler: rerun the primary net series by exec-ing the module up to `prim`
src = open("scratch/ultracode_research/verify_trend-following.py").read()
head = src.split('print("=" * 100)')[0]
ns: dict = {}
exec(head, ns)
prim = ns["run"](ns["sig_combo"], ns["UNIV"])
net = prim["net"]

frag = pd.read_parquet("data/rd2_fragility.parquet")["63d"].rolling(10, min_periods=1).mean()
fm = frag.resample("ME").mean().loc["2016-07-31":]
hi = fm >= 50
s16 = net.reindex(fm.index)

# group consecutive high months into episodes
grp = (hi != hi.shift()).cumsum()
ep_means = []
for g, idx in fm.index.to_series().groupby(grp):
    if hi.loc[idx].all() and len(idx) > 0:
        ep_means.append(s16.loc[idx].mean())
print(f"episodes: {len(ep_means)}, per-episode sleeve mean%/mo: "
      f"{[round(x*100,2) for x in ep_means]}")
other = s16[~hi].dropna()
tt = sps.ttest_ind(np.array(ep_means), other.values, equal_var=False)
print(f"episode-level Welch t={tt.statistic:+.2f} p={tt.pvalue:.3f} "
      f"(episode mean {np.mean(ep_means)*100:+.2f}%, other months {other.mean()*100:+.2f}%)")
print(f"episodes negative: {sum(1 for x in ep_means if x < 0)}/{len(ep_means)}")
