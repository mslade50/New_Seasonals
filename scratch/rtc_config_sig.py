"""Significance addendum: episode-level t vs baseline for fwd63 mean, and
exact binomial p for P(dd>=10% in 63td) vs the unconditional day-level rate."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
cfg = pd.read_parquet(ROOT / "scratch" / "rtc_config_history.parquet").sort_index()
spy = cfg["spy_close"].astype(float)
n = len(cfg)
dates = cfg.index

r63 = spy.shift(-63) / spy - 1
mdd63 = pd.Series(
    [spy.iloc[i + 1:i + 64].min() / spy.iloc[i] - 1 if i + 63 < n else np.nan
     for i in range(n)], index=dates)
base_mu, base_sd = r63.mean(), r63.std()
base_p10 = (mdd63 <= -0.10).mean()
base_p5 = (mdd63 <= -0.05).mean()

stats_json = json.loads((ROOT / "scratch" / "rtc_config_stats.json").read_text())
print(f"baseline fwd63 mu={base_mu*100:.2f}% sd={base_sd*100:.2f}% "
      f"P(dd5)={base_p5:.3f} P(dd10)={base_p10:.3f}")
print(f"{'class':<16}{'N63':>4}{'t_fwd63':>9}{'binom_p_dd10':>14}{'binom_p_dd5':>13}")
for name, d in stats_json["classes"].items():
    if "episode_dates" not in d:
        continue
    eps = pd.to_datetime(d["episode_dates"])
    rr = r63.loc[eps].dropna()
    mm = mdd63.loc[eps].dropna()
    if len(rr) < 5:
        continue
    t = (rr.mean() - base_mu) / (rr.std(ddof=1) / np.sqrt(len(rr)))
    k10 = int((mm <= -0.10).sum())
    k5 = int((mm <= -0.05).sum())
    p10 = stats.binomtest(k10, len(mm), base_p10, alternative="greater").pvalue
    p5 = stats.binomtest(k5, len(mm), base_p5, alternative="greater").pvalue
    print(f"{name:<16}{len(rr):>4}{t:>9.2f}{p10:>14.3f}{p5:>13.3f}")
