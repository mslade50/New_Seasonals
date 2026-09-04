"""Decompose the ramp: boost (<25) vs throttle (>=50); robustness checks."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
trades = pd.read_parquet(ROOT / "data" / "backtest_trades_full.parquet")
frag = pd.read_parquet(ROOT / "data" / "rd2_fragility.parquet")

frag_ma = frag["63d"].dropna().rolling(10, min_periods=1).mean()
frag_ma.index = pd.to_datetime(frag_ma.index).normalize()

trades["Signal Date"] = pd.to_datetime(trades["Signal Date"]).dt.normalize()
t = trades[trades["Signal Date"] >= frag_ma.index.min() + pd.Timedelta(days=20)].copy()
t = t.sort_values("Signal Date")
t["frag"] = pd.merge_asof(
    t[["Signal Date"]], frag_ma.rename("frag").reset_index(),
    left_on="Signal Date", right_on="Date",
)["frag"].values
t = t.dropna(subset=["frag", "R_Multiple"])
nb = t[~t["Strategy"].str.contains("Overbot Vol|OVS", case=False, na=False)].copy()


def ramp_current(f):
    if f <= 25:
        return 1.25 - (f / 25) * 0.25
    return max(0.10, 1.0 - ((f - 25) / 75) * 0.90)


def ramp_no_boost(f):  # cap at 1.0, same throttle
    return min(1.0, ramp_current(f))


def ramp_step50(f):  # flat 1.0, halve at >=50
    return 0.5 if f >= 50 else 1.0


def ramp_step50_deep(f):  # flat 1.0, quarter at >=50
    return 0.25 if f >= 50 else 1.0


print(f"non-OVS N={len(nb)}")
for name, fn in [("current (boost+throttle)", ramp_current),
                 ("no-boost, same throttle", ramp_no_boost),
                 ("step: 0.5x at frag>=50", ramp_step50),
                 ("step: 0.25x at frag>=50", ramp_step50_deep)]:
    m = nb.frag.map(fn)
    radj = nb.R_Multiple * m
    s = nb.assign(x=radj).sort_values("Exit Date").groupby("Exit Date")["x"].sum().cumsum()
    dd = (s - s.cummax()).min()
    print(f"{name:28s} totR {radj.sum():+7.1f}  avgR/unit {radj.sum()/m.sum():+.4f}  "
          f"avg mult {m.mean():.2f}  worstDD {dd:+.1f}")

s0 = nb.sort_values("Exit Date").groupby("Exit Date")["R_Multiple"].sum().cumsum()
print(f"{'baseline 1.0x':28s} totR {nb.R_Multiple.sum():+7.1f}  avgR/unit {nb.R_Multiple.mean():+.4f}  "
      f"avg mult 1.00  worstDD {(s0 - s0.cummax()).min():+.1f}")

# throttle robustness: drop 2021 then 2024 (the two big-N frag>=50 years)
nb["yr"] = nb["Signal Date"].dt.year
nb["ym"] = nb["Signal Date"].dt.to_period("M")
print("\nleave-one-year-out, avgR frag>=50 vs <50 (non-OVS):")
for drop in [None, 2021, 2024, 2026]:
    g = nb if drop is None else nb[nb.yr != drop]
    hi, lo = g[g.frag >= 50], g[g.frag < 50]
    hi_m = hi.groupby("ym")["R_Multiple"].mean()
    lo_m = lo.groupby("ym")["R_Multiple"].mean()
    tt = stats.ttest_ind(hi_m, lo_m, equal_var=False)
    print(f"  drop {str(drop):5s}: >=50 {hi.R_Multiple.mean():+.3f} (N={len(hi)})  "
          f"<50 {lo.R_Multiple.mean():+.3f} (N={len(lo)})  monthly-t={tt.statistic:+.2f} p={tt.pvalue:.3f}")

# per-strategy breakdown in frag>=50 (is one strategy driving it?)
print("\nfrag>=50 by strategy (non-OVS, N>=10):")
hi = nb[nb.frag >= 50]
g = hi.groupby("Strategy")["R_Multiple"].agg(["size", "mean", "sum"])
g = g[g["size"] >= 10].sort_values("sum")
lo_means = nb[nb.frag < 50].groupby("Strategy")["R_Multiple"].mean()
g["avgR_lo50"] = lo_means.reindex(g.index)
print(g.round(3).to_string())
