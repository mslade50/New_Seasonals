"""OVS hold-extension, v2: trigger only when losing by MORE than 0.5 ATR at
the T+2 close (R < -0.5 -- stop_atr is 1.0 so R units are ATR units), plus a
threshold sweep. Same replay mechanics as scratch/ovs_hold_extension_study.py."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

ledger = pd.read_parquet(ROOT / "data" / "backtest_trades_full.parquet")
ovs = ledger[ledger["Strategy"] == "Overbot Vol Spike"].copy()
for c in ("Signal Date", "Entry Date", "Exit Date"):
    ovs[c] = pd.to_datetime(ovs[c])

prices = pd.read_parquet(ROOT / "data" / "master_prices.parquet")
prices["date"] = pd.to_datetime(prices["date"])
prices = prices[prices["ticker"].isin(ovs["Ticker"].unique())]
px = {t: g.set_index("date").sort_index() for t, g in prices.groupby("ticker")}

rank252 = {}
for t, df in px.items():
    ret = df["Close"].pct_change(252, fill_method=None)
    rank252[t] = ret.expanding(min_periods=252).rank(pct=True) * 100.0
ovs["rank252"] = [
    rank252[t].loc[d] if (t in rank252 and d in rank252[t].index) else np.nan
    for t, d in zip(ovs["Ticker"], ovs["Signal Date"])
]
ovs["atr_pct"] = ovs["ATR"] / ovs["Signal Close"] * 100.0

# Replay the extension once for EVERY losing time exit; thresholds are then
# just masks over the precomputed deltas.
time_losers = (ovs["Exit Type"] == "Time") & (ovs["R_Multiple"] < 0)
delta = pd.Series(np.nan, index=ovs.index)
hit_target = pd.Series(False, index=ovs.index)
for idx, row in ovs[time_losers].iterrows():
    df = px.get(row["Ticker"])
    if df is None or row["Exit Date"] not in df.index:
        continue
    pos = df.index.get_loc(row["Exit Date"])
    cache_t2 = df["Close"].iloc[pos]
    if abs(cache_t2 / row["Exit Price"] - 1) > 0.01:
        continue
    ext = df.iloc[pos + 1 : pos + 4]
    if len(ext) < 3:
        continue
    tgt = row["Entry Price"] - row["tgt_atr"] * row["ATR"]
    new_exit = ext["Close"].iloc[-1]
    for _, day in ext.iterrows():
        if day["Low"] <= tgt:
            new_exit = tgt
            hit_target.loc[idx] = True
            break
    delta.loc[idx] = (cache_t2 - new_exit) / row["ATR"]

ovs["week"] = ovs["Signal Date"].dt.to_period("W")
ovs["year"] = ovs["Signal Date"].dt.year


def cluster_t(d: pd.Series, weeks: pd.Series) -> tuple[float, int]:
    cl = d.groupby(weeks).sum()
    n = len(cl)
    if n < 3:
        return np.nan, n
    return cl.mean() / (cl.std(ddof=1) / np.sqrt(n)), n


def loyo(d: pd.Series, weeks: pd.Series, years: pd.Series) -> float:
    worst = np.inf
    for y in years.unique():
        keep = years != y
        t, _ = cluster_t(d[keep], weeks[keep])
        worst = min(worst, t)
    return worst


def pf(r: pd.Series) -> float:
    g, l = r[r > 0].sum(), -r[r < 0].sum()
    return g / l if l > 0 else np.inf


def report(name: str, apply: pd.Series) -> None:
    d = delta[apply]
    w, y = ovs.loc[apply, "week"], ovs.loc[apply, "year"]
    new = ovs["R_Multiple"] + delta.where(apply, 0.0).fillna(0.0)
    base = ovs["R_Multiple"]
    t, nw = cluster_t(d, w)
    yr_sum = d.groupby(y).sum()
    bad = yr_sum[yr_sum < -1.0]
    print(f"\n=== {name} ===")
    print(f"  extensions: {apply.sum()}  ({hit_target[apply].sum()} hit target)"
          f"  deltaR sum {d.sum():+.1f}  mean {d.mean():+.3f}"
          f"  improved {(d > 0).mean() * 100:.0f}%  p5 {d.quantile(0.05):+.2f}  min {d.min():+.2f}")
    print(f"  t(week-clustered) {t:+.2f} over {nw} weeks  |  LOYO floor {loyo(d, w, y):+.2f}")
    print(f"  book totR {base.sum():+.1f} -> {new.sum():+.1f}  avgR {base.mean():+.3f} -> {new.mean():+.3f}"
          f"  win% {(base > 0).mean() * 100:.1f} -> {(new > 0).mean() * 100:.1f}"
          f"  PF {pf(base):.2f} -> {pf(new):.2f}  worst {base.min():.2f} -> {new.min():.2f}")
    print("  years < -1R: " + (", ".join(f"{yy}: {v:+.1f}R" for yy, v in bad.items()) if len(bad) else "none"))


replayed = delta.notna()

print("Threshold sweep (extend when R at T+2 < thresh), ALL names then ATR%>=3:")
print(f"{'thresh':>7} | {'n':>4} {'sumR':>7} {'t':>6} | {'n':>4} {'sumR':>7} {'t':>6}")
for th in [0.0, -0.25, -0.5, -0.75, -1.0]:
    m_all = replayed & (ovs["R_Multiple"] < th)
    m_hv = m_all & (ovs["atr_pct"] >= 3)
    t_all, _ = cluster_t(delta[m_all], ovs.loc[m_all, "week"])
    t_hv, _ = cluster_t(delta[m_hv], ovs.loc[m_hv, "week"])
    print(f"{th:>7} | {m_all.sum():>4} {delta[m_all].sum():>+7.1f} {t_all:>+6.2f}"
          f" | {m_hv.sum():>4} {delta[m_hv].sum():>+7.1f} {t_hv:>+6.2f}")

base_mask = replayed & (ovs["R_Multiple"] < -0.5)
report("R < -0.5: ALL", base_mask)
report("R < -0.5: 252d rank < 65", base_mask & (ovs["rank252"] < 65))
report("R < -0.5: ATR% < 3", base_mask & (ovs["atr_pct"] < 3))
report("R < -0.5: ATR% >= 3", base_mask & (ovs["atr_pct"] >= 3))
