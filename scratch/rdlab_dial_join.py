"""rdlab: join ledger with 63d fragility dial 10d-MA. PIT caveat: rd2_fragility.parquet
is append-only PIT from 2026-07-02; earlier history is a single recompute vintage."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

trades = pd.read_parquet(ROOT / "data" / "backtest_trades_full.parquet")
frag = pd.read_parquet(ROOT / "data" / "rd2_fragility.parquet")

frag = frag.sort_index()
frag["dial63_ma10"] = frag["63d"].rolling(10, min_periods=5).mean()
ma = frag["dial63_ma10"].dropna()
print(f"dial series: {ma.index.min().date()} .. {ma.index.max().date()}, n={len(ma)}")

trades["Signal Date"] = pd.to_datetime(trades["Signal Date"])
t = trades[trades["Signal Date"] >= ma.index.min()].copy()
t["dial"] = t["Signal Date"].map(ma)
# fill weekend/holiday signal dates with last available dial
t["dial"] = t["dial"].fillna(t["Signal Date"].map(lambda d: ma.asof(d)))
t = t.dropna(subset=["dial", "R_Multiple"])
print(f"trades in dial window: {len(t)} of {len(trades)} total "
      f"({t['Signal Date'].min().date()} .. {t['Signal Date'].max().date()})")

# ---------- (1) book avgR by dial decile ----------
# deciles of the DIAL DISTRIBUTION over trading days (not trade-weighted), so
# deciles mean "how unusual is the reading", then bucket trades into them.
edges = np.percentile(ma.values, np.arange(0, 101, 10))
edges[0], edges[-1] = -1, 101
t["decile"] = pd.cut(t["dial"], bins=edges, labels=False)
print("\n(1) Book avgR by dial decile (deciles of daily dial dist; edges shown)")
g = t.groupby("decile").agg(n=("R_Multiple", "size"), avgR=("R_Multiple", "mean"),
                            medR=("R_Multiple", "median"), totR=("R_Multiple", "sum"),
                            win=("R_Multiple", lambda s: (s > 0).mean()))
g["dial_lo"] = [f"{edges[int(i)]:.1f}" for i in g.index]
g["dial_hi"] = [f"{edges[int(i)+1]:.1f}" for i in g.index]
print(g.round(3).to_string())

# ---------- (2) per-strategy above/below 50 ----------
print("\n(2) Per-strategy avgR: dial < 50 vs >= 50")
rows = []
for strat, sub in t.groupby("Strategy"):
    lo, hi = sub[sub["dial"] < 50], sub[sub["dial"] >= 50]
    rows.append({"Strategy": strat, "n_lo": len(lo), "avgR_lo": lo["R_Multiple"].mean(),
                 "n_hi": len(hi), "avgR_hi": hi["R_Multiple"].mean(),
                 "totR_hi": hi["R_Multiple"].sum(),
                 "diff": (hi["R_Multiple"].mean() - lo["R_Multiple"].mean())
                 if len(hi) and len(lo) else np.nan})
df2 = pd.DataFrame(rows).sort_values("diff")
print(df2.round(3).to_string(index=False))

# ---------- (3) fraction of book R from dial >= 50 days ----------
tot = t["R_Multiple"].sum()
hi_r = t.loc[t["dial"] >= 50, "R_Multiple"].sum()
n_hi = (t["dial"] >= 50).sum()
day_frac = (ma >= 50).mean()
print(f"\n(3) Book R split: total {tot:.1f}R over {len(t)} trades")
print(f"    dial>=50: {hi_r:.1f}R ({100*hi_r/tot:.1f}% of R) from {n_hi} trades "
      f"({100*n_hi/len(t):.1f}% of trades); dial>=50 on {100*day_frac:.1f}% of days")
print(f"    dial<50 : {tot-hi_r:.1f}R from {len(t)-n_hi} trades")
# also flat-dollar basis
totd = t["PnL_flat_750k"].sum()
hid = t.loc[t["dial"] >= 50, "PnL_flat_750k"].sum()
print(f"    flat-$ basis: total ${totd:,.0f}; dial>=50 ${hid:,.0f} ({100*hid/totd:.1f}%)")

# ---------- (4) SPY forward returns by dial decile ----------
spy = pd.read_parquet(ROOT / "data" / "master_prices.parquet",
                      filters=[("ticker", "==", "SPY")])
spy = spy.set_index("date").sort_index()["Close"]
spy.index = pd.to_datetime(spy.index)
fwd21 = spy.shift(-21) / spy - 1
fwd63 = spy.shift(-63) / spy - 1
d = pd.DataFrame({"dial": ma}).join(pd.DataFrame({"f21": fwd21, "f63": fwd63}), how="inner")
d["decile"] = pd.cut(d["dial"], bins=edges, labels=False)
print("\n(4) SPY forward returns by dial decile (daily obs, overlapping — Ns inflated)")
g4 = d.groupby("decile").agg(n=("f21", "count"),
                             f21_mean=("f21", "mean"), f21_med=("f21", "median"),
                             f21_neg=("f21", lambda s: (s < 0).mean()),
                             f63_mean=("f63", "mean"), f63_med=("f63", "median"),
                             f63_neg=("f63", lambda s: (s < 0).mean()))
print((g4 * [1, 100, 100, 100, 100, 100, 100]).round(2).to_string())
print(f"baseline: f21 {100*d['f21'].mean():.2f}% f63 {100*d['f63'].mean():.2f}%  "
      f"(n={d['f21'].count()} overlapping days ~ {d['f21'].count()//21} indep 21d blocks)")
# above/below 50 split for the timing question
for lab, m in [("dial<50", d["dial"] < 50), ("dial>=50", d["dial"] >= 50)]:
    s = d[m]
    print(f"  {lab}: n={len(s)}, f21 {100*s['f21'].mean():+.2f}% (med {100*s['f21'].median():+.2f}%), "
          f"f63 {100*s['f63'].mean():+.2f}% (med {100*s['f63'].median():+.2f}%)")
