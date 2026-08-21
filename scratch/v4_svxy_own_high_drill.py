"""V4 follow-up (2026-08-21): the entry condition McKinley actually observed
is SVXY ITSELF near a 52-week high (and gapping up) — vol already crushed.
Is the post-opex window still +EV when the short-vol instrument is at its
own highs / VIX is already low?

Same window basis as the prereg grid (synthetic -0.5x legs from UVXY,
opex close -> +3 close, ex-Sep, 2011-10+). Conditioning, both lag-1:
  - the synthetic short-vol cumulative index's distance from its own
    252d max (the consistent-basis version of "SVXY at a 52w high")
  - ^VIX level (the economic variable underneath)
Current state reported from REAL SVXY bars (the cache; -1x pre-2018-02,
so the synthetic index is the conditioning series, not real SVXY).
"""
from __future__ import annotations

import sys
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from macro_calendar import event_dates  # noqa: E402


def load(tkr: str, cols=("Open", "Close")) -> pd.DataFrame:
    mp = pd.read_parquet(ROOT / "data" / "master_prices.parquet",
                         columns=["ticker", "date", *cols])
    df = mp[mp["ticker"] == tkr].set_index("date").sort_index()
    df.index = pd.to_datetime(df.index).normalize()
    return df[~df.index.duplicated(keep="last")][list(cols)]


u = load("UVXY")
ovn = -(u["Open"] / u["Close"].shift(1) - 1) / 3.0
intra = -(u["Close"] / u["Open"] - 1) / 3.0
idx = u.index

# Synthetic -0.5x cumulative index and its 52w-high distance.
daily = (1 + ovn.fillna(0)) * (1 + intra.fillna(0))
synth = daily.cumprod()
dist_own = (synth / synth.rolling(252).max() - 1) * 100.0

vix = load("^VIX", cols=("Close",))["Close"]


def window_ret(p: int, exit_k: int = 3) -> float:
    hi = p + exit_k
    if hi >= len(idx) or p < 1:
        return np.nan
    total = 1.0
    for i in range(p + 1, hi + 1):
        total *= (1 + ovn.iloc[i]) * (1 + intra.iloc[i])
    return total - 1


def sign_p(wins: int, n: int) -> float:
    return sum(comb(n, k) for k in range(wins, n + 1)) / 2 ** n


def cell(name: str, sub: pd.DataFrame) -> None:
    x = sub["ret"].dropna()
    if not len(x):
        print(f"{name:40s}  n=0")
        return
    n, mean, hit = len(x), x.mean() * 1e4, (x > 0).mean()
    t = (x.mean() / (x.std(ddof=1) / np.sqrt(n))) if n > 2 else np.nan
    p = sign_p(int((x > 0).sum()), n)
    print(f"{name:40s}  n={n:3d}  avg {mean:+7.1f} bps  t {t:5.2f}  "
          f"hit {hit:.0%}  sign-p {p:.3f}  worst {x.min()*100:+.1f}%")


rows = []
for d in event_dates("opex"):
    d = pd.Timestamp(d).normalize()
    if d.month == 9 or d not in idx.to_series().index:
        continue
    p = idx.get_loc(d)
    lag = idx[idx < d]
    own = dist_own.reindex(lag).iloc[-1] if len(lag) else np.nan
    vlag = vix.index[vix.index < d]
    v = vix.reindex(vlag).iloc[-1] if len(vlag) else np.nan
    eday = daily.iloc[p] - 1 if p < len(daily) else np.nan  # opex-day move
    rows.append({"date": d, "ret": window_ret(p), "own": own, "vix": v,
                 "eday": eday})
df = pd.DataFrame(rows).dropna(subset=["ret"])
print(f"windows: {len(df)}   own-high dist quartiles: "
      f"{np.nanpercentile(df['own'], [25, 50, 75]).round(1)}   "
      f"VIX quartiles: {np.nanpercentile(df['vix'], [25, 50, 75]).round(1)}")
print()
cell("ALL", df)
print("\nBy the short-vol index's own 252d-high distance (lag-1):")
cell("  at own high: > -1%", df[df["own"] > -1])
cell("  -5% .. -1%", df[(df["own"] > -5) & (df["own"] <= -1)])
cell("  -20% .. -5%", df[(df["own"] > -20) & (df["own"] <= -5)])
cell("  deeper than -20%", df[df["own"] <= -20])
print("\nBy lag-1 VIX level:")
cell("  VIX < 14", df[df["vix"] < 14])
cell("  14 <= VIX < 18", df[(df["vix"] >= 14) & (df["vix"] < 18)])
cell("  18 <= VIX < 25", df[(df["vix"] >= 18) & (df["vix"] < 25)])
cell("  VIX >= 25", df[df["vix"] >= 25])
print("\nToday's joint state:")
cell("  own > -2% (at high)", df[df["own"] > -2])
cell("  own > -2% AND 2018+", df[(df["own"] > -2) & (df["date"] >= "2018-03-01")])
last_own = dist_own.dropna()
last_v = vix.dropna()
cell(f"  own > -2% AND VIX band of today",
     df[(df["own"] > -2) & (abs(df["vix"] - last_v.iloc[-1]) <= 3)])
print("\nBought AFTER an up move (the entry MOC fills on the elevated "
      "close — 'gapped up into the fill'):")
cell("  opex day up for short-vol", df[df["eday"] > 0])
cell("  opex day strong (> +1%)", df[df["eday"] > 0.01])
cell("  strong entry day AND own > -2%",
     df[(df["eday"] > 0.01) & (df["own"] > -2)])
cell("  opex day DOWN (control)", df[df["eday"] <= 0])

sv = load("SVXY", cols=("Close",))["Close"]
sv_dist = (sv / sv.rolling(252).max() - 1) * 100.0
print(f"\nCurrent (lag-1 = {last_own.index[-1].date()}):")
print(f"  synthetic index vs own 252d high: {last_own.iloc[-1]:+.2f}%")
print(f"  REAL SVXY close vs own 252d high: {sv_dist.dropna().iloc[-1]:+.2f}%"
      f"  (close {sv.iloc[-1]:.2f})")
print(f"  VIX: {last_v.iloc[-1]:.2f}")
