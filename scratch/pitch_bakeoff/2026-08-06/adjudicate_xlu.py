"""Adjudicate the XLU disagreement between the two bake-off variants.

Fable shipped SHORT XLU / LONG SPY on its cell B (XLU 21d rank < 10, SPY 21d
rank > 50, SPY within 2% of its 252d high) at -111.8 bps on the XLU-SPY
spread, t -3.15, N 25 declustered. Opus killed the family, reporting that the
apparent edge was "one leg's beta plus overlapping windows" -- but Opus was
measuring a DIFFERENT trigger (relative 21d return percentile) which does not
fire today, so its verdict does not directly refute Fable's cell.

This runs Fable's exact trigger and asks the three questions neither run
answered together:

  1. LEG DECOMPOSITION. Is the spread edge the XLU short or the SPY long?
     Each leg is compared to its OWN unconditional drift over the same
     horizon, so "XLU fell" is separated from "XLU fell less than usual".
  2. THE MISSING CONTROL. What does the same long SPY / short XLU trade do
     on every day SPY is near a high with 21d rank > 50, WITHOUT requiring
     the XLU washout? If the washout adds nothing, the trade is a
     long-SPY-momentum bet wearing a spread costume.
  3. BETA. Dollar-neutral long SPY / short XLU is net long roughly half a
     unit of market beta. The beta-neutral version removes that.

Plus era split and a leave-one-year-out floor on the surviving cell.
Everything truncates at 2026-08-05, the last settled bar.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CUTOFF = pd.Timestamp("2026-08-05")
HORIZONS = (5, 10)
GAP = 10  # declustering gap in sessions, same as Fable used

mp = pd.read_parquet(ROOT / "data" / "master_prices.parquet",
                     columns=["ticker", "date", "Close"])


def load(tkr: str) -> pd.Series:
    df = mp[mp["ticker"] == tkr].set_index("date").sort_index()[["Close"]]
    df.index = pd.to_datetime(df.index).normalize()
    return df[~df.index.duplicated(keep="last")].loc[:CUTOFF, "Close"]


px = pd.DataFrame({"xlu": load("XLU"), "spy": load("SPY")}).dropna()
ret1 = px.pct_change()

r21 = px.pct_change(21)
rank_xlu = r21["xlu"].rolling(252).apply(lambda w: (w <= w.iloc[-1]).mean() * 100,
                                         raw=False)
rank_spy = r21["spy"].rolling(252).apply(lambda w: (w <= w.iloc[-1]).mean() * 100,
                                         raw=False)
near_high = (px["spy"] / px["spy"].rolling(252).max() - 1) >= -0.02

# Trailing 252d beta of XLU to SPY, lagged one day so it is knowable at the
# signal close.
cov = ret1["xlu"].rolling(252).cov(ret1["spy"])
var = ret1["spy"].rolling(252).var()
beta = (cov / var).shift(1)

fwd = {h: px.shift(-h) / px - 1 for h in HORIZONS}


def decluster(mask: pd.Series, gap: int = GAP) -> pd.DatetimeIndex:
    keep, last = [], None
    positions = {d: i for i, d in enumerate(mask.index)}
    for day, on in mask.items():
        if on and (last is None or positions[day] - last > gap):
            keep.append(day)
            last = positions[day]
    return pd.DatetimeIndex(keep)


def tstat(x: pd.Series) -> float:
    x = x.dropna()
    if len(x) < 3 or x.std(ddof=1) == 0:
        return float("nan")
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


def line(label: str, x: pd.Series) -> None:
    x = x.dropna()
    if len(x) == 0:
        print(f"  {label:52s} N=0")
        return
    print(f"  {label:52s} {x.mean()*1e4:+8.1f} bps  t {tstat(x):+5.2f}"
          f"  N {len(x):4d}  hit {(x > 0).mean():.2f}")


# The trade as shipped: SHORT XLU, LONG SPY, dollar neutral.
def trade(h: int) -> pd.Series:
    return fwd[h]["spy"] - fwd[h]["xlu"]


def trade_beta_neutral(h: int) -> pd.Series:
    """Long 1 SPY, short (1/beta) XLU so the pair carries no net market beta."""
    return fwd[h]["spy"] - fwd[h]["xlu"] / beta


cell_b = (rank_xlu < 10) & (rank_spy > 50) & near_high
control = (rank_spy > 50) & near_high            # drop ONLY the XLU washout
b_days = decluster(cell_b.fillna(False))
c_days = decluster(control.fillna(False))

print("=" * 88)
print("TODAY (signal 2026-08-05):  XLU 21d rank %.1f | SPY 21d rank %.1f | "
      "SPY near high %s | XLU beta %.2f"
      % (rank_xlu.iloc[-1], rank_spy.iloc[-1], bool(near_high.iloc[-1]),
         beta.iloc[-1]))
print("Fable cell B fires today:", bool(cell_b.iloc[-1]))
print("=" * 88)

print(f"\n1. THE SHIPPED TRADE, short XLU / long SPY  ({len(b_days)} episodes)")
for h in HORIZONS:
    line(f"fwd {h}d, Fable cell B", trade(h).reindex(b_days))

print("\n2. LEG DECOMPOSITION vs each leg's OWN unconditional drift")
for h in HORIZONS:
    x_cell = fwd[h]["xlu"].reindex(b_days).dropna()
    s_cell = fwd[h]["spy"].reindex(b_days).dropna()
    x_unc, s_unc = fwd[h]["xlu"].dropna(), fwd[h]["spy"].dropna()
    print(f"  h={h}d")
    print(f"    XLU in cell {x_cell.mean()*1e4:+8.1f} bps vs unconditional "
          f"{x_unc.mean()*1e4:+8.1f} bps  ->  lift {(x_cell.mean()-x_unc.mean())*1e4:+8.1f} bps")
    print(f"    SPY in cell {s_cell.mean()*1e4:+8.1f} bps vs unconditional "
          f"{s_unc.mean()*1e4:+8.1f} bps  ->  lift {(s_cell.mean()-s_unc.mean())*1e4:+8.1f} bps")
    short_leg = -(x_cell.mean() - x_unc.mean())
    long_leg = s_cell.mean() - s_unc.mean()
    total = short_leg + long_leg
    if total:
        print(f"    contribution to the pair's EXCESS: short XLU "
              f"{100*short_leg/total:+6.1f}% | long SPY {100*long_leg/total:+6.1f}%")

print(f"\n3. THE MISSING CONTROL: same trade whenever SPY is near a high with")
print(f"   21d rank > 50, WITHOUT the XLU washout  ({len(c_days)} episodes)")
for h in HORIZONS:
    line(f"fwd {h}d, control", trade(h).reindex(c_days))
    b = trade(h).reindex(b_days).dropna()
    c = trade(h).reindex(c_days).dropna()
    diff = b.mean() - c.mean()
    se = np.sqrt(b.var(ddof=1) / len(b) + c.var(ddof=1) / len(c))
    print(f"  {'washout adds (cell B minus control)':52s} {diff*1e4:+8.1f} bps"
          f"  Welch t {diff/se:+5.2f}")

print("\n4. BETA-NEUTRAL version of the shipped trade (cell B)")
for h in HORIZONS:
    line(f"fwd {h}d, beta-neutral", trade_beta_neutral(h).reindex(b_days))

print("\n5. ERA SPLIT of the shipped trade (cell B)")
for h in HORIZONS:
    series = trade(h).reindex(b_days)
    line(f"fwd {h}d, pre-2018", series[series.index < "2018-01-01"])
    line(f"fwd {h}d, 2018+", series[series.index >= "2018-01-01"])

print("\n5b. IS THE NEAR-HIGH LEG LOAD-BEARING? Same washout, SPY strong,")
print("    but SPY NOT within 2% of its 252d high")
a_not_b = decluster(((rank_xlu < 10) & (rank_spy > 50) & (~near_high)).fillna(False))
print(f"    ({len(a_not_b)} episodes: "
      + ", ".join(str(d.date()) for d in a_not_b) + ")")
for h in HORIZONS:
    line(f"fwd {h}d, washout WITHOUT near-high", trade(h).reindex(a_not_b))

print("\n6. LEAVE-ONE-YEAR-OUT on the shipped trade, h=5d")
series = trade(5).reindex(b_days).dropna()
years = sorted({d.year for d in series.index})
floors = []
for year in years:
    kept = series[[d.year != year for d in series.index]]
    floors.append((year, tstat(kept), kept.mean() * 1e4))
worst = min(floors, key=lambda r: abs(r[1]) if not np.isnan(r[1]) else 99)
print(f"  full sample t {tstat(series):+.2f}, mean {series.mean()*1e4:+.1f} bps")
print(f"  LOYO t range {min(f[1] for f in floors):+.2f} to "
      f"{max(f[1] for f in floors):+.2f}")
print(f"  weakest when dropping {worst[0]}: t {worst[1]:+.2f}, "
      f"mean {worst[2]:+.1f} bps")
top = series.sort_values(ascending=False)
print(f"  drop the single best episode ({top.index[0].date()}, "
      f"{top.iloc[0]*1e4:+.0f} bps): t {tstat(series.drop(top.index[0])):+.2f}")
print(f"  episodes by year: "
      + ", ".join(f"{y}:{sum(1 for d in series.index if d.year == y)}"
                  for y in years))
