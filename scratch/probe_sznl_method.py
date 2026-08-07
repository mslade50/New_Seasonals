"""Identify the method behind sznl_ranks.csv, so the 18 missing LIQUID names
can be added with the SAME definition rather than an invented one.

FIRST ATTEMPT FAILED and is worth recording. I assumed the pipeline in
build_atr_seasonal_ranks.py run on un-normalized SIMPLE returns at a SINGLE
window. Best correlation across 3 tickers x 2 years x 6 windows was 0.56-0.74
with 44-63 rank-point max errors, ~0% exact, and the winning window flipped
between 5/10/21 by ticker. Noise, not a match.

The real source is build_sznl_forecast.py, which emits exactly this file's
schema (Date, seasonal_rank, ticker) for the SECTOR_ETFS universe. Three
differences from the guess, each decisive:
  1. LOG forward returns, not simple
  2. ranks are AVERAGED ACROSS windows [5, 10, 21] before the cycle blend
  3. no day_count clipping (the ATR builder caps at 251; this does not)

Then: 25% all-years + 75% presidential-cycle, 5-day centered smooth.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

WINDOWS = [5, 10, 21]
PROBE = ["AAPL", "XOM", "JPM", "KO", "SPY"]
YEARS = [2025, 2026]

ref = pd.read_csv(ROOT / "sznl_ranks.csv", parse_dates=["Date"])
px = pd.read_parquet(ROOT / "data" / "master_prices.parquet")


def forward_returns(close: pd.Series) -> pd.DataFrame:
    d = pd.DataFrame({"Close": close})
    d["year"] = d.index.year
    d["day_count"] = d.groupby("year").cumcount() + 1
    for w in WINDOWS:
        d[f"Fwd_{w}d"] = np.log(d["Close"].shift(-w) / d["Close"])
    return d


def profile(d: pd.DataFrame, target_year: int):
    """Verbatim port of build_sznl_forecast.calculate_forecast_profile."""
    d = d[d["year"] < target_year].copy()
    if d.empty:
        return None
    fwd = [f"Fwd_{w}d" for w in WINDOWS]
    rank_all = d.groupby("day_count")[fwd].mean().rank(pct=True) * 100
    cyc = d[d["year"] % 4 == target_year % 4]
    if cyc.empty or cyc["year"].nunique() < 2:
        rank_cyc = rank_all.copy()
    else:
        rank_cyc = cyc.groupby("day_count")[fwd].mean().rank(pct=True) * 100
    max_day = max(int(d["day_count"].max()), 253)
    idx = pd.RangeIndex(1, max_day + 1)
    rank_all = rank_all.reindex(idx).interpolate(method="nearest").fillna(50)
    rank_cyc = rank_cyc.reindex(idx).interpolate(method="nearest").fillna(50)
    final = (rank_all.mean(axis=1) + 3 * rank_cyc.mean(axis=1)) / 4
    return final.rolling(5, center=True, min_periods=1).mean()


print(f"{'ticker':<7}{'year':<6}{'corr':>9}{'max_err':>10}{'within0.05':>12}{'within0.5':>11}")
rows = []
for t in PROBE:
    s = px[px.ticker == t].sort_values("date").set_index("date")["Close"]
    if s.empty:
        print(f"{t}: no price data"); continue
    d = forward_returns(s)
    for y in YEARS:
        act = ref[(ref.ticker == t) & (ref.Date.dt.year == y)].sort_values("Date")
        if act.empty:
            continue
        p = profile(d, y)
        if p is None:
            continue
        act = act.assign(day_count=np.arange(1, len(act) + 1))
        m = act.assign(pred=act.day_count.map(p)).dropna(subset=["pred"])
        if len(m) < 50:
            continue
        err = (m.seasonal_rank - m.pred).abs()
        corr = float(np.corrcoef(m.seasonal_rank, m.pred)[0, 1])
        rows.append((t, y, corr, float(err.max()),
                     float((err < 0.05).mean() * 100), float((err < 0.5).mean() * 100)))
        print(f"{t:<7}{y:<6}{corr:>9.4f}{err.max():>10.3f}"
              f"{(err < 0.05).mean() * 100:>11.1f}%{(err < 0.5).mean() * 100:>10.1f}%")

if rows:
    print(f"\nmedian corr {np.median([r[2] for r in rows]):.4f} | "
          f"worst max_err {max(r[3] for r in rows):.3f} | "
          f"median within-0.05 {np.median([r[4] for r in rows]):.1f}%")
    print("\nA match means corr ~1.0 and errors inside the file's 1-decimal "
          "rounding. Anything less and the method is still not identified.")
