"""Leader-expansion fade — validation pass on the FINAL config (2026-07-10).

Config under test (settled over the 2026-07-10 study chain):
  universe: LEV3X_ALL minus BULL_EQ (21 names: 13 bear-eq + TMF/TMV + 6 cmdty)
  filters:  2/5/10/21d rank > 80 (consec 1), 252d rank > 95 (leader REQUIRED)
  tape:     T+1 open gap: NextOpen > Close + 0.25 ATR
  entry:    Limit (Open +/- 0.75 ATR)  [OVS convention]
  exit:     2-day time exit, no stop
Validation, following the cycle-year-tilt discipline:
  1. episode clustering (overlapping-window day clusters, not just ISO week)
  2. clustered t-stat on episode means
  3. LOYO (leave-one-year-out) sign stability
  4. drop-best-episode / drop-best-year sensitivity
  5. episode bootstrap P(totR <= 0)
"""
import copy
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import data_provider
from strategy_config import STRATEGY_BOOK, ACCOUNT_VALUE, LEV3X_ALL, LEV3X_BEAR_EQ
from pages.strat_backtester import (
    load_seasonal_map,
    load_atr_seasonal_map,
    precompute_all_indicators,
    generate_candidates_fast,
    process_signals_fast,
)

START = pd.Timestamp("2003-01-01")

BULL_EQ = ['SPXL', 'TQQQ', 'UDOW', 'TNA', 'MIDU',
           'SOXL', 'FAS', 'TECL', 'LABU', 'CURE', 'ERX', 'DPST',
           'DRN', 'NAIL', 'RETL', 'WEBL', 'DFEN', 'YINN', 'BRZU', 'EDC', 'MEXX']
UNIVERSE = [t for t in LEV3X_ALL if t not in BULL_EQ]

def klass(tk):
    if tk in LEV3X_BEAR_EQ: return 'bear-eq'
    if tk in ('TMF', 'TMV'): return 'bond'
    return 'cmdty'

base = copy.deepcopy({s["name"]: s for s in STRATEGY_BOOK}["3x ETF Overbot Fade"])
v = copy.deepcopy(base)
v["name"] = "3x Leader Gap Fade (candidate)"
v["universe_tickers"] = UNIVERSE
pf = v["settings"]["perf_filters"]
for i in range(4):
    pf[i]["thresh"] = 80.0
pf[3]["consecutive"] = 1
flip = copy.deepcopy(pf[5])
flip["logic"] = ">"
flip["thresh"] = 95.0
v["settings"]["perf_filters"] = pf[:4] + [flip]
v["settings"]["use_t1_open_filter"] = True
v["settings"]["t1_open_filters"] = [
    {"reference": "Close", "atr_offset": 0.25, "logic": ">"}]
v["settings"]["entry_type"] = "Limit (Open +/- 0.75 ATR)"

md = data_provider.get_history(UNIVERSE + ["SPY", "^VIX"], start="2000-01-01")
vix_df = md.get("^VIX")
vix_series = None
if vix_df is not None and not vix_df.empty:
    vd = vix_df.copy()
    if isinstance(vd.columns, pd.MultiIndex):
        vd.columns = vd.columns.get_level_values(0)
    vd.columns = [c.capitalize() for c in vd.columns]
    vix_series = vd["Close"]

sznl_map = load_seasonal_map()
atr_sznl_map = load_atr_seasonal_map()
processed = precompute_all_indicators(md, [v], sznl_map, vix_series, atr_sznl_map)
cands, sd = generate_candidates_fast(processed, [v], sznl_map, START)
tr = process_signals_fast(cands, sd, processed, [v], ACCOUNT_VALUE, flat_sizing=True)
tr["Date"] = pd.to_datetime(tr["Date"])
tr["R"] = tr["PnL"] / tr["Risk $"].replace(0, np.nan)
tr["class"] = tr["Ticker"].map(klass)
tr = tr.sort_values("Date").reset_index(drop=True)

r = tr["R"]
pfac = r[r > 0].sum() / max(1e-9, -(r[r < 0].sum()))
yrs = (tr["Date"].max() - tr["Date"].min()).days / 365.25
print(f"FINAL CONFIG — {len(tr)} trades, {tr['Date'].min().date()} -> "
      f"{tr['Date'].max().date()} (~{len(tr)/yrs:.1f} tr/yr)")
print(f"  win={(r > 0).mean():.1%}  avgR={r.mean():+.3f}  medR={r.median():+.3f}  "
      f"totR={r.sum():+.1f}  PF={pfac:.2f}  minR={r.min():+.2f}  maxR={r.max():+.2f}")
print(f"  trade-level t = {r.mean() / (r.std() / np.sqrt(len(r))):.2f}")
for k in ['bear-eq', 'bond', 'cmdty']:
    g = tr[tr["class"] == k]
    print(f"    {k:<8} N={len(g):>3}  avgR={g['R'].mean():+.3f}  totR={g['R'].sum():+.1f}")

# --- 1/2. episode clustering: trades whose signal dates are within 5 td join ---
tr["gap_days"] = tr["Date"].diff().dt.days.fillna(999)
tr["episode"] = (tr["gap_days"] > 7).cumsum()
ep = tr.groupby("episode").agg(
    start=("Date", "min"), end=("Date", "max"), n=("R", "count"),
    meanR=("R", "mean"), sumR=("R", "sum"),
    tickers=("Ticker", lambda x: ",".join(sorted(set(x)))))
em = ep["meanR"]
t_ep = em.mean() / (em.std() / np.sqrt(len(em)))
print(f"\nEPISODES (new episode when >7 calendar days between signals): {len(ep)}")
print(f"  episode meanR: mean {em.mean():+.3f}  median {em.median():+.3f}  "
      f"clustered t = {t_ep:.2f}")
print(f"  positive episodes: {(ep['sumR'] > 0).sum()}/{len(ep)}")
for _, row in ep.iterrows():
    print(f"    {row['start']:%Y-%m-%d}..{row['end']:%m-%d}  n={row['n']:>2}  "
          f"sum={row['sumR']:+6.2f}  mean={row['meanR']:+6.2f}  {row['tickers']}")

# --- 3. LOYO ---
print("\nLOYO (drop each year, stats on the rest):")
years = sorted(tr["Date"].dt.year.unique())
worst_t, worst_y = 99.0, None
for y in years:
    rest = tr[tr["Date"].dt.year != y]
    er = rest.groupby("episode")["R"].mean()
    t_l = er.mean() / (er.std() / np.sqrt(len(er))) if len(er) > 2 else np.nan
    if t_l < worst_t:
        worst_t, worst_y = t_l, y
    print(f"  -{y}: N={len(rest):>3}  avgR={rest['R'].mean():+.3f}  "
          f"totR={rest['R'].sum():+6.1f}  episode-t={t_l:.2f}")
print(f"  LOYO floor: episode-t {worst_t:.2f} (dropping {worst_y})")

# --- 4. drop-best sensitivity ---
best_ep = ep["sumR"].idxmax()
rest = tr[tr["episode"] != best_ep]
er = rest.groupby("episode")["R"].mean()
t_b = er.mean() / (er.std() / np.sqrt(len(er)))
print(f"\nDrop BEST episode ({ep.loc[best_ep, 'start']:%Y-%m-%d}, "
      f"{ep.loc[best_ep, 'sumR']:+.1f}R): N={len(rest)}  "
      f"avgR={rest['R'].mean():+.3f}  totR={rest['R'].sum():+.1f}  episode-t={t_b:.2f}")
by_year = tr.groupby(tr["Date"].dt.year)["R"].sum()
best_year = by_year.idxmax()
rest_y = tr[tr["Date"].dt.year != best_year]
ery = rest_y.groupby("episode")["R"].mean()
t_by = ery.mean() / (ery.std() / np.sqrt(len(ery)))
print(f"Drop BEST year ({best_year}, {by_year.max():+.1f}R): N={len(rest_y)}  "
      f"avgR={rest_y['R'].mean():+.3f}  totR={rest_y['R'].sum():+.1f}  "
      f"episode-t={t_by:.2f}")

# --- 5. episode bootstrap ---
rng = np.random.default_rng(20260710)
sums = ep["sumR"].values
boots = np.array([rng.choice(sums, size=len(sums), replace=True).sum()
                  for _ in range(20000)])
print(f"\nEpisode bootstrap (20k resamples of {len(sums)} episode sums):")
print(f"  P(totR <= 0) = {(boots <= 0).mean():.3f}   "
      f"5th pctile totR = {np.percentile(boots, 5):+.1f}   "
      f"median = {np.percentile(boots, 50):+.1f}")
