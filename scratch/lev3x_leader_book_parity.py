"""Parity check: the committed '3x Leader Gap Fade' STRATEGY_BOOK entry must
reproduce the validation study's trades (scratch/lev3x_fade_leader_validation.py:
31 trades, totR +24.7, avgR +0.797). Run the BOOK entry through the engine."""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import data_provider
from strategy_config import STRATEGY_BOOK, ACCOUNT_VALUE
from pages.strat_backtester import (
    load_seasonal_map,
    load_atr_seasonal_map,
    precompute_all_indicators,
    generate_candidates_fast,
    process_signals_fast,
)

v = next(s for s in STRATEGY_BOOK if s["name"] == "3x Leader Gap Fade")
universe = list(v["universe_tickers"])

md = data_provider.get_history(universe + ["SPY", "^VIX"], start="2000-01-01")
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
cands, sd = generate_candidates_fast(processed, [v], sznl_map, pd.Timestamp("2003-01-01"))
tr = process_signals_fast(cands, sd, processed, [v], ACCOUNT_VALUE, flat_sizing=True)
tr["Date"] = pd.to_datetime(tr["Date"])
tr["R"] = tr["PnL"] / tr["Risk $"].replace(0, np.nan)
r = tr["R"]
print(f"book entry: {len(tr)} trades  totR={r.sum():+.1f}  avgR={r.mean():+.3f}  "
      f"win={(r > 0).mean():.1%}  minR={r.min():+.2f}")
print("expected  : 31 trades  totR=+24.7  avgR=+0.797  win=54.8%  minR=-2.95")
print(f"risk bps stamped: {tr['Risk bps'].unique()}")
print(f"first/last: {tr['Date'].min().date()} / {tr['Date'].max().date()}")
# per-trade risk sanity: 37.5 bps of 750k = $2,812.50, capped days scaled
day_risk = tr.groupby(tr["Date"].dt.normalize())["Risk $"].sum() / ACCOUNT_VALUE * 10000
print(f"max same-day placed risk: {day_risk.max():.1f} bps (cap 250)")
print(f"days at/over 250 pre-cap would appear as exactly 250 post-cap: "
      f"{(day_risk > 249.9).sum()} day(s)")
