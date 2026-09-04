"""Follow-up to lev3x_fade_class_study: episode-clustered stats for the
bear-eq loosening candidates. Same-direction 3x ETFs fire together in a
selloff, so per-trade N overstates independence. Cluster signals into
episodes (gap > 5 td starts a new one) and t-test episode-level R.
"""
import copy
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import data_provider
from strategy_config import STRATEGY_BOOK, ACCOUNT_VALUE, LEV3X_ALL
from pages.strat_backtester import (
    load_seasonal_map, load_atr_seasonal_map, precompute_all_indicators,
    generate_candidates_fast, process_signals_fast,
)

START = pd.Timestamp("2003-01-01")
BEAR_EQ = ['SPXS', 'SQQQ', 'SDOW', 'TZA',
           'SOXS', 'FAZ', 'TECS', 'LABD', 'ERY', 'DRV', 'WEBS', 'YANG', 'EDZ']

base = copy.deepcopy({s["name"]: s for s in STRATEGY_BOOK}["3x ETF Overbot Fade"])

def variant(name, thresh=None, consec21=None):
    v = copy.deepcopy(base)
    v["name"] = name
    v["universe_tickers"] = BEAR_EQ
    pf = v["settings"]["perf_filters"]
    if thresh is not None:
        for i in range(4):
            pf[i]["thresh"] = thresh
    if consec21 is not None:
        pf[3]["consecutive"] = consec21
    return v

variants = [
    variant("base"),
    variant("80+consec1", thresh=80.0, consec21=1),
    variant("75+consec3", thresh=75.0),
]

md = data_provider.get_history(list(LEV3X_ALL) + ["SPY", "^VIX"], start="2000-01-01")
vd = md["^VIX"].copy()
if isinstance(vd.columns, pd.MultiIndex):
    vd.columns = vd.columns.get_level_values(0)
vd.columns = [c.capitalize() for c in vd.columns]
vix_series = vd["Close"]

sznl_map = load_seasonal_map()
atr_sznl_map = load_atr_seasonal_map()
processed = precompute_all_indicators(md, variants, sznl_map, vix_series, atr_sznl_map)

def episodes(tr):
    tr = tr.sort_values("Date").copy()
    dates = pd.to_datetime(tr["Date"]).values
    ep = np.zeros(len(tr), dtype=int)
    for i in range(1, len(tr)):
        gap = np.busday_count(dates[i - 1].astype("M8[D]"), dates[i].astype("M8[D]"))
        ep[i] = ep[i - 1] + (1 if gap > 5 else 0)
    tr["ep"] = ep
    return tr

for v in variants:
    cands, sd = generate_candidates_fast(processed, [v], sznl_map, START)
    tr = process_signals_fast(cands, sd, processed, [v], ACCOUNT_VALUE, flat_sizing=True)
    if tr.empty:
        print(f"{v['name']}: no trades")
        continue
    tr["Date"] = pd.to_datetime(tr["Date"])
    tr["R"] = tr["PnL"] / tr["Risk $"].replace(0, np.nan)
    tr = episodes(tr)
    ep_mean = tr.groupby("ep")["R"].mean()   # avg R per episode
    ep_sum = tr.groupby("ep")["R"].sum()
    n = len(ep_mean)
    t = ep_mean.mean() / (ep_mean.std(ddof=1) / np.sqrt(n)) if n > 2 else np.nan
    print(f"\n{v['name']}: trades={len(tr)}  episodes={n}")
    print(f"  episode avgR: mean={ep_mean.mean():+.3f}  median={ep_mean.median():+.3f}  "
          f"t={t:+.2f}")
    print(f"  episode sumR: mean={ep_sum.mean():+.3f}  "
          f"neg episodes: {(ep_sum < 0).sum()}/{n}")
    print(f"  trades/episode: {len(tr)/n:.1f}   tickers: "
          f"{tr['Ticker'].value_counts().to_dict()}")
    yr = tr.groupby(tr["Date"].dt.year)["R"].agg(["sum", "count"])
    print("  by year: " + "  ".join(f"{y}:{s:+.1f}({int(c)})"
                                    for y, (s, c) in yr.iterrows()))
