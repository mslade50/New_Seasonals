"""Backtest the seasonal signal on TRADEABLE proxies (ETFs/futures) instead of the
cash indices, to see whether the macro edge survives on what we'd actually trade.
Scans each proxy for its own seasonal edge (so FX/tracking drag is included), then
compares each proxy to its cash index.

Reuses the backtest harness by pointing the macro universe at the proxies.
"""
import sys
import numpy as np
import pandas as pd

ROOT = r"C:\Users\McKinley Slade\dev\New_Seasonals"
sys.path.insert(0, ROOT)
import scripts.seasonal_edge as se
import scripts.backtest_seasonal_ideas as bt
from scripts.seasonal_sharpe import ratios

# index -> tradeable proxy (all 22 now that the European/extra set is backfilled)
PROXY = {"^GSPC": "SPY", "^NDX": "QQQ", "^DJI": "DIA", "^RUT": "IWM",
         "^N225": "EWJ", "^BVSP": "EWZ", "^MXX": "EWW", "^KS11": "EWY",
         "^TWII": "EWT", "^BSESN": "INDA", "^DJT": "IYT",
         "^GDAXI": "EWG", "^FCHI": "EWQ", "^FTSE": "EWU", "^STOXX50E": "FEZ",
         "^SOX": "SOXX", "^MID": "IJH", "^IXIC": "ONEQ", "^VIX": "VXX",
         "^AXJO": "EWA", "^GSPTSE": "EWC", "^HSI": "EWH"}
AVAIL = sorted(PROXY.values())

# union the backfilled ETF ranks into the rank source BEFORE the harness reads it
_extra = pd.read_parquet(ROOT + r"\data\proxy_extra_ranks.parquet")
_extra["Date"] = pd.to_datetime(_extra["Date"]); _extra["ticker"] = _extra["ticker"].str.upper()
_main = se.load_seasonal_ranks()
_combined = pd.concat([_main, _extra], ignore_index=True)
se.load_seasonal_ranks = lambda path=None: _combined

# point the macro universe at the proxies, scope the harness to them
se.MACRO_TICKERS = AVAIL
se.IDEA_UNIVERSE = AVAIL
import daily_seasonal_ideas as dsi  # it froze its own IDEA_UNIVERSE copy at import
dsi.IDEA_UNIVERSE = set(t.upper() for t in AVAIL)
bt.OUT = ROOT + r"\data\seasonal_proxy_backtest.parquet"
bt.CAND_OUT = ROOT + r"\data\seasonal_proxy_candidates.parquet"

df = bt.run("2010-01-01", "2026-02-25", "daily", ("A",), "t1_open", False,
            channels={"cross_asset"}, do_dedup=True)
print(f"Proxy backtest done: {len(df)} trades -> data/seasonal_proxy_backtest.parquet")
