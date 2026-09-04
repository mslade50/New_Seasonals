"""Verify the 3x Bear ETF Overbot Fade deploy end-to-end through the engine:
run the two 3x strategies exactly as they now sit in STRATEGY_BOOK and check
(a) the carve-out (no shared tickers, no same-day same-ticker trades),
(b) the bear strat books the expected ~51 trades,
(c) Size_Mult on multi-signal days reflects the same-day derate
    (0.9x on 2-signal days etc.), including frag-band interaction.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import data_provider
from strategy_config import STRATEGY_BOOK, ACCOUNT_VALUE, LEV3X_ALL, same_day_derate_mult
from pages.strat_backtester import (
    load_seasonal_map, load_atr_seasonal_map, precompute_all_indicators,
    generate_candidates_fast, process_signals_fast, frag_band_mult_at,
)

START = pd.Timestamp("2003-01-01")

strategies = [s for s in STRATEGY_BOOK
              if s["name"] in ("3x ETF Overbot Fade", "3x Bear ETF Overbot Fade")]
assert len(strategies) == 2
bear = next(s for s in strategies if "Bear" in s["name"])

md = data_provider.get_history(list(LEV3X_ALL) + ["SPY", "^VIX"], start="2000-01-01")
vd = md["^VIX"].copy()
if isinstance(vd.columns, pd.MultiIndex):
    vd.columns = vd.columns.get_level_values(0)
vd.columns = [c.capitalize() for c in vd.columns]

sznl_map = load_seasonal_map()
atr_sznl_map = load_atr_seasonal_map()
processed = precompute_all_indicators(md, strategies, sznl_map, vd["Close"], atr_sznl_map)

cands, sd = generate_candidates_fast(processed, strategies, sznl_map, START)
tr = process_signals_fast(cands, sd, processed, strategies, ACCOUNT_VALUE, flat_sizing=True)
tr["Date"] = pd.to_datetime(tr["Date"]).dt.normalize()
tr["R"] = tr["PnL"] / tr["Risk $"].replace(0, np.nan)

fails = 0

# (a) carve-out: no ticker overlap, no same-day same-ticker across the two
overlap = (set(strategies[0]["universe_tickers"])
           & set(strategies[1]["universe_tickers"]))
print(f"universe overlap: {sorted(overlap) or 'NONE'}")
fails += bool(overlap)
key_a = set(zip(tr[tr.Strategy == "3x ETF Overbot Fade"].Ticker,
                tr[tr.Strategy == "3x ETF Overbot Fade"].Date))
key_b = set(zip(tr[tr.Strategy == "3x Bear ETF Overbot Fade"].Ticker,
                tr[tr.Strategy == "3x Bear ETF Overbot Fade"].Date))
print(f"same-day same-ticker collisions: {len(key_a & key_b)}")
fails += bool(key_a & key_b)

# (b) trade counts + stats per strategy
for name, g in tr.groupby("Strategy"):
    r = g["R"].dropna()
    print(f"{name}: N={len(g)}  win={(r > 0).mean():.1%}  avgR={r.mean():+.3f}  "
          f"totR={r.sum():+.1f}")

# (c) derate applied: reconstruct expected Size_Mult per bear trade =
#     derate(n_signals_that_day) x frag_band_mult, compare to engine Size_Mult.
# n_signals must be counted on CANDIDATES (staged), not fills.
bear_idx = next(i for i, s in enumerate(strategies) if "Bear" in s["name"])
cand_counts = {}
for c in cands:
    if c[3] == bear_idx:
        d = pd.Timestamp(c[0]).normalize()
        cand_counts[d] = cand_counts.get(d, 0) + 1

bt = tr[tr.Strategy == "3x Bear ETF Overbot Fade"].copy()
bt["n_sig"] = bt["Date"].map(cand_counts)
bt["expected_mult"] = [
    same_day_derate_mult(bear["execution"], n) *
    frag_band_mult_at(bear["execution"], d.value)
    for n, d in zip(bt["n_sig"], bt["Date"])
]
mismatch = bt[~np.isclose(bt["Size_Mult"], bt["expected_mult"], atol=1e-9)]
print(f"\nbear trades with Size_Mult != derate x frag_band: {len(mismatch)}")
fails += bool(len(mismatch))
if len(mismatch):
    print(mismatch[["Ticker", "Date", "n_sig", "Size_Mult", "expected_mult"]]
          .to_string(index=False))

derated = bt[bt.n_sig > 1]
print(f"\nbear trades on multi-signal days: {len(derated)}")
print(derated[["Ticker", "Date", "n_sig", "Size_Mult", "R"]]
      .sort_values("Date").to_string(index=False))

print("\nVERIFY:", "PASS" if fails == 0 else f"FAIL ({fails})")
