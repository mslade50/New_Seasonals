"""C8c: does the systematic book already trade this state?"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import pandas as pd, numpy as np
import strategy_config as sc

led = pd.read_parquet(Path(__file__).resolve().parents[3] / "data"
                      / "backtest_trades_full.parquet")
print("ledger rows", len(led), "cols", list(led.columns)[:14])
sigcol = "Signal Date"
led[sigcol] = pd.to_datetime(led[sigcol])

UNIV = sorted(set(sc.LIQUID_PLUS_COMMODITIES))
px = close_panel(UNIV)
rk63 = pd.DataFrame({t: pct_rank(px[t], 63) for t in UNIV})
hi252 = pd.DataFrame({t: px[t].rolling(252, min_periods=200).max() for t in UNIV})
dd = pd.DataFrame({t: px[t] / hi252[t] - 1.0 for t in UNIV})
trig = ((rk63 >= 95) & (dd <= -0.15)).fillna(False)

tick = "Ticker"
led = led[led[tick].isin(UNIV)].copy()
pos = {d: i for i, d in enumerate(px.index)}
hits = []
for _, r in led.iterrows():
    d, t = r[sigcol], r[tick]
    if d in trig.index and t in trig.columns:
        hits.append(bool(trig.loc[d, t]))
hits = np.array(hits)
print("\nbook trades on liquid names: %d ; on a C8 trigger day: %d (%.2f%%)"
      % (len(hits), hits.sum(), 100*hits.mean()))
base_rate = trig.values.sum() / np.isfinite(rk63.values).sum()
print("base rate of C8 trigger name-days: %.2f%%  -> enrichment %.2fx"
      % (100*base_rate, hits.mean()/base_rate if base_rate else np.nan))
sub = led.iloc[:len(hits)][hits]
if len(sub):
    print("\nby strategy:")
    print(sub["Strategy"].value_counts().head(12).to_string())

print("\nstrategy_config strategies requiring a 52w HIGH (distinct by construction):")
for cfg in sc.STRATEGY_BOOK:
    txt = str(cfg)
    if "52" in txt or "high" in txt.lower():
        print("  -", cfg.get("name", cfg.get("strategy_name", "?")))
