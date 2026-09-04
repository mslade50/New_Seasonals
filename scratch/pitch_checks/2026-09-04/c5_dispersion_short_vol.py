"""C5 - Dispersion at the 92nd percentile read as a SHORT-index-vol signal.

Round 1: reconstruct the PRODUCTION dispersion signal from master_prices
(same code path the risk dashboard uses), then run the kill battery on
long SVXY / short ^VIX, plus the three mandatory C5 blockers:
  (a) separability from the fragility dial's own high band
  (b) separability from the production VIX Range Compression state
  (c) the reference class = the other seven fragility signals
"""
import sys, os
from pathlib import Path
ROOTP = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOTP))
sys.path.insert(0, str(ROOTP / "scripts"))
sys.path.insert(0, str(ROOTP / "pages"))
import numpy as np, pandas as pd
from pitch_lab import *  # noqa

DAY = Path(__file__).resolve().parent
CACHE = DAY / "_sigmask_cache.parquet"
PCT = DAY / "_disp_pctile.parquet"

def build_masks():
    from build_atr_downside_stats import build_inputs_from_master, compute_signal_masks
    spy_df, closes, sp500 = build_inputs_from_master()
    print(f"inputs: spy {spy_df.index.min().date()}..{spy_df.index.max().date()}  "
          f"sp500 cols={sp500.shape[1]}  rows={sp500.shape[0]}")
    sigs = compute_signal_masks(spy_df, closes, sp500)
    out = {}
    for k, v in sigs.items():
        h = v.get("signal_history")
        if h is None or len(h) == 0:
            print(f"  {k}: no history"); continue
        out[k] = pd.Series(h).astype(bool)
    m = pd.DataFrame(out).fillna(False)
    m.index = pd.to_datetime(m.index)
    m.to_parquet(CACHE)
    d = sigs["Dispersion"]
    pct = pd.DataFrame({"composite": pd.Series(d["composite_pctile"])})
    pct.index = pd.to_datetime(pct.index)
    pct.to_parquet(PCT)
    return m, pct

if CACHE.exists() and PCT.exists() and "--rebuild" not in sys.argv:
    masks = pd.read_parquet(CACHE); pct = pd.read_parquet(PCT)
else:
    masks, pct = build_masks()

print("\nsignal fire counts (day level, 25y master_prices):")
print(masks.sum().to_string())
comp = pct["composite"].dropna()
print(f"\ncomposite pctile: {comp.index.min().date()}..{comp.index.max().date()} "
      f"n={len(comp)}  latest={comp.iloc[-1]:.1f}  (surface map says 92)")

# ---------------------------------------------------------------- vehicles
px = close_panel(["SVXY", "^VIX", "SPY", "^VIX3M"])
print("panel", px.index.min().date(), px.index.max().date())

prod = masks["Dispersion"].reindex(px.index, fill_value=False)
raw85 = (comp > 85).reindex(px.index, fill_value=False)
raw90 = (comp > 90).reindex(px.index, fill_value=False)
print(f"\ntriggers on the price panel: prod(declustered+corr gate)={int(prod.sum())}  "
      f"raw comp>85={int(raw85.sum())}  raw comp>90={int(raw90.sum())}")
print("prod fire dates:", ", ".join(str(d.date()) for d in px.index[prod.values]))

# SVXY only exists from 2011-10; measure the unlevered short-VIX leg too.
for h in (5, 10, 21):
    battery(px, raw85, [("SVXY", 1.0)], h,
            f"C5 RAW comp>85 -> LONG SVXY", cost_bps=8.0,
            variants={"comp>90": raw90, "comp>92": (comp > 92).reindex(px.index, fill_value=False),
                      "prod signal": prod},
            min_gap=10)
for h in (5, 10, 21):
    battery(px, raw85, [("^VIX", -1.0)], h,
            f"C5 RAW comp>85 -> SHORT ^VIX (measurement only, untradeable)",
            cost_bps=0.0, min_gap=10)
battery(px, raw85, [("SPY", 1.0)], 10, "C5 RAW comp>85 -> LONG SPY (direction sanity)",
        cost_bps=2.0, min_gap=10)
