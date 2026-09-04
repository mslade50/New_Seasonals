"""C4 -- ^VIX3M LEVEL at a trailing-252d floor. Forward SPY and SVXY.

LEVEL percentile, not a return rank (registry: the two coincide 30.7% of the
time). Today: VIX3M 17.99, PIT level pctile 4.0, +1.52% above its 52w low,
fragility ma10(63d) 88.6.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import numpy as np, pandas as pd

TK = ["^VIX3M", "^VIX", "SPY", "SVXY"]
px = close_panel(TK)
px = px[px.index >= "2006-07-17"]

v3 = px["^VIX3M"]
# LEVEL percentile, trailing 252 sessions, PIT
lvl_pct = rolling_on_valid(v3, lambda x: x.rolling(252).rank(pct=True) * 100.0)
lo252 = rolling_on_valid(v3, lambda x: x.rolling(252).min())
above_low = v3 / lo252 - 1.0

print("today check:", v3.iloc[-1], "lvlpct", round(lvl_pct.iloc[-1], 2),
      "above52wlow%", round(100 * above_low.iloc[-1], 2))

# how often does a LEVEL-floor day coincide with a 5d-RETURN-rank floor?
r5rank = pct_rank(v3, 5)
m_lvl = (lvl_pct <= 5.0)
print("level-floor days:", int(m_lvl.sum()),
      "| of those also r5rank<=5:",
      int((m_lvl & (r5rank <= 5)).sum()))

# ---- primary mask
mask = m_lvl.fillna(False)
print("\nmask days:", int(mask.sum()), "span",
      px.index[mask][0].date(), "..", px.index[mask][-1].date())
print("by year:", mask.groupby(px.index.year).sum().to_dict())

variants = {
    "lvlpct<=2": (lvl_pct <= 2.0).fillna(False),
    "lvlpct<=5 (base)": mask,
    "lvlpct<=10": (lvl_pct <= 10.0).fillna(False),
    "lvlpct<=20": (lvl_pct <= 20.0).fillna(False),
    "within2% of 52w low": (above_low <= 0.02).fillna(False),
    "all days": pd.Series(True, index=px.index),
}

for h in (3, 5, 10):
    battery(px, mask, [("SPY", 1.0)], h, f"C4 LONG SPY | VIX3M lvlpct<=5", 3.0,
            variants=variants, min_gap=10, event_kinds=("jackson_hole",))

# ---- SVXY leg. post-leverage-cut only (Feb 2018) is the real instrument
sv = px["SVXY"].dropna()
cut = pd.Timestamp("2018-03-01")
mask_sv = mask & (px.index >= cut)
print("\n\n### SVXY leg, post-leverage-cut (2018-03+) only ###")
print("mask days post-cut:", int(mask_sv.sum()))
for h in (3, 5, 10):
    battery(px[px.index >= cut], mask_sv[px.index >= cut], [("SVXY", 1.0)], h,
            f"C4 LONG SVXY (post-cut) | VIX3M lvlpct<=5", 8.0,
            min_gap=10, event_kinds=("jackson_hole",))

# ---- joint with the fragility dial (CONTEXT only; registry closed the dial
# as a directional signal). Question: has this state EVER occurred at dial>=70?
frag = pd.read_parquet(ROOT / "data" / "rd2_fragility.parquet")
ma10 = frag["63d"].rolling(10).mean()
ma10 = ma10.reindex(px.index).ffill(limit=3)
sub = pd.DataFrame({"m": mask, "dial": ma10}).dropna()
trg = sub[sub["m"]]
print("\n\n### dial support for the cell (2016+ overlap) ###")
print("cell days with a dial reading:", len(trg))
if len(trg):
    print("dial min/median/max on cell days: %.1f / %.1f / %.1f"
          % (trg["dial"].min(), trg["dial"].median(), trg["dial"].max()))
    print("cell days at dial>=70:", int((trg["dial"] >= 70).sum()),
          " at dial>=85:", int((trg["dial"] >= 85).sum()))

# ---- book overlap
led = pd.read_parquet(ROOT / "data" / "backtest_trades_full.parquet")
led["Signal Date"] = pd.to_datetime(led["Signal Date"])
sig_days = set(px.index[mask])
ov = led[led["Signal Date"].isin(sig_days)]
print("\n### book overlap ###")
print("ledger signals in-state:", len(ov), "avgR:",
      round(ov["R_Multiple"].mean(), 3) if len(ov) else "n/a",
      "| book-wide avgR:", round(led["R_Multiple"].mean(), 3))
