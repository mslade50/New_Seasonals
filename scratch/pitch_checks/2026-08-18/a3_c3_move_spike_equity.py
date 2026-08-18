"""C3 - the same ^MOVE spike as a cross-asset EQUITY signal (short SPY).
Mechanism: rate vol LEADS equity vol; a MOVE spike is an early warning the
equity tape has not priced. Falsification inside the window: if ^VIX does not
rise over the same forward span, the claimed lead is dead on its own terms.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import numpy as np, pandas as pd

raw = load_prices(["^MOVE", "SPY", "^VIX", "TLT"])
mv, sp, vx = raw["^MOVE"]["Close"], raw["SPY"]["Close"], raw["^VIX"]["Close"]
mv_1d = mv.pct_change()
sp_d52h = sp / sp.rolling(252).max() - 1.0
idx = mv.index.intersection(sp.index).intersection(vx.index)
px = pd.DataFrame({"SPY": sp.reindex(idx), "VIX": vx.reindex(idx), "TLT": raw["TLT"]["Close"].reindex(idx)}).dropna()
idx = px.index
S = mv_1d.reindex(idx)
SPIKE = (S >= 0.08).fillna(False)
NEARHI = (sp_d52h.reindex(idx) >= -0.01).fillna(False)  # SPY within 1% of 52w high (live -0.67%)
print("spike days %d of %d | SPY near 52w high days %d | BOTH %d" %
      (SPIKE.sum(), len(idx), NEARHI.sum(), (SPIKE & NEARHI).sum()))

for H in (3, 5, 10):
    r = fwd_lag(px["SPY"], H, 1)
    v = r.notna()
    rows = []
    for lab, m in (("CTRL-b all days", pd.Series(True, index=idx)),
                   ("MOVE spike >=8%", SPIKE),
                   ("spike AND SPY near 52w high (live)", SPIKE & NEARHI),
                   ("SPY near 52w high, no spike", NEARHI & ~SPIKE)):
        d = idx[m.values & v.values]
        rows.append(summarize((-r).loc[d].values, f"SHORT SPY | {lab} (N={len(d)})"))
    show(rows, f"1. SHORT SPY h={H} lag=1 -- gate attribution")

# --- full battery, short SPY, h=5
battery(px, SPIKE, [("SPY", -1.0)], 5, "C3 MOVE spike >=8% -> SHORT SPY", 2.5,
        variants={"spike >=6%": (S >= 0.06).fillna(False),
                  "spike >=8%": SPIKE,
                  "spike >=8.70% (live)": (S >= 0.0870).fillna(False),
                  "spike >=10%": (S >= 0.10).fillna(False),
                  "spike >=12%": (S >= 0.12).fillna(False)},
        min_gap=10)

d = idx[SPIKE.values]
show(horizon_scan(px, d, [("SPY", -1.0)], hs=(1,2,3,5,10), min_gap=10),
     "2. horizon scan, SHORT SPY (episodes)")

# --- 3. THE MECHANISM TEST: does ^VIX actually rise after a MOVE spike?
print("\n=== 3. MECHANISM: forward ^VIX response (the claimed lead) ===")
epi = declusters(d, 10, idx)
rows = []
for H in (1, 2, 3, 5, 10):
    rv = fwd_lag(px["VIX"], H, 1)
    base = rv.dropna()
    e = rv.reindex(epi).dropna()
    rows.append({"h": H, "n": len(e), "VIX_fwd_cond_pct": round(100*e.mean(), 3),
                 "VIX_fwd_allday_pct": round(100*base.mean(), 3),
                 "edge_pct": round(100*(e.mean()-base.mean()), 3),
                 "hit_VIX_up_pct": round(100*(e > 0).mean(), 1),
                 "t": round(e.mean()/(e.std(ddof=1)/np.sqrt(len(e))), 2)})
print(pd.DataFrame(rows).to_string(index=False))
print("  (mechanism needs VIX_fwd_cond > VIX_fwd_allday by a real margin)")

# --- 4. cost on SPY
r5 = fwd_lag(px["SPY"], 5, 1)
e5 = (-r5).reindex(epi).dropna()
print("\n4. cost: 1 leg x ~2.5 bps round trip. episode mean %.3f%% = %.1f bps -> %.1fx cost"
      % (100*e5.mean(), 10000*e5.mean(), (10000*e5.mean())/2.5))

# --- 5. support
print("\n5. SUPPORT: SPY dist to 52w high on trigger days: mean %+.2f%% median %+.2f%% | LIVE -0.67%%"
      % (100*sp_d52h.reindex(d).mean(), 100*sp_d52h.reindex(d).median()))
print("   trigger days with SPY within 1%% of its 52w high: %d of %d (%.0f%%)"
      % (int((SPIKE & NEARHI).sum()), int(SPIKE.sum()), 100*(SPIKE & NEARHI).sum()/SPIKE.sum()))
