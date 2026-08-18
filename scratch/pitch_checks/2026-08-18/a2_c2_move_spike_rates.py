"""C2 - ^MOVE one-day spike as a rates-direction signal (long TLT).
Live 2026-08-17: ^MOVE +8.70% (96.7 pctile of its daily moves), LEVEL 75.63 =
43.2 pctile full history / 64.7 of the trailing year. TLT AT its 52w low.

Spike defined on the RETURN MAGNITUDE (not a rank of a rank), per the
2026-08-10 registry note. THE TEST IS GATE ATTRIBUTION against "TLT at its
52w low", which is simultaneously true today.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import numpy as np, pandas as pd

raw = load_prices(["^MOVE", "TLT", "SPY", "^VIX", "^TNX", "IEF"])
mv = raw["^MOVE"]["Close"]
tl = raw["TLT"]["Close"]

# own-series states
mv_1d = mv.pct_change()
mv_lvl_full = mv.expanding(252).apply(lambda a: (a[:-1] < a[-1]).mean()*100, raw=True)
tl_d52 = tl / tl.rolling(252).min() - 1.0          # % above 52w low
tl_r21 = tl.pct_change(21)

idx = mv.index.intersection(tl.index).intersection(raw["SPY"]["Close"].index)
px = pd.DataFrame({"TLT": tl.reindex(idx), "SPY": raw["SPY"]["Close"].reindex(idx),
                   "IEF": raw["IEF"]["Close"].reindex(idx)}).dropna()
idx = px.index
S = mv_1d.reindex(idx); D52 = tl_d52.reindex(idx); LVL = mv_lvl_full.reindex(idx)

SPIKE = (S >= 0.08).fillna(False)
ATLOW = (D52 <= 0.01).fillna(False)        # within 1% of the 52w low
print("live: MOVE 1d %+.2f%%  level-pctile(expanding) %.1f  TLT above 52w low %.3f%%"
      % (100*S.iloc[-1], LVL.iloc[-1], 100*D52.iloc[-1]))
print("spike days %d | TLT-at-52w-low days %d | BOTH %d  (of %d sessions)"
      % (SPIKE.sum(), ATLOW.sum(), (SPIKE & ATLOW).sum(), len(idx)))

for H in (3, 5, 10):
    ret = fwd_lag(px["TLT"], H, 1)
    v = ret.notna()
    rows = []
    for lab, m in (("CTRL-b all days", pd.Series(True, index=idx)),
                   ("MOVE spike >=8% ONLY", SPIKE),
                   ("TLT at 52w low ONLY", ATLOW),
                   ("spike AND at-low (live)", SPIKE & ATLOW),
                   ("spike AND NOT at-low", SPIKE & ~ATLOW),
                   ("at-low AND NO spike", ATLOW & ~SPIKE)):
        d = idx[m.values & v.values]
        rows.append(summarize(ret.loc[d].values, f"{lab} (N={len(d)})"))
    show(rows, f"GATE ATTRIBUTION both ways, LONG TLT h={H} lag=1")

# --- full battery on the spike alone, h=5
H = 5
mask = SPIKE
battery(px, mask, [("TLT", 1.0)], H, "C2 MOVE spike >=8% -> long TLT", 2.0,
        variants={"spike >=6%": (S >= 0.06).fillna(False),
                  "spike >=7%": (S >= 0.07).fillna(False),
                  "spike >=8%": SPIKE,
                  "spike >=8.70% (live)": (S >= 0.0870).fillna(False),
                  "spike >=10%": (S >= 0.10).fillna(False),
                  "spike >=12%": (S >= 0.12).fillna(False)},
        min_gap=10)

# --- horizon scan
d = idx[SPIKE.values]
show(horizon_scan(px, d, [("TLT", 1.0)], hs=(1,2,3,5,10), min_gap=10),
     "horizon scan, spike -> long TLT (episodes, min_gap 10td)")

# --- SUPPORT: is today inside the historical trigger population?
dd = idx[SPIKE.values]
print("\n=== SUPPORT: what did a historical MOVE-spike day look like? ===")
print("  MOVE LEVEL pctile (expanding, own series): trigger mean %.1f median %.1f | LIVE %.1f | pctile-of-live %.0f"
      % (LVL.reindex(dd).mean(), LVL.reindex(dd).median(), LVL.iloc[-1],
         100*(LVL.reindex(dd) < LVL.iloc[-1]).mean()))
print("  TLT %% above 52w low: trigger mean %+.2f%% median %+.2f%% | LIVE %+.3f%%"
      % (100*D52.reindex(dd).mean(), 100*D52.reindex(dd).median(), 100*D52.iloc[-1]))
print("  TLT 21d ret:        trigger mean %+.2f%% median %+.2f%% | LIVE %+.2f%%"
      % (100*tl_r21.reindex(dd).mean(), 100*tl_r21.reindex(dd).median(), 100*tl_r21.iloc[-1]))
# split by MOVE level regime -- the level/rank trap
lo = LVL.reindex(dd) <= 50
ret5 = fwd_lag(px["TLT"], 5, 1)
show([summarize(ret5.loc[dd[lo.values]].values, "spike from a LOW level (<=50 pctile, LIVE=43)"),
      summarize(ret5.loc[dd[~lo.values]].values, "spike from a HIGH level (>50 pctile)")],
     "SUPPORT split: spike from which LEVEL? (h=5)")
