"""Live-state recon for 2026-09-03. Numbers the surface map quotes."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import (close_panel, load_prices, load_events, rolling_on_valid,
                       anchor_positions, zscore)
warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)

# --- 1. the payroll anchor geometry -----------------------------------------
px = close_panel(["SPY", "^VIX", "^VIX3M", "SVXY", "UVXY", "^MOVE", "^TNX", "TLT", "IEF"])
cal = px["SPY"].dropna().index
print("freshest SPY bar:", cal[-1].date())
nfp = load_events(["nfp"])["date"]
nxt = [d for d in nfp if d > pd.Timestamp("2026-09-01")][:2]
print("next NFP dates:", [str(d.date()) for d in nxt])
pos, _ = anchor_positions(cal, nfp, -2)
anch = pd.DatetimeIndex([cal[i] for i in pos])
print("is 2026-09-02 a k=-2 NFP anchor?", pd.Timestamp("2026-09-02") in anch)
print("last 3 k=-2 anchors:", [str(d.date()) for d in anch[-3:]])

# --- 2. vix range compression gate ------------------------------------------
vix = px["^VIX"].dropna()
rng21 = vix.rolling(21).max() - vix.rolling(21).min()
rel = rng21 / vix.rolling(21).mean()
RNG = rel.rolling(252).rank(pct=True) * 100
ABS = rng21.rolling(252).rank(pct=True) * 100
print("\nVIX rel-range pctile (last 5):"); print(RNG.tail(5).round(2))
print("VIX abs-range pctile 9/2: %.2f" % ABS.iloc[-1])
print("VIX level %.2f  VIX3M %.2f  contango %.1f%%" % (
    vix.iloc[-1], px['^VIX3M'].dropna().iloc[-1],
    100*(px['^VIX3M'].dropna().iloc[-1]/vix.iloc[-1]-1)))

# --- 3. bond vol state -------------------------------------------------------
mv = px["^MOVE"].dropna()
lvl = mv.rolling(252).rank(pct=True).iloc[-1]*100
r5 = mv.pct_change(5).rolling(252).rank(pct=True).iloc[-1]*100
print("\n^MOVE level %.2f  LEVEL pctile(252) %.1f  5d-RETURN rank(252) %.1f  5d ret %.2f%%"
      % (mv.iloc[-1], lvl, r5, 100*(mv.iloc[-1]/mv.iloc[-6]-1)))
tnx = px["^TNX"].dropna()
print("^TNX %.3f  = 252d max? %s  (252d max %.3f)" % (tnx.iloc[-1], tnx.iloc[-1] >= tnx.rolling(252).max().iloc[-1]-1e-9, tnx.rolling(252).max().iloc[-1]))

# --- 4. SVXY own-series 52w distance (registry: never off a panel) -----------
sv = load_prices(["SVXY"])["SVXY"]["Close"].dropna()
print("\nSVXY own-series: last %.2f  252d max %.2f  dist %.3f%%" %
      (sv.iloc[-1], sv.rolling(252).max().iloc[-1],
       100*(sv.iloc[-1]/sv.rolling(252).max().iloc[-1]-1)))

# --- 5. energy z10 count (watchlist 19) --------------------------------------
ENER = ["XLE","XOP","USO","COP","CVX","VLO","OXY","SLB","EOG","HAL","WMB"]
ep = close_panel(ENER)
print("\nenergy complex z10 (pitch_lab.zscore, 10d):")
for t in ENER:
    s = ep[t].dropna()
    print("  %-5s z10 %+5.2f" % (t, zscore(s, 10).iloc[-1]))

# --- 6. industrial / rail washout --------------------------------------------
IND = ["XLI","NSC","UNP","CSX","DOV","ITW","PH","MMM","HON","SNA","IP","GE","CAT","EMR"]
ip_ = close_panel(IND)
print("\nindustrial complex: 5d rank(252) / z10 / dist 52wh")
for t in IND:
    s = ip_[t].dropna()
    r5_ = s.pct_change(5).rolling(252).rank(pct=True).iloc[-1]*100
    print("  %-5s r5 %5.1f  z10 %+5.2f  52wh %+7.2f%%" %
          (t, r5_, zscore(s,10).iloc[-1], 100*(s.iloc[-1]/s.rolling(252).max().iloc[-1]-1)))

# --- 7. credit / IG complex ---------------------------------------------------
cp = close_panel(["HYG","LQD","IEF","TLT"])
for t in ["HYG","LQD","IEF","TLT"]:
    s = load_prices([t])[t]["Close"].dropna()
    print("%-4s  dist252high %+6.2f%%  dist252low %+6.2f%%" %
          (t, 100*(s.iloc[-1]/s.rolling(252).max().iloc[-1]-1),
              100*(s.iloc[-1]/s.rolling(252).min().iloc[-1]-1)))

# --- 8. EWZ z10 both definitions (registry 2026-09-02) -----------------------
ez = load_prices(["EWZ"])["EWZ"]["Close"].dropna()
print("\nEWZ z10 pitch_lab.zscore %+0.2f  (tape/_metrics_for reports +2.23)" % zscore(ez,10).iloc[-1])
