"""Live readings for the watchlist verdicts in 00_surface_map.md (2026-09-17 close)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

TK = ["SPY", "IWM", "QQQ", "TLT", "IEF", "LQD", "HYG", "^TNX", "^MOVE", "^VIX", "^VIX3M", "^SKEW",
      "GLD", "GDX", "SLV", "USO", "UNG", "DBC", "UUP", "DX-Y.NYB", "EEM", "FXI", "XLE", "XOP", "COP",
      "CVX", "VLO", "OXY", "SLB", "EOG", "HAL", "WMB", "XLB", "XLC", "XLF", "XLI", "XLK", "XLP",
      "XLU", "XLV", "XLY", "XLRE", "KRE", "SVXY"]
px = close_panel(TK)
last = px.index[-1]
print("last bar", last.date())


def r(t, n):
    return float(pct_rank(px[t].dropna(), n).iloc[-1])


def z(t):
    return float(zscore(px[t].dropna(), 10).iloc[-1])


print(f"VIX {px['^VIX'].iloc[-1]:.2f} VIX3M {px['^VIX3M'].iloc[-1]:.2f} ratio {px['^VIX'].iloc[-1]/px['^VIX3M'].iloc[-1]:.3f}")
print(f"VIX 1d chg {px['^VIX'].pct_change().iloc[-1]*100:+.2f}%  VIX 21d rank {r('^VIX',21):.1f}")
mv = px["^MOVE"].dropna()
print(f"MOVE level pctile 252 {(mv.iloc[-252:] <= mv.iloc[-1]).mean()*100:.1f}  MOVE {mv.iloc[-1]:.1f}")
tnx = px["^TNX"].dropna()
print(f"TNX {tnx.iloc[-1]:.3f} 252max {tnx.iloc[-252:].max():.3f} 252-session chg {(tnx.iloc[-1]-tnx.iloc[-253])*100:+.1f}bp  21d chg {(tnx.iloc[-1]-tnx.iloc[-22])*100:+.1f}bp")
for t in ["SPY", "HYG", "TLT", "IEF", "LQD", "XLE", "USO", "DBC"]:
    s = px[t].dropna()
    print(f"{t}: dist 252 high {(s.iloc[-1]/s.iloc[-252:].max()-1)*100:+.2f}% dist 252 low {(s.iloc[-1]/s.iloc[-252:].min()-1)*100:+.2f}%")
print("SKEW r5", round(r("^SKEW", 5), 1), "DX r21", round(r("DX-Y.NYB", 21), 1), "GDX r5", round(r("GDX", 5), 1), "GLD r5", round(r("GLD", 5), 1))
print("XLU r21", round(r("XLU", 21), 2), "TLT r21", round(r("TLT", 21), 2), "IWM z10", round(z("IWM"), 2))
en = ["XLE", "XOP", "USO", "COP", "CVX", "VLO", "OXY", "SLB", "EOG", "HAL", "WMB"]
zz = {t: round(z(t), 2) for t in en}
print("energy z10 (pitch_lab.zscore):", zz, "count>=2:", sum(v >= 2 for v in zz.values()))
spdr = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "XLRE"]
for t in spdr:
    print(f"  {t} r5 {r(t,5):5.1f} r21 {r(t,21):5.1f} r63 {r(t,63):5.1f} triple10 {r(t,5)<=10 and r(t,21)<=10 and r(t,63)<=10}")
print(f"SPY 1d {px['SPY'].pct_change().iloc[-1]*100:+.2f}%  USO 1d {px['USO'].pct_change().iloc[-1]*100:+.2f}%  TLT 1d {px['TLT'].pct_change().iloc[-1]*100:+.2f}%")
d = pd.read_parquet(Path(__file__).resolve().parents[3] / "data" / "rd2_fragility.parquet")
print("dial tail", d["63d"].rolling(10).mean().tail(3).round(1).to_dict())
