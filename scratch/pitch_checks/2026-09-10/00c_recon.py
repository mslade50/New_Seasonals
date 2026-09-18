"""Reconcile the surface map's headline claims against master_prices directly."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import load_prices, close_panel

px = close_panel(["SPY", "^TNX", "TLT", "IEF", "LQD", "HYG", "DBC", "USO",
                  "XLE", "XOP", "^VIX", "^VIX3M", "UNG", "UUP", "DX-Y.NYB",
                  "EWJ", "EFA", "GLD", "SLV", "XLRE", "XLF", "XLY", "SMH"])
px = px.dropna(how="all")
spy_idx = px["SPY"].dropna().index
print("last SPY bar:", spy_idx[-1].date())
sep = spy_idx[(spy_idx >= "2026-09-01") & (spy_idx <= "2026-09-30")]
print("Sept 2026 sessions so far:", [d.date().isoformat() for d in sep],
      "-> tdom of 2026-09-10 would be", len(sep) + 1, "(09-10 bar not yet in cache)")

for t in ["^TNX", "DBC", "XLE", "XOP", "USO"]:
    s = px[t].reindex(spy_idx).dropna()
    w = s.iloc[-252:]
    print(f"{t:10s} last={s.iloc[-1]:.4f}  252d max={w.max():.4f} "
          f"dist={100*(s.iloc[-1]/w.max()-1):+.3f}%  at252dHigh={s.iloc[-1] >= w.max()-1e-9}")
for t in ["TLT", "IEF", "LQD", "UNG"]:
    s = px[t].reindex(spy_idx).dropna()
    w = s.iloc[-252:]
    print(f"{t:10s} last={s.iloc[-1]:.4f}  252d min={w.min():.4f} "
          f"above252dLow={100*(s.iloc[-1]/w.min()-1):+.3f}%")
s = px["HYG"].reindex(spy_idx).dropna()
print(f"HYG        last={s.iloc[-1]:.4f} 252d max={s.iloc[-252:].max():.4f} "
      f"dist={100*(s.iloc[-1]/s.iloc[-252:].max()-1):+.3f}%")

# VIX on SPY's calendar, per the 2026-09-09 registry correction
v = px["^VIX"].reindex(spy_idx).dropna()
print(f"^VIX on SPY calendar: last={v.iloc[-1]:.2f} 1d={100*(v.iloc[-1]/v.iloc[-2]-1):+.2f}% "
      f"5d={100*(v.iloc[-1]/v.iloc[-6]-1):+.2f}%")
v3 = px["^VIX3M"].reindex(spy_idx).dropna()
print(f"^VIX3M last={v3.iloc[-1]:.2f}  VIX/VIX3M={v.iloc[-1]/v3.iloc[-1]:.4f}")
