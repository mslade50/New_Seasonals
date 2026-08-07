"""Probe: metals/commodity coverage + does each trigger FIRE on 2026-08-06? Read-only."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import *  # noqa

import pandas as pd

TICK = ["GLD", "SLV", "GDX", "UNG", "USO", "DBC", "SPY"]
px = load_prices(TICK)
for t in TICK:
    if t in px:
        g = px[t]
        print(f"{t:5s} n={len(g):6d}  {g.index.min().date()} .. {g.index.max().date()}  last_close={g['Close'].iloc[-1]:.4f}")
    else:
        print(f"{t:5s} MISSING")

P = close_panel(["GLD", "SLV", "GDX", "UNG", "USO"])
print("\npanel", P.shape, P.index.min().date(), P.index.max().date())
asof = P.index[-1]
print("as-of:", asof.date())

r = lambda t, n: P[t].pct_change(n).loc[asof] * 100

print("\n--- recomputed tape (pct) ---")
for t in ["GLD", "SLV", "GDX", "UNG", "USO"]:
    s = P[t].dropna()
    hi52 = s.rolling(252).max().loc[asof]
    lo52 = s.rolling(252).min().loc[asof]
    last = s.loc[asof]
    print(f"{t:5s} last={last:8.3f} r5={r(t,5):7.2f} r21={r(t,21):7.2f} r63={r(t,63):7.2f} "
          f"r252={r(t,252):7.2f} z10={zscore(s,10).loc[asof]:6.2f} "
          f"dist52wHi={100*(last/hi52-1):7.2f} dist52wLo={100*(last/lo52-1):7.2f} "
          f"vsSMA200={100*(last/s.rolling(200).mean().loc[asof]-1):6.2f}")

print("\n--- TRIGGERS on 2026-08-06 close ---")
c5 = (P["GDX"].pct_change(21) - P["GLD"].pct_change(21)).loc[asof] * 100
print(f"C5 GDX21 - GLD21 = {c5:+.2f} pp   (need >= +8.00)  FIRES={c5 >= 8.0}")

c6a = (P["SLV"].pct_change(63) - P["GLD"].pct_change(63)).loc[asof] * 100
c6b = P["SLV"].pct_change(5).loc[asof] * 100
print(f"C6 SLV63 - GLD63 = {c6a:+.2f} pp (need <= -8.00), SLV r5 = {c6b:+.2f}% (need > 0)  "
      f"FIRES={(c6a <= -8.0) and (c6b > 0)}")

u = P["UNG"].dropna()
lo52u = u.rolling(252).min().loc[asof]
d = 100 * (u.loc[asof] / lo52u - 1)
mon = asof.month
print(f"C7 UNG dist above 52w low = {d:.3f}% (need <= 1.0), month={mon} (need 7-9)  "
      f"FIRES={(d <= 1.0) and (7 <= mon <= 9)}")

g = P["GDX"].dropna()
g21 = g.pct_change(21).loc[asof] * 100
g63rk = pct_rank(g, 63).loc[asof]
print(f"C8 GDX r21 = {g21:+.2f}% (need >= +12), 63d ret rank = {g63rk:.1f} (need < 30)  "
      f"FIRES={(g21 >= 12.0) and (g63rk < 30)}")
