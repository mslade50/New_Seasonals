"""Stage B1: today's number for every watchlist trigger that needs computing."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

ASOF = pd.Timestamp("2026-08-26")
ENERGY = ["XLE", "XOP", "USO", "COP", "CVX", "VLO", "OXY", "SLB", "EOG", "HAL", "WMB"]
f = load_prices(ENERGY + ["OIH", "^TNX", "XLU", "TLT", "GDX", "GLD", "KRE", "XLF"])

print("=== entry 22: energy names at z10 >= 2.0 (needs 2-3 to arm) ===")
c = 0
for t in ENERGY:
    if t not in f:
        print(f"  {t} MISSING"); continue
    s = f[t]["Close"].dropna(); s = s[s.index <= ASOF]
    z = zscore(s, 10).iloc[-1]
    c += z >= 2.0
    print(f"  {t:<6} z10 {z:+.2f}")
print(f"  COUNT at z10>=2.0: {c} of 11")

print("\n=== entry 24: OIH-minus-XOP 63d spread, PIT trailing-252 percentile ===")
a = f["OIH"]["Close"].dropna(); b = f["XOP"]["Close"].dropna()
j = pd.concat([a, b], axis=1).dropna(); j.columns = ["OIH", "XOP"]
j = j[j.index <= ASOF]
sp = j.OIH.pct_change(63, fill_method=None) - j.XOP.pct_change(63, fill_method=None)
pit = sp.rolling(252).apply(lambda w: (w[:-1] < w[-1]).mean() * 100, raw=True)
print(f"  spread {sp.iloc[-1]*100:+.2f}pp   PIT pctile {pit.iloc[-1]:.2f}  (rung <= 2.5)")

print("\n=== entry 21: ^TNX vs trailing-252 high ===")
s = f["^TNX"]["Close"].dropna(); s = s[s.index <= ASOF]
print(f"  ^TNX {s.iloc[-1]:.3f}  = {100*s.iloc[-1]/s.rolling(252).max().iloc[-1]:.2f}% of its 252d high (rung 99.75)")

print("\n=== entry 26: XLU 21d rank + TLT ranks ===")
for t in ["XLU", "TLT"]:
    s = f[t]["Close"].dropna(); s = s[s.index <= ASOF]
    print(f"  {t:<5} r21 {pct_rank(s,21).iloc[-1]:5.1f}  r5 {pct_rank(s,5).iloc[-1]:5.1f}")

print("\n=== entry 19: KRE complex breadth (5d rank <= 20 share) ===")
s = f["KRE"]["Close"].dropna(); s = s[s.index <= ASOF]
print(f"  KRE r5 {pct_rank(s,5).iloc[-1]:.1f}  r63 {pct_rank(s,63).iloc[-1]:.1f}")

print("\n=== entry 3: GDX/GLD 5d ranks ===")
for t in ["GDX", "GLD"]:
    s = f[t]["Close"].dropna(); s = s[s.index <= ASOF]
    print(f"  {t:<5} r5 {pct_rank(s,5).iloc[-1]:5.1f} (GDX rung >= 95)")
