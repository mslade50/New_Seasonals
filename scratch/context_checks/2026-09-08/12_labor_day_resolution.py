"""The 2026-09-06 brief led on VIX rising after Labor Day (22 of 26 since 2000,
mean +6.28%). That cell resolved TODAY. Score it, and do it on the last REAL
session, because ^VIX carries a phantom 2026-09-07 bar the index was closed for.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
from pitch_lab import load_prices, summarize, sign_test

px = load_prices(["^VIX", "^GSPC"])
vix = px["^VIX"]["Close"].dropna()
gspc = px["^GSPC"]["Close"].dropna()

# ^GSPC is the honest trading calendar: it has no holiday bars.
cal = gspc.index
print(f"^GSPC calendar {cal.min().date()} -> {cal.max().date()}, {len(cal)} sessions")
print(f"^VIX rows not on that calendar (phantom bars): "
      f"{len(vix.index.difference(cal))} -> "
      f"{[str(d.date()) for d in vix.index.difference(cal)][-6:]}")

# Labor Day = first Monday of September. The session AFTER it on the real calendar.
rows = []
for y in range(2000, 2027):
    sept = pd.date_range(f"{y}-09-01", f"{y}-09-08", freq="D")
    labor = [d for d in sept if d.dayofweek == 0][0]
    after = cal[cal > labor]
    if len(after) == 0:
        continue
    d1 = after[0]
    prev = cal[cal < labor]
    if len(prev) == 0:
        continue
    d0 = prev[-1]
    if d0 not in vix.index or d1 not in vix.index:
        continue
    r = vix.loc[d1] / vix.loc[d0] - 1.0
    rg = gspc.loc[d1] / gspc.loc[d0] - 1.0
    rows.append({"year": y, "prev": str(d0.date()), "after": str(d1.date()),
                 "vix0": round(float(vix.loc[d0]), 2),
                 "vix1": round(float(vix.loc[d1]), 2),
                 "vix_ret": r, "gspc_ret": rg})

df = pd.DataFrame(rows)
print(f"\n{len(df)} post-Labor-Day sessions scored on the real calendar")
print(df.assign(vix_pct=(100*df.vix_ret).round(2),
                gspc_pct=(100*df.gspc_ret).round(2))
        .drop(columns=["vix_ret", "gspc_ret"]).to_string(index=False))

v = df["vix_ret"].values
up = int((v > 0).sum())
print(f"\nVIX record: {up}-{len(v)-up} up, mean {100*v.mean():+.2f}%, "
      f"median {100*np.median(v):+.2f}%, sign p {sign_test(up, len(v)):.4f}")
prior = df[df.year < 2026]["vix_ret"].values
pu = int((prior > 0).sum())
print(f"through 2025 only: {pu}-{len(prior)-pu}, mean {100*prior.mean():+.2f}%")
print(f"2026 realised: {100*df[df.year==2026]['vix_ret'].iloc[0]:+.2f}%")

g = df["gspc_ret"].values
gu = int((g > 0).sum())
print(f"\n^GSPC same sessions: {gu}-{len(g)-gu} up, mean {100*g.mean():+.2f}%, "
      f"sign p(down) {sign_test(len(g)-gu, len(g)):.4f}")
print(f"2026 realised ^GSPC: {100*df[df.year==2026]['gspc_ret'].iloc[0]:+.2f}%")

print("\nWhat the state file says vs the real calendar:")
print(f"  state file ^VIX ret_1d: +2.75%  (uses the phantom 2026-09-07 bar at 15.30)")
print(f"  real last session 2026-09-04 close 14.53 -> 2026-09-08 close 15.72 = "
      f"{100*(15.72/14.53-1):+.2f}%")
