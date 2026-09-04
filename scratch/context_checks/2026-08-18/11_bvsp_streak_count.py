"""Verify the rarity claim before it is printed.

Drill 10's declustered episode lists suggested only one prior 11-session losing
streak since 2000. Declustering can hide or merge runs, so this counts maximal
runs directly, which is the form the brief will quote.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import close_panel  # noqa: E402

b = close_panel(["^BVSP"])["^BVSP"].dropna()
r = b.pct_change(fill_method=None)
neg = (r < 0).values

runs = []
start = None
for i, x in enumerate(neg):
    if x and start is None:
        start = i
    elif not x and start is not None:
        runs.append((start, i - 1))
        start = None
if start is not None:
    runs.append((start, len(neg) - 1))

R = pd.DataFrame([{"start": b.index[a].date(), "end": b.index[z].date(),
                   "len": z - a + 1,
                   "ret_pct": round(100 * (b.iloc[z] / b.iloc[a - 1] - 1), 2)
                   if a > 0 else np.nan}
                  for a, z in runs])
R = R[R["len"] >= 8].sort_values("len", ascending=False)
print(f"history: {b.index[0].date()} to {b.index[-1].date()}, {len(b)} sessions")
print(f"\nEVERY maximal down-run of 8+ sessions:\n")
print(R.to_string(index=False))

print(f"\n  runs of >= 9 : {int((R['len'] >= 9).sum())}")
print(f"  runs of >= 10: {int((R['len'] >= 10).sum())}")
print(f"  runs of >= 11: {int((R['len'] >= 11).sum())}")
print(f"  longest ever in sample: {int(R['len'].max())}")

cur = R[R["end"] == b.index[-1].date()]
print(f"\n  the live run: {cur.to_dict('records')}")
prior11 = R[(R["len"] >= 11) & (R["end"] != b.index[-1].date())]
print(f"  PRIOR runs of 11+ (excluding the live one): {len(prior11)}")
print(prior11.to_string(index=False) if len(prior11) else "   none")

print("\n" + "=" * 74)
print("what happened after each 8+ run ENDED (the run's last down close)")
print("=" * 74)
rows = []
for a, z in runs:
    L = z - a + 1
    if L < 8 or z + 1 >= len(b):
        continue
    rows.append({
        "ended": b.index[z].date(), "len": L,
        "run_ret_pct": round(100 * (b.iloc[z] / b.iloc[a - 1] - 1), 2) if a > 0 else np.nan,
        "next1_pct": round(100 * (b.iloc[z + 1] / b.iloc[z] - 1), 2),
        "next5_pct": round(100 * (b.iloc[z + 5] / b.iloc[z] - 1), 2) if z + 5 < len(b) else np.nan,
        "next10_pct": round(100 * (b.iloc[z + 10] / b.iloc[z] - 1), 2) if z + 10 < len(b) else np.nan,
    })
D = pd.DataFrame(rows)
print(D.to_string(index=False))
if len(D):
    print(f"\n  next session : mean {D['next1_pct'].mean():+.2f}%, "
          f"{int((D['next1_pct']>0).sum())}-{int((D['next1_pct']<=0).sum())}")
    print(f"  next 5       : mean {D['next5_pct'].mean():+.2f}%, "
          f"{int((D['next5_pct']>0).sum())}-{int((D['next5_pct']<=0).sum())}")
    print(f"  next 10      : mean {D['next10_pct'].mean():+.2f}%, "
          f"{int((D['next10_pct']>0).sum())}-{int((D['next10_pct']<=0).sum())}")
