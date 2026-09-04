"""The engine's `E:seasonal_doy` midterm cell for Sep-04 says TLT went 0-for-5
down (mean -1.23% at h1, -1.72% at h5) and ^TNX 6-0 up at h5 (+3.58%).

N of 5 is anecdote territory and a -1.23% average daily move in TLT is large
enough to be suspicious, so: pull the actual episodes, confirm they are five
distinct years and not one window counted five times, and check the all-years
cell that surrounds them.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

px = close_panel(["TLT", "IEF", "^TNX", "^GSPC"])
ref = px["^GSPC"].dropna().index
target = pd.Timestamp("2026-09-04")


def doy_anchors(years, window=2):
    """One pick per year: the session closest to the same calendar day."""
    out = []
    for y in years:
        want = pd.Timestamp(year=y, month=target.month, day=target.day)
        cand = ref[(ref >= want - pd.Timedelta(days=6)) &
                   (ref <= want + pd.Timedelta(days=6))]
        if len(cand) == 0:
            continue
        pick = min(cand, key=lambda d: abs((d - want).days))
        # anchor is the session BEFORE, so h1 is the analogue session itself
        i = ref.get_loc(pick)
        if i > 0:
            out.append(ref[i - 1])
    return pd.DatetimeIndex(sorted(out))


years_all = [y for y in range(1999, 2026)]
years_mid = [y for y in years_all if y % 4 == 2]
print("midterm years used:", years_mid)

for label, yrs in (("all years", years_all), ("midterm only", years_mid)):
    a = doy_anchors(yrs)
    print(f"\n=== {label}: {len(a)} anchors ===")
    print("  anchors:", [str(d.date()) for d in a])
    for tick in ("TLT", "IEF", "^TNX", "^GSPC"):
        for h in (1, 5):
            f = fwd_ret(px[tick].dropna(), h).reindex(a).dropna()
            if len(f) < 3:
                continue
            r = summarize(f.to_numpy())
            up = int((f > 0).sum())
            print(f"  {tick:6s} h{h} n={r['n']:3d} mean={r['mean_pct']:+7.3f}% "
                  f"med={r['median_pct']:+7.3f}% {up}-{len(f)-up} up "
                  f"p={sign_test(max(up,len(f)-up),len(f)):.4f}")

print("\nmidterm TLT episodes, session by session:")
a = doy_anchors(years_mid)
f1 = fwd_ret(px["TLT"].dropna(), 1)
f5 = fwd_ret(px["TLT"].dropna(), 5)
for d in a:
    print(f"  anchor {d.date()}  h1 {100*f1.get(d, float('nan')):+6.2f}%  "
          f"h5 {100*f5.get(d, float('nan')):+6.2f}%")

# Is a -1.2% daily TLT move even unusual? Give the reader the distribution.
t = px["TLT"].dropna().pct_change().dropna()
print(f"\nTLT daily move distribution: sd {100*t.std():.2f}%, "
      f"P(day <= -1.2%) = {100*(t <= -0.012).mean():.1f}%")
