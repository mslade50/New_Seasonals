"""Drill 01b — is the post-Labor-Day VIX pop just weekend decay unwinding?

Drill 01 found ^VIX +6.28% on the post-Labor-Day session, 22-4 up, vs +3.55%
for other holidays and +0.26% on all days. VIX has a well-known calendar-decay
pattern: implied vol bleeds across non-trading days and re-prices on the next
open. A three-day weekend is one extra calendar day, so SOME of this is
mechanical and the brief has to say which part.

Control ladder, all in September where possible:
  ordinary Monday (3 calendar days of decay)  ->  post-Labor-Day (4 days)
If the Labor Day number is just the extra day, ordinary Mondays should show a
proportionate pop. If it is bigger than that, something else is in it.

Also nails down the S&P side: the exact sign-test p for the down record, and
the year-by-year 2018+ list behind the 0-for-8 claim.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import close_panel, fwd_ret, summarize, sign_test, show  # noqa: E402

raw = close_panel(["^GSPC", "SPY", "^VIX", "^VIX3M"])
nyse = raw["^GSPC"].dropna().index
px = raw.reindex(nyse)


def labor_anchors(index):
    anc, post = [], []
    for yr in sorted(set(index.year)):
        sep = pd.Timestamp(yr, 9, 1)
        ld = sep + pd.Timedelta(days=(7 - sep.weekday()) % 7)
        b, a = index[index < ld], index[index > ld]
        if len(b) and len(a):
            anc.append(b[-1])
            post.append(a[0])
    return pd.DatetimeIndex(anc), pd.DatetimeIndex(post)


anc_ld, post_ld = labor_anchors(nyse)
nxt = pd.Series(nyse[1:], index=nyse[:-1])          # anchor -> following session
cal_gap = pd.Series([(b - a).days for a, b in zip(nyse[:-1], nyse[1:])],
                    index=nyse[:-1])

# ordinary weekend anchors: Friday -> Monday, 3 calendar days, no holiday
ord_wknd = cal_gap.index[(cal_gap.values == 3) & (cal_gap.index.weekday == 4)]
ord_wknd_sep = ord_wknd[ord_wknd.month == 9]
# ordinary mid-week anchors: 1 calendar day
ord_mid = cal_gap.index[cal_gap.values == 1]

print("=== ^VIX decay ladder: h1 from the anchor close ===")
f1v = fwd_ret(px["^VIX"].dropna(), 1)
rows = [
    summarize(f1v.reindex(anc_ld).dropna().values, "post-Labor-Day (4 cal days)"),
    summarize(f1v.reindex(ord_wknd_sep).dropna().values, "ordinary Sept weekend (3)"),
    summarize(f1v.reindex(ord_wknd).dropna().values, "ordinary weekend, any month (3)"),
    summarize(f1v.reindex(ord_mid).dropna().values, "ordinary mid-week (1)"),
]
show(rows, "^VIX")
for lbl, anc in (("post-Labor-Day", anc_ld), ("ordinary Sept weekend", ord_wknd_sep),
                 ("ordinary weekend", ord_wknd)):
    v = f1v.reindex(anc).dropna()
    w, n = int((v.values > 0).sum()), len(v)
    print(f"  {lbl:26s} {w}-{n - w} up, sign p(up) {sign_test(w, n):.5f}")

print("\n=== ^VIX3M (same ladder, longer tenor = less decay leverage) ===")
f1v3 = fwd_ret(px["^VIX3M"].dropna(), 1)
show([summarize(f1v3.reindex(anc_ld).dropna().values, "post-Labor-Day"),
      summarize(f1v3.reindex(ord_wknd).dropna().values, "ordinary weekend")], "^VIX3M")

print("\n=== S&P: the down record ===")
f1s = fwd_ret(px["^GSPC"].dropna(), 1)
v = f1s.reindex(anc_ld).dropna()
w, n = int((v.values > 0).sum()), len(v)
print(f"  full sample {post_ld[0].year}-{post_ld[-1].year}: {w}-{n - w} up, "
      f"mean {100 * v.mean():+.3f}%, median {100 * v.median():+.3f}%")
print(f"  sign p(down >= {n - w}) = {sign_test(n - w, n):.5f}")
print(f"  ordinary Sept weekend control: "
      f"{summarize(f1s.reindex(ord_wknd_sep).dropna().values, 'ctrl')}")
print("\n  year by year:")
for d, x in v.items():
    yr = nxt[d].year
    print(f"    {yr}  anchor {d.date()} -> post {nxt[d].date()}  {100 * x:+.2f}%"
          f"{'   [midterm]' if yr % 4 == 2 else ''}")
post18 = v[v.index >= "2018-01-01"]
w18 = int((post18.values > 0).sum())
print(f"\n  2018+: {w18}-{len(post18) - w18} up, mean {100 * post18.mean():+.3f}%, "
      f"sign p(down >= {len(post18) - w18}) = {sign_test(len(post18) - w18, len(post18)):.5f}")
print(f"  median of the 2018+ moves: {100 * post18.median():+.3f}% "
      f"(worst {100 * post18.min():+.2f}%, best {100 * post18.max():+.2f}%)")
print("  -> if the mean is episode-carried but every sign is negative, the RECORD "
      "is the honest claim, not the mean.")
