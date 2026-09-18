"""Drill 01 — the post-Labor-Day session specifically, not the pooled holiday cell.

Engine gave E:holiday_post pooling every market closure: ^VIX +3.83% (t=7.10),
SPY dead flat. Tuesday 2026-09-08 is the post-Labor-Day session. Labor Day is
the only three-day weekend landing in September, the weakest seasonal month,
so the pooled cell mixes it with Thanksgiving, July 4 and New Year.

Anchor convention: the session BEFORE the event, so h1 is the post-holiday
session's own close-to-close move (Friday close -> Tuesday close).

NOTE the calendar trap this rewrite fixes: FX and futures TRADE on US market
holidays, so a union-of-tickers index has no Labor Day gap at all and naive
detection lands on the holiday session itself. Everything below is reindexed
onto the ^GSPC (NYSE) session calendar first, which makes the FX number a
genuine Friday-close-to-Tuesday-close move that skips the thin Monday bar.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    close_panel, fwd_ret, summarize, era_split, cluster_note, sign_test, show,
)

SUBJECTS = ["^GSPC", "SPY", "QQQ", "IWM", "^VIX", "GC=F", "JPY=X", "TLT", "HYG"]
raw = close_panel(SUBJECTS)
nyse = raw["^GSPC"].dropna().index          # the real NYSE session calendar
px = raw.reindex(nyse)
print(f"NYSE calendar {nyse[0].date()} .. {nyse[-1].date()}  n={len(nyse)}")


def labor_days(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """First Monday of September, per year."""
    out = []
    for yr in sorted(set(index.year)):
        sep = pd.Timestamp(yr, 9, 1)
        out.append(sep + pd.Timedelta(days=(7 - sep.weekday()) % 7))
    return pd.DatetimeIndex(out)


ld = labor_days(nyse)
# confirm the market really was closed on each one
open_on_ld = [d for d in ld if d in set(nyse)]
print(f"Labor Mondays {len(ld)}, of which NYSE was OPEN on {len(open_on_ld)} "
      f"(expect 0): {[str(d.date()) for d in open_on_ld]}")

pos = pd.Series(range(len(nyse)), index=nyse)
# anchor = last session strictly before Labor Day (the Friday)
anc_ld, post_ld = [], []
for d in ld:
    before = nyse[nyse < d]
    after = nyse[nyse > d]
    if len(before) and len(after):
        anc_ld.append(before[-1])
        post_ld.append(after[0])
anc_ld = pd.DatetimeIndex(anc_ld)
post_ld = pd.DatetimeIndex(post_ld)
print(f"post-Labor-Day sessions: {len(post_ld)} "
      f"{post_ld[0].date()} .. {post_ld[-1].date()}")
gap = np.array([(b - a).days for a, b in zip(anc_ld, post_ld)])
print(f"anchor->post calendar gap days: {sorted(set(gap.tolist()))} (4 = normal)")

# every OTHER weekday closure, for the contrast the engine pooled
gap_bd = np.array([len(pd.bdate_range(p, c)) - 1
                   for p, c in zip(nyse[:-1], nyse[1:])])
post_all = pd.DatetimeIndex(nyse[1:][gap_bd > 1])
anc_all = pd.DatetimeIndex([nyse[pos[d] - 1] for d in post_all])
anc_other = anc_all.difference(anc_ld)
print(f"all post-holiday anchors {len(anc_all)}, non-Labor-Day {len(anc_other)}")

for tkr in SUBJECTS:
    s = px[tkr].dropna()
    f1, f5 = fwd_ret(s, 1), fwd_ret(s, 5)
    rows = []
    for label, anc in (("post-Labor-Day", anc_ld),
                       ("other holidays", anc_other),
                       ("all holidays", anc_all)):
        v = f1.reindex(anc).dropna().values
        if len(v):
            rows.append(summarize(v, label))
    base = f1.dropna()
    rows.append(summarize(base.values, "all days (control)"))
    rows.append(summarize(base[base.index.month == 9].values, "all Sept days"))
    show(rows, f"=== {tkr} h1 (close-to-close) ===")

    v_ld = f1.reindex(anc_ld).dropna()
    if len(v_ld) >= 5:
        w, n = int((v_ld.values > 0).sum()), len(v_ld)
        print(f"  record {w}-{n - w} up, sign p {sign_test(w, n):.4f}")
        for e in era_split(v_ld.index, v_ld.values):
            print(f"    era {e['label']}: n={e['n']} mean {e['mean_pct']:+.3f}% "
                  f"hit {e['hit']:.1f}")
        print(f"    conc: {cluster_note(v_ld.index, v_ld.values)}")
        v5 = f5.reindex(anc_ld).dropna()
        print(f"    h5 (the holiday-shortened week): n={len(v5)} "
              f"mean {100 * v5.mean():+.3f}% hit {100 * (v5.values > 0).mean():.1f}")
        mid = v_ld[v_ld.index.year % 4 == 2]
        if len(mid):
            wm = int((mid.values > 0).sum())
            print(f"    MIDTERM n={len(mid)} mean {100 * mid.mean():+.3f}% "
                  f"record {wm}-{len(mid) - wm} up sign p {sign_test(wm, len(mid)):.4f}")
            print(f"      {[(str(d.date()), round(100 * x, 2)) for d, x in mid.items()]}")
    print()
