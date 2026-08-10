"""D5 final -- (a) verify the event->session mapping, (b) redo the whole k x h
grid against the CALENDAR-MATCHED control instead of the all-days control.

If no (k, h) survives the right control, the choice of k stops mattering and
the parent is dead everywhere, not just at the k the calendar forces today.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

px = close_panel(["TLT"])
tl = px["TLT"].dropna()
idx = tl.index
c = tl.values
N = len(c)
ev = load_events()
cpi_dates = ev[ev.event == "cpi"]["date"]

print("=" * 96)
print("A. EVENT -> SESSION MAPPING AUDIT")
print("=" * 96)
pre, holiday, future, ok = 0, [], 0, 0
for x in cpi_dates:
    if x < idx[0]:
        pre += 1
        continue
    if x > idx[-1]:
        future += 1
        continue
    p = int(idx.searchsorted(x, "left"))
    if p < N and idx[p] != x:
        holiday.append((x.date(), idx[p].date()))
    else:
        ok += 1
print(f"  CPI events total {len(cpi_dates)}")
print(f"  before TLT inception {idx[0].date()}: {pre}")
print(f"  after last bar {idx[-1].date()}: {future}")
print(f"  exact trading-day match: {ok}")
print(f"  mapped forward (CPI date not a session): {len(holiday)} {holiday}")

ym = pd.Series(idx.year * 100 + idx.month, index=idx)
tdom = ym.groupby(ym.values).cumcount().values + 1

print("\n" + "=" * 96)
print("B. k x h GRID vs the CALENDAR-MATCHED control (per-entry excess over the")
print("   mean h-return of the same trading-day-of-month, CPI window removed)")
print("=" * 96)
rows = []
for K in range(1, 6):
    for H in (1, 2, 3, 5, 10):
        r = np.full(N, np.nan)
        r[:N - H] = c[H:] / c[:-H] - 1.0
        pos = []
        for x in cpi_dates:
            p = int(idx.searchsorted(x, "left"))
            if 8 <= p < N - 14 and idx[0] <= x <= idx[-1]:
                pos.append(p)
        pos = np.array(sorted(set(pos)))
        entry = pos - K
        win = set()
        for p in pos:
            win.update(range(p - K, p - K + H + 1))
        inw = np.array([i in win for i in range(N)])
        tset = set(int(x) for x in tdom[entry])
        bucket = {}
        for j in tset:
            m = (tdom == j) & ~inw & ~np.isnan(r)
            bucket[j] = r[m].mean() if m.sum() else np.nan
        exc, yrs, mids = [], [], []
        for p in entry:
            if np.isnan(r[p]):
                continue
            exc.append(r[p] - bucket[int(tdom[p])])
            yrs.append(idx[p].year)
            mids.append((idx[p].year % 4) == 2)
        exc = np.array(exc)
        yrs = np.array(yrs)
        mids = np.array(mids)
        w = int((exc > 0).sum())
        raw = r[entry]
        raw = raw[~np.isnan(raw)]
        rows.append({
            "k": K, "h": H, "N": len(exc),
            "raw_pct": round(100 * raw.mean(), 3),
            "tdom_ctrl": round(100 * (raw.mean() - exc.mean()), 3),
            "EXCESS_bps": round(100 * 100 * exc.mean(), 1),
            "hit": round(100 * w / len(exc), 1),
            "sign_p": round(sign_test(w, len(exc)), 3),
            "t": round(exc.mean() / (exc.std(ddof=1) / np.sqrt(len(exc))), 2),
            "ex08_bps": round(100 * 100 * exc[yrs != 2008].mean(), 1),
            "y18p_bps": round(100 * 100 * exc[yrs >= 2018].mean(), 1),
            "mid_bps": round(100 * 100 * exc[mids].mean(), 1),
            "x_cost": round(100 * 100 * exc.mean() / 2.5, 1),
        })
g = pd.DataFrame(rows)
print(g.to_string(index=False))
print("\n  cost bar: an edge needs >=5x a 2.5 bps TLT round trip = >=12.5 bps.")
surv = g[(g.EXCESS_bps >= 12.5) & (g.ex08_bps >= 12.5) & (g.hit >= 55)]
print(f"  cells clearing EXCESS>=12.5bps AND ex-2008>=12.5bps AND hit>=55%: "
      f"{len(surv)}")
if len(surv):
    print(surv.to_string(index=False))
print(f"\n  TODAY'S EXECUTABLE ROW (k=2, MOC tonight):")
print(g[g.k == 2].to_string(index=False))
