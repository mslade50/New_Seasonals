"""D4b - the decisive controls for the VIX-expiry week.

(a) Is it just mid-month drift? Control = SPY h3 return by trading-day-of-month position.
(b) Paired within-month control: cell return minus the mean h3 return of every OTHER
    session in the same calendar month. Kills the "bull sample" objection entirely.
(c) Per-year concentration / drop-best-year.
(d) Can it be traded TODAY? Entry is 3 sessions before 2026-08-19 = 2026-08-14, which is
    5 sessions AFTER today's close. Test the lead that WOULD correspond to a 08-07 entry.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

sys.path.insert(0, str(Path(__file__).parent))
from _common import *  # noqa: F401,F403

BD = CustomBusinessDay(calendar=USFederalHolidayCalendar())
px = load_prices(["SPY"])["SPY"]
spy = px["Close"][px.index <= "2026-08-06"]
idx = spy.index
pos = pd.Series(range(len(idx)), index=idx)
ev = load_events()
vix = ev[ev.event == "vix_expiry"].date


def snap(d):
    if d > idx[-1]:
        return None
    s = idx[idx <= d]
    return s[-1] if len(s) and (d - s[-1]).days <= 4 else None


def cells(lead: int) -> pd.DataFrame:
    rows = []
    for d in vix:
        b = snap(d)
        if b is None:
            continue
        pb = pos[b]
        if pb - lead < 0:
            continue
        rows.append({"anchor": b, "entry": idx[pb - lead], "year": b.year, "month": b.month,
                     "ret": spy.iloc[pb] / spy.iloc[pb - lead] - 1.0})
    return pd.DataFrame(rows)


vx = cells(3)
h3 = (spy.shift(-3) / spy - 1.0)

# ---------- (a) trading-day-of-month profile ----------
tdom = pd.Series(index=idx, dtype=float)
for (y, m), g in spy.groupby([idx.year, idx.month]):
    tdom.loc[g.index] = np.arange(1, len(g) + 1)
prof = pd.DataFrame({"tdom": tdom, "h3": h3}).dropna()
g = prof.groupby("tdom").h3.agg(["mean", "count"])
g["mean_pct"] = 100 * g["mean"]
print("### (a) SPY h3 forward return by trading-day-of-month (entry day position)")
print(g[["mean_pct", "count"]].head(23).round(3).to_string())
entry_tdom = tdom.reindex(vx.entry).values
print(f"\nvix-week ENTRY tdom distribution: "
      f"{pd.Series(entry_tdom).value_counts().sort_index().to_dict()}")
mid = prof[prof.tdom.isin(pd.Series(entry_tdom).dropna().unique())]
print(f"h3 mean on ALL days at those same tdom positions: {100*mid.h3.mean():+.3f}% "
      f"(n={len(mid)})  vs vix-week cells {100*vx.ret.mean():+.3f}%")
print(f"  -> mid-month-position control eats "
      f"{100*mid.h3.mean()/ (100*vx.ret.mean()):.0%} of the raw cell mean")

# ---------- (b) paired within-month control ----------
paired = []
for r in vx.itertuples():
    same = h3[(idx.year == r.entry.year) & (idx.month == r.entry.month)].dropna()
    same = same[same.index != r.entry]
    if len(same) >= 5:
        paired.append(r.ret - same.mean())
paired = np.asarray(paired)
s = summarize(paired, "vix-week MINUS same-month mean h3 (paired excess)")
s["boot_P<=0"] = bootstrap_p_le0(paired)
show([s], "(b) paired within-month excess return")
for cut in ("2013-01-01", "2018-01-01"):
    d = pd.DatetimeIndex(vx.entry[:len(paired)])
    show(era_split(d, paired, cut=cut), f"(b) paired excess, era cut {cut[:4]}")

# ---------- (c) per-year concentration ----------
vx["yr"] = vx.year
yr = vx.groupby("yr").ret.mean() * 100
print("\n### (c) per-year mean (%):")
print(yr.round(2).to_string())
pos_yr = (yr > 0).sum()
print(f"positive years: {pos_yr}/{len(yr)}")
for k in (1, 2, 3):
    best = yr.nlargest(k).index
    sub = vx[~vx.yr.isin(best)]
    print(f"drop best {k} year(s) {list(best)}: mean {100*sub.ret.mean():+.3f}%, "
          f"t {summarize(sub.ret.values)['t']:.2f}, n={len(sub)}")

# ---------- (d) tradeable today? ----------
print("\n### (d) can this be entered at TODAY's close (2026-08-07)?")
nxt = pd.Timestamp("2026-08-19")
sess = pd.date_range("2026-08-07", nxt, freq=BD)
print(f"  sessions 08-07 .. 08-19 inclusive: {[d.strftime('%m-%d') for d in sess]}")
print(f"  entry for a 3-session lead is {sess[-4].date()} = "
      f"{len(sess)-4} sessions AFTER today's close -> NOT a today trade.")
lead_today = len(sess) - 1
print(f"  a 2026-08-07 MOC entry held to the 08-19 settle is lead={lead_today}.")
w = cells(lead_today)
u = (spy.shift(-lead_today) / spy - 1.0).dropna()
s = summarize(w.ret.values, f"lead={lead_today} (a TODAY entry)")
s["boot_P<=0"] = bootstrap_p_le0(w.ret.values)
s["vs_ctrl_pp"] = s["mean_pct"] - 100 * u.mean()
show([s, summarize(u.values, f"ctrl: SPY any day h{lead_today}")], "(d) the actual today-trade")
show(era_split(pd.DatetimeIndex(w.entry), w.ret.values), "(d) era split of the today-trade")
