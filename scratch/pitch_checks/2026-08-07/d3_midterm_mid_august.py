"""D3 - "Midterm mid-August": SPY/IWM entered MOC at the Nth trading session of August,
held 10 and 21 td, midterm years (year %% 4 == 2) vs all years, 2000-2026.

NOTE the spec says "7th trading session"; 2026-08-07 is actually the 5TH session of
August 2026 (verified in d0_verify_today.py). Both N=5 and N=7 are tested, plus N=3..9
as the sensitivity grid, because the entry-session index IS the only free parameter here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _common import *  # noqa: F401,F403

TICKERS = ["SPY", "IWM"]
H = [10, 21]
px = load_prices(TICKERS)
closes = {t: px[t]["Close"][px[t].index <= "2026-08-06"] for t in TICKERS}


def aug_entries(s: pd.Series, nth: int) -> pd.DataFrame:
    """Nth August trading session of each year (from the instrument's own sessions)."""
    rows = []
    for y, g in s.groupby(s.index.year):
        a = g[g.index.month == 8]
        if len(a) < nth:
            continue
        rows.append({"year": y, "entry": a.index[nth - 1]})
    return pd.DataFrame(rows)


def cell(t: str, nth: int, h: int, midterm: bool | None) -> tuple[pd.DatetimeIndex, np.ndarray]:
    s = closes[t]
    pos = pd.Series(range(len(s)), index=s.index)
    e = aug_entries(s, nth)
    if midterm is True:
        e = e[e.year % 4 == 2]
    elif midterm is False:
        e = e[e.year % 4 != 2]
    d, r = [], []
    for row in e.itertuples():
        p = pos[row.entry]
        if p + h < len(s):
            d.append(row.entry)
            r.append(s.iloc[p + h] / s.iloc[p] - 1.0)
    return pd.DatetimeIndex(d), np.asarray(r)


print("### D3 midterm mid-August | entry MOC at Nth Aug session, exit MOC +h td")
for t in TICKERS:
    print(f"  {t} history starts {closes[t].index[0].date()}")

# ---------- 1. pattern vs TWO controls ----------
for t in TICKERS:
    rows = []
    s = closes[t]
    for nth in (5, 7):
        for h in H:
            d, r = cell(t, nth, h, True)
            rows.append(summarize(r, f"{t} N{nth} h{h} MIDTERM"))
            d2, r2 = cell(t, nth, h, None)
            rows.append(summarize(r2, f"{t} N{nth} h{h} ALL YEARS"))
            d3, r3 = cell(t, nth, h, False)
            rows.append(summarize(r3, f"{t} N{nth} h{h} NON-midterm"))
    for h in H:
        u = (s.shift(-h) / s - 1.0).dropna()
        rows.append(summarize(u.values, f"ctrl A: {t} any day h{h}"))
        rows.append(summarize(u[u.index.month == 8].values, f"ctrl B: {t} any AUG day h{h}"))
        um = u[u.index.year % 4 == 2]
        rows.append(summarize(um.values, f"ctrl C: {t} any MIDTERM day h{h}"))
        rows.append(summarize(um[um.index.month == 8].values, f"ctrl D: {t} midterm AUG day h{h}"))
    show(rows, f"1. {t} conditional vs controls")

# ---------- 2/3. per-episode detail, drop-best, bootstrap, era ----------
for t in TICKERS:
    for nth in (5, 7):
        for h in H:
            d, r = cell(t, nth, h, True)
            if not len(r):
                continue
            print(f"\n{t} N{nth} h{h} MIDTERM episodes (N={len(r)}, non-overlapping by "
                  f"construction, 1/4 yrs):")
            print("   " + ", ".join(f"{x.year}:{100*v:+.2f}%" for x, v in zip(d, r)))
            s = summarize(r, f"{t} N{nth} h{h} midterm")
            s["boot_P<=0"] = bootstrap_p_le0(r)
            s["drop_best_mean"] = 100 * np.delete(r, np.argmax(r)).mean()
            s["drop_worst_mean"] = 100 * np.delete(r, np.argmin(r)).mean()
            s["drop2best_mean"] = 100 * np.sort(r)[:-2].mean() if len(r) > 2 else np.nan
            rows = [s] + era_split(d, r)
            show(rows, "")

# ---------- 4. sensitivity: entry session index 3..9, and horizon ----------
for t in TICKERS:
    rows = []
    for nth in range(3, 10):
        for h in (5, 10, 21):
            d, r = cell(t, nth, h, True)
            s = summarize(r) if len(r) else {}
            da, ra = cell(t, nth, h, None)
            rows.append({"tkr": t, "Nth": nth, "h": h, "mid_n": len(r),
                         "mid_mean": s.get("mean_pct", np.nan), "mid_t": s.get("t", np.nan),
                         "mid_hit": s.get("hit", np.nan),
                         "all_n": len(ra),
                         "all_mean": summarize(ra).get("mean_pct", np.nan) if len(ra) else np.nan,
                         "all_t": summarize(ra).get("t", np.nan) if len(ra) else np.nan})
    show(rows, f"4. {t} sensitivity: entry session 3..9 x horizon (midterm vs all-years)")

# ---------- 5/6. cost + CPI split ----------
cpi = set(load_events(["cpi"])["date"])
rows = []
for t in TICKERS:
    for nth in (5, 7):
        for h in H:
            d, r = cell(t, nth, h, True)
            if not len(r):
                continue
            s = closes[t]
            pos = pd.Series(range(len(s)), index=s.index)
            fl = np.array([any(x < c <= s.index[pos[x] + h] for c in cpi) for x in d])
            rows.append(summarize(r[fl], f"{t} N{nth} h{h} CPI inside"))
            rows.append(summarize(r[~fl], f"{t} N{nth} h{h} no CPI"))
show(rows, "6. CPI-in-window split (midterm cells)")

print("\n5. cost sanity (~1 bp round trip):")
for t in TICKERS:
    for nth in (5, 7):
        for h in H:
            d, r = cell(t, nth, h, True)
            if not len(r):
                continue
            u = (closes[t].shift(-h) / closes[t] - 1.0).dropna()
            edge = 100 * r.mean() - 100 * u[u.index.year % 4 == 2].mean()
            print(f"   {t} N{nth} h{h}: raw {100*r.mean():+.2f}% ({100*r.mean()/0.01:.0f}x cost), "
                  f"edge over midterm-day control {edge:+.2f}pp ({edge/0.01:.0f}x cost)")
