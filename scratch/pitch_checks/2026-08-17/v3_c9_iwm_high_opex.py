"""C9 round 1 -- IWM at a 52w high while SPY is not, into August opex.

HONESTY DECLARATION, up front, per the brief:
The pitched MECHANISM is dealer positioning / index rebalancing into opex.
This repo has NO dealer gamma history, NO options open-interest history and
NO futures roll positioning. data/option_surface_history.parquet and
data/option_positioning_history.parquet begin 2026-08-05 (see section 0),
which is 9 sessions, so nothing about gamma can be falsified here at all.
What IS measurable is the PRICE-STATE cell and the IWM-minus-SPY relative
leg. This script measures those and nothing else, and the mechanism half of
the thesis stays UNVERIFIED by construction.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

ROOT = Path(__file__).resolve().parents[3]

print("=" * 78)
print("0. WHAT POSITIONING DATA ACTUALLY EXISTS")
print("=" * 78)
for f in ("option_surface_history.parquet", "option_positioning_history.parquet"):
    p = ROOT / "data" / f
    if not p.exists():
        print(f"  {f}: ABSENT")
        continue
    d = pd.read_parquet(p)
    dc = next((c for c in d.columns if "date" in c.lower()), None)
    span = (f"{pd.to_datetime(d[dc]).min().date()} .. "
            f"{pd.to_datetime(d[dc]).max().date()}") if dc else "no date col"
    print(f"  {f}: {len(d)} rows, {span}, cols={list(d.columns)[:8]}")
print("  -> zero usable history for a gamma/dealer-positioning claim.")

px = close_panel(["IWM", "SPY", "QQQ"])
ALL = px.index
oh = {t: 1.0 - px[t] / px[t].rolling(252).max() for t in ("IWM", "SPY", "QQQ")}

print("\n" + "=" * 78)
print("1. LIVE STATE")
print("=" * 78)
print(f"  asof {ALL[-1].date()}   IWM off 52w high {100*oh['IWM'].iloc[-1]:.3f}%   "
      f"SPY {100*oh['SPY'].iloc[-1]:.3f}%   QQQ {100*oh['QQQ'].iloc[-1]:.3f}%")

# opex week: sessions from opex-4 through opex inclusive
opex = pd.to_datetime(load_events(["opex"])["date"].unique())
posn = pd.Series(range(len(ALL)), index=ALL)
opex_week = pd.Series(False, index=ALL)
opex_day = pd.Series(False, index=ALL)
for d in opex:
    p = ALL.searchsorted(d)
    if p >= len(ALL):
        continue
    opex_week.iloc[max(0, p - 4):p + 1] = True
    opex_day.iloc[p] = True
print(f"  today in opex week? {bool(opex_week.iloc[-1])}  "
      f"(opex 2026-08-21 = +4 td)   opex-week days in sample: "
      f"{int(opex_week.sum())} of {len(ALL)} ({100*opex_week.mean():.0f}%)")

IWM_AT = 0.10   # % off its own 252d high
SPY_NOT = 0.10  # SPY strictly further off than this
state = (100 * oh["IWM"] <= IWM_AT) & (100 * oh["SPY"] > SPY_NOT)
print(f"\n  trigger 'IWM within {IWM_AT}% of 52w high AND SPY more than "
      f"{SPY_NOT}% off': {int(state.sum())} days")
print(f"  x opex week: {int((state & opex_week).sum())} days")

variants = {}
for a in (0.10, 0.25, 0.50):
    for b in (0.10, 0.25, 0.50, 1.00):
        variants[f"IWM<={a} & SPY>{b}"] = ((100 * oh["IWM"] <= a) &
                                           (100 * oh["SPY"] > b))
variants["IWM<=0.10 (SPY gate OFF)"] = (100 * oh["IWM"] <= 0.10)
variants["IWM<=0.25 (SPY gate OFF)"] = (100 * oh["IWM"] <= 0.25)

print("\n" + "=" * 78)
print("2. THE PRICE-STATE CELL, WITHOUT the opex gate")
print("=" * 78)
for legs, cost in (([("IWM", 1.0)], 2.0), ([("IWM", 1.0), ("SPY", -1.0)], 4.0)):
    for h in (4, 5, 10):
        battery(px, state, legs, h, f"2. {legs} h={h}  (no opex gate)",
                cost_bps=cost, variants=variants, min_gap=max(h, 5))

print("\n" + "=" * 78)
print("3. GATE ATTRIBUTION -- does the OPEX WEEK gate move anything?")
print("=" * 78)
cells = {
    "state & opex week (TODAY)": state & opex_week,
    "state, NOT opex week": state & ~opex_week,
    "state (gate off)": state,
    "opex week alone": opex_week,
    "IWM-at-high alone (no SPY gate)": (100 * oh["IWM"] <= IWM_AT),
    "IWM-at-high & opex week": (100 * oh["IWM"] <= IWM_AT) & opex_week,
}
for legs in ([("IWM", 1.0)], [("IWM", 1.0), ("SPY", -1.0)]):
    for h in (4, 5, 10):
        ret = vehicle_ret(px, legs, h, 1)
        valid = ret.dropna().index
        rows = []
        for lbl, m in cells.items():
            d = pd.DatetimeIndex(ALL[m.values]).intersection(valid)
            if len(d) == 0:
                rows.append({"label": lbl, "n": 0})
                continue
            e = declusters(d, max(h, 5), valid)
            r = summarize(ret.loc[e].values, lbl)
            r["n_days"] = len(d)
            rows.append(r)
        rows.append(summarize(ret.loc[valid].values, "ALL DAYS"))
        show(rows, f"3. {legs} h={h} gate attribution (episodes)")

print("\n" + "=" * 78)
print("4. ERA SPLIT of the relative leg -- IWM's leadership regime")
print("=" * 78)
for h in (5, 10):
    ret = vehicle_ret(px, [("IWM", 1.0), ("SPY", -1.0)], h, 1)
    valid = ret.dropna().index
    d = pd.DatetimeIndex(ALL[state.values]).intersection(valid)
    e = declusters(d, max(h, 5), valid)
    show(era_split(e, ret.loc[e].values), f"4. IWM-SPY h={h} episodes")
    print(f"  unconditional IWM-SPY h={h}: pre-2018 "
          f"{100*ret[valid[valid < '2018-01-01']].mean():+.3f}%  2018+ "
          f"{100*ret[valid[valid >= '2018-01-01']].mean():+.3f}%")
    yrs = pd.Series(ret.loc[e].values, index=e.year).groupby(level=0).mean()
    print(f"  by year (mean %): {dict((int(y), round(100*v, 2)) for y, v in yrs.items())}")

print("\n" + "=" * 78)
print("5. BETA HONESTY -- is 'IWM minus SPY' a spread or a beta bet?")
print("=" * 78)
r_i = px["IWM"].pct_change()
r_s = px["SPY"].pct_change()
ok = r_i.notna() & r_s.notna()
beta = np.polyfit(r_s[ok].values, r_i[ok].values, 1)[0]
print(f"  IWM daily beta to SPY (full sample): {beta:.3f}")
for h in (5, 10):
    ret_i = vehicle_ret(px, [("IWM", 1.0)], h, 1)
    ret_s = vehicle_ret(px, [("SPY", 1.0)], h, 1)
    valid = (ret_i.notna() & ret_s.notna())
    v = ALL[valid.values]
    d = pd.DatetimeIndex(ALL[state.values]).intersection(v)
    e = declusters(d, max(h, 5), v)
    resid = ret_i - beta * ret_s
    show([summarize(ret_i.loc[e].values, "IWM raw"),
          summarize(ret_s.loc[e].values, "SPY same days"),
          summarize((ret_i - ret_s).loc[e].values, "IWM-SPY 1:1"),
          summarize(resid.loc[e].values, f"IWM - {beta:.2f}*SPY"),
          summarize(resid.loc[v].values, "beta-neutral ALL DAYS")],
         f"5. h={h} decomposition (episodes)")
