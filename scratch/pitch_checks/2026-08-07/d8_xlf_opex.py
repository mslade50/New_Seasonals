"""D8 - XLF (sector leader) vs SPY into monthly opex, equal dollar, 10 td.

Convention: trigger on the SIGNAL close D (2026-08-06); order is MOC at the
close of D+1 (2026-08-07); return runs close D+1 -> close D+1+h. Pair =
XLF leg - SPY leg (long XLF / short SPY); negative mean = the reverse trade.
Two forms: (a) a generic h=10 hold, (b) an OPEX-ANCHORED hold that exits at
the next monthly opex close, restricted to 8-12 td out so it matches the real
2026-08-07 -> 2026-08-21 order (10 td, verified in d0_fires_today.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _common import bootstrap_p_le0, close_panel, declusters, era_split, load_events, pct_rank, show, summarize  # noqa: E402

px = close_panel(["XLF", "SPY"]).dropna()
cal = px.index
CPI = pd.DatetimeIndex(load_events(["cpi"])["date"])
OPEX = pd.DatetimeIndex(load_events(["opex"])["date"])


def trig(rk: float, dist: float) -> pd.Series:
    s = px["XLF"]
    return (pct_rank(s, 63) >= rk) & ((1 - s / s.rolling(252).max()) <= dist / 100.0)


def legs(h: int, mask: pd.Series):
    dts, p, a, b = [], [], [], []
    for dt in cal[mask.reindex(cal).fillna(False).values]:
        i = cal.get_loc(dt)
        if i + 1 + h >= len(cal):
            continue
        ra = px["XLF"].iloc[i + 1 + h] / px["XLF"].iloc[i + 1] - 1.0
        rb = px["SPY"].iloc[i + 1 + h] / px["SPY"].iloc[i + 1] - 1.0
        dts.append(dt); p.append(ra - rb); a.append(ra); b.append(rb)
    return pd.DatetimeIndex(dts), np.array(p), np.array(a), np.array(b)


H = 10
m = trig(95, 1.0)
dts, pair, xlf, spy = legs(H, m)
print(f"XLF/SPY {cal[0].date()}..{cal[-1].date()}; trigger days {int(m.sum())}, graded {len(dts)}; "
      f"fires 2026-08-06: {bool(m.loc[pd.Timestamp('2026-08-06')])}")

uP = (px["XLF"].shift(-H) / px["XLF"] - px["SPY"].shift(-H) / px["SPY"]).dropna()
show([summarize(pair, "COND pair XLF-SPY"), summarize(xlf, "  leg: XLF"), summarize(spy, "  leg: SPY"),
      summarize(uP.loc[dts[0]:dts[-1]].values, "CTRL pair same span"),
      summarize(uP.values, "CTRL pair all-days full")],
     "1. conditional vs unconditional pair drift (h=10), day-level")

ep = declusters(dts, H, cal)
em = np.isin(dts.values, ep.values)
pe = pair[em]
print(f"\n3. decluster min-gap {H} td: day-level N={len(pair)} -> episode-level N={len(pe)}")
show([summarize(pair, "day-level"), summarize(pe, "EPISODE-level"),
      summarize(np.sort(pe)[:-1], "episodes drop-best"), summarize(np.sort(pe)[1:], "episodes drop-worst"),
      summarize(xlf[em], "  leg XLF (episodes)"), summarize(spy[em], "  leg SPY (episodes)")]
     + era_split(dts[em], pe), "2+3. episodes, drop-best, leg attribution, era")
print(f"   bootstrap P(mean<=0) = {bootstrap_p_le0(pe):.3f}  P(mean>=0) = {1 - bootstrap_p_le0(pe):.3f}")
print(f"   episode years: {sorted(pd.Series(ep).dt.year.unique().tolist())}")

rows = []
for h in (5, 10, 21):
    for rk in (90, 95, 98):
        row = {"h": h, "rank63>=": rk}
        for ds in (0.5, 1.0, 2.0):
            dd, pp, _, _ = legs(h, trig(rk, ds))
            ee = pp[np.isin(dd.values, declusters(dd, h, cal).values)] if len(dd) else np.array([])
            s = summarize(ee)
            row[f"<={ds}%"] = f"{s.get('mean_pct', float('nan')):.2f}/t{s.get('t', float('nan')):.1f}/n{s.get('n', 0)}"
        rows.append(row)
show(rows, "4. sensitivity (EPISODE-level): pair mean% / t / N")

se = summarize(pe)
print(f"\n5. cost: pair ~4 bps round trip. Episode mean {100 * se['mean_pct']:.1f} bps "
      f"-> {abs(100 * se['mean_pct']) / 4.0:.1f}x (need ~5x)")

has = np.array([bool(((CPI > cal[cal.get_loc(d) + 1]) & (CPI <= cal[cal.get_loc(d) + 1 + H])).any())
                for d in dts[em]])
show([summarize(pe[has], "episode, CPI inside"), summarize(pe[~has], "episode, no CPI inside")],
     "6. split by whether a CPI print falls inside the 10 td hold")

# ---- opex-anchored form ---------------------------------------------------
dts2, p2, a2, b2, gap = [], [], [], [], []
for dt in cal[m.reindex(cal).fillna(False).values]:
    i = cal.get_loc(dt)
    if i + 1 >= len(cal):
        continue
    en = cal[i + 1]
    nxt = OPEX[OPEX > en]
    if len(nxt) == 0:
        continue
    j = int(np.searchsorted(cal.values, np.datetime64(nxt[0])))
    if j >= len(cal) or cal[j] != nxt[0]:
        continue
    k = j - (i + 1)
    if not (8 <= k <= 12):
        continue
    dts2.append(dt); gap.append(k)
    a2.append(px["XLF"].iloc[j] / px["XLF"].iloc[i + 1] - 1.0)
    b2.append(px["SPY"].iloc[j] / px["SPY"].iloc[i + 1] - 1.0)
d2 = pd.DatetimeIndex(dts2)
p2 = np.array(a2) - np.array(b2)
e2 = np.isin(d2.values, declusters(d2, 10, cal).values)
print(f"\n7. OPEX-ANCHORED (exit at next opex close, 8-12 td out; today k=10): "
      f"day N={len(p2)}, episode N={int(e2.sum())}, median gap {int(np.median(gap)) if gap else 0} td")
show([summarize(p2, "opex-anchored day-level"), summarize(p2[e2], "opex-anchored EPISODES"),
      summarize(np.array(a2)[e2], "  leg XLF"), summarize(np.array(b2)[e2], "  leg SPY"),
      summarize(np.sort(p2[e2])[:-1], "episodes drop-best")]
     + era_split(d2[e2], p2[e2]), "7. opex-anchored form")
print(f"   bootstrap P(mean<=0) = {bootstrap_p_le0(p2[e2]):.3f}")
