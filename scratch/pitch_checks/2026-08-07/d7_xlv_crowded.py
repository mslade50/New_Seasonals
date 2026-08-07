"""D7 - crowded/extended XLV vs SPY, equal dollar, 10 td.

Convention: trigger measured on the SIGNAL close D (2026-08-06); the order is
MOC at the close of D+1 (2026-08-07); the return runs close D+1 -> close D+1+h.
Pair return = XLV leg - SPY leg (equal dollar, long XLV / short SPY); a
negative mean is the reverse trade. Ranks and the 52w high are recomputed on
the adjusted close panel (both relative, so dividend-basis invariant).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _common import bootstrap_p_le0, close_panel, declusters, era_split, load_events, pct_rank, show, summarize  # noqa: E402

TK = ["XLV", "SPY", "PFE", "ABT", "BDX"]
px = close_panel(TK).dropna()
cal = px.index
CPI = pd.DatetimeIndex(load_events(["cpi"])["date"])


def legs(sig: str, h: int, mask: pd.Series):
    """(dates, pair, sig_leg, spy_leg) entering at close D+1, holding h td."""
    dts, p, a, b = [], [], [], []
    n = len(cal)
    for dt in cal[mask.reindex(cal).fillna(False).values]:
        i = cal.get_loc(dt)
        if i + 1 + h >= n:
            continue
        ra = px[sig].iloc[i + 1 + h] / px[sig].iloc[i + 1] - 1.0
        rb = px["SPY"].iloc[i + 1 + h] / px["SPY"].iloc[i + 1] - 1.0
        dts.append(dt); p.append(ra - rb); a.append(ra); b.append(rb)
    return pd.DatetimeIndex(dts), np.array(p), np.array(a), np.array(b)


def trig(t: str, rk: float, dist: float) -> pd.Series:
    s = px[t]
    return (pct_rank(s, 63) >= rk) & ((1 - s / s.rolling(252).max()) <= dist / 100.0)


H = 10
m = trig("XLV", 90, 2.0)
dts, pair, xlv, spy = legs("XLV", H, m)
print(f"XLV/SPY {cal[0].date()}..{cal[-1].date()};  trigger days: {int(m.sum())}, "
      f"graded: {len(dts)}; fires 2026-08-06: {bool(m.loc[pd.Timestamp('2026-08-06')])}")

# ---- 1. exists? vs two controls ------------------------------------------
uP = (px["XLV"].shift(-H) / px["XLV"] - px["SPY"].shift(-H) / px["SPY"]).dropna()
show([summarize(pair, "COND pair XLV-SPY"), summarize(xlv, "  leg: XLV"), summarize(spy, "  leg: SPY"),
      summarize(uP.loc[dts[0]:dts[-1]].values, "CTRL pair, same span all-days"),
      summarize(uP.values, "CTRL pair, all-days full")],
     "1. conditional vs unconditional pair drift (h=10), day-level")

# ---- 2/3. decluster, era, worst, drop-best, bootstrap --------------------
ep = declusters(dts, H, cal)
em = np.isin(dts.values, ep.values)
pe, xe = pair[em], xlv[em]
print(f"\n3. decluster min-gap {H} td: day-level N={len(pair)} -> episode-level N={len(pe)}")
show([summarize(pair, "day-level"), summarize(pe, "EPISODE-level"),
      summarize(np.sort(pe)[:-1], "episodes drop-best"), summarize(np.sort(pe)[1:], "episodes drop-worst")]
     + era_split(dts[em], pe), "2+3. episodes, era split, drop-best (pair)")
print(f"   bootstrap P(mean<=0) episodes = {bootstrap_p_le0(pe):.3f}   "
      f"P(mean>=0) = {1 - bootstrap_p_le0(pe):.3f}")
print(f"   episode years: {sorted(pd.Series(ep).dt.year.unique().tolist())}")

# ---- 4. sensitivity grid --------------------------------------------------
rows = []
for h in (5, 10, 21):
    for rk in (85, 90, 95):
        row = {"h": h, "rank63>=": rk}
        for ds in (1.0, 2.0, 3.0):
            dd, pp, _, _ = legs("XLV", h, trig("XLV", rk, ds))
            ee = pp[np.isin(dd.values, declusters(dd, h, cal).values)]
            s = summarize(ee)
            row[f"<={ds}%"] = f"{s.get('mean_pct', float('nan')):.2f}/t{s.get('t', float('nan')):.1f}/n{s.get('n', 0)}"
        rows.append(row)
show(rows, "4. sensitivity (EPISODE-level): pair mean% / t / N")

# ---- 5. cost --------------------------------------------------------------
se = summarize(pe)
print(f"\n5. cost: pair pays ~2-4 bps round trip (both legs). Episode mean "
      f"{100 * se['mean_pct']:.1f} bps -> {abs(100 * se['mean_pct']) / 4.0:.1f}x a 4 bps pair cost (need ~5x)")

# ---- 6. CPI inside the window --------------------------------------------
has = np.array([bool(((CPI > cal[cal.get_loc(d) + 1]) & (CPI <= cal[cal.get_loc(d) + 1 + H])).any()) for d in dts[em]])
show([summarize(pe[has], "episode, CPI inside window"), summarize(pe[~has], "episode, no CPI inside")],
     "6. split by whether a CPI print falls inside the 10 td hold")

# ---- individual crowded names --------------------------------------------
rows = []
for t in ("PFE", "ABT", "BDX"):
    mt = trig(t, 90, 2.0)
    dd, pp, aa, _ = legs(t, H, mt)
    if len(dd) == 0:
        rows.append({"label": f"{t} (no fires)", "n": 0}); continue
    ee = declusters(dd, H, cal)
    k = np.isin(dd.values, ee.values)
    rows.append(summarize(pp[k], f"{t}-SPY pair, episodes"))
    rows.append(summarize(aa[k], f"{t} outright, episodes"))
    fires = bool(mt.loc[pd.Timestamp("2026-08-06")])
    rows[-1]["label"] += f" (fires today: {fires})"
show(rows, "extra: same trigger on the individual crowded names")
