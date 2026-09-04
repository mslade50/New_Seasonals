"""C1b round 1 — the same bottom-pole vol state expressed as long SVXY.

Registry obligations this check must discharge before believing anything:
 (i)  the 2026-08-13 LAGGING-MARKER lesson: check SVXY's TRAILING 21d return
      on trigger days. A "cheap vol" premise dies if the vehicle has already
      run.
 (ii) SVXY's leverage changed from -1x to -0.5x on 2018-02-28 (after the
      2018-02-05 blow-up). Any sample straddling that date is two different
      instruments. Measure the regime break, do not assume it.
 (iii) V4 collision: the book already owns post-opex long SVXY.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pitch_lab import (battery, close_panel, declusters, show,  # noqa: E402
                       sign_test, summarize, vehicle_ret)

px = close_panel(["SPY", "SVXY", "^SKEW", "^VIX"]).dropna()
print(f"panel (SVXY-limited) {px.index[0].date()} .. {px.index[-1].date()}  "
      f"n={len(px)}")
sk, vx, sp, sv = px["^SKEW"], px["^VIX"], px["SPY"], px["SVXY"]


def lvl_pctile(s, lb=252):
    return s.rolling(lb).rank(pct=True) * 100.0


sk_p, vx_p = lvl_pctile(sk), lvl_pctile(vx)
sp_hi = sp / sp.rolling(252).max() - 1.0

CORE = (sk_p <= 5) & (vx_p <= 10) & (sp_hi >= -0.005)
LOOSE = (sk_p <= 10) & (vx_p <= 20) & (sp_hi >= -0.01)

print("\n" + "=" * 78)
print("0. THE INSTRUMENT BREAK — is pre-2018 SVXY the same thing as today's?")
print("=" * 78)
r_sv = sv.pct_change()
r_vx = vx.pct_change()
for lbl, s, e in [("-1x era  (2011-10 .. 2018-02-27)", "2011-10-04", "2018-02-27"),
                  ("-0.5x era (2018-03-01 .. today)", "2018-03-01", "2026-08-13")]:
    m = (px.index >= s) & (px.index <= e)
    a, b = r_sv[m].dropna(), r_vx[m].dropna()
    j = a.index.intersection(b.index)
    beta = np.polyfit(b.loc[j], a.loc[j], 1)[0]
    print(f"  {lbl}: n={m.sum():5d}  SVXY daily sd {100*a.std():5.2f}%  "
          f"beta to VIX daily {beta:+.3f}  worst day {100*a.min():+7.2f}%")

print("\n" + "=" * 78)
print("1. TRIGGER INVENTORY on the SVXY panel")
print("=" * 78)
for lbl, m in [("CORE (5/10/-0.5%)", CORE), ("LOOSE (10/20/-1%)", LOOSE)]:
    d = px.index[m.values]
    e = declusters(d, 21, px.index)
    print(f"  {lbl}: {len(d)} days, {len(e)} episodes -> "
          f"{[str(x.date()) for x in e]}")
    pre = [x for x in e if x < pd.Timestamp('2018-02-28')]
    print(f"      of which on the OLD -1x instrument: {len(pre)} "
          f"({[str(x.date()) for x in pre]})")

print("\n" + "=" * 78)
print("2. LAGGING-MARKER CHECK — SVXY trailing return ON trigger days")
print("=" * 78)
tr21 = sv / sv.shift(21) - 1.0
tr63 = sv / sv.shift(63) - 1.0
sv_hi = sv / sv.rolling(252).max() - 1.0
for lbl, m in [("CORE", CORE), ("LOOSE", LOOSE)]:
    d = px.index[m.values]
    e = declusters(d, 21, px.index)
    print(f"  {lbl} episodes:")
    for x in e:
        print(f"    {x.date()}  SVXY trailing21d {100*tr21.loc[x]:+7.2f}%  "
              f"trailing63d {100*tr63.loc[x]:+7.2f}%  "
              f"dist52wh {100*sv_hi.loc[x]:+6.2f}%")
    print(f"    -> median trailing21d {100*tr21.loc[e].median():+.2f}%  "
          f"vs all-days median {100*tr21.dropna().median():+.2f}%")
print(f"  TODAY 2026-08-13: SVXY trailing21d {100*tr21.iloc[-1]:+.2f}%  "
      f"trailing63d {100*tr63.iloc[-1]:+.2f}%  dist52wh {100*sv_hi.iloc[-1]:+.2f}%")

print("\n" + "=" * 78)
print("3. THE CELL, measured anyway (LOOSE, the only mask with any N)")
print("=" * 78)
VARIANTS = {
    "CORE 5/10/-0.5%": CORE,
    "LOOSE 10/20/-1%": LOOSE,
    "GATE-OFF: VIX<=20 & hi>=-1% (no skew)": (vx_p <= 20) & (sp_hi >= -0.01),
    "GATE-OFF: SKEW<=10 alone": (sk_p <= 10),
    "GATE-OFF: hi>=-1% alone": (sp_hi >= -0.01),
}
for h in (3, 5, 10):
    battery(px, LOOSE, [("SVXY", 1.0)], h,
            f"C1b long SVXY, LOOSE bottom-pole vol state",
            cost_bps=8.0, variants=VARIANTS, min_gap=21,
            event_kinds=("cpi", "fomc_decision"))

print("\n" + "=" * 78)
print("4. IS IT JUST EQUITY BETA? SVXY vs a beta-matched SPY leg")
print("=" * 78)
j = r_sv.dropna().index.intersection(sp.pct_change().dropna().index)
beta_spy = np.polyfit(sp.pct_change().loc[j], r_sv.loc[j], 1)[0]
print(f"  SVXY beta to SPY (full SVXY history, daily) = {beta_spy:.2f}")
for h in (5, 10):
    ret_sv = vehicle_ret(px, [("SVXY", 1.0)], h, 1)
    ret_sp = vehicle_ret(px, [("SPY", beta_spy)], h, 1)
    resid = ret_sv - ret_sp
    d = px.index[LOOSE.values & resid.notna().values]
    e = declusters(d, 21, px.index)
    show([summarize(ret_sv.loc[e].values, f"h={h} SVXY raw"),
          summarize(ret_sp.loc[e].values, f"h={h} beta*SPY ({beta_spy:.2f}x)"),
          summarize(resid.loc[e].values, f"h={h} SVXY residual vs beta*SPY"),
          summarize(resid.dropna().values, f"h={h} residual, ALL days")],
         f"beta decomposition, LOOSE episodes (N={len(e)})")

print("\n" + "=" * 78)
print("5. V4 COLLISION — how much of the hold overlaps the book's live trade?")
print("=" * 78)
print("  V4 POSTOPEX_VOL: long SVXY 10% NAV, opex MOC -> +3 sessions MOC.")
print("  2026 Aug opex = 08-21 (+5 td from a 08-14 entry).")
print("  A h>=6 hold from 2026-08-14 contains 08-21..08-26 = the ENTIRE V4 leg.")
for h in (3, 5, 10):
    print(f"    h={h}: exit {'inside' if h >= 5 else 'before'} opex -> "
          f"{'OVERLAPS V4' if h >= 6 else 'clear of V4'}")
