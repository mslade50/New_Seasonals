"""C1 round 1 — the BOTTOM pole of skew: ^SKEW at a bottom-percentile LEVEL
with ^VIX also bottom-decile and SPY at/near its 52w high.

Direction is NOT assumed. The cell is measured long SPY and the sign is
whatever it is; a negative mean is a short read, a mean indistinguishable
from the controls is no read at all.

LEVEL percentile, not return rank (2026-08-10 ^MOVE registry lesson).
Entry lag=1 MOC-tomorrow throughout. Fractions in, percent out.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pitch_lab import (battery, close_panel, declusters, sign_test,  # noqa: E402
                       summarize, show, vehicle_ret)

TK = ["SPY", "QQQ", "^SKEW", "^VIX"]
px = close_panel(TK).dropna()
print(f"panel {px.index[0].date()} .. {px.index[-1].date()}  n={len(px)}")


def lvl_pctile(s: pd.Series, lookback: int = 252) -> pd.Series:
    return s.rolling(lookback).rank(pct=True) * 100.0


sk_p = lvl_pctile(px["^SKEW"])
vx_p = lvl_pctile(px["^VIX"])
sp_hi = px["SPY"] / px["SPY"].rolling(252).max() - 1.0

TODAY = dict(skew=sk_p.iloc[-1], vix=vx_p.iloc[-1], sphi=100 * sp_hi.iloc[-1])
print(f"today: SKEW lvl pctile {TODAY['skew']:.1f}, VIX lvl pctile "
      f"{TODAY['vix']:.1f}, SPY {TODAY['sphi']:+.2f}% off 52wh")

# --- UNITS ASSERTION at the boundary -------------------------------------
_r = px["SPY"].pct_change().dropna()
assert abs(_r).max() < 0.5, "returns must be FRACTIONS before summarize()"


def mask(skt, vxt, hit):
    return (sk_p <= skt) & (vx_p <= vxt) & (sp_hi >= hit)


CORE = mask(5, 10, -0.005)     # the reasonable neighbourhood of today
STRICT = mask(2, 5, 0.0)       # today's literal reading

print("\nepisode inventory (min_gap 21td):")
for lbl, m in [("STRICT (2/5/0.0%)", STRICT), ("CORE (5/10/-0.5%)", CORE)]:
    d = px.index[m.values]
    e = declusters(d, 21, px.index)
    print(f"  {lbl}: {len(d)} days, {len(e)} episodes -> "
          f"{[str(x.date()) for x in e]}")

VARIANTS = {
    "SKEW<=2 VIX<=5 hi>=0 (strict/today)": STRICT,
    "SKEW<=5 VIX<=10 hi>=-0.5% (core)": CORE,
    "SKEW<=10 VIX<=10 hi>=-1%": mask(10, 10, -0.01),
    "SKEW<=20 VIX<=20 hi>=-1%": mask(20, 20, -0.01),
    "GATE-OFF: SKEW<=5 alone": (sk_p <= 5),
    "GATE-OFF: VIX<=10 alone": (vx_p <= 10),
    "GATE-OFF: SPY hi>=-0.5% alone": (sp_hi >= -0.005),
    "GATE-OFF: SKEW<=5 + VIX<=10 (no SPY leg)": (sk_p <= 5) & (vx_p <= 10),
    "GATE-OFF: SKEW<=5 + hi>=-0.5% (no VIX leg)": (sk_p <= 5) & (sp_hi >= -0.005),
    "GATE-OFF: VIX<=10 + hi>=-0.5% (no SKEW leg)": (vx_p <= 10) & (sp_hi >= -0.005),
}

for h in (5, 10):
    battery(px, CORE, [("SPY", 1.0)], h,
            f"C1 CORE long SPY  SKEW lvl<=5 & VIX lvl<=10 & SPY within 0.5% of 52wh",
            cost_bps=2.0, variants=VARIANTS, min_gap=21,
            event_kinds=("cpi", "fomc_decision"))

battery(px, STRICT, [("SPY", 1.0)], 5,
        "C1 STRICT long SPY (today's literal reading: 2 / 5 / at the high)",
        cost_bps=2.0, min_gap=21, event_kinds=("cpi", "fomc_decision"))

# --- the era question, stated directly ------------------------------------
print("\n" + "=" * 78)
print("ERA: when does this state OCCUR at all? (day counts by year)")
print("=" * 78)
for lbl, m in [("STRICT", STRICT), ("CORE", CORE)]:
    byyr = pd.Series(1, index=px.index[m.values]).groupby(
        px.index[m.values].year).sum()
    print(f"  {lbl}: {dict(byyr)}")

# --- midterm cross (2026 is year%4==2) ------------------------------------
print("\n" + "=" * 78)
print("MIDTERM cross (2026 %% 4 == 2)")
print("=" * 78)
for h in (5, 10):
    ret = vehicle_ret(px, [("SPY", 1.0)], h, 1)
    d = px.index[CORE.values & ret.notna().values]
    e = declusters(d, 21, px.index)
    mid = np.array([x.year % 4 == 2 for x in e])
    show([summarize(ret.loc[e[mid]].values, f"h={h} midterm episodes"),
          summarize(ret.loc[e[~mid]].values, f"h={h} non-midterm episodes")],
         f"CORE h={h} by cycle year")
