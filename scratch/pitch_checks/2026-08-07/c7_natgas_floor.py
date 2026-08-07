"""C7 "Nat gas at the floor": LONG UNG when close is within 1% of a 52w low, Jul-Sep.

Horizons 5 and 10 td, entry MOC D+1.
Cost: UNG ~5 bps/side => ~10 bps round trip, PLUS structural contango roll decay
(measured here as UNG's unconditional drift over the same horizon).
See data/pitch_negative_registry.md: USO is already killed for roll decay.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _study import *  # noqa

import numpy as np
import pandas as pd

LAG = 1
COST_BPS = 10.0
P = close_panel(["UNG"]).dropna()
s = P["UNG"]
ASOF = s.index[-1]
lo52 = s.rolling(252).min()
dist = (s / lo52 - 1.0) * 100
print(f"sample {s.index.min().date()} .. {ASOF.date()}  n={len(s)}")
print(f"FIRES TODAY: dist above 52w low = {dist.loc[ASOF]:.3f}% (<=1.0), month={ASOF.month} (7-9) "
      f"-> {(dist.loc[ASOF] <= 1.0) and (7 <= ASOF.month <= 9)}")

# ---- structural roll drag: what does simply owning UNG cost? ----
rows = []
for hh in (5, 10, 21, 63, 252):
    rows.append(summarize(fwd_lag(s, hh, LAG).dropna().values, f"UNG unconditional h{hh}"))
show(rows, "0. STRUCTURAL ROLL DRAG (unconditional UNG drift, all days)")
yrs = (s.index[-1] - s.index[0]).days / 365.25
print(f"   buy-and-hold {s.iloc[-1]/s.iloc[0]-1:+.2%} over {yrs:.1f}y = {(s.iloc[-1]/s.iloc[0])**(1/yrs)-1:+.2%}/yr")


def trigger(tol=1.0, months=(7, 8, 9)):
    m = (dist <= tol)
    if months:
        m &= s.index.month.isin(months)
    return dist.index[m.fillna(False)]


for H in (5, 10):
    print(f"\n{'='*70}\nHORIZON {H} td\n{'='*70}")
    fw = fwd_lag(s, H, LAG)
    trig = trigger()
    vt = pd.DatetimeIndex(trig).intersection(fw.dropna().index)
    print(f"[1] day-level triggers: {len(vt)}")
    show(report(fw, trig, "C7", H), f"1. conditional vs 2 controls (h{H})")
    show(era_split(vt, fw.loc[vt].values), "2. era split (day-level)")

    ep, ev_ = episodes(fw, trig, H)
    print(f"[3] declustered episodes (min gap {H} td): {len(ep)} (day-level {len(vt)})")
    show([summarize(ev_, f"C7 episodes h{H}")], "3. episode-level")
    show(era_split(ep, ev_), "3b. episode era split")
    print("   boot P(mean<=0) =", round(bootstrap_p_le0(ev_), 4))
    print("   concentration:", cluster_note(ep, ev_))
    print("   episode dates:", [str(d.date()) for d in ep])
    print(f"[5] cost {COST_BPS:.0f}bps rt; mean {100*np.nanmean(ev_):+.3f}% "
          f"-> net of commission {100*np.nanmean(ev_) - COST_BPS/100:+.3f}%; "
          f"edge vs unconditional drift {100*np.nanmean(ev_) - 100*fw.dropna().mean():+.3f}pp")
    m = cpi_in_window(ep, fw.dropna().index, H, LAG)
    show([summarize(ev_[m], "CPI inside"), summarize(ev_[~m], "no CPI")], f"6. CPI split (h{H})")

rows = []
for tol in (0.5, 1.0, 2.0, 3.0):
    for hh in (5, 10, 21):
        fwh = fwd_lag(s, hh, LAG)
        e, v = episodes(fwh, trigger(tol), hh)
        r = summarize(v, f"tol={tol}% Jul-Sep h={hh}")
        r["boot_p"] = bootstrap_p_le0(v)
        rows.append(r)
for hh in (5, 10, 21):
    fwh = fwd_lag(s, hh, LAG)
    e, v = episodes(fwh, trigger(1.0, months=None), hh)
    r = summarize(v, f"tol=1.0% ALL MONTHS h={hh}")
    r["boot_p"] = bootstrap_p_le0(v)
    rows.append(r)
show(rows, "4. sensitivity: tolerance, horizon, and whether Jul-Sep is fitted")
