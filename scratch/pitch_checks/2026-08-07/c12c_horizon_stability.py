"""C12 horizon check. The brief calls the hold "8 td", but the real window is
2026-08-07 close -> 2026-08-18 close = SEVEN sessions (Aug 10,11,12,13,14,17,18).
If the gated cell is only alive at k=8 it is fitted to a miscounted horizon.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import *  # noqa

import numpy as np
import pandas as pd

px = load_prices(["SVXY", "^VIX", "SPY"])
sv, vix = px["SVXY"]["Close"].dropna(), px["^VIX"]["Close"].dropna()
cal = sv.index
pos = pd.Series(range(len(cal)), index=cal)
ev = load_events()
vxp = [d for d in ev.loc[ev.event == "vix_expiry", "date"] if d <= cal[-1]]
vr5 = pct_rank(vix, 5).reindex(cal)

# confirm the live horizon off a real US-business-day calendar
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
fwd = pd.bdate_range("2026-08-10", "2026-08-18", freq=bd)
print(f"sessions from the 2026-08-07 close to the 2026-08-18 close: {len(fwd)}")
print(f"  {[str(d.date()) for d in fwd]}")
print("  -> the live hold is k=7, not k=8\n")

rows = []
for k in range(4, 13):
    W = []
    for E in vxp:
        prior = cal[cal < E]
        if len(prior) == 0:
            continue
        xi = pos[prior[-1]]
        if xi - k >= 0:
            W.append((cal[xi - k], cal[xi]))
    ent = pd.DatetimeIndex([a for a, _ in W])
    V = np.array([sv.loc[b] / sv.loc[a] - 1.0 for a, b in W])
    g = vr5.reindex(ent).to_numpy() <= 25
    fk = fwd_ret(sv, k)
    ok = (vr5 <= 25).to_numpy() & fk.notna().to_numpy()
    mod = ent[g] >= pd.Timestamp("2018-06-01")
    s = summarize(V[g], f"gated k={k}")
    s["ctl_anyday_gated"] = 100 * fk.to_numpy()[ok].mean()
    s["edge_vs_ctl"] = s["mean_pct"] - s["ctl_anyday_gated"]
    s["p_le0"] = bootstrap_p_le0(V[g])
    s["mod_n"] = int(mod.sum())
    s["mod_mean"] = 100 * V[g][mod].mean()
    rows.append(s)
show(rows, "gated pre-expiry cell across horizons (mod_* = 2018-06+ only)")

print("\nRead: a cell whose mean swings 2x between k=7 and k=8 - one session -")
print("is not measuring a mechanism, it is measuring which day the sample lands on.")
