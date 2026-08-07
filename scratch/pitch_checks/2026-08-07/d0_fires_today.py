"""D0 - verify every trigger fact for D5/D6/D7/D8 from the real data.

Recomputes ranks, 52w distances and trading-day counts on a US-federal-holiday
business calendar. Confirms the CPI/opex dates from macro_events.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

sys.path.insert(0, str(Path(__file__).parent))
from _common import close_panel, load_events, pct_rank, zscore  # noqa: E402

TODAY = pd.Timestamp("2026-08-06")  # freshest bar
BD = CustomBusinessDay(calendar=USFederalHolidayCalendar())

TK = ["GLD", "DX-Y.NYB", "^TNX", "XLV", "XLF", "SPY", "PFE", "ABT", "BDX"]
px = close_panel(TK)

print("=== last bars ===")
print(px.tail(2).to_string())

rows = []
for t in TK:
    s = px[t].dropna()
    hi52 = s.rolling(252).max()
    rows.append({
        "ticker": t,
        "close": round(float(s.loc[TODAY]), 3),
        "ret_5d": round(100 * float(s.pct_change(5).loc[TODAY]), 2),
        "ret_21d": round(100 * float(s.pct_change(21).loc[TODAY]), 2),
        "ret_63d": round(100 * float(s.pct_change(63).loc[TODAY]), 2),
        "rank_21d": round(float(pct_rank(s, 21).loc[TODAY]), 1),
        "rank_63d": round(float(pct_rank(s, 63).loc[TODAY]), 1),
        "z10": round(float(zscore(s, 10).loc[TODAY]), 2),
        "pct_below_52wh": round(100 * (1 - float(s.loc[TODAY]) / float(hi52.loc[TODAY])), 2),
    })
print("\n=== recomputed state (2026-08-06 close) ===")
print(pd.DataFrame(rows).to_string(index=False))

# ---- calendar facts -------------------------------------------------------
ev = load_events()
nxt = ev[ev["date"] > TODAY].groupby("event").first()
print("\n=== next event of each kind after 2026-08-06 ===")
print(nxt.loc[["cpi", "ppi", "nfp", "opex", "vix_expiry"], ["date", "detail"]].to_string())

cpi = pd.Timestamp("2026-08-12")
opex = pd.Timestamp("2026-08-21")
entry = pd.Timestamp("2026-08-07")

fwd = pd.date_range(TODAY, "2026-09-05", freq=BD)
print("\n=== forward business calendar (US federal holidays) ===")
print(" ".join(d.strftime("%m-%d(%a)") for d in fwd[:16]))


def td_between(a: pd.Timestamp, b: pd.Timestamp) -> int:
    return len(pd.date_range(a, b, freq=BD)) - 1


print(f"\nentry MOC {entry.date()} -> exit MOC {(cpi - BD).date()} (close before CPI): "
      f"{td_between(entry, cpi - BD)} forward sessions")
print(f"  D5/D6 brief says '3 sessions' -> ACTUAL k = {td_between(entry, cpi - BD)}")
print(f"entry MOC {entry.date()} -> opex close {opex.date()}: {td_between(entry, opex)} forward sessions")
print(f"CPI {cpi.date()} is {td_between(entry, cpi)} sessions after the {entry.date()} close "
      f"({td_between(TODAY, cpi)} after the {TODAY.date()} close)")

# ---- trigger booleans -----------------------------------------------------
print("\n=== trigger checks on the 2026-08-06 close ===")


def state(t):
    s = px[t].dropna()
    return (float(pct_rank(s, 63).loc[TODAY]),
            100 * (1 - float(s.loc[TODAY]) / float(s.rolling(252).max().loc[TODAY])))


for t, rk, dist in [("XLV", 90.0, 2.0), ("XLF", 95.0, 1.0), ("PFE", 90.0, 2.0),
                    ("ABT", 90.0, 2.0), ("BDX", 90.0, 2.0)]:
    r, d = state(t)
    print(f"  {t:5s} rank63={r:5.1f} (>= {rk}? {r >= rk})  below52wh={d:5.2f}% "
          f"(<= {dist}%? {d <= dist})  FIRES={bool(r >= rk and d <= dist)}")

g = px["GLD"].dropna()
print(f"  GLD  5d ret={100 * float(g.pct_change(5).loc[TODAY]):.2f}% (>0? "
      f"{float(g.pct_change(5).loc[TODAY]) > 0})")
d = px["DX-Y.NYB"].dropna()
r21, r63 = float(pct_rank(d, 21).loc[TODAY]), float(pct_rank(d, 63).loc[TODAY])
print(f"  DX   rank21={r21:.1f} (<20? {r21 < 20})  rank63={r63:.1f} (>90? {r63 > 90})  "
      f"FIRES={bool(r21 < 20 and r63 > 90)}")
