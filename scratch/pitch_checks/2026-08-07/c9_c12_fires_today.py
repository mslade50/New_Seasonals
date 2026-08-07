"""Do C9/C10/C11/C12 triggers ACTUALLY fire on the 2026-08-06 close?

Recomputes every rank / z-score from master_prices. Zero trust in the brief.
Also verifies the 2026-08-07 NFP + 2026-08-12 CPI + 2026-08-19 vix_expiry
dates directly from macro_events.csv.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import *  # noqa

import numpy as np
import pandas as pd

D = pd.Timestamp("2026-08-06")
TK = ["SPY", "QQQ", "SMH", "AAPL", "SVXY", "^VIX"]
px = load_prices(TK)
panel = pd.DataFrame({t: px[t]["Close"] for t in TK})

print("=== bar coverage ===")
for t in TK:
    s = px[t]["Close"].dropna()
    print(f"  {t:6s} first={s.index[0].date()} last={s.index[-1].date()} n={len(s)}")

# pitch_state z10 convention: ret10 / (vol21 * sqrt(10)); _common.zscore is the
# 252d standardized form. Report both so neither definition can hide a miss.
def z10_state(s):
    return s.pct_change(10) / (s.pct_change().rolling(21).std() * np.sqrt(10))

print("\n=== recomputed levels at 2026-08-06 (mine, not the brief's) ===")
rows = []
for t in TK:
    s = px[t]["Close"].dropna()
    rows.append(dict(
        tkr=t, close=s.loc[D],
        ret5=100 * (s.loc[D] / s.shift(5).loc[D] - 1),
        ret21=100 * (s.loc[D] / s.shift(21).loc[D] - 1),
        ret63=100 * (s.loc[D] / s.shift(63).loc[D] - 1),
        rank5=pct_rank(s, 5).loc[D], rank21=pct_rank(s, 21).loc[D],
        rank63=pct_rank(s, 63).loc[D],
        z10_state=z10_state(s).loc[D], z10_252=zscore(s, 10).loc[D],
        off_52wh=100 * (s.loc[D] / s.rolling(252).max().loc[D] - 1),
    ))
show(rows, "levels")

print("\n=== trigger evaluation ===")
spy, qqq, smh, aapl, vix = (px[t]["Close"].dropna() for t in
                            ["SPY", "QQQ", "SMH", "AAPL", "^VIX"])

# --- C9: post-NFP into CPI. Trigger = today is an NFP date (+ optional rank gate)
ev = load_events()
nfp = set(ev.loc[ev.event == "nfp", "date"])
cpi = set(ev.loc[ev.event == "cpi", "date"])
vxp = set(ev.loc[ev.event == "vix_expiry", "date"])
opx = set(ev.loc[ev.event == "opex", "date"])
print(f"C9: 2026-08-07 in nfp dates ? {pd.Timestamp('2026-08-07') in nfp}")
print(f"    next cpi after 2026-08-07 = "
      f"{min(d for d in cpi if d > pd.Timestamp('2026-08-07')).date()}")
print(f"    2026-08-19 in vix_expiry ? {pd.Timestamp('2026-08-19') in vxp}   "
      f"2026-08-21 in opex ? {pd.Timestamp('2026-08-21') in opx}")
spy_r5 = pct_rank(spy, 5).loc[D]
print(f"    conditioner (a) SPY rank5={spy_r5:.1f} >= 90 ? {spy_r5 >= 90}")
print(f"    conditioner (b) midterm year 2026 (2026%4==2) ? {2026 % 4 == 2}")
print(f"  -> C9 FIRES: {pd.Timestamp('2026-08-07') in nfp}  "
      f"(entry is tonight's NFP-day MOC, so the gate is the calendar)")

# --- C10: SMH rank63 <= 10 AND SMH rank5 >= 80
s63, s5 = pct_rank(smh, 63).loc[D], pct_rank(smh, 5).loc[D]
c10 = (s63 <= 10) and (s5 >= 80)
print(f"\nC10: SMH rank63={s63:.1f} <= 10 ? {s63 <= 10}   "
      f"SMH rank5={s5:.1f} >= 80 ? {s5 >= 80}")
print(f"  -> C10 FIRES: {c10}")

# --- C11: AAPL rank5 <= 10 AND QQQ rank5 >= 85
a5, q5 = pct_rank(aapl, 5).loc[D], pct_rank(qqq, 5).loc[D]
c11 = (a5 <= 10) and (q5 >= 85)
print(f"\nC11: AAPL rank5={a5:.1f} <= 10 ? {a5 <= 10}   "
      f"QQQ rank5={q5:.1f} >= 85 ? {q5 >= 85}")
print(f"  -> C11 FIRES: {c11}")

# --- C12: VIX rank5 <= 25, with 8 td to vix expiry
v5 = pct_rank(vix, 5).loc[D]
svxy = px["SVXY"]["Close"].dropna()
cal = spy.index
td_to_exp = int((cal > pd.Timestamp("2026-08-07")).sum()
                - (cal > pd.Timestamp("2026-08-19")).sum())
c12 = v5 <= 25
print(f"\nC12: ^VIX rank5={v5:.1f} <= 25 ? {c12}")
print(f"     SVXY last bar {svxy.index[-1].date()} close={svxy.iloc[-1]:.2f}, "
      f"off 52wh {100*(svxy.iloc[-1]/svxy.rolling(252).max().iloc[-1]-1):.2f}%")
print(f"  -> C12 FIRES: {c12}")

print("\n=== brief-vs-recompute deltas (sanity) ===")
brief = {"SPY_rank5": 96.0, "SPY_rank21": 75.4, "SPY_z10": 1.46,
         "QQQ_rank5": 93.3, "QQQ_rank21": 40.1,
         "SMH_rank5": 85.3, "SMH_rank21": 14.3, "SMH_rank63": 2.0,
         "AAPL_rank5": 3.2, "VIX_rank5": 17.1, "SVXY_ret63": 16.17}
mine = {"SPY_rank5": pct_rank(spy, 5).loc[D], "SPY_rank21": pct_rank(spy, 21).loc[D],
        "SPY_z10": z10_state(spy).loc[D],
        "QQQ_rank5": q5, "QQQ_rank21": pct_rank(qqq, 21).loc[D],
        "SMH_rank5": s5, "SMH_rank21": pct_rank(smh, 21).loc[D], "SMH_rank63": s63,
        "AAPL_rank5": a5, "VIX_rank5": v5,
        "SVXY_ret63": 100 * (svxy.loc[D] / svxy.shift(63).loc[D] - 1)}
for k in brief:
    print(f"  {k:12s} brief={brief[k]:7.2f}  mine={mine[k]:7.2f}  d={mine[k]-brief[k]:+.2f}")
