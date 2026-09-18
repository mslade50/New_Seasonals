"""c4 follow-up: the one conditioner that looked positive in round 1 was the
^VIX SPY-residual with VIX/VIX3M(opex-1) < 0.85 (Sept 6-1, other months 43-16).
Spot ^VIX is not tradeable: a steep curve (low VIX/VIX3M) is exactly when the
futures already price the mean reversion as roll-down. Does it transfer to the
TRADEABLE hedged short SVXY, and is it an opex/September effect or generic?
Also the c2 September row: crush vs no-crush inside September only.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pitch_lab import *  # noqa
from kA_common import *  # noqa
import numpy as np
import pandas as pd

px = build_panel()
cal = px.index
vix = px["^VIX"]
vts = (px["^VIX"] / px["^VIX3M"])
opex = pd.DatetimeIndex(sorted(set(load_events(["opex"])["date"]) & set(cal)))
opex = opex[opex < pd.Timestamp("2026-09-18")]
post = pd.Series(cal >= POST, index=cal)
pre = pd.Series((cal < BREAK) & (cal >= pd.Timestamp("2011-10-10")), index=cal)
vts_prev = vts.shift(1)   # VIX/VIX3M at opex-1 (known before the opex MOC)
print(f"live VIX/VIX3M 2026-09-17 = {vts.loc['2026-09-17']:.3f}")

for h in (1, 2, 3):
    b = hedge_beta(px, "SVXY", h, 0, post)
    hs = vehicle_ret(px, [("SVXY", -1.0), ("SPY", b)], h, 0)
    bp = hedge_beta(px, "SVS", h, 0, pre)
    hp = vehicle_ret(px, [("SVS", -1.0), ("SPY", bp)], h, 0)
    rows = []
    for era, ser, m_era in [("post", hs, post), ("PRE synth", hp, pre)]:
        ok = ser.notna() & m_era
        io = pd.Series(cal.isin(opex), index=cal)
        sp = io & pd.Series(cal.month == 9, index=cal)
        for lbl, m in [("Sept opex & vts<0.85", sp & (vts_prev < 0.85)),
                       ("Sept opex & vts>=0.85", sp & (vts_prev >= 0.85)),
                       ("all opex & vts<0.85", io & (vts_prev < 0.85)),
                       ("all opex & vts>=0.85", io & (vts_prev >= 0.85)),
                       ("ALL DAYS vts<0.85 (generic)", vts_prev < 0.85),
                       ("ALL DAYS vts>=0.85", vts_prev >= 0.85)]:
            d = cal[(m & ok).values]
            rows.append(rec_row(ser.loc[d].values, f"{era} h={h} {lbl}", 12.0))
    show(rows, f"hedged short SVXY (entry opex close) by VIX/VIX3M at opex-1, h={h}")

print("\n=== c2 September row: short SPY from the Sept opex close, crush vs no crush ===")
v3_lag = vix.shift(1) / vix.shift(4) - 1
for h in (1, 2, 3, 5):
    r = -vehicle_ret(px, [("SPY", 1.0)], h, 0)
    sep = opex[opex.month == 9]
    cr = sep[(v3_lag.reindex(sep) <= -0.10).values]
    nc = sep.difference(cr)
    rows = [rec_row(r.reindex(cr).values, f"h={h} Sept & crush (A)", 3.0),
            rec_row(r.reindex(nc).values, f"h={h} Sept NO crush", 3.0)]
    show(rows)
    v = r.reindex(cr).dropna()
    print("   ", signed_concentration(v.index, v.values, k=1))
