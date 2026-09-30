"""X1 round 1: long MXN (short USDMXN) after a carry-unwind session.

Pre-specified rule: USDMXN=X one-day change >= +1.25% AND (^MOVE up OR ^VIX up)
on the same date. Long MXN = MXNUSD = 1/USDMXN (the 6M futures side). Entry is
lag=1 (the NEXT FX close after the signal bar), h = 1..10. Yahoo FX closes are
stamped at the London/NY roll, not an exchange close; lean on h >= 2.

Carry: a spot-only series understates long-MXN by the forward points. MXN_TR adds
an APPROXIMATE yearly Banxico-minus-Fed policy differential (hand table below,
annual averages from memory, +/- 1pp) accrued per calendar day. Carry is additive
and nearly constant, so it moves the absolute return, not the edge vs own drift.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

DIFF = {2003: 4.5, 2004: 5.6, 2005: 6.0, 2006: 2.3, 2007: 2.3, 2008: 6.0, 2009: 5.2, 2010: 4.3,
        2011: 4.4, 2012: 4.3, 2013: 3.9, 2014: 3.1, 2015: 2.9, 2016: 3.8, 2017: 5.8, 2018: 5.9,
        2019: 5.7, 2020: 4.9, 2021: 4.3, 2022: 6.0, 2023: 6.2, 2024: 5.7, 2025: 4.3, 2026: 3.2}

P = load_prices(["USDMXN=X", "^MOVE", "^VIX", "DX-Y.NYB", "^TNX"])
fx = P["USDMXN=X"]["Close"].dropna()
fx = fx[fx > 0]
cal = fx.index
mxn = 1.0 / fx
dly = mxn.pct_change()
days = pd.Series(cal, index=cal).diff().dt.days.fillna(1)
carry = days * pd.Series([DIFF.get(d.year, 4.5) for d in cal], index=cal) / 100.0 / 365.0
tr = (1 + (dly.fillna(0) + carry)).cumprod()
px = pd.DataFrame({"MXN": mxn, "MXN_TR": tr})

usd1 = fx.pct_change()


def vchg(t):
    s = P[t]["Close"].dropna()
    return (s / s.shift(1) - 1).reindex(cal)


mv, vx = vchg("^MOVE"), vchg("^VIX")
volup = (mv > 0) | (vx > 0)
has_vol = mv.notna() | vx.notna()
m_base = usd1 >= 0.0125
trig = m_base & volup
print(f"live 2026-09-24: USDMXN d1 {100*usd1.iloc[-1]:+.2f}%  MOVE {100*mv.iloc[-1]:+.2f}%  "
      f"VIX {100*vx.iloc[-1]:+.2f}%  trigger={bool(trig.iloc[-1])}")
print(f"USDMXN +1.25% is the {100*(usd1.dropna() < 0.0125).mean():.1f}th pctile of daily moves "
      f"(2003+); +1.48% is the {100*(usd1.dropna() < usd1.iloc[-1]).mean():.1f}th")
print(f"trigger days {int(trig.sum())}, base (+1.25% any) days {int(m_base.sum())}, "
      f"base with vol data {int((m_base & has_vol).sum())}")

variants = {
    "base +1.25% no vol gate": m_base,
    "+1.25% vol NOT up": m_base & has_vol & ~volup,
    "+1.25% MOVE up": m_base & (mv > 0),
    "+1.25% VIX up": m_base & (vx > 0),
    "+1.25% both up": m_base & (mv > 0) & (vx > 0),
    "+1.00% & vol up": (usd1 >= 0.010) & volup,
    "+1.50% & vol up": (usd1 >= 0.015) & volup,
    "+2.00% & vol up": (usd1 >= 0.020) & volup,
}
for h in (2, 5, 10):
    battery(px, trig, [("MXN", 1.0)], h, f"X1 long MXN spot, h={h}", cost_bps=3,
            variants=variants, min_gap=10, event_kinds=("nfp",))
    battery(px, trig, [("MXN_TR", 1.0)], h, f"X1 long MXN CARRY-ADJ, h={h}", cost_bps=3,
            min_gap=10, event_kinds=("nfp",))

# horizon table, episodes declustered at 10
tr_dates = px.index[trig.reindex(px.index, fill_value=False).values]
show(horizon_scan(px, tr_dates, [("MXN_TR", 1.0)], hs=tuple(range(1, 11)), min_gap=10),
     "horizon scan, carry-adj, episodes (gap 10)")
show(horizon_scan(px, tr_dates, [("MXN", 1.0)], hs=tuple(range(1, 11)), min_gap=10),
     "horizon scan, spot, episodes (gap 10)")

# gate attribution at the episode level, h=5 carry-adj
r5 = fwd_lag(px["MXN_TR"], 5)
rows = []
for lbl, m in [("trigger (vol up)", trig), ("base no gate", m_base), ("vol NOT up", m_base & has_vol & ~volup)]:
    d = px.index[m.reindex(px.index, fill_value=False).values & r5.notna().values]
    e = declusters(d, 10, px.index)
    v = r5.loc[e].values
    r = summarize(v, lbl)
    w = int((v > 0).sum())
    r["rec"] = f"{w}-{len(v)-w}"
    r["sign_p"] = round(sign_test(w, len(v)), 4)
    rows.append(r)
show(rows, "gate attribution h=5 carry-adj, episodes gap 10")

# is it 'the dollar breakout fails'? trigger days with DX within 0.5% of its 252 high
dx = P["DX-Y.NYB"]["Close"].dropna()
dxhi = dx / rolling_on_valid(dx, lambda x: x.rolling(252).max()) - 1
dxz = zscore(dx, 10)
dxb = ((dxhi >= -0.005) & (dxz >= 2)).reindex(cal).fillna(False)
dxnear = (dxhi >= -0.005).reindex(cal).fillna(False)
rows = []
for lbl, m in [("trigger & DX breakout (z10>=2, <=0.5% off hi)", trig & dxb),
               ("trigger & DX within 0.5% of 252 hi", trig & dxnear),
               ("trigger & DX NOT near hi", trig & ~dxnear)]:
    d = px.index[m.values & r5.notna().values]
    e = declusters(d, 10, px.index)
    rows.append(summarize(r5.loc[e].values, f"{lbl} (N days {len(d)})"))
show(rows, "overlap with the killed DX-breakout pole, h=5 carry-adj episodes")
