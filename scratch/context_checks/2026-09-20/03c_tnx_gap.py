"""Exact shape of the 10-year's return to a 5% handle."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import close_panel, load_events, sign_test, summarize, show  # noqa

px = close_panel(["^TNX", "SPY", "^GSPC", "TLT", "IWM", "^VIX"]).dropna(subset=["^TNX"])
tnx = px["^TNX"]
ge5 = tnx[tnx >= 5.0]
prev = ge5[ge5.index < "2020-01-01"]
print("last 5%+ close before 2026:", str(prev.index[-1].date()), round(float(prev.iloc[-1]), 3))
print("the 2026 one:", str(ge5.index[-1].date()), round(float(ge5.iloc[-1]), 3))
gap_sessions = int(((tnx.index > prev.index[-1]) & (tnx.index < ge5.index[-1])).sum())
gap_years = (ge5.index[-1] - prev.index[-1]).days / 365.25
print(f"gap: {gap_sessions} sessions / {gap_years:.1f} years between them")

ev = load_events(["fomc_decision"])
fomc = set(pd.to_datetime(ev["date"]).dt.normalize())
print("was 2026-09-16 a scheduled FOMC decision day?", ge5.index[-1].normalize() in fomc)

print("\nlast 12 ^TNX closes:")
for d, v in tnx.iloc[-12:].items():
    print(f"  {d.date()} {float(v):.3f}")

# the low of the intervening period
mid = tnx[(tnx.index > prev.index[-1]) & (tnx.index <= ge5.index[-1])]
print(f"\nover those {gap_years:.1f} years the 10-year ranged "
      f"{float(mid.min()):.3f} ({mid.idxmin().date()}) to {float(mid.max()):.3f} ({mid.idxmax().date()})")

# How did equities do the LAST time the 10y crossed back above 5 from below?
# Cross = first close >= 5.0 after 60+ sessions below.
above = tnx >= 5.0
cross = above & (~above.rolling(60).max().astype(bool).shift(1).fillna(False))
cd = tnx.index[cross.reindex(tnx.index).fillna(False)]
print("\nfirst 5%+ close after 60+ sessions below:", [str(d.date()) for d in cd])
for h in (5, 21, 63):
    rows = []
    for name in ("SPY", "^GSPC", "^TNX"):
        s = px[name]
        f = s.shift(-h) / s - 1.0
        v = f.reindex(cd).dropna().values
        if len(v):
            r = summarize(v, f"{name} h{h}")
            up = int((v > 0).sum())
            r["rec"] = f"{up}-{len(v)-up}"
            rows.append(r)
    show(rows, f"crosses, h={h}")
print("\nNOTE n is tiny here by construction; this is a rarity statement, not a forecast.")

# Where does ^TNX sit vs its own 252d max, and how often has it been this close?
hi = tnx.rolling(252).max()
d = tnx / hi - 1.0
print(f"\nlive dist to 252d max: {100*float(d.iloc[-1]):+.3f}%")
near = d >= -0.005
print(f"sessions within 0.5% of the 252d max: {int(near.sum())} of {len(tnx)} "
      f"({100*near.mean():.1f}%)")
