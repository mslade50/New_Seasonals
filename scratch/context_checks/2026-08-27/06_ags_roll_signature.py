"""Six agricultural subjects fired price triggers tonight. Test all of them
for the continuous-contract roll signature at once.

Signature, from drills 04 and 05: the session's move is almost entirely a GAP
with little or no intraday follow-through, volume jumps by a large multiple,
and the immediately preceding bars carry DUPLICATED volume, the tell that the
expiring contract had stopped trading and the cache was carrying a stale bar
forward.

A roll gap is a change of instrument, not a change of price, so any price
trigger that fires on one is measuring plumbing.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import load_prices  # noqa: E402

# every ags subject in tonight's fired price cells, plus the equity/vol
# controls that fired nothing suspicious
AGS = ["ZC=F", "ZW=F", "ZS=F", "KC=F", "CT=F", "CC=F", "SB=F"]
CONTROL = ["SPY", "GC=F", "CL=F", "HG=F", "^VIX"]
px = load_prices(AGS + CONTROL)

rows = []
for t in AGS + CONTROL:
    d = px[t]
    if len(d) < 70:
        continue
    last, prev = d.iloc[-1], d.iloc[-2]
    o, c0 = last.get("Open"), prev.get("Close")
    tot = 100 * (last["Close"] / c0 - 1)
    gap = 100 * (o / c0 - 1) if o and o > 0 else np.nan
    intra = 100 * (last["Close"] / o - 1) if o and o > 0 else np.nan
    share = (abs(intra) / (abs(gap) + abs(intra)) * 100
             if np.isfinite(gap) and (abs(gap) + abs(intra)) > 0 else np.nan)
    vol = last.get("Volume", np.nan)
    v63 = d["Volume"].tail(64).iloc[:-1].median() if "Volume" in d else np.nan
    prior5 = d["Volume"].tail(6).iloc[:-1] if "Volume" in d else pd.Series(dtype=float)
    rows.append({
        "ticker": t,
        "ags": t in AGS,
        "session%": round(tot, 2),
        "gap%": round(gap, 2) if np.isfinite(gap) else None,
        "intraday%": round(intra, 2) if np.isfinite(intra) else None,
        "intraday_share%": round(share, 0) if np.isfinite(share) else None,
        "vol_x63d": round(vol / v63, 2) if v63 and np.isfinite(v63) else None,
        "dup_vol_prior5": bool(prior5.duplicated().any()) if len(prior5) else None,
        "open_is_zero": bool(o == 0) if o is not None else None,
    })

df = pd.DataFrame(rows)
print("=== 2026-08-27 bar anatomy ===")
print(df.to_string(index=False))

a = df[df["ags"]]
c = df[~df["ags"]]
print(f"\nags subjects: {len(a)}")
print(f"  median |session move|      {a['session%'].abs().median():.2f}%")
print(f"  median intraday share      {a['intraday_share%'].median():.0f}%")
print(f"  carrying duplicated volume in the prior 5 bars: "
      f"{int(a['dup_vol_prior5'].sum())} of {len(a)}")
print(f"  volume multiple, median    {a['vol_x63d'].median():.2f}x")
print(f"  open printed as 0.00:      {int(a['open_is_zero'].sum())}")
print(f"\ncontrols: {len(c)}")
print(f"  median |session move|      {c['session%'].abs().median():.2f}%")
print(f"  median intraday share      {c['intraday_share%'].median():.0f}%")
print(f"  carrying duplicated volume: {int(c['dup_vol_prior5'].fillna(False).sum())} of {len(c)}")

# how unusual is a whole-complex duplicated-volume session?
print("\n=== how often do 5+ ags carry duplicated consecutive volume? ===")
panel = {}
for t in AGS:
    if "Volume" in px[t]:
        v = px[t]["Volume"]
        panel[t] = (v == v.shift(1)) & v.notna() & (v > 0)
P = pd.DataFrame(panel).fillna(False)
cnt = P.sum(axis=1)
hits = cnt[cnt >= 5]
print(f"  sessions with >=5 of {len(panel)} ags repeating the prior volume: {len(hits)}")
if len(hits):
    print("  years:", pd.Series(hits.index.year).value_counts().sort_index().to_dict())
    print("  most recent 10:", [str(d.date()) for d in hits.index[-10:]])
print(f"  most recent session's count: {int(cnt.iloc[-1])} "
      f"(prior session {int(cnt.iloc[-2])})")

# the gap-dominance test across the complex, tonight vs history
print("\n=== gap-dominated sessions (intraday share < 25%, |move| > 2%) ===")
for t in AGS:
    d = px[t]
    if "Open" not in d:
        continue
    o, cl = d["Open"], d["Close"]
    prevc = cl.shift(1)
    ok = (o > 0) & prevc.notna()
    tot = (cl / prevc - 1).abs() * 100
    gp = ((o / prevc - 1) * 100).abs()
    itr = ((cl / o - 1) * 100).abs()
    shr = itr / (gp + itr) * 100
    m = ok & (tot > 2) & (shr < 25)
    print(f"  {t:>6}: {int(m.sum()):4d} such sessions in {int(ok.sum())}, "
          f"tonight qualifies: {bool(m.iloc[-1])}")
