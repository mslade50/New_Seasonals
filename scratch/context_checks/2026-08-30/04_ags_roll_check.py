"""Gate for every ags nugget: is Friday's grain bar a price move or a
continuous-contract roll?

Friday printed ZW=F +5.49% to a 52-week high, ZC=F +5.10% to a 52-week high,
ZS=F +2.49%, CC=F +7.81%, PA=F +8.10%, KC=F -8.57%. The 2026-08-27 brief ran
this exact check on the previous session and tagged corn's bar as plumbing, so
nothing in tonight's ags lane can be believed until the same test is run on
Friday's bar.

Signature, carried over from 2026-08-27 drills 04 to 06: the session's move is
almost entirely a GAP with little intraday follow-through, volume jumps by a
large multiple, and preceding bars carry duplicated volume, the tell that the
expiring contract stopped trading and the cache carried a stale bar forward.

A roll gap is a change of instrument, not a change of price. Any trigger that
fires on one is measuring plumbing.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import load_prices  # noqa: E402

AGS = ["ZC=F", "ZW=F", "ZS=F", "KC=F", "CT=F", "CC=F", "SB=F", "PA=F"]
CONTROL = ["SPY", "GC=F", "CL=F", "HG=F", "^VIX"]
px = load_prices(AGS + CONTROL)

ASOF = pd.Timestamp("2026-08-28")

rows = []
for t in AGS + CONTROL:
    d = px[t]
    if ASOF not in d.index:
        rows.append({"ticker": t, "note": "NO BAR on asof"})
        continue
    i = d.index.get_loc(ASOF)
    last, prev = d.iloc[i], d.iloc[i - 1]
    o, c0 = last.get("Open"), prev["Close"]
    tot = 100 * (last["Close"] / c0 - 1)
    gap = 100 * (o / c0 - 1) if o and o > 0 else np.nan
    intra = 100 * (last["Close"] / o - 1) if o and o > 0 else np.nan
    denom = abs(gap) + abs(intra) if np.isfinite(gap) else np.nan
    share = (abs(intra) / denom * 100) if denom and denom > 0 else np.nan
    vol = last.get("Volume", np.nan)
    hist = d.iloc[max(0, i - 64):i]
    v63 = hist["Volume"].median() if "Volume" in d else np.nan
    prior5 = d.iloc[max(0, i - 5):i]["Volume"] if "Volume" in d else pd.Series(dtype=float)
    rows.append({
        "ticker": t,
        "ags": t in AGS,
        "session%": round(tot, 2),
        "gap%": round(gap, 2) if np.isfinite(gap) else None,
        "intraday%": round(intra, 2) if np.isfinite(intra) else None,
        "intraday_share%": round(share, 0) if np.isfinite(share) else None,
        "vol_x63d": round(vol / v63, 2) if v63 and np.isfinite(v63) and v63 > 0 else None,
        "dup_vol_prior5": bool(prior5.duplicated().any()) if len(prior5) else None,
    })

df = pd.DataFrame(rows)
print("=== 2026-08-28 bar anatomy ===")
print(df.to_string(index=False))

print()
print("=== how the same subjects looked on 2026-08-27 (the bar the previous")
print("    brief flagged), for continuity ===")
prev_rows = []
for t in AGS:
    d = px[t]
    pd_asof = pd.Timestamp("2026-08-27")
    if pd_asof not in d.index:
        continue
    i = d.index.get_loc(pd_asof)
    last, prev = d.iloc[i], d.iloc[i - 1]
    o, c0 = last.get("Open"), prev["Close"]
    tot = 100 * (last["Close"] / c0 - 1)
    gap = 100 * (o / c0 - 1) if o and o > 0 else np.nan
    intra = 100 * (last["Close"] / o - 1) if o and o > 0 else np.nan
    prev_rows.append({"ticker": t, "session%": round(tot, 2),
                      "gap%": round(gap, 2) if np.isfinite(gap) else None,
                      "intraday%": round(intra, 2) if np.isfinite(intra) else None})
print(pd.DataFrame(prev_rows).to_string(index=False))

print()
print("=== the 5-session run into Friday, close by close ===")
for t in ["ZW=F", "ZC=F", "ZS=F"]:
    d = px[t]
    i = d.index.get_loc(ASOF)
    w = d.iloc[i - 6:i + 1][["Open", "High", "Low", "Close", "Volume"]]
    w = w.assign(chg_pct=(100 * (w["Close"] / w["Close"].shift(1) - 1)).round(2))
    print(f"\n  {t}")
    print(w.to_string())

print()
print("=== verdict inputs ===")
a = df[df["ags"] == True]  # noqa: E712
for _, r in a.iterrows():
    if r.get("intraday_share%") is None:
        continue
    flag = []
    if r["intraday_share%"] is not None and r["intraday_share%"] < 25:
        flag.append("move is mostly GAP")
    if r["vol_x63d"] is not None and r["vol_x63d"] > 3:
        flag.append("volume spike")
    if r["dup_vol_prior5"]:
        flag.append("duplicated prior volume")
    print(f"  {r['ticker']:6s} {r['session%']:+6.2f}%  "
          f"intraday share {r['intraday_share%']:.0f}%  "
          f"vol x{r['vol_x63d']}  ->  {', '.join(flag) if flag else 'looks like a real session'}")
