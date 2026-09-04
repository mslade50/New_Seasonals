"""Gate for the coffee nugget: is today's -9.21% a price move or a roll?

KC=F printed the largest single-session move in the 98-name tape, -9.21%, and
its 5-day return is the 0.8th percentile of its own year. Two triggers fired
on it (P5 bottom 5%, P6 two-ATR down).

The 2026-08-30 brief ran this exact test on the 2026-08-28 bar and found
coffee's session was 96% gap, i.e. a continuous-contract roll, so nothing in
tonight's KC=F lane is believable until the same test clears on today's bar.

Signature: the move is almost entirely a GAP with little intraday
follow-through, volume jumps by a large multiple of its own median, and the
preceding bars carry duplicated volume (the tell that the expiring contract
stopped trading and the cache carried a stale bar forward).

A roll gap is a change of instrument, not a change of price.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import load_prices  # noqa: E402

ASOF = pd.Timestamp("2026-08-31")
SUSPECTS = ["KC=F", "ZC=F", "ZW=F", "ZS=F", "CT=F", "CC=F", "SB=F", "LE=F", "HE=F", "PA=F"]
CONTROL = ["SPY", "GC=F", "CL=F", "HG=F"]

px = load_prices(SUSPECTS + CONTROL)

rows = []
for t in SUSPECTS + CONTROL:
    d = px.get(t)
    if d is None or ASOF not in d.index:
        rows.append({"ticker": t, "note": "NO BAR"})
        continue
    i = d.index.get_loc(ASOF)
    bar, prev = d.iloc[i], d.iloc[i - 1]
    c0 = prev["Close"]
    o = bar.get("Open")
    tot = 100 * (bar["Close"] / c0 - 1)
    gap = 100 * (o / c0 - 1) if o and o > 0 else np.nan
    intra = 100 * (bar["Close"] / o - 1) if o and o > 0 else np.nan
    denom = abs(gap) + abs(intra)
    gap_share = 100 * abs(gap) / denom if denom and np.isfinite(denom) and denom > 0 else np.nan

    vol = bar.get("Volume", np.nan)
    med63 = d["Volume"].iloc[max(0, i - 63):i].median() if "Volume" in d else np.nan
    vmult = vol / med63 if med63 and med63 > 0 else np.nan
    # duplicated-volume tell over the five bars ending today
    lastv = d["Volume"].iloc[i - 4:i + 1].tolist() if "Volume" in d else []
    dupes = len(lastv) - len(set(lastv)) if lastv else 0

    rows.append({
        "ticker": t, "tot_pct": round(tot, 2),
        "gap_pct": round(gap, 2) if np.isfinite(gap) else None,
        "intra_pct": round(intra, 2) if np.isfinite(intra) else None,
        "gap_share_pct": round(gap_share, 0) if np.isfinite(gap_share) else None,
        "vol_x_med63": round(vmult, 2) if np.isfinite(vmult) else None,
        "dup_vols_last5": dupes,
    })

print("=== 2026-08-31 bar decomposition (roll signature: gap_share high, vol_x extreme, dupes>0) ===")
print(pd.DataFrame(rows).to_string(index=False))

print("\n=== KC=F last 10 bars ===")
k = px["KC=F"]
i = k.index.get_loc(ASOF)
sub = k.iloc[i - 9:i + 1][["Open", "High", "Low", "Close", "Volume"]].copy()
sub["ret_pct"] = (100 * (sub["Close"] / k["Close"].shift(1).iloc[i - 9:i + 1] - 1)).round(2)
print(sub.to_string())

# How often has KC=F moved this hard in one session, and does the tape agree it
# was a real move? Compare against the distribution of |1d| moves.
r = k["Close"].pct_change()
big = r[r <= -0.06]
print(f"\nKC=F sessions <= -6% since {k.index[0].date()}: {len(big)}")
print("  most recent 8:", [f"{d.date()} {100*v:+.1f}%" for d, v in big.tail(8).items()])
