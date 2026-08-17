"""C3 round 2 -- the only surviving direction (long SVXY h=10) torn down.

Round 1 (v1_c3_termstructure.py) killed the complacency-fade direction in
every vehicle (UVXY -2.792% on 8-20, short QQQ h=10 -0.916% on 18-38) and
left long SVXY h=10 at +1.672% on 20 episodes, 15-5.

This script asks the four round-2 questions of THAT cell:
  1. concentration -- by episode AND by year
  2. definition neighbours -- offset ladder (is the trigger a LAGGING marker?)
     + lookback window + threshold
  3. era / instrument-break handling (post-break only by construction)
  4. gate attribution vs 'VIX level is low' with the LOCAL control attached
  5. base-rate-corrected sign test (2026-08-11 trap: SVXY drifts up)
  6. calendar overlap with the LIVE event sleeve V4 (long SVXY, opex -> +3 td)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

SVXY_BREAK = pd.Timestamp("2018-03-01")
H = 10

px = close_panel(["^VIX", "^VIX3M", "SVXY", "SPY"])
ratio = (px["^VIX"] / px["^VIX3M"]).dropna()


def level_rank(s, lb=252):
    return s.rolling(lb).rank(pct=True) * 100.0


rr = level_rank(ratio, 252)
vixr = level_rank(px["^VIX"], 252)
post = px.loc[px.index >= SVXY_BREAK]
ret = vehicle_ret(post, [("SVXY", 1.0)], H, 1)
valid = ret.dropna().index

trig = pd.DatetimeIndex(post.index[(rr <= 2.0).reindex(post.index, fill_value=False).values])
trig = trig.intersection(valid)
epi = declusters(trig, H, valid)
ep = ret.loc[epi].values

print("=" * 78)
print("1. CONCENTRATION (episode and YEAR)")
print("=" * 78)
byyr = pd.Series(ep, index=epi.year).groupby(level=0).agg(["sum", "count", "mean"])
byyr["sum"] = (100 * byyr["sum"]).round(2)
byyr["mean"] = (100 * byyr["mean"]).round(2)
print(byyr.to_string())
tot = 100 * ep.sum()
top2y = byyr["sum"].sort_values(ascending=False).head(2)
print(f"  total {tot:+.2f}pp over {len(ep)} episodes; top-2 YEARS "
      f"{dict(top2y)} = {100*top2y.sum()/tot:.0f}% of it")
srt = np.sort(ep)
print(f"  drop-best-episode mean {100*srt[:-1].mean():+.3f}%   "
      f"drop-best-2 {100*srt[:-2].mean():+.3f}%   drop-best-3 {100*srt[:-3].mean():+.3f}%")
loyo = {int(y): round(100 * ep[epi.year != y].mean(), 3) for y in sorted(set(epi.year))}
print(f"  leave-one-year-out means: {loyo}")
print(f"  LOYO floor {min(loyo.values()):+.3f}%")

print("\n" + "=" * 78)
print("2. DEFINITION NEIGHBOURS")
print("=" * 78)
print("\n2a. OFFSET LADDER -- enter k sessions from the trigger day "
      "(k<0 = BEFORE it). A plateau or an earlier peak = lagging marker.")
pos = pd.Series(range(len(post.index)), index=post.index)
rows = []
for k in range(-10, 6):
    shifted = []
    for d in trig:
        p = pos.get(d)
        if p is None:
            continue
        q = p + k
        if 0 <= q < len(post.index):
            shifted.append(post.index[q])
    s = pd.DatetimeIndex(sorted(set(shifted))).intersection(valid)
    e = declusters(s, H, valid)
    r = summarize(ret.loc[e].values, f"offset {k:+d}")
    rows.append(r)
show(rows, "2a. offset ladder, long SVXY h=10 (episodes)")
real = [r for r in rows if r["label"] == "offset +0"][0]["mean_pct"]
better = sum(1 for r in rows if r.get("mean_pct", -9e9) > real)
print(f"  the TRUE anchor (offset 0) ranks {better+1} of {len(rows)} offsets")

print("\n2b. LOOKBACK + THRESHOLD neighbours (episode level)")
rows = []
for lb in (126, 252, 504):
    rlb = level_rank(ratio, lb)
    for thr in (1.0, 2.0, 5.0, 10.0):
        m = (rlb <= thr)
        s = pd.DatetimeIndex(post.index[m.reindex(post.index, fill_value=False).values]).intersection(valid)
        if len(s) == 0:
            continue
        e = declusters(s, H, valid)
        r = summarize(ret.loc[e].values, f"lb={lb} thr<={thr}")
        rows.append(r)
# raw LEVEL definition, no ranking at all
for lvl in (0.78, 0.80, 0.85):
    m = (ratio <= lvl)
    s = pd.DatetimeIndex(post.index[m.reindex(post.index, fill_value=False).values]).intersection(valid)
    e = declusters(s, H, valid)
    rows.append(summarize(ret.loc[e].values, f"raw level <= {lvl}"))
show(rows, "2b. definition neighbours")

print("\n" + "=" * 78)
print("3. GATE ATTRIBUTION with the LOCAL control attached")
print("=" * 78)
vlow = (vixr <= 10)
for lbl, m in (("ratio<=2 & vix-low (TODAY'S cell)", (rr <= 2.0) & vlow),
               ("vix-low alone", vlow),
               ("ratio<=2 alone", (rr <= 2.0))):
    s = pd.DatetimeIndex(post.index[m.reindex(post.index, fill_value=False).values]).intersection(valid)
    e = declusters(s, H, valid)
    loc = local_control(valid, s)
    v = ret.loc[e].values
    w = int((v > 0).sum())
    base = float((ret.loc[valid] > 0).mean())
    print(f"\n  {lbl}: N_ep={len(e)}  mean {100*v.mean():+.3f}%  "
          f"LOCAL ctrl {100*ret.loc[loc].mean():+.3f}%  "
          f"EDGE {100*(v.mean()-ret.loc[loc].mean()):+.3f}pp")
    print(f"    record {w}-{len(v)-w}; sign p vs coin {sign_test(w, len(v)):.4f}; "
          f"sign p vs SVXY's OWN {100*base:.1f}% base rate "
          f"{sign_test(w, len(v), p=base):.4f}")

print("\n" + "=" * 78)
print("4. BOOK OVERLAP -- live event sleeve V4_POSTOPEX_VOL (long SVXY 10% NAV,")
print("   opex MOC -> +3 sessions MOC, every month except September)")
print("=" * 78)
opex = load_events(["opex"])["date"]
allidx = post.index
posn = pd.Series(range(len(allidx)), index=allidx)
v4_days = set()
for d in opex:
    p = posn.get(pd.Timestamp(d))
    if p is None or d.month == 9:
        continue
    for q in range(p + 1, min(p + 4, len(allidx))):
        v4_days.add(allidx[q])
# would a 10td SVXY long entered at trigger+1 overlap a V4 hold?
ov = []
for d in epi:
    p = posn.get(d)
    span = set(allidx[p + 1: p + 2 + H])
    ov.append(len(span & v4_days))
ov = np.array(ov)
print(f"  historical episodes whose 10td hold touches a V4 window: "
      f"{(ov > 0).sum()} of {len(ov)} ({100*(ov>0).mean():.0f}%), "
      f"mean {ov.mean():.1f} overlapping sessions")
print("  TODAY: entry MOC 2026-08-18, hold 08-19..09-01 (10 td).")
print("         V4 fires MOC 2026-08-21 and holds 08-24..08-26.")
print("         -> 3 of 10 held sessions are DOUBLE-LONG SVXY beside a live sleeve leg.")
