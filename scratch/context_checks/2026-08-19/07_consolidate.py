"""Three loose ends before composing.
(1) Jackson Hole run-in, with the right column name this time.
(2) The BTC 5d-rank cell's h5 era split and concentration.
(3) One explicit definition of the gold+silver joint-thrust episode set, on one
    panel, plus a look at the 2026-01-28 bar whose VIX change printed 0.00%."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    close_panel, cluster_note, declusters, era_split, fwd_ret, load_events,
    load_prices, sign_test, summarize,
)

print("########## 1. Jackson Hole ##########")
ev = load_events()
jh = ev[ev["event"].astype(str).str.contains("jackson", case=False, na=False)]
print("  rows:", len(jh))
dates = pd.DatetimeIndex(pd.to_datetime(jh["date"]))
print("  dates:", [d.date().isoformat() for d in dates])
mkt = close_panel(["SPY", "GC=F", "DX-Y.NYB", "^VIX", "TLT"]).dropna(subset=["SPY"])
ai = mkt.index
pos = {d: i for i, d in enumerate(ai)}
anch = []
for d in dates:
    prior = ai[ai < d]
    if len(prior) < 10:
        continue
    i = pos[prior[-1]] - 6
    if i >= 0:
        anch.append(ai[i])
anch = pd.DatetimeIndex([a for a in anch if a < ai[-1]])
print("  anchors 7 td before the opening:", len(anch),
      [a.date().isoformat() for a in anch])
for nm in ("SPY", "GC=F", "DX-Y.NYB", "^VIX", "TLT"):
    s = mkt[nm].dropna()
    f = fwd_ret(s, 7).reindex(anch).dropna()
    if len(f) < 5:
        print(f"  {nm:<10} n={len(f)} too few")
        continue
    st = summarize(f.values)
    nup = int((f > 0).sum())
    allc = summarize(fwd_ret(s, 7).dropna().values)
    print(f"  {nm:<10} run-in 7td n={st['n']:<3} mean={st['mean_pct']:+.3f}%  "
          f"{nup}-{len(f)-nup} up  t={st['t']:+.2f}  "
          f"sign_p={sign_test(nup, len(f)):.4f}  all-days {allc['mean_pct']:+.3f}%")

print("\n########## 2. BTC 5d-rank cell, h5 detail ##########")
btc = close_panel(["BTC-USD"])["BTC-USD"].dropna()
rank = (btc.pct_change(5).rolling(252).rank(pct=True) * 100)
idx = pd.DatetimeIndex([d for d in btc.index[(rank >= 95).fillna(False)]
                        if d < btc.index[-1]])
trig = declusters(idx, 5, btc.index)
f5 = fwd_ret(btc, 5).reindex(trig).dropna()
print("  n", len(f5))
for lab, m in (("pre-2018", f5.index < pd.Timestamp("2018-01-01")),
               ("2018+", f5.index >= pd.Timestamp("2018-01-01"))):
    v = f5[m]
    st = summarize(v.values)
    nup = int((v > 0).sum())
    print(f"  {lab:<9} n={st['n']:<3} mean={st['mean_pct']:+.3f}%  "
          f"{nup}-{len(v)-nup} up  t={st['t']:+.2f}  sign_p={sign_test(nup, len(v)):.4f}")
print(" ", cluster_note(f5.index, f5.values))
print("  today BTC 5d rank", round(float(rank.iloc[-1]), 1))

print("\n########## 3. gold+silver joint thrust, one definition ##########")
# Panel dropped on gold and silver only; SPY/VIX joined where they exist.
p = close_panel(["GC=F", "SI=F", "SPY", "^VIX"]).dropna(subset=["GC=F", "SI=F"])
g, s = p["GC=F"].pct_change(), p["SI=F"].pct_change()
sp, vx = p["SPY"].pct_change(), p["^VIX"].pct_change()
mask = ((g >= 0.04) & (s >= 0.04)).fillna(False)
raw = pd.DatetimeIndex([d for d in p.index[mask] if d < p.index[-1]])
trig = declusters(raw, 5, p.index)
print(f"  panel {p.index[0].date()} -> {p.index[-1].date()}, n={len(p)}")
print(f"  raw {len(raw)}, declustered at 5 td {len(trig)}")
print("  date         gold    silver   SPY      VIX")
for d in trig:
    print(f"  {d.date()}  {g[d]*100:+6.2f}%  {s[d]*100:+6.2f}%  "
          f"{sp.get(d, np.nan)*100:+6.2f}%  {vx.get(d, np.nan)*100:+7.2f}%")
spv = np.array([sp.get(d, np.nan) for d in trig], dtype=float) * 100
vxv = np.array([vx.get(d, np.nan) for d in trig], dtype=float) * 100
print(f"  SPY that session: mean {np.nanmean(spv):+.2f}%, "
      f"down on {int((spv < 0).sum())} of {int(np.isfinite(spv).sum())}, "
      f"|move| < 0.5% on {int((np.abs(spv) < 0.5).sum())}")
print(f"  VIX that session: up on {int((vxv > 0).sum())} of "
      f"{int(np.isfinite(vxv).sum())}, mean {np.nanmean(vxv):+.2f}%")
print(f"  today: gold {float(g.iloc[-1])*100:+.2f}%  silver {float(s.iloc[-1])*100:+.2f}%"
      f"  SPY {float(sp.iloc[-1])*100:+.2f}%  VIX {float(vx.iloc[-1])*100:+.2f}%")
for h in (1, 5, 21):
    f = fwd_ret(p["GC=F"], h).reindex(trig).dropna()
    st = summarize(f.values)
    nup = int((f > 0).sum())
    print(f"  gold after, h{h:<3} n={st['n']:<3} mean={st['mean_pct']:+.3f}%  "
          f"med={st['median_pct']:+.3f}%  {nup}-{len(f)-nup} up  "
          f"sign_p={sign_test(nup, len(f)):.4f}")

print("\n  --- the 2026-01-28 bar, whose VIX change printed 0.00% ---")
raw_px = load_prices(["^VIX", "SPY", "GC=F"])
v = raw_px["^VIX"]
w = v.loc["2026-01-23":"2026-01-30"]
print(w[["Open", "High", "Low", "Close"]].round(3).to_string())
sy = raw_px["SPY"].loc["2026-01-23":"2026-01-30"]
print(sy[["Open", "High", "Low", "Close", "Volume"]].round(2).to_string())
