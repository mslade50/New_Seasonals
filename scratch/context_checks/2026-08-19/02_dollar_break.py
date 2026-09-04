"""The dollar index closed below its 200d mean for the first time in 63+
sessions, on a 2-ATR down session, with its 21d return at the 0.4 percentile of
its own year. Price the three states separately and jointly. Then ask what the
equity tape was doing on the historical precious-metal thrust days."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    close_panel, cluster_note, declusters, era_split, fwd_ret, local_control,
    load_prices, sign_test, summarize,
)

TK = ["DX-Y.NYB", "GC=F", "SPY", "^VIX", "SI=F", "EEM", "TLT"]
px = close_panel(TK).dropna(subset=["DX-Y.NYB"])
dx = px["DX-Y.NYB"]
r = dx.pct_change()
sma = dx.rolling(200).mean()
above = dx > sma
print("DXY", dx.index[0].date(), "->", dx.index[-1].date(), "n", len(dx))
print("today close", round(float(dx.iloc[-1]), 3), "sma200",
      round(float(sma.iloc[-1]), 3), "ret", round(float(r.iloc[-1]) * 100, 2), "%")

# state 1: first close below the 200d in 63+ sessions
cross_dn = above.shift(1).fillna(False) & ~above
sess_since = pd.Series(np.nan, index=dx.index)
last = None
firsts = []
for d in dx.index:
    if bool(cross_dn.get(d, False)):
        if last is None or dx.index.get_loc(d) - dx.index.get_loc(last) >= 63:
            firsts.append(d)
        last = d
firsts = pd.DatetimeIndex(firsts)
print("\n=== state 1: first 200d break in 63+ sessions ===")
print("episodes", len(firsts), "| most recent before today:",
      firsts[-2].date() if len(firsts) > 1 else None)

# state 2: 21d return in the bottom 5% of its trailing year
r21 = dx.pct_change(21)
rank21 = r21.rolling(252).rank(pct=True) * 100
print("today 21d rank", round(float(rank21.iloc[-1]), 2))

# state 3: session move <= -2 ATR
hi_lo = load_prices(["DX-Y.NYB"])["DX-Y.NYB"]
tr = pd.concat([
    hi_lo["High"] - hi_lo["Low"],
    (hi_lo["High"] - hi_lo["Close"].shift()).abs(),
    (hi_lo["Low"] - hi_lo["Close"].shift()).abs()], axis=1).max(axis=1)
atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
move_atr = (hi_lo["Close"] - hi_lo["Close"].shift()) / atr.shift()
move_atr = move_atr.reindex(dx.index)
print("today move in ATR", round(float(move_atr.iloc[-1]), 2))

cells = {
    "S1 first 200d break in 63+": pd.DatetimeIndex(firsts),
    "S2 21d rank <= 1": dx.index[(rank21 <= 1).fillna(False)],
    "S3 session <= -2 ATR": dx.index[(move_atr <= -2).fillna(False)],
    "S1+S2 break with 21d rank <= 5": pd.DatetimeIndex(
        [d for d in firsts if (rank21.get(d, np.nan) or np.nan) <= 5]),
}
for lab, idx in cells.items():
    idx = pd.DatetimeIndex([d for d in idx if d < dx.index[-1]])
    trig = declusters(idx, 10, dx.index)
    print(f"\n--- {lab}  (raw {len(idx)}, declustered {len(trig)}) ---")
    if len(trig) == 0:
        continue
    for h in (1, 5, 21):
        f = fwd_ret(dx, h).reindex(trig).dropna()
        if len(f) == 0:
            continue
        s = summarize(f.values, f"h{h}")
        nup = int((f > 0).sum())
        allc = summarize(fwd_ret(dx, h).dropna().values)
        print(f"  DXY h{h:<3} n={s['n']:<3} mean={s['mean_pct']:+.3f}%  "
              f"{nup}-{len(f)-nup} up  t={s['t']:+.2f}  "
              f"sign_p={sign_test(nup, len(f)):.4f}  all-days {allc['mean_pct']:+.3f}%")
    for h in (5, 21):
        fg = fwd_ret(px["GC=F"], h).reindex(trig).dropna()
        if len(fg):
            s = summarize(fg.values)
            nup = int((fg > 0).sum())
            print(f"  gold h{h:<3} n={s['n']:<3} mean={s['mean_pct']:+.3f}%  "
                  f"{nup}-{len(fg)-nup} up  t={s['t']:+.2f}")
    f1 = fwd_ret(dx, 21).reindex(trig).dropna()
    if len(f1) >= 6:
        print("  era h21:", [(e["label"], e["n"], round(e["mean_pct"], 3))
                             for e in era_split(f1.index, f1.values)])
        print("  concentration h21:", cluster_note(f1.index, f1.values))
    if lab.startswith("S1"):
        for d in trig:
            print(f"    {d.date()}  21d rank "
                  f"{rank21.get(d, float('nan')):.1f}  fwd21 "
                  f"{fwd_ret(dx, 21).get(d, float('nan'))*100:+.2f}%")

# What was the equity tape doing on the gold+silver joint thrust days?
print("\n=== the company gold keeps: SPY and VIX on prior gold+silver 4% days ===")
g1 = px["GC=F"].pct_change()
s1 = px["SI=F"].pct_change()
spy1 = px["SPY"].pct_change()
vix1 = px["^VIX"].pct_change()
joint = declusters(
    pd.DatetimeIndex([d for d in g1.index[((g1 >= 0.04) & (s1 >= 0.04)).fillna(False)]
                      if d < g1.index[-1]]), 5, g1.index)
rows = []
for d in joint:
    rows.append((d.date(), g1.get(d, np.nan) * 100, s1.get(d, np.nan) * 100,
                 spy1.get(d, np.nan) * 100, vix1.get(d, np.nan) * 100))
for a, b, c, e, f in rows:
    print(f"  {a}  gold {b:+.2f}%  silver {c:+.2f}%  SPY {e:+.2f}%  VIX {f:+.2f}%")
sp = np.array([x[3] for x in rows], dtype=float)
vx = np.array([x[4] for x in rows], dtype=float)
print(f"  SPY on those days: mean {np.nanmean(sp):+.2f}%  "
      f"{int((sp > 0).sum())}-{int((sp < 0).sum())} up  "
      f"|SPY| < 0.5% on {int((np.abs(sp) < 0.5).sum())} of {len(sp)}")
print(f"  VIX on those days: mean {np.nanmean(vx):+.2f}%  "
      f"up on {int((vx > 0).sum())} of {int(np.isfinite(vx).sum())}")
print(f"  today: SPY {float(spy1.iloc[-1])*100:+.2f}%  "
      f"VIX {float(vix1.iloc[-1])*100:+.2f}%  gold {float(g1.iloc[-1])*100:+.2f}%  "
      f"silver {float(s1.iloc[-1])*100:+.2f}%")
