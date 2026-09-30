"""Nasdaq back at a 52-week high on a 2%+ day.

A. ^IXIC first 52w closing high in 30+ calendar days (the BH-passing engine cell),
   split by the size of the breakout session and by drought length.
B. ^NDX 2-ATR up days (the engine's N=21 cell), split by distance to the 52w high.
C. The crossing: ^NDX up 2%+ closing within 1% of its 52w high, declustered.
Forward returns lag 0 close-to-close from the signal close.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import load_prices, fwd_ret, summarize, era_split, sign_test, cluster_note, show, declusters, local_control  # noqa
from pitch_grammar import wilder_atr  # noqa

px = load_prices(["^IXIC", "^NDX", "QQQ", "^GSPC", "SPY", "^VIX"])
START = pd.Timestamp("1999-01-01")


def first_high(close: pd.Series, days: int) -> pd.Series:
    hi = close >= close.rolling(252, min_periods=252).max()
    out = pd.Series(False, index=close.index)
    last = None
    for d in close.index[hi.fillna(False).values]:
        if last is None or (d - last).days > days:
            out.loc[d] = True
        last = d
    return out


def fwd_table(close: pd.Series, dates, label: str, hs=(1, 5, 21)) -> list[dict]:
    rows = []
    for h in hs:
        r = fwd_ret(close, h).reindex(dates).dropna()
        s = summarize(r.values, f"{label} h{h}")
        s["up"] = int((r > 0).sum())
        s["sign_p_up"] = sign_test(int((r > 0).sum()), len(r))
        s["sign_p_dn"] = sign_test(int((r < 0).sum()), len(r))
        rows.append(s)
    return rows


# ---------------------------------------------------------------- A
ix = px["^IXIC"]["Close"].astype(float)
ix = ix[ix.index >= START - pd.Timedelta(days=400)]
r1 = ix.pct_change()
p1 = first_high(ix, 30)
p1 = p1[p1.index >= START]
ev = p1.index[p1.values]
print(f"^IXIC P1 events since 1999: {len(ev)} (incl. today {ev[-1].date()})")

hi_mask = ix >= ix.rolling(252, min_periods=252).max()
hi_dates = ix.index[hi_mask.fillna(False).values]
prev_hi = hi_dates[hi_dates < ix.index[-1]][-1]
print(f"previous ^IXIC 52w closing high: {prev_hi.date()}  ({(ix.index[-1]-prev_hi).days} cal days)")

det = pd.DataFrame({"ret_day": r1.reindex(ev) * 100})
gaps = []
for d in ev:
    prior = hi_dates[hi_dates < d]
    gaps.append((d - prior[-1]).days if len(prior) else np.nan)
det["drought_days"] = gaps
for h in (1, 5, 21):
    det[f"h{h}"] = fwd_ret(ix, h).reindex(ev) * 100
print(det.round(2).to_string())

hist = ev[:-1]
show(fwd_table(ix, hist, "all P1"), "A. ^IXIC first 52w high in 30+ days")
big = [d for d in hist if r1[d] >= 0.02]
small = [d for d in hist if r1[d] < 0.02]
show(fwd_table(ix, pd.DatetimeIndex(big), "day>=+2%") + fwd_table(ix, pd.DatetimeIndex(small), "day<+2%"),
     "A2. split by breakout-day size")
long_d = [d for d, g in zip(hist, gaps[:-1]) if g >= 90]
short_d = [d for d, g in zip(hist, gaps[:-1]) if g < 90]
show(fwd_table(ix, pd.DatetimeIndex(long_d), "drought>=90") + fwd_table(ix, pd.DatetimeIndex(short_d), "drought 30-89"),
     "A3. split by drought length")
h1 = fwd_ret(ix, 1).reindex(hist).dropna()
show(era_split(h1.index, h1.values), "A4. era split h1")
h5 = fwd_ret(ix, 5).reindex(hist).dropna()
show(era_split(h5.index, h5.values), "A4b. era split h5")
print("A5 concentration h1:", cluster_note(h1.index, h1.values))
alld = fwd_ret(ix, 1)
alld = alld[alld.index >= START].dropna()
print(f"A6 control all days h1: mean {100*alld.mean():.3f}%  hit {100*(alld>0).mean():.1f}%")
for h in (5, 21):
    a = fwd_ret(ix, h)
    a = a[a.index >= START].dropna()
    print(f"   control all days h{h}: mean {100*a.mean():.3f}%  hit {100*(a>0).mean():.1f}%")
lc = local_control(ix.index[ix.index >= START], hist)
lcv = fwd_ret(ix, 1).reindex(lc).dropna()
print(f"A7 local +/-126td control h1: mean {100*lcv.mean():.3f}%  hit {100*(lcv>0).mean():.1f}%  n {len(lcv)}")
# all days at a 52w high (not first) as a second control
athi = hi_dates[(hi_dates >= START)]
athi = athi.difference(ev)
v = fwd_ret(ix, 1).reindex(athi).dropna()
print(f"A8 any other 52w-high close h1: mean {100*v.mean():.3f}%  hit {100*(v>0).mean():.1f}% n {len(v)}")

# ---------------------------------------------------------------- B
nd = px["^NDX"]
c = nd["Close"].astype(float)
atr = pd.Series(wilder_atr(nd["High"].to_numpy(), nd["Low"].to_numpy(), c.to_numpy()), index=nd.index)
up2 = (c.diff() >= 2.0 * atr.shift(1)) & (c.diff() > 0)
up2 = up2[up2.index >= START]
dates2 = up2.index[up2.values]
dist = (c / c.rolling(252, min_periods=252).max() - 1) * 100
b = pd.DataFrame({"ret": c.pct_change().reindex(dates2) * 100, "dist_hi": dist.reindex(dates2),
                  "atr_mult": (c.diff() / atr.shift(1)).reindex(dates2),
                  "h1": fwd_ret(c, 1).reindex(dates2) * 100, "h5": fwd_ret(c, 5).reindex(dates2) * 100,
                  "h21": fwd_ret(c, 21).reindex(dates2) * 100})
print("\nB. ^NDX 2-ATR up days since 1999")
print(b.round(2).to_string())

# ---------------------------------------------------------------- C
r = c.pct_change()
near = dist >= -1.0
cand = (r >= 0.02) & near
cand = cand[cand.index >= START]
cd = cand.index[cand.values]
cd_hist = cd[cd < c.index[-1]]
epi = declusters(cd_hist, 10, c.index)
print(f"\nC. ^NDX +2%+ day closing within 1% of its 52w high: {len(cd_hist)} days, {len(epi)} declustered (10td)")
cc = pd.DataFrame({"ret": r.reindex(epi) * 100, "dist_hi": dist.reindex(epi),
                   "h1": fwd_ret(c, 1).reindex(epi) * 100, "h5": fwd_ret(c, 5).reindex(epi) * 100,
                   "h21": fwd_ret(c, 21).reindex(epi) * 100})
print(cc.round(2).to_string())
show(fwd_table(c, epi, "NDX +2% near high"), "C2 summary")
e1 = fwd_ret(c, 1).reindex(epi).dropna()
show(era_split(e1.index, e1.values), "C3 era h1")
e5 = fwd_ret(c, 5).reindex(epi).dropna()
show(era_split(e5.index, e5.values), "C3b era h5")
print("C4 concentration h5:", cluster_note(e5.index, e5.values))
a1 = fwd_ret(c, 1)
a1 = a1[a1.index >= START].dropna()
a5 = fwd_ret(c, 5)
a5 = a5[a5.index >= START].dropna()
print(f"C5 NDX all days: h1 {100*a1.mean():.3f}% hit {100*(a1>0).mean():.1f}%, h5 {100*a5.mean():.3f}% hit {100*(a5>0).mean():.1f}%")
# all +2% days not near the high, as the contrast
far = (r >= 0.02) & (dist < -1.0)
far = far[far.index >= START]
fd = declusters(far.index[far.values], 10, c.index)
show(fwd_table(c, fd, "NDX +2% NOT near high"), "C6 contrast")
