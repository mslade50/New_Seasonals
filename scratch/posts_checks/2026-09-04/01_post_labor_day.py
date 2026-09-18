"""The session after Labor Day (next session is Tuesday 2026-09-08), anchored
on the Friday close that just printed. Forms:

  A. lag0 h1  : Friday close -> Tuesday close (the brief convention, not tradeable now)
  B. MOO->MOC : Tuesday open -> Tuesday close (tradeable; gap forfeited)
  C. lag1 hN  : Tuesday close -> +N sessions (the pitch convention)
  D. lag0 h5  : the whole post-Labor-Day week from Friday's close

Instruments SPY QQQ IWM TLT. Controls: all Tuesdays after any 3-day weekend,
all September sessions, all days. Splits: era 2018, midterm years, payrolls
on the eve (only some years), concentration.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    cluster_note, era_split, fwd_lag, load_events, load_prices, sign_test,
    summarize, wilder_atr,
)

warnings.filterwarnings("ignore")
ASOF = pd.Timestamp("2026-09-04")
raw = load_prices(["SPY", "QQQ", "IWM", "TLT", "^GSPC"])
ref = raw["^GSPC"]["Close"].dropna().index
pos = {d: i for i, d in enumerate(ref)}

for name in ("SPY", "QQQ", "IWM", "TLT"):
    d = raw[name]
    c = d["Close"].dropna()
    a = pd.Series(wilder_atr(d["High"], d["Low"], d["Close"]), index=d.index).reindex(c.index)
    print(f"{name} close {c.iloc[-1]:.2f} bar {c.index[-1].date()}  Wilder-14 ATR {a.iloc[-1]:.4f} "
          f"({100*a.iloc[-1]/c.iloc[-1]:.2f}%)")

# Labor Day = first Monday of September. Anchor = last session BEFORE it.
anchors = []
for y in range(2000, 2027):
    sept = pd.Timestamp(y, 9, 1)
    ld = sept + pd.Timedelta(days=(7 - sept.weekday()) % 7)  # first Monday
    before = ref[ref < ld]
    if len(before):
        anchors.append(before[-1])
anchors = pd.DatetimeIndex(anchors)
anchors = anchors[anchors <= ASOF]
print("anchors:", [a.date().isoformat() for a in anchors])
print("tonight is the anchor:", ASOF in set(anchors))

nfp = set(load_events(["nfp"])["date"])
eve_nfp = pd.DatetimeIndex([a for a in anchors if a in nfp])
print("Labor Day eves that were ALSO payrolls:", [a.date().isoformat() for a in eve_nfp])

# control: any Friday before a 3-day weekend (next session is a Tuesday)
gap3 = pd.DatetimeIndex([ref[i] for i in range(len(ref) - 1)
                         if (ref[i + 1] - ref[i]).days >= 4 and ref[i + 1].weekday() == 1])
gap3 = gap3[(gap3 <= ASOF)]
gap3_ex = gap3.difference(anchors)


def block(name, s, dates, h=1, lag=1, notes=False):
    f = fwd_lag(s, h, lag)
    v = f.reindex(pd.DatetimeIndex(dates)).dropna()
    if len(v) == 0:
        print(f"  {name:<50} n=0")
        return v
    st = summarize(v.values)
    nup = int((v > 0).sum())
    drift = 100 * f.dropna().mean()
    sep = f[[d.month == 9 for d in f.index]].dropna()
    print(f"  {name:<50} n={st['n']:<4} mean={st['mean_pct']:+.3f}%  med={st['median_pct']:+.3f}%  "
          f"{nup}-{len(v)-nup} ({st['hit']:.1f}%)  t={st['t']:+.2f}  sp={sign_test(nup, len(v)):.4f}  "
          f"| all-days {drift:+.3f}%  sept-days {100*sep.mean():+.3f}% hit {100*(sep>0).mean():.1f}%  "
          f"| worst {st['worst_pct']:+.2f}% ({v.idxmin().date()})  best {st['best_pct']:+.2f}%")
    if notes:
        print("    era:", [(e["label"], e["n"], round(e.get("mean_pct", np.nan), 3), round(e.get("hit", np.nan), 1))
                           for e in era_split(v.index, v.values)])
        print("    concentration:", cluster_note(v.index, v.values))
        mid = v[[d.year % 4 == 2 for d in v.index]]
        print(f"    midterm n={len(mid)} {int((mid>0).sum())}-{int((mid<=0).sum())} mean={100*mid.mean():+.3f}%  "
              f"{[(d.year, round(100*x,2)) for d,x in mid.items()]}")
        print("    all:", [(d.year, round(100 * x, 2)) for d, x in v.items()])
    return v


def open_to_close(d, dates, h_close=1):
    c, o = d["Close"].dropna(), d["Open"].reindex(d["Close"].dropna().index)
    p = {x: i for i, x in enumerate(c.index)}
    out, gap = {}, {}
    for a in dates:
        if a in p and p[a] + h_close < len(c):
            out[a] = c.iloc[p[a] + h_close] / o.iloc[p[a] + 1] - 1
            gap[a] = o.iloc[p[a] + 1] / c.iloc[p[a]] - 1
    return pd.Series(out), pd.Series(gap)


def oc_block(name, d, dates, h_close=1):
    v, g = open_to_close(d, dates, h_close)
    if len(v) == 0:
        print(f"  {name:<50} n=0")
        return v
    st = summarize(v.values)
    nup = int((v > 0).sum())
    allv, _ = open_to_close(d, d["Close"].dropna().index[252:-6], h_close)
    print(f"  {name:<50} n={st['n']:<4} mean={st['mean_pct']:+.3f}%  med={st['median_pct']:+.3f}%  "
          f"{nup}-{len(v)-nup} ({st['hit']:.1f}%)  t={st['t']:+.2f}  sp={sign_test(nup, len(v)):.4f}  "
          f"| all-days same form {100*allv.mean():+.3f}% hit {100*(allv>0).mean():.1f}%  "
          f"| gap {100*g.mean():+.3f}% ({int((g>0).sum())}-{int((g<=0).sum())})  | worst {st['worst_pct']:+.2f}%")
    print("    all:", [(d_.year, round(100 * x, 2)) for d_, x in v.items()])
    return v


for name in ("SPY", "QQQ", "IWM", "TLT"):
    d = raw[name]
    c = d["Close"].dropna()
    print(f"\n=== {name} ===")
    block("A. post-LD session, Fri close -> Tue close (lag0 h1)", c, anchors, 1, 0, notes=True)
    block("A. ctrl: any 3-day-weekend Friday -> Tuesday, ex LD", c, gap3_ex, 1, 0)
    oc_block("B. MOO Tue -> MOC Tue", d, anchors)
    oc_block("B. ctrl: any post-3-day-weekend Tue MOO->MOC", d, gap3_ex)
    for h in (1, 2, 3, 5):
        block(f"C. Tue close -> +{h} (lag1)", c, anchors, h, 1, notes=(h == 3))
    block("C. ctrl: 3-day-weekend Tue close -> +3, ex LD", c, gap3_ex, 3, 1)
    block("D. post-LD week, Fri close -> +5 (lag0 h5)", c, anchors, 5, 0, notes=True)
    block("D. ctrl: 3-day-weekend Fri -> +5, ex LD", c, gap3_ex, 5, 0)
    block("D2. post-LD 2 weeks, Fri close -> +10 (lag0)", c, anchors, 10, 0)
