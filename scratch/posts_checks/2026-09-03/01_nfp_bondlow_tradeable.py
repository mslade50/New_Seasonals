"""Idea candidates for Friday 2026-09-04 (payrolls AND the Labor Day eve) from
tonight's brief cell: a payrolls eve with IEF within 1% of its 252d low
(tonight 0.47%). The brief reported, anchored on the eve close (lag 0):
S&P higher on the print session 14 of 19 (+0.44%), TLT higher 15 of 19 over
the next five sessions. Both are lag-0 from a close that has already printed,
so the tradeable forms all enter tomorrow:

  SPY  A. brief reproduction, eve close -> print close (lag 0, h1)
       B. MOO print -> MOC print (open->close; the overnight gap is forfeited)
       C. lag-1, print close -> h sessions later
  TLT  D. brief reproduction, eve close -> +5 (lag 0, h5)
       E. MOO print -> close of the 5th session counting the print
       F. lag-1, print close -> +5 sessions (the pitch convention)

Controls: every payrolls eve (ungated), the same bond state on NON-payrolls
days (declustered 5), era split, concentration, worst, midterm split, and a
placebo anchor ladder for TLT form F (slide the anchor -5..+5 sessions around
the print, gate evaluated at the slid anchor). This morning's pitch killed
TLT k=-2 -> print close (ungated) with exactly that ladder, so the same test
is owed here. Also prints the midterm-year Sep-4 cell the brief flagged as
contradicting (6 of 6 down over five sessions).
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    cluster_note, declusters, era_split, fwd_lag, load_events, load_prices,
    local_control, sign_test, summarize, wilder_atr,
)

warnings.filterwarnings("ignore")
ASOF = pd.Timestamp("2026-09-03")
raw = load_prices(["SPY", "TLT", "IEF", "^GSPC"])
spy, tlt, ief = raw["SPY"], raw["TLT"], raw["IEF"]
ref = raw["^GSPC"]["Close"].dropna().index
pos = {d: i for i, d in enumerate(ref)}

for name, d in (("SPY", spy), ("TLT", tlt)):
    c = d["Close"].dropna()
    a = pd.Series(wilder_atr(d["High"], d["Low"], d["Close"]), index=d.index).reindex(c.index)
    print(f"{name} close {c.iloc[-1]:.2f} bar {c.index[-1].date()}  Wilder-14 ATR {a.iloc[-1]:.4f} "
          f"({100*a.iloc[-1]/c.iloc[-1]:.2f}%)")

iefc = ief["Close"].dropna()
dist_low = 100 * (iefc / iefc.rolling(252).min() - 1.0)
print(f"IEF {dist_low.iloc[-1]:.2f}% above its 252d low tonight")
GATE = dist_low <= 1.0

nfp = load_events(["nfp"])["date"]
nfp = pd.DatetimeIndex(sorted(set(nfp) & set(ref)))
eves_all = pd.DatetimeIndex([ref[pos[d] - 1] for d in nfp if pos.get(d, 0) > 0])
eves_all = eves_all[eves_all <= ASOF]
eves_on = eves_all[GATE.reindex(eves_all).fillna(False).values]
eves_off = eves_all[~GATE.reindex(eves_all).fillna(False).values & dist_low.reindex(eves_all).notna().values]
print(f"tonight is a payrolls eve: {ASOF in set(eves_all)}   eves with IEF data: {len(eves_on)+len(eves_off)}  "
      f"gate ON: {len(eves_on)}  OFF: {len(eves_off)}")
print("gate-ON eves:", [d.date().isoformat() for d in eves_on])


def block(name, s, dates, h=1, lag=1, notes=False):
    f = fwd_lag(s, h, lag)
    v = f.reindex(pd.DatetimeIndex(dates)).dropna()
    if len(v) == 0:
        print(f"  {name:<52} n=0")
        return v
    st = summarize(v.values)
    nup = int((v > 0).sum())
    drift = 100 * f.dropna().mean()
    loc = f.reindex(local_control(s.index, v.index, 126)).dropna()
    print(f"  {name:<52} n={st['n']:<4} mean={st['mean_pct']:+.3f}%  med={st['median_pct']:+.3f}%  "
          f"{nup}-{len(v)-nup} ({st['hit']:.1f}%)  t={st['t']:+.2f}  sp={sign_test(nup, len(v)):.4f}  "
          f"| drift {drift:+.3f}%  local {100*loc.mean():+.3f}% hit {100*(loc>0).mean():.1f}%  "
          f"| worst {st['worst_pct']:+.2f}% ({v.idxmin().date()})")
    if notes:
        print("    era:", [(e["label"], e["n"], round(e.get("mean_pct", np.nan), 3), round(e.get("hit", np.nan), 1))
                           for e in era_split(v.index, v.values)])
        print("    concentration:", cluster_note(v.index, v.values))
        mid = v[[d.year % 4 == 2 for d in v.index]]
        print(f"    midterm n={len(mid)} {int((mid>0).sum())}-{int((mid<=0).sum())} mean={100*mid.mean():+.3f}%")
        print("    all:", [(d.date().isoformat(), round(100 * x, 2)) for d, x in v.items()])
    return v


def open_to_close(d, dates, h_close=1):
    """Enter at the OPEN of the session after each anchor, exit at the close h_close sessions
    after the anchor (h_close=1 -> same-session MOC)."""
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
        print(f"  {name:<52} n=0")
        return v
    st = summarize(v.values)
    nup = int((v > 0).sum())
    allv, _ = open_to_close(d, d["Close"].dropna().index[252:-6], h_close)
    print(f"  {name:<52} n={st['n']:<4} mean={st['mean_pct']:+.3f}%  med={st['median_pct']:+.3f}%  "
          f"{nup}-{len(v)-nup} ({st['hit']:.1f}%)  t={st['t']:+.2f}  sp={sign_test(nup, len(v)):.4f}  "
          f"| all-days same form {100*allv.mean():+.3f}% hit {100*(allv>0).mean():.1f}%  "
          f"| gap forfeited {100*g.mean():+.3f}% ({int((g>0).sum())}-{int((g<=0).sum())})  | worst {st['worst_pct']:+.2f}%")
    return v


spyc, tltc = spy["Close"].dropna(), tlt["Close"].dropna()
# the gate on non-payrolls days: gate-ON sessions not within 3 td of any eve
near = set()
for a in eves_all:
    i = pos[a]
    near.update(ref[max(0, i - 3): i + 4])
G_days = dist_low.index[GATE.values]
G_days = pd.DatetimeIndex([d for d in G_days if d <= ASOF and d not in near and d in pos])
G_dc = declusters(G_days, 5, ref)

print("\n=== SPY ===")
block("A. gate ON, eve close -> print close (lag0 h1)", spyc, eves_on, 1, 0, notes=True)
block("A. gate OFF eves, lag0 h1", spyc, eves_off, 1, 0)
block("A. ALL eves, lag0 h1", spyc, eves_all, 1, 0)
block("A. gate ON, NON-payrolls days (dc5), lag0 h1", spyc, G_dc, 1, 0)
oc_block("B. gate ON, MOO print -> MOC print", spy, eves_on)
oc_block("B. ALL eves, MOO print -> MOC print", spy, eves_all)
oc_block("B. gate ON, NON-payrolls days (dc5), MOO -> MOC", spy, G_dc)
for h in (1, 2, 3, 5):
    block(f"C. gate ON, print close -> +{h} (lag1)", spyc, eves_on, h, 1)
block("C. ALL eves, print close -> +5 (lag1)", spyc, eves_all, 5, 1)

print("\n=== TLT ===")
block("D. gate ON, eve close -> +5 (lag0 h5)", tltc, eves_on, 5, 0, notes=True)
block("D. gate OFF eves, lag0 h5", tltc, eves_off, 5, 0)
block("D. ALL eves, lag0 h5", tltc, eves_all, 5, 0)
block("D. gate ON, NON-payrolls days (dc5), lag0 h5", tltc, G_dc, 5, 0)
block("D. gate ON, lag0 h1 (the print session itself)", tltc, eves_on, 1, 0)
oc_block("E. gate ON, MOO print -> close +5 (5 sessions incl print)", tlt, eves_on, 5)
oc_block("E. ALL eves, MOO print -> close +5", tlt, eves_all, 5)
for h in (3, 5, 10):
    block(f"F. gate ON, print close -> +{h} (lag1)", tltc, eves_on, h, 1, notes=(h == 5))
block("F. ALL eves, print close -> +5 (lag1)", tltc, eves_all, 5, 1)
block("F. gate ON, NON-payrolls days (dc5), lag1 h5", tltc, G_dc, 5, 1)

print("\n=== placebo anchor ladder, TLT lag-1 h5, gate evaluated at the slid anchor (j=0 is live) ===")
rows = []
for j in range(-5, 6):
    anc = pd.DatetimeIndex([ref[pos[a] + j] for a in eves_all if 0 <= pos[a] + j < len(ref)])
    anc = anc[GATE.reindex(anc).fillna(False).values]
    v = fwd_lag(tltc, 5, 1).reindex(anc).dropna()
    if len(v):
        rows.append((j, len(v), 100 * v.mean(), 100 * (v > 0).mean()))
for j, n, m, hit in rows:
    print(f"  j={j:+d}  n={n:<3} mean={m:+.3f}%  hit={hit:.1f}%")
live = [r for r in rows if r[0] == 0][0]
print(f"  live rank by mean: {sorted([r[2] for r in rows], reverse=True).index(live[2]) + 1} of {len(rows)}")

print("\n=== the contradicting cell: session nearest Sep 4, midterm years, TLT lag0 h5 ===")
doy = []
for y in range(2002, 2026, 4):
    cands = tltc.index[(tltc.index.year == y)]
    if len(cands):
        d = cands[np.argmin(np.abs((cands - pd.Timestamp(y, 9, 4)).days))]
        doy.append(d)
v = fwd_lag(tltc, 5, 0).reindex(pd.DatetimeIndex(doy)).dropna()
print("  ", [(d.date().isoformat(), round(100 * x, 2)) for d, x in v.items()], f"mean {100*v.mean():+.2f}%")
v = fwd_lag(tltc, 5, 1).reindex(pd.DatetimeIndex(doy)).dropna()
print("   lag1 form:", [(d.date().isoformat(), round(100 * x, 2)) for d, x in v.items()], f"mean {100*v.mean():+.2f}%")
