"""B3 round 1 -- short SPY at FOMC-10td in a midterm year.

The map's largest inversion (-0.820pp, N=53). This is close to the event
sleeve's T2 and the overlap is named precisely below.

  T2  : short SPY 10% NAV, MOC at decision-4, exit MOO decision-day open,
        midterm years ONLY, gated on SPY 21d-return rank (252d, lag-1) < 50.
  this: short SPY MOC at decision-10, exit MOC on the decision close,
        midterm years, NO rank gate. Live rank today = 67.5, so T2 is OFF.

The decisive question the sleeve's own prereg poses: if rank21 < 50 is a real
gate, the OVERBOUGHT half (rank >= 50) -- which is the half live today -- has
to be where the short is WORST. Measured here.

Plus: era histogram, concentration by VALUE on the traded (short) side with
drop-best-2/3, and the T2 return-overlap statistic.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

ASOF = pd.Timestamp("2026-08-31")
H, K_ENTRY = 10, -10

px = load_prices(["SPY", "IWM", "QQQ"])
S = {t: px[t]["Close"].dropna()[lambda s: s.index <= ASOF] for t in px}
OP = {t: px[t]["Open"].dropna()[lambda s: s.index <= ASOF] for t in px}
spy = S["SPY"]

ev = load_events(["fomc_decision"])
FOM = pd.DatetimeIndex(sorted(ev["date"].unique()))
FOM = FOM[FOM <= ASOF]
MID = pd.DatetimeIndex([d for d in FOM if d.year % 4 == 2])
NONMID = FOM.difference(MID)


def window(s, anchors, k=K_ENTRY, h=H):
    pos, kept = anchor_positions(s.index, anchors, offset=k)
    d, e, v = [], [], []
    a = s.values
    for p, dd in zip(pos, kept):
        if p < 0 or p + h >= len(a):
            continue
        d.append(dd)
        e.append(s.index[p])
        v.append(a[p + h] / a[p] - 1.0)
    return pd.DatetimeIndex(d), pd.DatetimeIndex(e), np.asarray(v, float)


def drift(s, h=H, span=None):
    r = (s.shift(-h) / s - 1.0).dropna()
    if span:
        r = r[(r.index >= span[0]) & (r.index <= span[1])]
    return r.values


print("=" * 78)
print("1. THE SHORT, stated as the traded return (short = -1 x long)")
print("=" * 78)
d, e, v = window(spy, MID)
short = -v
base = drift(spy, H, (d[0], d[-1]))
wins = int((short > 0).sum())
print("  midterm anchors n=%d | LONG mean %+.3f%% | own drift %+.3f%% | edge %+.3fpp"
      % (len(v), 100 * v.mean(), 100 * base.mean(), 100 * (v.mean() - base.mean())))
print("  SHORT mean %+.3f%% BEFORE cost, record %d-%d, sign p %.4f"
      % (100 * short.mean(), wins, len(short) - wins, sign_test(wins, len(short))))
print("  ... but a short PAYS the drift: the short's own unconditional return over")
print("      the same span is %+.3f%%, so the short is only 'ahead' by the edge."
      % (-100 * base.mean()))
print("  bootstrap P(short mean <= 0) = %.3f" % bootstrap_p_le0(short))

print("\n" + "=" * 78)
print("2. THE SLEEVE'S OWN GATE: split by SPY 21d return rank (252d, lag-1).")
print("   T2 trades rank < 50. Live today = 67.5, i.e. the OVERBOUGHT half.")
print("=" * 78)
r21 = pct_rank(spy, 21, 252).shift(1)
rr = pd.Series(r21.reindex(e).values, index=e)
print("  rank at the midterm entry sessions: min %.1f med %.1f max %.1f | NaN %d"
      % (np.nanmin(rr), np.nanmedian(rr), np.nanmax(rr), int(rr.isna().sum())))
rows = []
for lbl, m in (("rank21 < 50 (T2's half)", rr < 50),
               ("rank21 >= 50 (LIVE half)", rr >= 50),
               ("rank21 >= 65 (live 67.5)", rr >= 65),
               ("rank21 < 30", rr < 30),
               ("rank21 >= 80", rr >= 80)):
    mv = m.fillna(False).values
    sv = short[mv]
    if len(sv) == 0:
        rows.append({"cell": lbl, "n": 0})
        continue
    w = int((sv > 0).sum())
    rows.append({"cell": lbl, "n": len(sv), "short_pct": 100 * sv.mean(),
                 "long_pct": -100 * sv.mean(), "hit": 100 * w / len(sv),
                 "sign_p": sign_test(w, len(sv)),
                 "worst_pct": 100 * sv.min(), "best_pct": 100 * sv.max()})
show(rows, "short SPY, midterm FOMC-10td -> decision close, by rank21 gate")
lo, hi = short[(rr < 50).fillna(False).values], short[(rr >= 50).fillna(False).values]
if len(lo) > 1 and len(hi) > 1:
    se = np.sqrt(lo.var(ddof=1) / len(lo) + hi.var(ddof=1) / len(hi))
    print("  gate separation: rank<50 %+.3f%% vs rank>=50 %+.3f%%  diff %+.3fpp, welch t %+.2f"
          % (100 * lo.mean(), 100 * hi.mean(), 100 * (lo.mean() - hi.mean()),
             (lo.mean() - hi.mean()) / se))
    print("  DOES THE GATE FILTER?  (registry: if it does not move the result,")
    print("  nothing may be attributed to it.)")

print("\n" + "=" * 78)
print("3. ERA HISTOGRAM on the SHORT side -- which years own it?")
print("=" * 78)
by = pd.Series(short, index=pd.DatetimeIndex(d).year).groupby(level=0)
print(pd.DataFrame({"n": by.size(), "sum_pct": 100 * by.sum(),
                    "mean_pct": 100 * by.mean(),
                    "hit": 100 * by.apply(lambda x: (x > 0).mean())}).round(3).to_string())
show(era_split(pd.DatetimeIndex(d), short), "era split, short side")
ordr = np.argsort(-short)
print("  concentration BY VALUE on the traded (short) side:")
print("    total %+.2fpp | best2 %+.2fpp (%s) = %.0f%% of total | best3 %+.2fpp"
      % (100 * short.sum(), 100 * short[ordr[:2]].sum(),
         [str(pd.Timestamp(d[i]).date()) for i in ordr[:2]],
         100 * short[ordr[:2]].sum() / short.sum() if short.sum() else np.nan,
         100 * short[ordr[:3]].sum()))
print("    mean %+.3f%% | drop-best-2 %+.3f%% | drop-best-3 %+.3f%%"
      % (100 * short.mean(), 100 * short[ordr[2:]].mean(), 100 * short[ordr[3:]].mean()))
print("    edge over the short's own drift: full %+.3fpp | drop-best-2 %+.3fpp | drop-best-3 %+.3fpp"
      % (100 * (short.mean() + base.mean()), 100 * (short[ordr[2:]].mean() + base.mean()),
         100 * (short[ordr[3:]].mean() + base.mean())))

print("\n" + "=" * 78)
print("4. T2 OVERLAP, stated as a number rather than a claim")
print("=" * 78)
# T2 leg: MOC decision-4 -> MOO decision-day open
pos, kept = anchor_positions(spy.index, MID, offset=-4)
t2, ours, dts = [], [], []
cv, ov = spy.values, OP["SPY"].reindex(spy.index).values
for p, dd in zip(pos, kept):
    if p + 4 >= len(cv):
        continue
    t2.append(-(ov[p + 4] / cv[p] - 1.0))
    q = p - 6                       # dec-10
    if q < 0:
        continue
    ours.append(-(cv[p + 4] / cv[q] - 1.0))
    dts.append(dd)
n = min(len(t2), len(ours))
t2, ours = np.array(t2[-n:]), np.array(ours[-n:])
print("  n=%d aligned midterm decisions" % n)
print("  T2 (short dec-4 close -> decision open) mean %+.3f%%, hit %.0f%%"
      % (100 * t2.mean(), 100 * (t2 > 0).mean()))
print("  ours (short dec-10 close -> decision close) mean %+.3f%%, hit %.0f%%"
      % (100 * ours.mean(), 100 * (ours > 0).mean()))
print("  correlation of the two legs' returns = %.3f" % np.corrcoef(t2, ours)[0, 1])
print("  calendar overlap: T2 occupies 4 of our 10 held sessions plus the decision")
print("  open; the sessions dec-10..dec-5 are ours alone.")
solo = []
for p, dd in zip(pos, kept):
    q = p - 6
    if q < 0 or p >= len(cv):
        continue
    solo.append(-(cv[p] / cv[q] - 1.0))     # short dec-10 -> dec-4 close, T2-free
solo = np.array(solo)
w = int((solo > 0).sum())
print("  the T2-FREE portion alone (short dec-10 close -> dec-4 close, 6 td):")
print("    n=%d mean %+.3f%% record %d-%d sign p %.4f"
      % (len(solo), 100 * solo.mean(), w, len(solo) - w, sign_test(w, len(solo))))
b6 = -drift(spy, 6, (d[0], d[-1])).mean()
print("    vs the short's own 6-td drift over the same span %+.3f%% -> edge %+.3fpp"
      % (100 * b6, 100 * (solo.mean() - b6)))

print("\n" + "=" * 78)
print("5. NON-MIDTERM contrast (is 'midterm' the conditioner or the sample?)")
print("=" * 78)
dn, en, vn = window(spy, NONMID)
bn = drift(spy, H, (dn[0], dn[-1]))
print("  non-midterm LONG mean %+.3f%% (n=%d) vs own drift %+.3f%% -> edge %+.3fpp"
      % (100 * vn.mean(), len(vn), 100 * bn.mean(), 100 * (vn.mean() - bn.mean())))
se = np.sqrt(v.var(ddof=1) / len(v) + vn.var(ddof=1) / len(vn))
print("  midterm vs non-midterm difference %+.3fpp, welch t %+.2f"
      % (100 * (v.mean() - vn.mean()), (v.mean() - vn.mean()) / se))
