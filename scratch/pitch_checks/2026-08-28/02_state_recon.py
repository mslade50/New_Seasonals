"""Point-in-time recon of the joint states the 2026-08-28 candidates are named
after. Every candidate has to be TRUE before it is worth a check, and the
2026-08-25 EEM kill ("false premise, and the trigger is not live") is why this
runs first.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
from pitch_lab import load_prices, pct_rank, zscore

ASOF = pd.Timestamp("2026-08-27")
NAMES = ["SPY", "QQQ", "IWM", "DIA", "EEM", "EFA", "EWZ", "EWJ", "FXI", "GLD", "GDX",
         "SLV", "SVXY", "^VIX", "^VIX3M", "XBI", "IBB", "XLP", "XLU", "XLRE", "XLV",
         "XLK", "XLF", "XLE", "XOP", "OIH", "SMH", "TLT", "IEF", "LQD", "HYG", "XME",
         "FCX", "USO", "UNG", "DX-Y.NYB", "^TNX"]
px = load_prices(NAMES)


def S(t):
    return px[t]["Close"].dropna()


def ret(t, n):
    s = S(t).loc[:ASOF]
    return float(s.iloc[-1] / s.iloc[-1 - n] - 1.0) * 100


def rk(t, n, lb=252):
    return float(pct_rank(S(t), n, lb).loc[:ASOF].iloc[-1])


def dhi(t, lb=252):
    s = S(t).loc[:ASOF]
    return float(s.iloc[-1] / s.rolling(lb).max().iloc[-1] - 1.0) * 100


def dlo(t, lb=252):
    s = S(t).loc[:ASOF]
    return float(s.iloc[-1] / s.rolling(lb).min().iloc[-1] - 1.0) * 100


print("=" * 88)
print("C1  round-trip breakout: index near a 52w high with a LOW 63d return rank")
print("=" * 88)
for t in ["SPY", "QQQ", "IWM", "DIA", "EFA", "EEM", "EWJ", "XLK", "SMH"]:
    print("  %-5s  dist52wH %+7.2f%%   63d ret %+7.2f%% (rank %5.1f)   21d rank %5.1f"
          % (t, dhi(t), ret(t, 63), rk(t, 63), rk(t, 21)))
# how rare is the SPY joint state historically?
spy = S("SPY")
d = spy / spy.rolling(252).max() - 1.0
r63 = pct_rank(spy, 63)
r21 = pct_rank(spy, 21)
joint = (d >= -0.01) & (r63 <= 25)
print("  SPY joint (within 1%% of 52wH AND 63d rank <= 25): %d days of %d valid (%.2f%%)"
      % (int(joint.sum()), int((d.notna() & r63.notna()).sum()),
         100 * joint.sum() / max(1, (d.notna() & r63.notna()).sum())))
print("  near-high alone: %d days | 63d-rank-low alone: %d days"
      % (int((d >= -0.01).sum()), int((r63 <= 25).sum())))
print("  live today: dist %+.3f%%  r63 %.1f  r21 %.1f" % (dhi("SPY"), rk("SPY", 63), rk("SPY", 21)))

print()
print("=" * 88)
print("C2  SVXY at a fresh 52-week high (vehicle changed leverage 2018-02-28)")
print("=" * 88)
sv = S("SVXY")
svhi = sv.rolling(252).max()
at_hi = (sv >= svhi * 0.9999)
post = sv.index >= pd.Timestamp("2018-03-01")
print("  SVXY close %.2f, 252d max %.2f, dist %+.3f%%" % (sv.loc[:ASOF].iloc[-1],
      svhi.loc[:ASOF].iloc[-1], dhi("SVXY")))
print("  at-52w-high days: full %d | post-2018-03 %d" % (int(at_hi.sum()),
      int((at_hi & post).sum())))
print("  fresh (first in >=10 td) post-2018-03: computed in the check")
print("  VIX %.2f (21d rank %.1f), VIX3M %.2f (dist52wL %+.2f%%)"
      % (S("^VIX").loc[:ASOF].iloc[-1], rk("^VIX", 21),
         S("^VIX3M").loc[:ASOF].iloc[-1], dlo("^VIX3M")))

print()
print("=" * 88)
print("C3/C4  international: thrust from inside a drawdown / the V that turned")
print("=" * 88)
for t in ["EWZ", "EEM", "EFA", "FXI", "EWJ"]:
    print("  %-5s  5d %+6.2f%% (rank %5.1f)  21d %+6.2f%% (rank %5.1f)  63d rank %5.1f  "
          "dist52wH %+7.2f%%  z10 %+5.2f"
          % (t, ret(t, 5), rk(t, 5), ret(t, 21), rk(t, 21), rk(t, 63), dhi(t),
             float(zscore(S(t), 10).loc[:ASOF].iloc[-1])))
eem = S("EEM")
e21, e63 = pct_rank(eem, 21), pct_rank(eem, 63)
m = (e21 >= 90) & (e63 <= 5)
print("  EEM joint (21d rank >=90 AND 63d rank <=5): %d days ever" % int(m.sum()))

print()
print("=" * 88)
print("C5  biotech thrust: XBI / IBB near a 52w high inside a top-decile year")
print("=" * 88)
for t in ["XBI", "IBB", "XLV"]:
    print("  %-5s  dist52wH %+7.2f%%  252d %+7.2f%%  21d rank %5.1f  63d rank %5.1f  z10 %+5.2f"
          % (t, dhi(t), ret(t, 252), rk(t, 21), rk(t, 63),
             float(zscore(S(t), 10).loc[:ASOF].iloc[-1])))

print()
print("=" * 88)
print("C7  defensive complex at 21d rank floors while SPY sits at a high")
print("=" * 88)
DEF = ["XLP", "XLU", "XLRE"]
for t in DEF:
    print("  %-5s  21d %+6.2f%% (rank %5.1f)  5d rank %5.1f  dist52wH %+7.2f%%"
          % (t, ret(t, 21), rk(t, 21), rk(t, 5), dhi(t)))
panel = pd.DataFrame({t: pct_rank(S(t), 21) for t in DEF}).dropna()
spyd = (spy / spy.rolling(252).max() - 1.0).reindex(panel.index)
allthree = (panel <= 20).all(axis=1)
joint7 = allthree & (spyd >= -0.01)
print("  all three <= 20th pct: %d days | AND SPY within 1%% of its high: %d days"
      % (int(allthree.sum()), int(joint7.sum())))
print("  live today: %s  SPY dist %+.2f%%" % (panel.loc[:ASOF].iloc[-1].round(1).to_dict(),
                                              dhi("SPY")))

print()
print("=" * 88)
print("C11  gold AND the S&P both in the top decile of their 21-day returns")
print("=" * 88)
g21, s21 = pct_rank(S("GLD"), 21), pct_rank(spy, 21)
both = (g21 >= 90) & (s21 >= 90)
al = g21.notna() & s21.notna()
print("  GLD 21d rank %.1f (%+.2f%%) | SPY 21d rank %.1f (%+.2f%%)"
      % (rk("GLD", 21), ret("GLD", 21), rk("SPY", 21), ret("SPY", 21)))
print("  joint top-decile days: %d of %d valid (%.2f%%)"
      % (int(both.sum()), int(al.sum()), 100 * both.sum() / max(1, al.sum())))
print("  GLD alone: %d | SPY alone: %d | independence expectation: %.0f days"
      % (int((g21 >= 90).sum()), int((s21 >= 90).sum()),
         (g21 >= 90).sum() * (s21 >= 90).sum() / max(1, al.sum())))
print("  GLD dist52wH %+.2f%% (a 21d thrust still %.1f%% under its own high)"
      % (dhi("GLD"), -dhi("GLD")))

print()
print("=" * 88)
print("C8  energy: 5-day pullback inside a 21-day thrust; services led the last session")
print("=" * 88)
for t in ["XLE", "XOP", "OIH", "USO", "UNG"]:
    print("  %-5s 1d %+6.2f%%  5d %+6.2f%% (rank %5.1f)  21d %+6.2f%% (rank %5.1f)  "
          "dist52wH %+7.2f%%" % (t, ret(t, 1), ret(t, 5), rk(t, 5), ret(t, 21), rk(t, 21), dhi(t)))

print()
print("=" * 88)
print("C6  calendar: today IS Jackson Hole (JH+0), month-end is Monday (ME-1)")
print("=" * 88)
ev = pd.read_csv(Path(__file__).resolve().parents[3] / "data/macro_events.csv")
ev["date"] = pd.to_datetime(ev["date"])
jh = ev[ev["event"] == "jackson_hole"]["date"]
print("  jackson_hole anchors in the file: %d, from %s to %s"
      % (len(jh), jh.min().date(), jh.max().date()))
print("  today %s is in the anchor list: %s" % (ASOF.date() + pd.Timedelta(days=1),
      bool((jh == pd.Timestamp("2026-08-28")).any())))
