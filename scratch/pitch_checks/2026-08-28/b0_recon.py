"""Stage-C checker B: point-in-time recon for C3 / C4 / C5.

Confirms each named state is actually TRUE on the 2026-08-27 close before any
falsification budget is spent (the 2026-08-25 "false premise" kill is why).
Also prints the reference-class populations each candidate will be judged
against, so the round-2 scripts do not re-derive them.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import pandas as pd
from pitch_lab import load_prices, pct_rank, zscore

ASOF = pd.Timestamp("2026-08-27")

INTL = ["EWZ", "EEM", "EFA", "FXI", "EWJ", "EWT", "EWY", "EWW", "INDA", "KWEB", "VGK"]
BROAD = ["SPY", "QQQ", "IWM", "DIA"]
SECT = ["XLK", "XLV", "XLF", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC",
        "XLE", "SMH", "XBI", "IBB", "KRE", "IHI", "ITB", "XME", "XOP", "OIH",
        "XRT", "XHB", "IYT", "ITA", "IYR", "GDX"]
EXTRA = ["UUP", "DX-Y.NYB"]
ALL = sorted(set(INTL + BROAD + SECT + EXTRA))
px = load_prices(ALL)


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


print("=" * 96)
print("C3 recon -- 5d thrust rank >=90 while >=10% below the 252d high (intl family)")
print("=" * 96)
print("  %-6s %8s %7s %7s %7s %7s %9s %7s" %
      ("tkr", "5d%", "r5", "r21", "r63", "z10", "dist52wH", "lastbar"))
for t in INTL:
    s = S(t)
    print("  %-6s %+8.2f %7.1f %7.1f %7.1f %+7.2f %+9.2f  %s" %
          (t, ret(t, 5), rk(t, 5), rk(t, 21), rk(t, 63),
           float(zscore(s, 10).loc[:ASOF].iloc[-1]), dhi(t),
           str(s.index[-1].date())))
print()
for t in INTL:
    s = S(t)
    r5 = pct_rank(s, 5)
    d = s / s.rolling(252).max() - 1.0
    joint = (r5 >= 90) & (d <= -0.10)
    bare = (r5 >= 90)
    print("  %-6s  joint(r5>=90 & dd<=-10%%) %4d days | bare r5>=90 %4d days | "
          "share %.1f%%  LIVE=%s" %
          (t, int(joint.sum()), int(bare.sum()),
           100 * joint.sum() / max(1, bare.sum()),
           bool(joint.loc[:ASOF].iloc[-1])))

print()
print("=" * 96)
print("C4 recon -- 21d rank >=90 AND 63d rank <=10 (the V that turned)")
print("=" * 96)
fam = BROAD + ["EFA", "EEM", "EWJ", "FXI", "EWZ"] + SECT
live = []
for t in fam:
    s = S(t)
    r21, r63 = pct_rank(s, 21), pct_rank(s, 63)
    m = (r21 >= 90) & (r63 <= 10)
    on = bool(m.loc[:ASOF].iloc[-1])
    if on:
        live.append(t)
    print("  %-6s r21 %5.1f  r63 %5.1f  ret63 %+7.2f%%  joint days %4d  "
          "bare r21>=90 %4d  LIVE=%s" %
          (t, rk(t, 21), rk(t, 63), ret(t, 63), int(m.sum()),
           int((r21 >= 90).sum()), on))
print("  LIVE TODAY:", live)

print()
print("=" * 96)
print("C5 recon -- 21d AND 63d rank both >=95 while within 1% of the 252d high")
print("=" * 96)
live5 = []
for t in SECT + BROAD + ["EEM", "EFA", "EWJ"]:
    s = S(t)
    r21, r63 = pct_rank(s, 21), pct_rank(s, 63)
    d = s / s.rolling(252).max() - 1.0
    m = (r21 >= 95) & (r63 >= 95) & (d >= -0.01)
    on = bool(m.loc[:ASOF].iloc[-1])
    if on:
        live5.append(t)
    if on or t in ("IBB", "XBI", "IHI", "SMH", "XLV"):
        print("  %-6s r21 %5.1f  r63 %5.1f  dist %+6.2f%%  252d %+7.2f%%  "
              "joint days %4d  LIVE=%s" %
              (t, rk(t, 21), rk(t, 63), dhi(t), ret(t, 252), int(m.sum()), on))
print("  LIVE TODAY:", live5)
