"""Live-state probe for the surface map, 2026-09-14 (asof close 2026-09-11).

Prints every number the surface map cites for watchlist verdicts and tape
extremes, from the same master_prices the checks use.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

ASOF = pd.Timestamp("2026-09-11")
SPDR = ["XLK", "XLF", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB", "XLE"]
IND = ["SMH", "IBB", "XBI", "IHI", "KRE", "ITA", "XME", "XRT", "XHB", "OIH",
       "XOP", "IYR", "XLRE", "XLC", "ITB", "QQQ", "IWM", "EFA", "EEM", "FXI",
       "EWZ", "EWJ", "DIA"]
ENERGY = ["XLE", "XOP", "USO", "COP", "CVX", "VLO", "OXY", "SLB", "EOG", "HAL", "WMB"]
TK = sorted(set(SPDR + IND + ENERGY + ["SPY", "TLT", "IEF", "LQD", "HYG", "GLD",
            "GDX", "SLV", "DBC", "UUP", "DX-Y.NYB", "^VIX", "^VIX3M", "^MOVE",
            "^SKEW", "^TNX", "SVXY", "UNG"]))
px = load_prices(TK)


def C(t):
    return px[t]["Close"]


def at(s, d=ASOF):
    return float(s.loc[:d].iloc[-1])


print("last bars:", {t: str(C(t).index[-1].date()) for t in ["SPY", "^TNX", "TLT", "HYG", "USO", "^MOVE"]})

tnx = C("^TNX")
print("\nW18 curve: TNX close %.4f, 252-max %.4f, 252-session change %+.1f bp (arm >= +88)"
      % (at(tnx), at(tnx.rolling(252).max()), 100 * (at(tnx) - float(tnx.loc[:ASOF].iloc[-253]))))
for d in tnx.loc[:ASOF].index[-5:]:
    s = tnx.loc[:d]
    print("   %s close %.3f  at252max=%s  chg252 %+.1f bp  chg21 %+.1f bp" % (
        d.date(), s.iloc[-1], bool(s.iloc[-1] >= s.iloc[-252:].max() - 1e-9),
        100 * (s.iloc[-1] - s.iloc[-253]), 100 * (s.iloc[-1] - s.iloc[-22])))

print("\nW19 energy count, z10>=2.0:")
zl = {t: at(zscore(C(t), 10)) for t in ENERGY}
print("   pitch_lab.zscore:", {k: round(v, 2) for k, v in sorted(zl.items(), key=lambda kv: -kv[1])},
      "count", sum(v >= 2.0 for v in zl.values()))

print("\nW13 gold: DX 21d rank %.1f (arm <=15), TNX 21-session change %+.1f bp (arm >= +20)"
      % (at(pct_rank(C("DX-Y.NYB"), 21)), 100 * (at(tnx) - float(tnx.loc[:ASOF].iloc[-22]))))
mv = C("^MOVE")
print("W30 MOVE level pctile (trailing 252): %.1f (band [40,50))"
      % at(mv.rolling(252).apply(lambda a: (a <= a[-1]).mean() * 100, raw=True)))

print("\nSector ranks (5/21/63 PIT) + dist 52w high + dist 200d:")
for t in SPDR + IND:
    c = C(t)
    print("   %-5s r5 %5.1f r21 %5.1f r63 %5.1f  d52h %6.2f%%  d200 %6.2f%%  ret5 %+.2f%%" % (
        t, at(pct_rank(c, 5)), at(pct_rank(c, 21)), at(pct_rank(c, 63)),
        100 * (at(c) / at(c.rolling(252).max()) - 1), 100 * (at(c) / at(c.rolling(200).mean()) - 1),
        100 * (at(c) / float(c.loc[:ASOF].iloc[-6]) - 1)))

spy = C("SPY")
print("\nSector 5d return minus SPY 5d:")
s5 = 100 * (at(spy) / float(spy.loc[:ASOF].iloc[-6]) - 1)
for t in SPDR:
    c = C(t)
    print("   %-4s %+.2fpp" % (t, 100 * (at(c) / float(c.loc[:ASOF].iloc[-6]) - 1) - s5))
xlv_rel = (C("XLV") / spy).dropna()
rel5 = xlv_rel.pct_change(5)
print("XLV/SPY 5d rel return %.2f%%, pctile over full history %.1f, 252d PIT rank %.1f"
      % (100 * at(rel5), 100 * (rel5.dropna() <= at(rel5)).mean(), at(pct_rank(xlv_rel, 5))))

print("\nCredit: HYG r5 %.1f z10 %.2f  d52h %.2f%%; LQD r5 %.1f; IEF r5 %.1f"
      % (at(pct_rank(C("HYG"), 5)), at(zscore(C("HYG"), 10)),
         100 * (at(C("HYG")) / at(C("HYG").rolling(252).max()) - 1),
         at(pct_rank(C("LQD"), 5)), at(pct_rank(C("IEF"), 5))))
hy_ief = (C("HYG") / C("IEF")).dropna()
print("HYG/IEF ratio 5d rank %.1f" % at(pct_rank(hy_ief, 5)))

print("\nVol: VIX %.2f VIX3M %.2f ratio %.3f; VIX 1d %+.1f%%; SKEW %.1f r21 %.1f"
      % (at(C("^VIX")), at(C("^VIX3M")), at(C("^VIX")) / at(C("^VIX3M")),
         100 * (at(C("^VIX")) / float(C("^VIX").loc[:ASOF].iloc[-2]) - 1),
         at(C("^SKEW")), at(pct_rank(C("^SKEW"), 21))))
print("USO: 1d %+.2f%% r21 %.1f z10 %.2f d52h %.2f%%"
      % (100 * (at(C("USO")) / float(C("USO").loc[:ASOF].iloc[-2]) - 1),
         at(pct_rank(C("USO"), 21)), at(zscore(C("USO"), 10)),
         100 * (at(C("USO")) / at(C("USO").rolling(252).max()) - 1)))
print("IWM r21 %.1f r63 %.1f z10 %.2f; SPY d52h %.2f%%"
      % (at(pct_rank(C("IWM"), 21)), at(pct_rank(C("IWM"), 63)), at(zscore(C("IWM"), 10)),
         100 * (at(spy) / at(spy.rolling(252).max()) - 1)))
