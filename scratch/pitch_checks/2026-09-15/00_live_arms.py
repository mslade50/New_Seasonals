"""Live-state probe for the surface map, 2026-09-15 (asof close 2026-09-14).

Prints the numbers the surface map cites for watchlist verdicts and for the
tape extremes that seed today's candidates, from the same master_prices the
checks use.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

ASOF = pd.Timestamp("2026-09-14")
SPDR = ["XLK", "XLF", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB", "XLE"]
IND = ["SMH", "IBB", "XBI", "IHI", "KRE", "ITA", "XME", "XRT", "XHB", "OIH",
       "XOP", "IYR", "XLRE", "XLC", "ITB", "QQQ", "IWM", "EFA", "EEM", "FXI",
       "EWZ", "EWJ", "DIA"]
ENERGY = ["XLE", "XOP", "USO", "COP", "CVX", "VLO", "OXY", "SLB", "EOG", "HAL", "WMB"]
BANKS = ["JPM", "BAC", "C", "WFC", "GS", "MS", "BNY", "STT", "USB", "SCHW", "KEY", "RF"]
THRUST = ["GLW", "INTC", "AMD", "HPQ", "QCOM", "MU", "AMAT", "AVGO", "NVDA", "ADI", "TXN"]
TK = sorted(set(SPDR + IND + ENERGY + BANKS + THRUST + [
    "SPY", "TLT", "IEF", "LQD", "HYG", "GLD", "GDX", "SLV", "DBC", "UUP", "DX-Y.NYB",
    "^VIX", "^VIX3M", "^MOVE", "^SKEW", "^TNX", "SVXY", "UNG", "CL=F", "FDX"]))
px = load_prices(TK)


def C(t):
    return px[t]["Close"]


def at(s, d=ASOF):
    return float(s.loc[:d].iloc[-1])


def watr(t):
    d = px[t]
    return pd.Series(np.asarray(wilder_atr(d["High"], d["Low"], d["Close"]), dtype=float), index=d.index)


def ret(t, n):
    c = C(t).loc[:ASOF]
    return 100 * (c.iloc[-1] / c.iloc[-1 - n] - 1)


missing = [t for t in TK if t not in px]
print("missing:", missing)
print("last bars:", {t: str(C(t).index[-1].date()) for t in ["SPY", "^TNX", "TLT", "BAC", "OIH", "CL=F", "^MOVE", "^VIX"]})

print("\nW22 utilities x rates: XLU r21 %.2f (arm <=5)  TLT r21 %.2f (arm <25)  TLT r5 %.1f"
      % (at(pct_rank(C("XLU"), 21)), at(pct_rank(C("TLT"), 21)), at(pct_rank(C("TLT"), 5))))

smh = C("SMH")
r252 = smh.pct_change(252)
print("W25 SMH: r63 %.2f (<=5)  r5 %.2f (<15)  252d ret %.1f%%, 252d-ret PIT pctile over trailing 252 %.1f"
      % (at(pct_rank(smh, 63)), at(pct_rank(smh, 5)), 100 * at(r252), at(pct_rank(smh, 252))))

tnx = C("^TNX")
print("\nTNX close %.3f 252max %.3f chg5 %+.1f bp chg21 %+.1f bp"
      % (at(tnx), at(tnx.rolling(252).max()), 100 * (at(tnx) - float(tnx.loc[:ASOF].iloc[-6])),
         100 * (at(tnx) - float(tnx.loc[:ASOF].iloc[-22]))))
mv = C("^MOVE")
print("MOVE %.1f level pctile(252) %.1f r5 %.1f" % (
    at(mv), at(mv.rolling(252).apply(lambda a: (a <= a[-1]).mean() * 100, raw=True)), at(pct_rank(mv, 5))))
print("DX r21 %.1f; UUP r5 %.1f" % (at(pct_rank(C("DX-Y.NYB"), 21)), at(pct_rank(C("UUP"), 5))))

print("\nW19 energy count z10>=2.0 (pitch_lab.zscore):")
zl = {t: at(zscore(C(t), 10)) for t in ENERGY}
print("  ", {k: round(v, 2) for k, v in sorted(zl.items(), key=lambda kv: -kv[1])},
      "count", sum(v >= 2.0 for v in zl.values()))
print("CL=F: close %.2f, d52h %.2f%%, r5 %.1f, 5d %+.2f%%; USO 5d %+.2f%%; OIH r5 %.1f 5d %+.2f%% 1d %+.2f%%; SLB 1d %+.2f%%"
      % (at(C("CL=F")), 100 * (at(C("CL=F")) / at(C("CL=F").rolling(252).max()) - 1),
         at(pct_rank(C("CL=F"), 5)), ret("CL=F", 5), ret("USO", 5), at(pct_rank(C("OIH"), 5)),
         ret("OIH", 5), ret("OIH", 1), ret("SLB", 1)))

print("\nSector ranks (5/21/63 PIT) + d52h + d200:")
for t in SPDR + IND:
    c = C(t)
    print("   %-5s r5 %5.1f r21 %5.1f r63 %5.1f  d52h %6.2f%%  d200 %6.2f%%  1d %+.2f%% 5d %+.2f%%" % (
        t, at(pct_rank(c, 5)), at(pct_rank(c, 21)), at(pct_rank(c, 63)),
        100 * (at(c) / at(c.rolling(252).max()) - 1), 100 * (at(c) / at(c.rolling(200).mean()) - 1),
        ret(t, 1), ret(t, 5)))

print("\nXLV - XLK 1d gap %+.2fpp; SPY d52h %.2f%%; SPY Wilder ATR %% %.2f" % (
    ret("XLV", 1) - ret("XLK", 1), 100 * (at(C("SPY")) / at(C("SPY").rolling(252).max()) - 1),
    100 * at(watr("SPY")) / at(C("SPY"))))

print("\nBanks 1d / 5d / Wilder-ATR multiple of 1d move:")
for t in BANKS + ["KRE", "XLF"]:
    a = at(watr(t).shift(1))
    c = C(t).loc[:ASOF]
    print("   %-5s 1d %+.2f%%  5d %+.2f%%  1d/ATR %+.2f  r5 %.1f r63 %.1f" % (
        t, ret(t, 1), ret(t, 5), (c.iloc[-1] - c.iloc[-2]) / a, at(pct_rank(C(t), 5)), at(pct_rank(C(t), 63))))

print("\nThrust names: r5 at 09-11, 1d on 09-14 in ATR:")
prev = pd.Timestamp("2026-09-11")
for t in THRUST:
    a = at(watr(t).shift(1))
    c = C(t).loc[:ASOF]
    print("   %-5s r5@0911 %5.1f  5d@0911 %+.2f%%  1d %+.2f%%  1d/ATR %+.2f" % (
        t, at(pct_rank(C(t), 5), prev), 100 * (float(c.iloc[-2]) / float(c.iloc[-7]) - 1),
        ret(t, 1), (c.iloc[-1] - c.iloc[-2]) / a))

vix = C("^VIX")
print("\nVIX %.2f 1d %+.2f%% 2d %+.2f%% 5d %+.2f%%; VIX3M %.2f ratio %.3f; SVXY 1d %+.2f%%"
      % (at(vix), ret("^VIX", 1), ret("^VIX", 2), ret("^VIX", 5), at(C("^VIX3M")),
         at(vix) / at(C("^VIX3M")), ret("SVXY", 1)))
rng = (vix.rolling(21).max() - vix.rolling(21).min()) / vix.rolling(21).mean()
print("VIX 21d rel range %.3f, trailing-252 pctile %.1f" % (
    at(rng), at(rng.rolling(252).apply(lambda a: (a <= a[-1]).mean() * 100, raw=True))))
print("GLD 5d %+.2f SLV 5d %+.2f GDX 5d %+.2f XME 5d %+.2f; HYG r5 %.1f; HYG/IEF r5 %.1f" % (
    ret("GLD", 5), ret("SLV", 5), ret("GDX", 5), ret("XME", 5), at(pct_rank(C("HYG"), 5)),
    at(pct_rank((C("HYG") / C("IEF")).dropna(), 5))))
print("FDX r21 %.1f r63 %.1f 5d %+.2f%%" % (at(pct_rank(C("FDX"), 21)), at(pct_rank(C("FDX"), 63)), ret("FDX", 5)))
