"""Read every watchlist trigger and every candidate state on today's tape.

Count-first: nothing is measured until the live geometry is confirmed off
master_prices directly, not off the tape summary file.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

ASOF = pd.Timestamp("2026-09-01")   # freshest bar; entry is MOC 2026-09-02 (lag=1)

NAMES = ["SPY", "QQQ", "IWM", "^GSPC", "^NDX", "DIA", "TLT", "IEF", "LQD", "HYG", "^TNX",
         "GLD", "SLV", "GDX", "NEM", "CEF", "USO", "UNG", "DBC", "UUP", "DX-Y.NYB",
         "EFA", "EEM", "FXI", "EWZ", "^VIX", "^VIX3M", "^MOVE", "SVXY", "UVXY",
         "XLE", "XLF", "XLI", "XLK", "XLV", "XLU", "XLP", "XLY", "XLB", "XLRE",
         "SMH", "IBB", "XBI", "KRE", "ITA", "VNQ", "XME", "XOP", "OIH", "IHI",
         "COP", "CVX", "VLO", "OXY", "SLB", "EOG", "HAL", "WMB",
         "GD", "NOC", "RTX", "BA", "LMT", "HON", "TJX"]
px = close_panel(NAMES)
px = px[px.index <= ASOF]
last = px.index[-1]
print("panel last bar %s  cols %d  rows %d" % (last.date(), px.shape[1], len(px)))
assert last == ASOF, last


def r(t, n):
    s = px[t].dropna()
    return float(s.iloc[-1] / s.iloc[-1 - n] - 1) * 100


def rk(t, n, lb=252):
    """Trailing-lb percentile rank of the n-day return. pct_rank takes the
    PRICE series and does the pct_change itself; feeding it a return series
    double-differences and produced a wrong first draft of this map."""
    return float(pct_rank(px[t], n=n, lookback=lb).iloc[-1])


def d52h(t):
    s = px[t].dropna().iloc[-252:]
    return float(s.iloc[-1] / s.max() - 1) * 100


def d52l(t):
    s = px[t].dropna().iloc[-252:]
    return float(s.iloc[-1] / s.min() - 1) * 100


def sma_dist(t, n=200):
    s = px[t].dropna()
    return float(s.iloc[-1] / s.rolling(n).mean().iloc[-1] - 1) * 100


print("\n=== 1. LIVE GEOMETRY (own computation, not the tape file) ===")
rows = []
for t in ["SPY", "QQQ", "IWM", "TLT", "IEF", "LQD", "HYG", "^TNX", "GLD", "SLV", "GDX", "NEM",
          "USO", "DBC", "XLE", "XOP", "^VIX", "^MOVE", "XLI", "ITA", "XLRE", "XLU", "SMH",
          "XLK", "XLV", "IBB"]:
    rows.append(dict(t=t, r1=round(r(t, 1), 2), r5=round(r(t, 5), 2), r21=round(r(t, 21), 2),
                     rk5=round(rk(t, 5), 1), rk21=round(rk(t, 21), 1), rk63=round(rk(t, 63), 1),
                     h52=round(d52h(t), 2), l52=round(d52l(t), 2)))
show(rows, "live geometry")

print("\n=== 2. WATCHLIST TRIGGERS ===")
wl = load_watchlist()
ents = wl["entries"] if isinstance(wl, dict) else wl
print("%d active entries" % len(ents))

tnx_21 = (px["^TNX"].iloc[-1] - px["^TNX"].iloc[-22]) * 100
tnx_252 = (px["^TNX"].iloc[-1] - px["^TNX"].iloc[-253]) * 100


def line(i, verdict, detail):
    print("[%2d] %-9s %s\n        %s" % (i, verdict, ents[i]["title"][:72], detail))


line(0, "PASS", "midterm year; NFP 2026-09-04 is +2 td. Parks to 2027-01.")
line(1, "PASS", "episode count 4 vs 8 required. HYG %+.2f%% off high, LQD %+.2f%% above low: state live, still uncountable." % (d52h("HYG"), d52l("LQD")))
line(2, "PASS", "next CPI 2026-09-11, +6 td; the overnight entry is 5 sessions away.")
line(3, "PASS", "GDX r5 %.1f vs the >=95 leg (miners BROKE, did not thrust); GLD %+.2f%% off high vs the added within-10%% leg." % (rk("GDX", 5), d52h("GLD")))
line(4, "CHECK", "USO 1d %+.2f%% -> band [5,6) LIVE. ATR + event legs below." % r("USO", 1))
line(5, "PASS", "TLT %+.2f%% above low vs <=0.5%% rung (IEF %+.2f, LQD %+.2f both pass)." % (d52l("TLT"), d52l("IEF"), d52l("LQD")))
line(6, "PASS", "^SKEW not carried in master_prices; tape r5 68.7 vs >=95. midterm block stands.")
line(7, "PASS", "USO r5 %.1f vs >=90 leg; r63 %.1f vs <=20 leg." % (rk("USO", 5), rk("USO", 63)))
line(8, "PASS", "IHI r21 %.1f vs the rank-100 rung." % rk("IHI", 21))
line(9, "PASS", "FXI r5 %.1f vs <=20 trigger." % rk("FXI", 5))
line(10, "PASS", "parks to trading days 4-12 of November 2026.")
line(11, "PASS", "SPY %+.2f%% off high vs <=0.5%%; TLT %+.2f%% above low vs <=1%%." % (d52h("SPY"), d52l("TLT")))
line(12, "CHECK", "VIX 1d %+.2f%% vs >=+5 pop -> LIVE. SPY 1d %+.2f%% vs > -0.75%% -> LIVE. calm-tape leg below." % (r("^VIX", 1), r("SPY", 1)))
line(13, "PASS", "DX 21d rank %.1f vs <=15; 21-session yield change %+.1f bp vs the +20 bp floor." % (rk("DX-Y.NYB", 21), tnx_21))
line(14, "PASS", "one-day XLV-XLK gap %+.2fpp vs >=+3.0pp." % (r("XLV", 1) - r("XLK", 1)))
line(15, "PASS", "^TNX r21 %.1f vs >=65 leg." % rk("^TNX", 21))
line(16, "PASS", "TLT 1d %+.2f%% vs >=+1.5%% thrust rung." % r("TLT", 1))
line(17, "PASS", "KRE r5 %.1f; arm is an ex-crisis cost threshold no new episode moves." % rk("KRE", 5))
line(18, "CHECK", "252-session yield change %+.1f bp vs the +78 bp arm; ^TNX %+.2f%% off its 252d max." % (tnx_252, d52h("^TNX")))
line(19, "CHECK", "energy z10>=2.0 count below (arm is a count in [2,3]).")
line(20, "PASS", "SPY %+.2f%% off high vs >2.0%%; raw-21d fragility 62.8 vs <=50." % d52h("SPY"))
line(21, "CHECK", "sector r5<=5 within 5%% of high below; dial 87.5 vs episode max 68.6 is a standing blocker.")
line(22, "PASS", "XLU r21 %.1f vs <=5 leg; TLT r21 %.1f vs <25 rung." % (rk("XLU", 21), rk("TLT", 21)))
line(23, "PASS", "parks to a non-midterm year; DX r21 %.1f." % rk("DX-Y.NYB", 21))
line(24, "PASS", "SPY %+.2f%% off high vs >=2.0%%; dial ma10 87.5 vs <50." % d52h("SPY"))
line(25, "PASS", "SMH r63 %.1f (state live) but r5 %.1f vs the <15 arm, and refclass Q is unmoved." % (rk("SMH", 63), rk("SMH", 5)))
line(26, "PASS", "HYG %+.2f%% off high vs the <=0.25%% rung -> tight rung NOT live today." % d52h("HYG"))
line(27, "PASS", "midterm; conference passed 2026-08-28. Parks to 2027-09.")
line(28, "CHECK", "pooled r21>=90 & r63<=10 holders below.")
line(29, "CHECK", "metals break: SLV 1d %+.2f%%, GLD %+.2f%%, GDX %+.2f%% -> conjunction LIVE. depth + lag debt below." % (r("SLV", 1), r("GLD", 1), r("GDX", 1)))
line(30, "CHECK", "^MOVE trailing-252 LEVEL pctile below vs the [40,50) band.")
line(31, "PASS", "parks to December.")
line(32, "CHECK", "XLE %+.2f%% off 252d max on a SPY %+.2f%% session -> state live; standing blocker is not a number." % (d52h("XLE"), r("SPY", 1)))

print("\n=== 3. NEW-STATE COUNT-FIRST ===")

# 3a. USO band + ATR (watchlist 4 arm)
raw = load_prices(["USO"])["USO"]
raw = raw[raw.index <= ASOF]
atr = wilder_atr(raw["High"], raw["Low"], raw["Close"], 14)
uso_1d = r("USO", 1)
atr_abs = float(atr[-1])
atr_pct = atr_abs / float(raw["Close"].iloc[-1]) * 100
print("USO 1d %+.3f%%  Wilder-14 ATR %.4f (%.3f%% of price) -> move = %.2f ATR"
      % (uso_1d, atr_abs, atr_pct, uso_1d / atr_pct))
print("  band [5,6): %s   >=1.50 ATR: %s"
      % ("YES" if 5.0 <= uso_1d < 6.0 else "NO", "YES" if uso_1d / atr_pct >= 1.50 else "NO"))
print("  PPI 2026-09-10 (+5 td), CPI 2026-09-11 (+6 td) -> both OUTSIDE a 3-session hold from 2026-09-02")

# 3b. metals complex break
print("\n metals:")
for t in ["SLV", "GLD", "GDX", "NEM", "CEF"]:
    print("  %-4s 1d %+.2f%%  5d %+.2f%%  r21 %.1f" % (t, r(t, 1), r(t, 5), rk(t, 21)))
print("  SLV depth bucket <=-4%%: %s (live %+.2f%%)"
      % ("YES" if r("SLV", 1) <= -4.0 else "NO", r("SLV", 1)))

# 3c. VIX calm-tape leg (watchlist 12)
vix = px["^VIX"].dropna()
lvl_rank21 = float((vix.iloc[-21:] <= vix.iloc[-1]).mean() * 100)
print("\n vol:")
print("  ^VIX level pct-rank over trailing 21 sessions: %.1f   21d-RETURN rank(252): %.1f"
      % (lvl_rank21, rk("^VIX", 21)))
print("  VIX 1d %+.2f%%  SPY 1d %+.2f%%  VIX level %.2f  VIX3M %.2f"
      % (r("^VIX", 1), r("SPY", 1), vix.iloc[-1], px["^VIX3M"].dropna().iloc[-1]))

# 3d. energy z10 count (watchlist 19)
ENER = ["XLE", "XOP", "USO", "COP", "CVX", "VLO", "OXY", "SLB", "EOG", "HAL", "WMB"]
cnt = [(t, round(float(zscore(px[t].dropna(), 10).iloc[-1]), 2)) for t in ENER]
n2 = sum(1 for _, z in cnt if z >= 2.0)
print("\n energy z10>=2.0 count = %d of 11 -> top: %s" % (n2, sorted(cnt, key=lambda x: -x[1])[:5]))

# 3e. sector washout within 5% of high (watchlist 21)
SEC = ["XLE", "XLF", "XLI", "XLK", "XLV", "XLU", "XLP", "XLY", "XLB"]
hits = [(t, round(rk(t, 5), 1), round(d52h(t), 2)) for t in SEC if rk(t, 5) <= 5]
print(" sectors with r5<=5: %s  (arm also needs within 5%% of the 52w high)" % hits)
print(" XLRE r5 %.1f / %+.2f%% off high (XLRE is outside the nine-SPDR cell)"
      % (rk("XLRE", 5), d52h("XLRE")))

# 3f. pooled laggard state (watchlist 28)
POOL = ["SPY", "QQQ", "IWM", "EFA", "EEM", "FXI", "VNQ", "GLD", "SLV", "DBC", "TLT", "LQD", "HYG",
        "XLE", "XLF", "XLI", "XLK", "XLV", "XLU", "XLP", "XLY", "XLB", "XLRE", "SMH", "IBB",
        "XBI", "KRE", "ITA", "XME"]
holders = [(t, round(rk(t, 21), 1), round(rk(t, 63), 1), round(rk(t, 5), 1)) for t in POOL
           if rk(t, 21) >= 90 and rk(t, 63) <= 10]
print(" r21>=90 & r63<=10 holders: %s" % (holders if holders else "NONE"))

# 3g. ^MOVE level band (watchlist 30)
mv = px["^MOVE"].dropna()
print(" ^MOVE trailing-252 LEVEL pctile %.1f vs the [40,50) band"
      % float((mv.iloc[-252:] <= mv.iloc[-1]).mean() * 100))

# 3h. defense / industrial washout (NEW today, no watchlist entry)
print("\n defense + industrials (no watchlist entry covers this):")
for t in ["ITA", "GD", "NOC", "RTX", "BA", "LMT", "XLI"]:
    if t in px.columns and px[t].notna().any():
        print("  %-4s z10 %+.2f  r5 %5.1f  r21 %5.1f  r63 %5.1f  52wH %+.2f%%  200d %+.2f%%"
              % (t, float(zscore(px[t].dropna(), 10).iloc[-1]), rk(t, 5), rk(t, 21), rk(t, 63),
                 d52h(t), sma_dist(t)))

# 3i. growth vs defensive 63d dispersion (NEW today)
print("\n growth vs defensive 63d ranks:")
print("  XLK %.1f | SMH %.1f | QQQ %.1f || XLV %.1f | IBB %.1f | XLP %.1f"
      % (rk("XLK", 63), rk("SMH", 63), rk("QQQ", 63), rk("XLV", 63), rk("IBB", 63), rk("XLP", 63)))
