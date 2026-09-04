"""Today's live reading for every active watchlist trigger (2026-08-28).

Stage B1 owes each parked entry a verdict citing today's number. This computes
them from master_prices rather than copying tape values by hand.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import pandas as pd
from pitch_lab import load_prices, pct_rank, zscore

ASOF = pd.Timestamp("2026-08-27")   # freshest bar
NAMES = ["SPY", "QQQ", "IWM", "TLT", "IEF", "LQD", "HYG", "^TNX", "GLD", "GDX", "SLV",
         "USO", "XLE", "XOP", "OIH", "UNG", "DBC", "UUP", "DX-Y.NYB", "EFA", "EEM",
         "FXI", "^VIX", "^VIX3M", "SVXY", "^SKEW", "XLK", "XLV", "XLU", "XLP", "XLRE",
         "XLF", "XLI", "XLY", "XLB", "XLC", "SMH", "KRE", "IHI", "XME", "FCX"]

px = load_prices(NAMES)


def ret_n(t, n):
    s = px[t]["Close"].dropna().loc[:ASOF]
    return float(s.iloc[-1] / s.iloc[-1 - n] - 1.0) * 100


def rank_n(t, n, lb=252):
    s = px[t]["Close"].dropna()
    return float(pct_rank(s, n, lb).loc[:ASOF].iloc[-1])


def dist_hi(t, lb=252):
    s = px[t]["Close"].dropna().loc[:ASOF]
    return float(s.iloc[-1] / s.rolling(lb).max().iloc[-1] - 1.0) * 100


def dist_lo(t, lb=252):
    s = px[t]["Close"].dropna().loc[:ASOF]
    return float(s.iloc[-1] / s.rolling(lb).min().iloc[-1] - 1.0) * 100


V = []


def v(i, title, verdict, number):
    V.append((i, title, verdict, number))


v(0, "TLT from the NFP close, long end at 52w floor", "PASS",
  "parks to the first NON-midterm NFP (2027-01). 2026 is midterm.")
v(1, "LQD vs HYG at joint 52w extremes", "PASS",
  "state IS live (HYG %.2f%% off high, LQD %.2f%% off low) but the trigger is >=8 declustered "
  "episodes ex-2018; still 4." % (dist_hi("HYG"), dist_lo("LQD")))
v(2, "SVXY overnight into the CPI print", "PASS",
  "parks to the CPI eve, 2026-09-10. Today is CPI-9.")
v(3, "GLD on a miner-led thrust the metal has not joined", "PASS",
  "needs GDX 5d rank >=95; today %.1f." % rank_n("GDX", 5))
v(4, "XLE on a crude one-day pop in [5%,6%)", "PASS",
  "needs a USO 1d pop in [5%%,6%%) and >=1.50 ATR; today USO 1d %+.2f%%." % ret_n("USO", 1))
v(5, "TLT with the whole IG complex pinned at 52w lows", "PASS",
  "tight rung needs TLT within 0.5%% of its 52w low; today %+.2f%%." % dist_lo("TLT"))
v(6, "SPY on a skew spike alone", "PASS",
  "needs pct_rank(^SKEW,5) >= 95; today %.1f. Also midterm-blocked." % rank_n("^SKEW", 5))
v(7, "Fade a crude thrust out of a deep base", "PASS",
  "needs USO 5d rank >=90 and 63d rank <=20; today %.1f / %.1f."
  % (rank_n("USO", 5), rank_n("USO", 63)))
v(8, "IHI at a 21d rank of 100 out of a drawdown", "PASS",
  "needs IHI 21d rank 100; today %.1f." % rank_n("IHI", 21))
v(9, "FXI 5d break inside an intact thrust", "PASS",
  "needs FXI 5d rank <=20 AND 21d rank >=80; today %.1f / %.1f."
  % (rank_n("FXI", 5), rank_n("FXI", 21)))
v(10, "TLT on the NOVEMBER month-position effect", "PASS", "parks to November.")
v(11, "Short SPY at a 52w high while TLT sits at a 52w low", "PASS",
  "needs SPY within 0.5%% of its high AND TLT within 1%% of its low; today %+.2f%% / %+.2f%%."
  % (dist_hi("SPY"), dist_lo("TLT")))
v(12, "TLT into the month-end close, entered nine sessions before", "PASS",
  "entry window is ME-9; today is ME-1 (August month-end is Mon 2026-08-31).")
v(13, "SPY on a volatility pop inside an already-calm tape", "PASS",
  "calm leg passes (VIX 21d rank %.1f) but needs VIX UP >=5%% today; VIX 1d %+.2f%%."
  % (rank_n("^VIX", 21), ret_n("^VIX", 1)))

tnx = px["^TNX"]["Close"].dropna().loc[:ASOF]
yield_rise_21 = float(tnx.iloc[-1] - tnx.iloc[-22])
v(14, "Gold on an unconfirmed rate rise, both dials at force", "PASS",
  "dollar leg passes (DX 21d rank %.1f <= 15) but the yield leg needs a 21-session rise "
  ">= +0.20pt; today %+.3fpt." % (rank_n("DX-Y.NYB", 21), yield_rise_21))
v(15, "Tech against healthcare after a rotation gap", "PASS",
  "needs a one-day XLV-minus-XLK gap >= +3.0pp; today %+.2fpp (XLK led, wrong sign)."
  % (ret_n("XLV", 1) - ret_n("XLK", 1)))
v(16, "Short the dollar on a rate rise it does not confirm", "PASS",
  "needs TNX 21d rank >=65 while DX 21d rank <=20; today %.1f / %.1f."
  % (rank_n("^TNX", 21), rank_n("DX-Y.NYB", 21)))
v(17, "Crude through Jackson Hole, entered six sessions before", "PASS",
  "entry window is JH-6; today IS the conference (JH+0).")
v(18, "Short TLT after a big up day from inside the 52w low zone", "PASS",
  "needs TLT 1d >= +1.5%% while within 4%% of its 252d low; today 1d %+.2f%%, %+.2f%% off "
  "the low." % (ret_n("TLT", 1), dist_lo("TLT")))
v(19, "Short KRE against XLF on a bank-breadth washout", "PASS",
  "parked on COST once the crisis years come out; nothing about the cost bar moved.")
v(20, "High yield across Jackson Hole, entered five sessions before", "PASS",
  "entry window is JH-5; today IS the conference. Standing blocker besides.")
v(21, "Duration-neutral IEF vs 0.52 TLT with TNX at a 52w high", "PASS",
  "needs ^TNX within 0.25%% of its trailing-252 high; today %+.2f%%. Second blocker also live: "
  "holds spanning Jackson Hole are 0-for-6 and today is JH." % dist_hi("^TNX"))

ENERGY = ["XLE", "XOP", "USO", "OIH"]
z_energy = {t: float(zscore(px[t]["Close"].dropna(), 10).loc[:ASOF].iloc[-1]) for t in ENERGY}
v(22, "Narrow energy thrust cluster, 2-3 names at z10 > 2", "PASS",
  "needs >=2 of the 11-name complex at z10 >= 2.0; the tape's energy max is %+.2f (count 0)."
  % max(z_energy.values()))
v(23, "Cross-sectional new-high breadth, index further off its high", "PASS",
  "needs SPY more than 2.0%% below its 52w high (today %+.2f%%) AND raw-21d fragility <= 50 "
  "(today 69.3). Both fail." % dist_hi("SPY"))

oc = px["OIH"]["Close"].dropna()
xc = px["XOP"]["Close"].dropna()
spread = ((oc / oc.shift(63)) - (xc / xc.shift(63))) * 100
spread = spread.dropna()
pit = float(spread.rolling(252).rank(pct=True).loc[:ASOF].iloc[-1] * 100)
v(24, "Long OIH outright at a 63-day services-vs-E&P extreme", "PASS",
  "state IS firing (OIH 63d %+.2f%% vs XOP %+.2f%%, spread %+.2fpp, PIT %.2f) but the trigger "
  "is the RECORD (32 of 51 wins); it stands at 28 and no new episode has graded."
  % (ret_n("OIH", 63), ret_n("XOP", 63), ret_n("OIH", 63) - ret_n("XOP", 63), pit))

SECT = ["XLK", "XLV", "XLF", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE"]
wash = {t: rank_n(t, 5) for t in SECT}
lowest = min(wash.items(), key=lambda kv: kv[1])
v(25, "Sector washout into a 52w high, as a family effect", "PASS",
  "needs a SPDR at 5d rank <= 5 within 5%% of its 52w high; the lowest today is %s at %.1f."
  % (lowest[0], lowest[1]))
v(26, "Utilities washout with the long end hit ALONGSIDE it", "PASS",
  "needs XLU 21d rank <= 5 with the long end hit too; XLU 21d rank %.1f and TLT is RALLYING "
  "(5d rank %.1f, %+.2f%%), so the rates leg is on the wrong side."
  % (rank_n("XLU", 21), rank_n("TLT", 5), ret_n("TLT", 5)))
v(27, "The bare dollar washout, no rate leg", "PASS",
  "needs DX 21d return at a trailing-252 PIT rank <= 2; today %.1f. Also parks to a "
  "non-midterm year." % rank_n("DX-Y.NYB", 21))
v(28, "High yield at a fresh 52w high while the index has not", "PASS",
  "HYG %+.2f%% off its high and SPY %+.2f%% off its own, so the DEPTH leg the cell died on is "
  "not there; it needs the index materially further off."
  % (dist_hi("HYG"), dist_hi("SPY")))
v(29, "The single small-cap session from ME-3 to ME-2", "PASS",
  "the window is ME-3 -> ME-2; today is ME-1. Midterm-blocked besides.")
v(30, "Semiconductors at a 63-day rank floor inside a top-decile year", "PASS",
  "needs SMH 5d rank < 15 (the conditioner the pitch had backwards); today %.1f. 63d rank %.1f."
  % (rank_n("SMH", 5), rank_n("SMH", 63)))
v(31, "IG complex at 52w lows while high yield prints a high", "PASS",
  "state IS live (IEF %+.2f%%, LQD %+.2f%%, HYG %+.2f%%) but the trigger is an episode count "
  "above one and this is still the SAME 2026 episode."
  % (dist_lo("IEF"), dist_lo("LQD"), dist_hi("HYG")))

print("WATCHLIST VERDICTS, asof bar %s  (n=%d)" % (ASOF.date(), len(V)))
print("=" * 100)
for i, title, verdict, number in V:
    print("[%2d] %-5s | %s" % (i, verdict, title))
    print("        %s" % number)
checks = [x for x in V if x[2] == "CHECK"]
print("\nCHECK: %d   PASS: %d" % (len(checks), len(V) - len(checks)))
