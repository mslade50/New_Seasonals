"""Reference close + Wilder-14 ATR as of the freshest bar, for whatever
survives stage C. The pitch convention is Wilder-14, never the scanner's
simple-mean ATR.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import pandas as pd
from pitch_grammar import wilder_atr
from pitch_lab import load_prices

ASOF = pd.Timestamp("2026-08-27")
NAMES = sys.argv[1:] or ["SPY", "QQQ", "IWM", "EFA", "EEM", "EWZ", "IBB", "XBI",
                         "SVXY", "GLD", "GDX", "XLE", "XLP", "XLU", "XLRE", "TLT"]

px = load_prices(NAMES)
print("%-8s %10s %10s %8s" % ("ticker", "close", "atr14", "atr%"))
for t in NAMES:
    if t not in px:
        print("%-8s MISSING" % t)
        continue
    d = px[t].loc[:ASOF]
    atr = float(wilder_atr(d["High"], d["Low"], d["Close"], 14)[-1])
    c = float(d["Close"].iloc[-1])
    print("%-8s %10.4f %10.4f %7.2f%%" % (t, c, atr, 100 * atr / c))
