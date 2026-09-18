import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
from pitch_lab import load_prices
from pitch_grammar import wilder_atr

TK = ["IEF", "TLT", "HYG", "GLD", "SPY"]
P = load_prices(TK)
for t in TK:
    df = P[t]
    a = wilder_atr(df["High"], df["Low"], df["Close"])
    last = df.index[-1]
    c = float(df["Close"].iloc[-1])
    print(f"{t}: last {last.date()} close {c:.2f} Wilder14 ATR {float(a[-1]):.3f} ({100*float(a[-1])/c:.2f}%)")
