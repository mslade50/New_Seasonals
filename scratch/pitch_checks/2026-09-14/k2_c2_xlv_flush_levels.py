"""C2 near-miss reference levels: 09-11 close and Wilder-14 ATR for XLV and SPY, plus
trailing betas (the hedge-ratio question)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
from pitch_grammar import wilder_atr

d = load_prices(["XLV", "SPY"])
for t, f in d.items():
    atr = wilder_atr(f["High"], f["Low"], f["Close"])
    a = float(atr[-1])
    print(t, f.index[-1].date(), "close", round(float(f["Close"].iloc[-1]), 2), "ATR14w", round(a, 2),
          f"({100*a/float(f['Close'].iloc[-1]):.2f}%)")
px = close_panel(["XLV", "SPY"])
dr = px.pct_change()
for n in (63, 126, 252):
    b = dr["XLV"].iloc[-n:].cov(dr["SPY"].iloc[-n:]) / dr["SPY"].iloc[-n:].var()
    c = dr["XLV"].iloc[-n:].corr(dr["SPY"].iloc[-n:])
    print(f"trailing {n}d beta {b:.2f} corr {c:.2f}")
