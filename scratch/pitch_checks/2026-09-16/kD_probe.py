import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_grammar import wilder_atr

px = load_prices(["DX-Y.NYB", "UUP"])
for t, g in px.items():
    a = wilder_atr(g["High"].values, g["Low"].values, g["Close"].values)
    print(t, g.index[-1].date(), "close", round(float(g["Close"].iloc[-1]), 3), "ATR14w", round(float(a[-1]), 4),
          "ATR%", round(100 * float(a[-1]) / float(g["Close"].iloc[-1]), 3))
ev = load_events(None)
print(sorted(ev["event"].unique()))
print(ev.query("date >= '2026-09-16' and date <= '2026-09-25'")[["date", "event", "detail"]].to_string())
