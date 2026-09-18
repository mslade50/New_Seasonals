"""Extra live readings for watchlist verdicts (2026-09-16 close)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

TK = ["IHI", "FXI", "EEM", "XLV", "XLK", "XLF", "KRE", "JPM", "BAC", "C", "WFC", "GS", "MS", "USB", "PNC",
      "SCHW", "BNY", "STT", "HYG", "IEF", "TLT", "SLV", "GDX", "GLD", "SPY", "EWJ", "VLO", "USO", "XLE"]
px = close_panel(TK)
last = px.index[-1]
print("last bar", last.date())


def r(t, n):
    return float(pct_rank(px[t].dropna(), n).iloc[-1])


print("IHI r21", round(r("IHI", 21), 1), "FXI r21", round(r("FXI", 21), 1), "FXI r5", round(r("FXI", 5), 1))
d1 = px.pct_change(fill_method=None).iloc[-1] * 100
print("1d:", {t: round(float(d1[t]), 2) for t in TK})
banks = ["JPM", "BAC", "C", "WFC", "GS", "MS", "USB", "PNC", "SCHW", "BNY", "STT"]
r5 = {b: round(r(b, 5), 1) for b in banks}
print("bank r5", r5, "share<=20", sum(v <= 20 for v in r5.values()) / len(banks))
print("bank r63 median", float(np.median([r(b, 63) for b in banks])))
for t in ["GS", "BAC", "MS", "JPM", "USB", "PNC", "WFC", "C", "SCHW"]:
    p = load_prices([t])[t]
    atr = pd.Series(wilder_atr(p["High"], p["Low"], p["Close"], 14), index=p.index)
    move = p["Close"].iloc[-1] - p["Close"].iloc[-2]
    print(f"{t} move {move:+.2f} prior-day wilder ATR {atr.iloc[-2]:.2f} -> {move/atr.iloc[-2]:+.2f} ATR; "
          f"open->close {p['Close'].iloc[-1]-p['Open'].iloc[-1]:+.2f} gap {p['Open'].iloc[-1]-p['Close'].iloc[-2]:+.2f}")
print("HYG z10 pitch_lab", round(float(zscore(px["HYG"].dropna(), 10).iloc[-1]), 2), "IEF r5", round(r("IEF", 5), 1))
tape = json.load(open(Path(__file__).resolve().parents[3] / "data" / "pitch_tape.json"))["tickers"]
lows = sorted([k for k, v in tape.items() if v.get("dist_52w_low_pct") is not None and v["dist_52w_low_pct"] <= 1.0])
highs = sorted([k for k, v in tape.items() if v.get("dist_52w_high_pct") is not None and v["dist_52w_high_pct"] >= -1.0])
print(f"tape within 1% of 52w LOW ({len(lows)}):", lows)
print(f"tape within 1% of 52w HIGH ({len(highs)}):", highs)
