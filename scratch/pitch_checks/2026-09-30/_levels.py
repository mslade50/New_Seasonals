import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import load_prices, close_panel  # noqa: E402

tickers = ["^TNX", "^MOVE", "^VIX", "^VIX3M", "DX-Y.NYB", "MXN=X", "USO", "CL=F", "NG=F", "UNG",
           "GLD", "SLV", "GDX", "TLT", "IEF", "HYG", "SPY", "XLF", "KRE", "EEM", "XLRE"]
px = close_panel(load_prices(tickers))
tail = px.tail(7)
print(tail.round(3).T.to_string())
print("\n1d % change (last row):")
print((px.pct_change().iloc[-1] * 100).round(2).to_string())
