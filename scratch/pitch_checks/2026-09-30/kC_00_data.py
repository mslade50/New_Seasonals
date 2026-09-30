import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

T = ["UNG", "NG=F", "GLD", "GC=F", "USO", "CL=F", "XLE", "^TNX", "GDX", "DX-Y.NYB", "DX=F", "UUP", "SPY", "BNO"]
px = load_prices(T)
for t, d in px.items():
    print(t, d.index[0].date(), d.index[-1].date(), len(d), list(d.columns)[:6])
    print(d.tail(4)[["Close"]].T.round(3).to_string())
