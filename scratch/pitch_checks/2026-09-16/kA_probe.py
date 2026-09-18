import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

tk = ["TLT", "IEF", "GLD", "GC=F", "DX-Y.NYB", "DX=F", "UUP", "^TNX", "^IRX", "SPY"]
px = load_prices(tk)
for t, g in px.items():
    print(t, g.index[0].date(), g.index[-1].date(), len(g), round(float(g["Close"].iloc[-1]), 3))
ev = load_events(["fomc_decision"])
print(ev.tail(12).to_string())
print(len(ev), ev.date.min().date(), ev.detail.value_counts().head(10))
print(load_events(["cpi", "ppi", "nfp", "quad_witching", "vix_expiry"]).query("date >= '2026-09-14' and date <= '2026-10-02'").to_string())
