import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

# [64] MOVE band on the entry script's own conventions (kD_v1_svxy_movespike_b.py)
IDX = load_prices(["SPY"])["SPY"].index
px = close_panel(["SPY", "^MOVE", "^VIX"]).reindex(IDX)
rs = px["SPY"].pct_change()
rm = rolling_on_valid(px["^MOVE"], lambda x: x.pct_change())
t = rm.dropna().index[-1]
full = rm.dropna()
p18 = full[full.index >= "2018-03-01"]
rank252 = rolling_on_valid(rm, lambda x: x.rolling(252).rank(pct=True) * 100)
print(f"[64] last {t.date()} MOVE 1d {100 * rm[t]:+.4f}% | SPY 1d {100 * rs[t]:+.4f}% (damage needs <= -0.75%)")
for lbl, s in [("full-history (script q)", full), ("2018-03+", p18)]:
    print(f"     {lbl}: pctile {100 * (s <= rm[t]).mean():.2f} | q90 {100 * s.quantile(.90):+.4f}% q97 {100 * s.quantile(.97):+.4f}%"
          f" | in band [q90,q97): {bool(s.quantile(.90) <= rm[t] < s.quantile(.97))}")
print(f"     PIT trailing-252 rank {rank252[t]:.1f}")

# [21] SPDR washout near a high
SP = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
c = close_panel(SP)
for s in SP:
    x = c[s].dropna()
    print(f"[21] {s} r5 {pct_rank(x, 5).iloc[-1]:5.1f} off-hi {100 * (x.iloc[-1] / x.iloc[-252:].max() - 1):+.2f}%")

# [51] large-bank close-to-close drop in prior-day Wilder ATR units (intraday split not available here)
BANKS = ["JPM", "BAC", "C", "WFC", "GS", "MS", "BK", "USB", "PNC", "SCHW"]
raw = load_prices(BANKS)
for b in BANKS:
    if b not in raw:
        continue
    d = raw[b].dropna()
    a = pd.Series(wilder_atr(d["High"], d["Low"], d["Close"]), index=d.index)
    mv = (d["Close"].iloc[-1] - d["Close"].iloc[-2]) / a.iloc[-2]
    gap = (d["Open"].iloc[-1] - d["Close"].iloc[-2]) / a.iloc[-2]
    print(f"[51] {b:<5} 1d {100 * (d['Close'].iloc[-1] / d['Close'].iloc[-2] - 1):+.2f}% = {mv:+.2f} ATR (gap {gap:+.2f} ATR)")
