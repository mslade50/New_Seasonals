"""Stage B1 recon: state of every asset class, every live calendar anchor,
and the PIT rank of every trigger the surface map will cite.

Nothing here decides anything. It prints the numbers the map quotes so that
no dismissal in 00_surface_map.md is written from recall.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import json
import numpy as np
import pandas as pd

ASOF = pd.Timestamp("2026-08-24")
BAR = pd.Timestamp("2026-08-21")

CLASSES = {
    "us_large": ["SPY", "QQQ", "^GSPC", "^NDX", "DIA"],
    "us_small": ["IWM"],
    "rates": ["TLT", "IEF", "^TNX"],
    "credit": ["HYG", "LQD"],
    "gold_miners": ["GLD", "GDX"],
    "other_metals": ["SLV", "FCX", "XME", "XLB"],
    "energy": ["USO", "UNG", "DBC", "XLE", "XOP"],
    "dollar_fx": ["UUP", "DX-Y.NYB"],
    "international": ["EFA", "EEM", "FXI"],
    "volatility": ["^VIX", "^VIX3M", "^MOVE", "SVXY", "UVXY"],
    "sectors": ["XLK", "XLV", "XLU", "XLP", "XLF", "XLY", "XLI", "XLE", "XLB", "XLC", "XLRE"],
}
ALL = sorted({t for v in CLASSES.values() for t in v})

px = load_prices(ALL)


def state(t):
    d = px.get(t)
    if d is None:
        return None
    c = d["Close"].dropna()
    c = c[c.index <= BAR]
    if len(c) < 300:
        return None
    last = c.iloc[-1]
    hi252 = c.rolling(252).max().iloc[-1]
    lo252 = c.rolling(252).min().iloc[-1]
    sma200 = c.rolling(200).mean().iloc[-1]
    out = {"close": last,
           "d52wh_pct": 100 * (last / hi252 - 1),
           "d52wl_pct": 100 * (last / lo252 - 1),
           "d200_pct": 100 * (last / sma200 - 1)}
    for n in (1, 5, 21, 63, 252):
        out[f"r{n}"] = 100 * (last / c.iloc[-1 - n] - 1) if len(c) > n else np.nan
        if n > 1:
            out[f"rank{n}"] = float(pct_rank(c, n).iloc[-1])
    return out


print("=" * 100)
print(f"CLASS STATE  (bar {BAR.date()}, PIT trailing-252 ranks)")
print("=" * 100)
for cls, tks in CLASSES.items():
    print(f"\n[{cls}]")
    for t in tks:
        s = state(t)
        if s is None:
            print(f"  {t:10s} -- not in cache")
            continue
        print(f"  {t:10s} px {s['close']:10.2f} | 52wh {s['d52wh_pct']:7.2f}% 52wl {s['d52wl_pct']:7.2f}% "
              f"200d {s['d200_pct']:7.2f}% | r5 {s['r5']:6.2f}(rk {s['rank5']:5.1f}) "
              f"r21 {s['r21']:7.2f}(rk {s['rank21']:5.1f}) r63 {s['r63']:7.2f}(rk {s['rank63']:5.1f})")

# ---------------------------------------------------------------- calendar
print("\n" + "=" * 100)
print("CALENDAR ANCHORS, and the offset collisions between them")
print("=" * 100)
ev = load_events()
ev["date"] = pd.to_datetime(ev["date"])
win = ev[(ev["date"] >= ASOF - pd.Timedelta(days=21)) & (ev["date"] <= ASOF + pd.Timedelta(days=35))]
dates = px["SPY"]["Close"].dropna().index
dates = dates[dates <= BAR]
print(win.to_string(index=False))

# month-end position
spy = px["SPY"]["Close"].dropna()
allc = spy.index
# trading sessions left in August 2026 per the cached calendar + known schedule
print(f"\nlast cached session: {allc[-1].date()}")
print("August 2026 remaining sessions (calendar arithmetic): 24 25 26 27 28 31 -> ME anchor 2026-08-31")
print("today 2026-08-24 is ME-5 (5 sessions before the month-end close)")
print("Jackson Hole 2026-08-28 -> today is JH-4")
print("opex was 2026-08-21 -> today is opex+1")

# ------------------------------------------------- cross-sectional breadth
print("\n" + "=" * 100)
print("CROSS-SECTIONAL: how many tape names sit AT a 52-week high today")
print("=" * 100)
tape = json.load(open(Path(__file__).resolve().parents[3] / "data" / "pitch_tape.json"))["tickers"]
at_high = [t for t, v in tape.items() if v["dist_52w_high_pct"] >= -0.25]
at_low = [t for t, v in tape.items() if v["dist_52w_low_pct"] <= 2.0]
print(f"within 0.25% of a 52w high: {len(at_high)}/{len(tape)}  -> {sorted(at_high)}")
print(f"within 2.0% of a 52w low : {len(at_low)}/{len(tape)}  -> {sorted(at_low)}")

# --------------------------------------------------------------- dial state
frag = pd.read_parquet(Path(__file__).resolve().parents[3] / "data" / "rd2_fragility.parquet")
print("\nfragility (sizing parquet) tail:")
print(frag.tail(6).to_string())
