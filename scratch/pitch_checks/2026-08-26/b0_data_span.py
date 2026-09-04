"""b0: data span / usability recon for C3, C4, C5, C11.

Establish FIRST what history exists. ^MOVE in particular may be short or
gappy (C11's brief says a data kill is possible and must be reported early).
No conclusions here beyond usability.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import _valid_pct_change
import pandas as pd, numpy as np

pd.set_option("display.width", 220)

TICKERS = [
    # C3 dollar
    "UUP", "DX-Y.NYB",
    # C4 international
    "EEM", "EFA", "SPY", "FXI", "EWJ", "EWZ", "INDA",
    # C5 metals complex
    "GLD", "SLV", "GDX", "GDXJ", "NEM", "FCX", "XME", "HG=F", "SI=F", "GC=F", "SIL",
    # C11 rates x vol
    "^MOVE", "^VIX", "^VIX3M", "TLT", "IEF", "SHY", "LQD",
]

px = load_prices(TICKERS)
rows = []
for t in TICKERS:
    if t not in px:
        rows.append({"ticker": t, "n": 0, "first": "MISSING", "last": "MISSING"})
        continue
    s = px[t]["Close"].dropna()
    # gap analysis: biggest run of missing business days
    idx = s.index
    bd = pd.bdate_range(idx[0], idx[-1])
    covered = len(idx) / len(bd)
    d = pd.Series(idx).diff().dt.days
    rows.append({
        "ticker": t, "n": len(s), "first": str(idx[0].date()), "last": str(idx[-1].date()),
        "bd_cover": round(covered, 3), "max_gap_days": int(d.max()) if len(d) > 1 else 0,
        "n_gaps_gt7d": int((d > 7).sum()),
        "last_px": round(float(s.iloc[-1]), 4),
    })
print(pd.DataFrame(rows).to_string(index=False))

# ^MOVE specifically: usable span after dropping leading sparse era
if "^MOVE" in px:
    m = px["^MOVE"]["Close"].dropna()
    print("\n=== ^MOVE yearly observation counts ===")
    print(m.groupby(m.index.year).size().to_string())
    print("\n^MOVE last 5:")
    print(m.tail(5).to_string())

# today's live states, re-derived (never trust the map)
print("\n=== TODAY'S LIVE STATE, re-derived ===")
panel = close_panel([t for t in TICKERS if t in px])
# C3
for t in ["UUP", "DX-Y.NYB"]:
    s = px[t]["Close"].dropna()
    rk = pct_rank(s, 21, 252)
    print(f"  {t}: ret21={_valid_pct_change(s,21).iloc[-1]*100:+.2f}%  rank21={rk.iloc[-1]:.2f}")
# C4
for t in ["EEM", "EFA", "SPY"]:
    s = px[t]["Close"].dropna()
    print(f"  {t}: rank5={pct_rank(s,5,252).iloc[-1]:.1f} rank21={pct_rank(s,21,252).iloc[-1]:.1f} "
          f"rank63={pct_rank(s,63,252).iloc[-1]:.1f}")
# C5
comp = ["GLD", "SLV", "GDX", "NEM", "FCX", "XME"]
print("  C5 complex rank21 (252d PIT):")
cnt = 0
for t in comp:
    s = px[t]["Close"].dropna()
    r = pct_rank(s, 21, 252).iloc[-1]
    cnt += int(r >= 95)
    print(f"    {t}: rank21={r:.1f}  ret21={_valid_pct_change(s,21).iloc[-1]*100:+.2f}%")
print(f"    COUNT >=95 today = {cnt} of {len(comp)}")
# metal vs equity: did the METAL move?
for t in ["GLD", "SLV", "GC=F", "SI=F", "HG=F"]:
    if t in px:
        s = px[t]["Close"].dropna()
        print(f"    metal {t}: ret21={_valid_pct_change(s,21).iloc[-1]*100:+.2f}% "
              f"ret5={_valid_pct_change(s,5).iloc[-1]*100:+.2f}% last={s.index[-1].date()}")
# C11
if "^MOVE" in px:
    mv = px["^MOVE"]["Close"]
    vx = px["^VIX"]["Close"]
    both = pd.DataFrame({"mv": mv, "vx": vx}).dropna()
    ratio = both["mv"] / both["vx"]
    cur = ratio.iloc[-1]
    print(f"  MOVE/VIX = {cur:.4f}  252d pctile {(ratio.tail(252) <= cur).mean()*100:.1f} "
          f"full pctile {(ratio <= cur).mean()*100:.1f}  n={len(ratio)} "
          f"span {ratio.index[0].date()}..{ratio.index[-1].date()}")
    print(f"  ^MOVE level {both['mv'].iloc[-1]:.2f} (252d pctile "
          f"{(both['mv'].tail(252) <= both['mv'].iloc[-1]).mean()*100:.1f}, full "
          f"{(both['mv'] <= both['mv'].iloc[-1]).mean()*100:.1f})")
    print(f"  ^VIX level {both['vx'].iloc[-1]:.2f} (252d pctile "
          f"{(both['vx'].tail(252) <= both['vx'].iloc[-1]).mean()*100:.1f})")
