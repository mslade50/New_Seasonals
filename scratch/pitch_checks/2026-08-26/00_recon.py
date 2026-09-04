"""Recon: where do today's candidate states sit in their own history?
Survey input only. Nothing here is a check; every number gets re-derived
inside the candidate's own script."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import pandas as pd, numpy as np

pd.set_option("display.width", 200)

# ---------- 1. CBOE put/call composition (ZERO registry mentions) ----------
pc = pd.read_parquet(ROOT / "data/cboe_putcall.parquet")
pc = pc.sort_index()
print("=== CBOE put/call, last 5 ===")
print(pc.tail(5))
ratio = pc["index"] / pc["equity"]
for name, s in [("index", pc["index"]), ("equity", pc["equity"]), ("etp", pc["etp"]),
                ("spx", pc["spx"]), ("total", pc["total"]), ("index/equity", ratio)]:
    s10 = s.rolling(10).mean()
    cur, cur10 = s.iloc[-1], s10.iloc[-1]
    p_full = (s <= cur).mean() * 100
    p252 = (s.tail(252) <= cur).mean() * 100
    p10_252 = (s10.tail(252) <= cur10).mean() * 100
    print(f"  {name:<13} last={cur:6.3f} ma10={cur10:6.3f} | pctile full={p_full:5.1f} 252d={p252:5.1f} | ma10 252d pctile={p10_252:5.1f}")

# ---------- 2. gold/silver ratio (ZERO registry mentions) ----------
px = close_panel(["GLD", "SLV", "GDX", "SPY", "UUP", "DX-Y.NYB", "HYG", "LQD",
                  "^VIX", "^VIX3M", "TLT", "IEF", "QQQ", "SMH", "IWM", "^MOVE", "EEM"])
gs = px["GLD"] / px["SLV"]
print("\n=== gold/silver ratio (GLD/SLV price ratio) ===")
print(f"  today {gs.iloc[-1]:.4f} | 252d pctile {(gs.tail(252) <= gs.iloc[-1]).mean()*100:.1f} "
      f"| full pctile {(gs <= gs.iloc[-1]).mean()*100:.1f}")
r21 = gs.pct_change(21)
print(f"  21d change in ratio {r21.iloc[-1]*100:+.2f}% | pctile of 21d changes (252d) "
      f"{(r21.tail(252) <= r21.iloc[-1]).mean()*100:.1f}")

# ---------- 3. dollar 21d washout ----------
for t in ["UUP", "DX-Y.NYB"]:
    s = px[t].dropna()
    r = s.pct_change(21)
    rk = pct_rank(s, 21, 252)
    print(f"\n=== {t} 21d state ===")
    print(f"  ret21 {r.iloc[-1]*100:+.2f}%  rank21(252d) {rk.iloc[-1]:.1f} "
          f"| days since rank21<=2: {(rk.tail(504) <= 2).sum()} in last 504")

# ---------- 4. VIX term structure ----------
ts = px["^VIX3M"] / px["^VIX"]
print(f"\n=== VIX3M/VIX = {ts.iloc[-1]:.4f} | 252d pctile {(ts.tail(252)<=ts.iloc[-1]).mean()*100:.1f} "
      f"| full pctile {(ts<=ts.iloc[-1]).mean()*100:.1f}")

# ---------- 5. credit leading: HYG at 52w high while SPY is not ----------
def dist_high(s, w=252):
    return s / s.rolling(w).max() - 1.0
dh = {t: dist_high(px[t]) for t in ["HYG", "SPY", "LQD", "QQQ", "IWM"]}
print("\n=== distance from 252d high ===")
for t, s in dh.items():
    print(f"  {t:<5} {s.iloc[-1]*100:+6.2f}%")
state = (dh["HYG"] >= -0.0005) & (dh["SPY"] <= -0.01)
print(f"  HYG at high & SPY <=-1% off: {int(state.sum())} days, "
      f"{len(declusters(px.index[state.fillna(False)], 10, px.index))} declustered episodes")

# ---------- 6. MOVE vs VIX ----------
mv = px["^MOVE"] / px["^VIX"]
print(f"\n=== MOVE/VIX = {mv.iloc[-1]:.3f} | 252d pctile {(mv.tail(252)<=mv.iloc[-1]).mean()*100:.1f} "
      f"| full pctile {(mv<=mv.iloc[-1]).mean()*100:.1f}")

# ---------- 7. NVDA print anchor availability ----------
ec = pd.read_parquet(ROOT / "data/earnings_calendar.parquet")
print("\n=== earnings calendar cols ===", ec.columns.tolist(), ec.shape)
nv = ec[ec.iloc[:, 0].astype(str).str.upper() == "NVDA"] if ec.shape[1] else ec
print(nv.tail(6))
