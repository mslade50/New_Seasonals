"""C6 -- GDX after a 99th-pctile 21d thrust AND a >=2% down day. Both directions.

Registry constraints checked here:
  - a rank is not a magnitude (IHI denominator roll-off). Magnitude quoted.
  - the closed GDX thrust cell: "GDX one-week moves above +10% pay +0.483% at a
    51.6% hit against +1.113% at 72.7% below that" -- the violent half is wrong.
  - GLD fwd correlates +0.766 with GDX over 5 days, so GLD is not independent.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import numpy as np, pandas as pd

TK = ["GDX", "GLD", "NEM", "SPY"]
px = close_panel(TK)
px = px[px.index >= "2006-05-22"]

g = px["GDX"]
r21 = g / g.shift(21) - 1.0
rk21 = rolling_on_valid(r21, lambda x: x.rolling(252).rank(pct=True) * 100.0)
r1 = g.pct_change(fill_method=None)

# ---- (a) is today's extreme a genuine move or a denominator roll-off?
print("=== denominator-roll check (last 6 sessions) ===")
d = pd.DataFrame({"close": g, "r1_pct": 100 * r1, "r21_pct": 100 * r21,
                  "rk21": rk21, "d_r21_pp": 100 * r21.diff(),
                  "rolled_off_bar_pct": 100 * (g.shift(21) / g.shift(22) - 1.0)})
print(d.tail(6).round(2).to_string())

print("\ntoday: r21 = %.2f%%  rank %.1f  |  1d %.2f%%" %
      (100 * r21.iloc[-1], rk21.iloc[-1], 100 * r1.iloc[-1]))
print("MAGNITUDE gate check: how many days ever had r21 >= +38%?",
      int((r21 >= 0.38).sum()))

# ---- (b) the cell
m_rank = (rk21 >= 99.0)
m_down = (r1 <= -0.02)
mask = (m_rank & m_down).fillna(False)
print("\n=== occurrence counts ===")
print("rank>=99 days:", int(m_rank.sum()),
      "| AND 1d<=-2%:", int(mask.sum()))
print("does the down-day gate filter? %.1f%% of rank days survive"
      % (100 * mask.sum() / max(1, m_rank.sum())))
print("by year:", mask.groupby(px.index.year).sum().pipe(
    lambda s: {k: int(v) for k, v in s.items() if v}))
print("r21 on cell days: min %.1f%% median %.1f%% max %.1f%%"
      % (100 * r21[mask].min(), 100 * r21[mask].median(), 100 * r21[mask].max()))

variants = {
    "rank>=99 only (NO gate)": m_rank.fillna(False),
    "rank>=99 & 1d<=-2 (base)": mask,
    "rank>=95 & 1d<=-2": ((rk21 >= 95) & m_down).fillna(False),
    "rank>=99 & 1d<=-1": ((rk21 >= 99) & (r1 <= -0.01)).fillna(False),
    "MAG r21>=25% & 1d<=-2": ((r21 >= 0.25) & m_down).fillna(False),
    "MAG r21>=35% & 1d<=-2": ((r21 >= 0.35) & m_down).fillna(False),
    "all days": pd.Series(True, index=px.index),
}

for tkr, cost in (("GDX", 5.0), ("GLD", 3.0), ("NEM", 5.0)):
    for h in (3, 5, 10):
        battery(px, mask, [(tkr, 1.0)], h,
                f"C6 LONG {tkr} | GDX r21 rank>=99 & 1d<=-2%", cost,
                variants=variants if tkr == "GDX" else None,
                min_gap=10, event_kinds=("jackson_hole",))

# ---- midterm split (mandatory)
print("\n\n### midterm split, GDX h=5 and h=10 ###")
for h in (5, 10):
    ret = vehicle_ret(px, [("GDX", 1.0)], h, 1)
    sig = px.index[mask.values & ret.notna().values]
    epi = declusters(sig, 10, px.index)
    mid = np.array([d.year % 4 == 2 for d in epi])
    show([summarize(ret.loc[epi[mid]].values, f"h={h} MIDTERM (N={int(mid.sum())})"),
          summarize(ret.loc[epi[~mid]].values, f"h={h} non-midterm (N={int((~mid).sum())})")],
         f"midterm split h={h}")

# ---- book overlap
led = pd.read_parquet(ROOT / "data" / "backtest_trades_full.parquet")
led["Signal Date"] = pd.to_datetime(led["Signal Date"])
ov = led[led["Signal Date"].isin(set(px.index[mask]))]
print("\n### book overlap ###")
print("ledger signals in-state:", len(ov), "avgR:",
      round(ov["R_Multiple"].mean(), 3) if len(ov) else "n/a",
      "| book-wide", round(led["R_Multiple"].mean(), 3))
if len(ov):
    print(ov.groupby("Strategy")["R_Multiple"].agg(["count", "mean"]).round(2).to_string())
