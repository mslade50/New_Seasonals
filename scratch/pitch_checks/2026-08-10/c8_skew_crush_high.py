"""C8 round 1 -- crushed ^SKEW at a 52w SPY high, controlled for equity P/C.

Mechanism as stated: a low SKEW LEVEL means no crash-hedging demand, so no
dealer put wall under the tape. That is a statement about the LEVEL, so the
pre-specified trigger is the SKEW level percentile, not its 21d return rank.
(rank21 is carried as a definition neighbour, not the primary.)

Pre-specified: mask = SKEW level in the bottom decile of its trailing 252d
AND SPY within 0.5% of its 252d high. Vehicle SPY, entry lag=1 MOC-tomorrow,
h=5. NO direction prior -- the long is printed and the sign is read off it.

CRITICAL CONTROL: equity P/C complacency is already live book machinery
(fragility signal + PC_FEAR_BANDS). If the SKEW cell is the P/C cell wearing a
hat, it is both a kill and a re-run of the book.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

import pc_fear  # noqa: E402

px = close_panel(["SPY", "^SKEW", "^VIX"]).dropna()
idx = px.index
spy, skew, vix = px["SPY"], px["^SKEW"], px["^VIX"]

# ---------------------------------------------------------------- state
skew_lvl = skew.rolling(252).rank(pct=True) * 100        # LEVEL percentile
skew_r21 = pct_rank(skew, 21)                            # 21d RETURN rank
spy_hi = spy.rolling(252).max()
spy_dist = spy / spy_hi - 1.0
vix_lvl = vix.rolling(252).rank(pct=True) * 100

# equity P/C percentile, lag-1 (pc_fear's own statistic)
pcs = pc_fear.pct_series()
pc = pcs.reindex(idx, method="ffill").shift(1)

print("today:", idx[-1].date(),
      f"SKEW={skew.iloc[-1]:.2f} lvlpct={skew_lvl.iloc[-1]:.1f} "
      f"r21={skew_r21.iloc[-1]:.1f} | SPY dist52wh={100*spy_dist.iloc[-1]:+.3f}% "
      f"| VIX={vix.iloc[-1]:.2f} lvlpct={vix_lvl.iloc[-1]:.1f} "
      f"| P/C pct={pc.iloc[-1]:.1f}")

NEAR = -0.005            # within 0.5% of the 52w high
SKEW_P = 10.0            # bottom decile LEVEL

m_hi = spy_dist >= NEAR
m_sk = skew_lvl <= SKEW_P
m = (m_hi & m_sk).fillna(False)

print(f"\ntrigger days: SPY-near-high {int(m_hi.sum())}, "
      f"SKEW bottom-decile {int(m_sk.sum())}, BOTH {int(m.sum())}")

# ---------------------------------------------------------------- cluster depth today
run = 0
for v in m.values[::-1]:
    if v:
        run += 1
    else:
        break
runhi = 0
for v in m_hi.values[::-1]:
    if v:
        runhi += 1
    else:
        break
print(f"CLUSTER DEPTH TODAY: consecutive BOTH-sessions = {run}; "
      f"consecutive SPY-near-high sessions = {runhi}  "
      f"(mid-cluster entry is not a fresh trigger)")

# ---------------------------------------------------------------- round 1
variants = {
    "SKEW lvl <=5": ((skew_lvl <= 5) & m_hi).fillna(False),
    "SKEW lvl <=20": ((skew_lvl <= 20) & m_hi).fillna(False),
    "SKEW lvl <=30": ((skew_lvl <= 30) & m_hi).fillna(False),
    "near-high 0.1%": (m_sk & (spy_dist >= -0.001)).fillna(False),
    "near-high 2.0%": (m_sk & (spy_dist >= -0.02)).fillna(False),
    "SKEW r21<=10 (nbr defn)": ((skew_r21 <= 10) & m_hi).fillna(False),
    "NO skew gate (SPY high alone)": m_hi.fillna(False),
    "NO high gate (SKEW alone)": m_sk.fillna(False),
}

for h in (5, 10):
    battery(px, m, [("SPY", 1.0)], h=h,
            title=f"LONG SPY | SKEW bottom-decile LEVEL + SPY within 0.5% of 52wh, h={h}",
            cost_bps=2.0, lag=1, min_gap=h,
            event_kinds=("cpi",), variants=variants if h == 5 else None)

print("\n" + "=" * 78)
print("SCAN (multiplicity applies): horizon 1..21")
print("=" * 78)
trig = idx[m.values]
show(horizon_scan(px, trig, [("SPY", 1.0)], hs=(1, 2, 3, 5, 7, 10, 15, 21)),
     "SKEW-crush x SPY-high, long SPY")

# ---------------------------------------------------------------- P/C control
print("\n" + "=" * 78)
print("THE DECISIVE CONTROL: is this the equity P/C complacency cell?")
print("=" * 78)
ok = pc.notna()
print(f"P/C coverage: {int(ok.sum())} of {len(idx)} sessions "
      f"({idx[ok][0].date()} ..)")
pc_lo = (pc <= 10).fillna(False)          # complacency by the book's statistic
print(f"corr(SKEW lvl pctile, P/C pctile) over common days = "
      f"{skew_lvl[ok].corr(pc[ok]):.3f}")
both = (m & pc_lo).fillna(False)
sk_only = (m & ~pc_lo & ok).fillna(False)
pc_only = (pc_lo & m_hi & ~m_sk).fillna(False)
print(f"\noverlap on SPY-near-high days (P/C era only):")
print(f"  SKEW-low & P/C-low : {int(both.sum())}")
print(f"  SKEW-low, P/C not  : {int(sk_only.sum())}")
print(f"  P/C-low, SKEW not  : {int(pc_only.sum())}")
print(f"  TODAY: SKEW lvlpct {skew_lvl.iloc[-1]:.1f}, P/C pct {pc.iloc[-1]:.1f}"
      f"  -> today's cell is "
      f"{'BOTH' if both.iloc[-1] else ('SKEW-only' if sk_only.iloc[-1] else 'other')}")

for h in (5, 10):
    r = vehicle_ret(px, [("SPY", 1.0)], h, 1)
    val = r.notna()
    rows = []
    for lbl, mm in [("SKEW-low & P/C-low", both), ("SKEW-low, P/C NOT low", sk_only),
                    ("P/C-low, SKEW NOT low", pc_only),
                    ("SPY-high, P/C era, neither", (m_hi & ~m_sk & ~pc_lo & ok).fillna(False))]:
        d = idx[mm.values & val.values]
        e = declusters(d, h, idx)
        rows.append(summarize(r.loc[e].values, f"{lbl} (epi N={len(e)}, days {len(d)})"))
    base = r[val & ok]
    rows.append(summarize(base.values, "CTRL all days, P/C era"))
    show(rows, f"P/C decomposition, h={h}, episodes")

# ---------------------------------------------------------------- midterm
print("\n" + "=" * 78)
print("MIDTERM SPLIT (2026 is midterm)")
print("=" * 78)
for h in (5, 10):
    r = vehicle_ret(px, [("SPY", 1.0)], h, 1)
    val = r.notna()
    d = idx[m.values & val.values]
    e = declusters(d, h, idx)
    v = r.loc[e].values
    yr = pd.DatetimeIndex(e).year
    mid = (yr % 4 == 2)
    base = r[val]
    bmid = base.index.year % 4 == 2
    show([summarize(v[mid], f"midterm episodes (N={int(mid.sum())})"),
          summarize(v[~mid], f"non-midterm episodes (N={int((~mid).sum())})"),
          summarize(base[bmid].values, "CTRL all days midterm"),
          summarize(base[~bmid].values, "CTRL all days non-midterm")],
         f"h={h}")
    if mid.sum():
        print("  midterm episode dates:",
              ", ".join(str(x.date()) for x in pd.DatetimeIndex(e)[mid]))

# ---------------------------------------------------------------- weekday placebo
print("\n" + "=" * 78)
print("WEEKDAY PLACEBO (registry: the weekend-risk-at-a-high cell is a fossil)")
print("=" * 78)
h = 5
r = vehicle_ret(px, [("SPY", 1.0)], h, 1)
val = r.notna()
d = idx[m.values & val.values]
rows = []
for wd, nm in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri"]):
    sel = d[d.weekday == wd]
    if len(sel):
        rows.append(summarize(r.loc[sel].values, f"anchor {nm} (days N={len(sel)})"))
show(rows, "day-level by anchor weekday, h=5")

# ---------------------------------------------------------------- year histogram
print("\nEPISODE YEAR HISTOGRAM (h=5):")
e = declusters(idx[m.values & val.values], 5, idx)
v = r.loc[e].values
hist = pd.Series(100 * v, index=pd.DatetimeIndex(e)).groupby(
    pd.DatetimeIndex(e).year).agg(["count", "sum", "mean"])
print(hist.round(2).to_string())
