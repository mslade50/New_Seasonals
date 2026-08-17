"""C10 round 1 - GDX (and NEM) at a 21d rank of 100 inside a 22% drawdown.

Registry collisions carried in by construction:
  - "miner-over-metal is beta" (2026-08-10): NO relative GDX/GLD form is
    checked as a candidate; the regression beta is reported for the record.
  - "silver thrust from deep inside a drawdown - the filter does not filter"
    (2026-08-10): the drawdown gate is run as an ATTRIBUTION, not as a premise,
    and the same 8->10% threshold nudge that killed the silver version is run.
  - "cluster depth is not a one-way objection" (2026-08-10): depth is measured
    and TODAY's depth is quoted.
  - rank triggers can jump on a denominator roll (2026-08-13): a MAGNITUDE gate
    is quoted beside the rank everywhere.

Convention lag=1 (signal close 2026-08-14, entry MOC 2026-08-17).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

TICKERS = ["GDX", "GLD", "SLV", "NEM", "SPY", "GDXJ"]
px = close_panel(TICKERS).dropna(subset=["GDX"])
idx = px.index
print(f"panel {idx[0].date()} .. {idx[-1].date()}  N={len(idx)}")

# ---------------------------------------------------------------------------
# live state
# ---------------------------------------------------------------------------
state = {}
for t in ["GDX", "NEM", "GLD", "SLV"]:
    s = px[t].dropna()
    r21 = s.pct_change(21)
    rk21 = pct_rank(s, 21)
    hi52 = s.rolling(252).max()
    dd = s / hi52 - 1.0
    state[t] = dict(rk21=rk21.iloc[-1], r21=100 * r21.iloc[-1],
                    dd=100 * dd.iloc[-1], rk63=pct_rank(s, 63).iloc[-1])
    print(f"  {t}: 21d rank {state[t]['rk21']:.1f}  21d ret {state[t]['r21']:+.2f}%  "
          f"off 52wh {state[t]['dd']:+.2f}%  63d rank {state[t]['rk63']:.1f}")

# ---------------------------------------------------------------------------
# triggers
# ---------------------------------------------------------------------------
def build(t: str):
    s = px[t]
    rk21 = pct_rank(s, 21)
    r21 = s.pct_change(21)
    dd = s / s.rolling(252).max() - 1.0
    return rk21, r21, dd


rk21_g, r21_g, dd_g = build("GDX")
rk21_n, r21_n, dd_n = build("NEM")

# depth: trigger days inside the trailing 10 sessions, inclusive
def depth_of(mask: pd.Series, win: int = 10) -> pd.Series:
    return mask.astype(int).rolling(win, min_periods=1).sum()


base_g = rk21_g >= 100.0          # maximal 21d thrust, no other condition
dep_g = depth_of(base_g)
print(f"\nGDX rank-100 trigger days total: {int(base_g.sum())}")
print(f"TODAY's depth (rank-100 days in the trailing 10 sessions incl. 08-14): "
      f"{int(dep_g.iloc[-1])}")
recent = idx[base_g.values][-6:]
print("  most recent rank-100 days:", ", ".join(str(d.date()) for d in recent))

# ---------------------------------------------------------------------------
# 1. UNCONDITIONAL maximal thrust, both horizons, no drawdown gate at all
# ---------------------------------------------------------------------------
for h in (5, 10):
    battery(px, base_g, [("GDX", 1.0)], h,
            f"C10 GDX 21d rank==100 (NO drawdown gate)", cost_bps=5.0,
            variants={
                # --- drawdown gate ATTRIBUTION (the already-dead silver form)
                "+ dd <= -20% (today -22.3%)": base_g & (dd_g <= -0.20),
                "+ dd <= -15%": base_g & (dd_g <= -0.15),
                "+ dd <= -10%": base_g & (dd_g <= -0.10),
                "+ NEAR high, dd > -10%": base_g & (dd_g > -0.10),
                # --- magnitude gate beside the rank (denominator-roll guard)
                "+ 21d ret >= 20%": base_g & (r21_g >= 0.20),
                "+ 21d ret >= 26% (today)": base_g & (r21_g >= 0.26),
                "+ 21d ret >= 30%": base_g & (r21_g >= 0.30),
                # --- rank nudge
                "rank >= 99 instead of 100": rk21_g >= 99.0,
                "rank >= 97": rk21_g >= 97.0,
                # --- today's joint state
                "rank100 + dd<=-20 + ret>=26 (LIVE)":
                    base_g & (dd_g <= -0.20) & (r21_g >= 0.26),
                # --- depth
                "depth == 1 (episode FIRST)": base_g & (dep_g <= 1),
                "depth >= 3 (mid-cluster, today)": base_g & (dep_g >= 3),
            },
            min_gap=21, event_kinds=("jackson_hole", "fomc_decision"))

# ---------------------------------------------------------------------------
# 2. same rule on NEM (the more extreme name) - reference class of one peer
# ---------------------------------------------------------------------------
battery(px, rk21_n >= 100.0, [("NEM", 1.0)], 10,
        "C10b NEM 21d rank==100 (NO drawdown gate)", cost_bps=5.0,
        variants={"+ dd <= -20%": (rk21_n >= 100.0) & (dd_n <= -0.20),
                  "+ 21d ret >= 29% (today)": (rk21_n >= 100.0) & (r21_n >= 0.29),
                  "depth == 1": (rk21_n >= 100.0) & (depth_of(rk21_n >= 100.0) <= 1)},
        min_gap=21, event_kinds=("jackson_hole",))

# ---------------------------------------------------------------------------
# 3. SHORT side / exhaustion: is the answer symmetric?
# ---------------------------------------------------------------------------
print("\n\n===== direction check: exhaustion vs continuation, GDX rank100 =====")
rows = []
for h in (1, 2, 3, 5, 10):
    ret = fwd_lag(px["GDX"], h, 1)
    d = idx[base_g.values & ret.notna().values]
    e = declusters(d, 21, idx)
    r = summarize(ret.loc[e].values, f"h={h} rank100 episodes")
    r["ctl_all_pct"] = round(100 * ret.dropna().mean(), 3)
    r["edge_pp"] = round(r["mean_pct"] - 100 * ret.dropna().mean(), 3)
    rows.append(r)
show(rows, "GDX horizon scan, episodes")

# ---------------------------------------------------------------------------
# 4. for the record: the beta-neutral GDX/GLD residual on THIS trigger
# ---------------------------------------------------------------------------
print("\n\n===== registry duty: beta-neutral GDX-vs-GLD residual on this trigger =====")
d1g = px["GDX"].pct_change()
d1l = px["GLD"].pct_change()
ok = d1g.notna() & d1l.notna()
beta = np.polyfit(d1l[ok].values, d1g[ok].values, 1)[0]
print(f"  full-sample daily regression beta GDX~GLD = {beta:.3f}  "
      f"corr = {d1g[ok].corr(d1l[ok]):.3f}")
for h in (5, 10):
    rg = fwd_lag(px["GDX"], h, 1)
    rl = fwd_lag(px["GLD"], h, 1)
    d = idx[base_g.values & rg.notna().values & rl.notna().values]
    e = declusters(d, 21, idx)
    show([summarize(rg.loc[e].values, f"h={h} GDX leg"),
          summarize(rl.loc[e].values, f"h={h} GLD leg"),
          summarize((rg - rl).loc[e].values, f"h={h} equal-dollar GDX-GLD"),
          summarize((rg - beta * rl).loc[e].values,
                    f"h={h} BETA-NEUTRAL residual (b={beta:.2f})")],
         f"h={h} spread decomposition")

# ---------------------------------------------------------------------------
# 5. Jackson Hole sits at +9 td: does a rank-100 thrust into a JH window differ?
# ---------------------------------------------------------------------------
print("\n\n===== Jackson Hole inside the hold =====")
for h in (5, 10):
    ret = fwd_lag(px["GDX"], h, 1)
    d = idx[base_g.values & ret.notna().values]
    e = declusters(d, 21, idx)
    fl = event_in_window(e, idx, h, 1, ("jackson_hole",))
    show([summarize(ret.loc[e].values[fl], f"h={h} JH in hold (N={int(fl.sum())})"),
          summarize(ret.loc[e].values[~fl], f"h={h} JH out (N={int((~fl).sum())})")],
         f"h={h} JH split")
