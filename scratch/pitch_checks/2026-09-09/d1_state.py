"""C13 step 1: build the MOVE-vs-equity-vol divergence as a CONTINUOUS spread.
Report ^MOVE's true first usable bar, true N, the live spread value and its own
point-in-time percentile BEFORE anything else."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import _valid_pct_change, rolling_on_valid
import numpy as np, pandas as pd

TK = ["^MOVE", "^VIX", "^VIX3M", "SPY", "QQQ", "IWM", "TLT", "IEF", "SVXY"]
px = close_panel(TK)
print("panel span", px.index[0].date(), px.index[-1].date(), "rows", len(px))

mv = px["^MOVE"].dropna()
print("\n^MOVE first bar", mv.index[0].date(), " last", mv.index[-1].date(),
      " valid bars", len(mv))
# holes: how many SPY sessions inside MOVE's span lack a MOVE print?
spy_span = px["SPY"].dropna().loc[mv.index[0]:mv.index[-1]]
print("SPY sessions in MOVE span:", len(spy_span),
      " MOVE missing on", len(spy_span.index.difference(mv.index)), "of them")
# staleness: repeated identical closes
rep = (mv.diff() == 0).sum()
print("MOVE unchanged-close sessions:", int(rep), f"({100*rep/len(mv):.1f}%)")
print("first 5 MOVE bars:\n", mv.head(5))

# --- state components -------------------------------------------------------
# MOVE trailing-252 LEVEL percentile
move_pct = rolling_on_valid(mv, lambda x: x.rolling(252).rank(pct=True) * 100.0)
# SPY 21d realized vol, annualised, then its trailing-252 LEVEL percentile
spy = px["SPY"].dropna()
r = spy.pct_change()
rv21 = r.rolling(21).std() * np.sqrt(252) * 100.0
rv_pct = rolling_on_valid(rv21, lambda x: x.rolling(252).rank(pct=True) * 100.0)
# VIX level percentile
vix = px["^VIX"].dropna()
vix_pct = rolling_on_valid(vix, lambda x: x.rolling(252).rank(pct=True) * 100.0)

move_pct = move_pct.reindex(px.index)
rv_pct = rv_pct.reindex(px.index)
vix_pct = vix_pct.reindex(px.index)

# PRE-SPECIFIED: spread A = MOVE pctile - SPY 21d realized-vol pctile.
# (the candidate text names realized vol first and the live premise is stated
#  in realized-vol terms; VIX form carried as spread B for sensitivity only)
sprA = (move_pct - rv_pct).rename("sprA")
sprB = (move_pct - vix_pct).rename("sprB")

for nm, s in [("A MOVE-RVOL", sprA), ("B MOVE-VIX", sprB)]:
    v = s.dropna()
    live = v.iloc[-1]
    # point-in-time percentile of the spread within its own trailing 252
    pit = rolling_on_valid(v, lambda x: x.rolling(252).rank(pct=True) * 100.0)
    allt = 100.0 * (v <= live).mean()
    print(f"\n=== spread {nm} ===")
    print(f"  N valid days {len(v)}  first {v.index[0].date()}")
    print(f"  LIVE (2026-09-08) = {live:+.1f}")
    print(f"  its trailing-252 PIT percentile = {pit.iloc[-1]:.1f}")
    print(f"  its ALL-HISTORY percentile      = {allt:.1f}")
    print("  deciles:", np.round(np.percentile(v, [10,25,50,75,90,95,99]), 1))
    print(f"  days at/above live: {(v >= live).sum()} of {len(v)} "
          f"({100*(v>=live).mean():.1f}%)")

# raw component values today
print("\ncomponents today: MOVE lvl pct %.1f | SPY rv21 %.2f%% ann lvl pct %.1f "
      "| VIX lvl pct %.1f" % (move_pct.dropna().iloc[-1], rv21.dropna().iloc[-1],
                              rv_pct.dropna().iloc[-1], vix_pct.dropna().iloc[-1]))

out = pd.DataFrame({"move_pct": move_pct, "rv_pct": rv_pct, "vix_pct": vix_pct,
                    "sprA": sprA, "sprB": sprB})
out.to_parquet(Path(__file__).parent / "d1_state.parquet")
print("\nwrote d1_state.parquet")
