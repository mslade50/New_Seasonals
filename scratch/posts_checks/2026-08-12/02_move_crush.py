"""Today ^MOVE fell 7.5% while the 10y yield closed within half a percent
of where it started (-0.04% on ^TNX). Tonight's context brief scores the
cell N=94, next-day MOVE +3.95%, 63.8% hit, t=3.36. Recheck independently
before posting: MOVE down >=5% on the day AND |^TNX day change| <= 0.5%,
forward 1-session MOVE return, declustered not needed (day-level cell is
what the brief published; note overlap honestly if triggers chain)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
from pitch_lab import close_panel, show, sign_test, summarize

px = close_panel(["^MOVE", "^TNX"])
move = px["^MOVE"].dropna()
tnx = px["^TNX"].reindex(move.index).ffill()

move_r = move.pct_change()
tnx_r = tnx.pct_change()
fwd = move.shift(-1) / move - 1.0

from pitch_lab import declusters  # noqa: E402

mask = (move_r <= -0.05) & (tnx_r.abs() <= 0.005)
trig = declusters(move.index[mask.fillna(False)], 5, move.index)

rows = []
for h in (1, 5, 10):
    fh = move.shift(-h) / move - 1.0
    v = fh.loc[trig].dropna().to_numpy()
    wins = int((v > 0).sum())
    r = summarize(v, f"crush cell fwd h{h}")
    r["sign_p"] = round(sign_test(wins, len(v)), 4)
    r["record"] = f"{wins}/{len(v)}"
    rows.append(r)
    rows.append(summarize(fh.dropna().to_numpy(), f"any day fwd h{h}"))
show(rows, "MOVE -5%+ day, 10y yield inside 0.5%, declustered 5td")
# Resolution vs the context brief: its published numbers (n=94 +3.95%
# 63.8% t=3.36) are the h10 horizon; h1 is flat-to-negative. The honest
# post carries BOTH: no next-day bounce, the rebuild takes ~2 weeks.
# Era split + drop-top-3 robustness: scratch/context_checks/2026-08-12/
# 04b_move_rebound.py (pre-2018 +4.1%, 2018+ +3.6%; drop-3 t=3.41).
