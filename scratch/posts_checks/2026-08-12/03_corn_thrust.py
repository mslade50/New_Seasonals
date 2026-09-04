"""Corn (ZC=F) apparently printed a top-of-history up day at a 52-week
high today (context brief: +10.1%, top 0.05% of 6522 sessions). Verify
today's move and rank it, then check what big corn up-days AT a 52w high
did next (5 sessions forward, declustered), vs all days. Descriptive."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
from pitch_lab import close_panel, declusters, show, sign_test, summarize

px = close_panel(["ZC=F"])
s = px["ZC=F"].dropna()
r = s.pct_change()
print(f"last bar {s.index[-1].date()} close {s.iloc[-1]:.2f} ret {r.iloc[-1]*100:+.2f}%")
print(f"rank of today's ret in {len(r.dropna())} sessions: "
      f"{(r.dropna() < r.iloc[-1]).mean()*100:.2f} pctile")

high52 = s.rolling(252).max()
at_high = s >= high52 * 0.999
fwd5 = s.shift(-5) / s - 1.0

for thr in (0.03, 0.05):
    mask = (r >= thr) & at_high
    trig = declusters(s.index[mask.fillna(False)], 5, s.index)
    vals = fwd5.loc[trig].dropna()
    v = vals.to_numpy()
    if not len(v):
        print(f"thr {thr}: no triggers")
        continue
    wins = int((v > 0).sum())
    row = summarize(v, f"corn +{thr:.0%} day at 52w high, fwd 5d")
    row["sign_p"] = round(sign_test(wins, len(v)), 4)
    row["record"] = f"{wins}/{len(v)}"
    show([row, summarize(fwd5.dropna().to_numpy(), "corn any day fwd 5d")],
         f"threshold {thr:.0%}")
    print("dates:", [str(d.date()) for d in trig[-12:]])

# Idea-viability leg: an evening post can only enter tomorrow. Re-run the
# +3% cell with entry at T+1 OPEN and T+1 CLOSE, exit 5 sessions after the
# trigger close (same calendar as the cc cell), so the tradeable slice is
# measured honestly rather than assumed.
from pitch_lab import load_prices
ohlc = load_prices(["ZC=F"])["ZC=F"]
opn = ohlc["Open"].reindex(s.index)
mask3 = (r >= 0.03) & at_high
trig3 = declusters(s.index[mask3.fillna(False)], 5, s.index)
for entry_lbl, entry_px_series, entry_off in (("T+1 open", opn, 1), ("T+1 close", s, 1)):
    vals = []
    for d in trig3:
        pos = s.index.searchsorted(d)
        if pos + 5 >= len(s.index) or pos + entry_off >= len(s.index):
            continue
        e = entry_px_series.iloc[pos + entry_off]
        x = s.iloc[pos + 5]
        if np.isnan(e) or np.isnan(x):
            continue
        vals.append(x / e - 1.0)
    v = np.array(vals)
    if not len(v):
        continue
    wins = int((v > 0).sum())
    row = summarize(v, f"entry {entry_lbl} -> exit trigger+5d close")
    row["sign_p"] = round(sign_test(wins, len(v)), 4)
    row["record"] = f"{wins}/{len(v)}"
    show([row], f"tradeable variant: {entry_lbl}")

# VERDICT: the +10.13% print is roughly half ROLL ARTIFACT, not price.
# Cross-check run 2026-08-12 evening: wheat ZW=F +3.61%, soybeans ZS=F
# +3.12%, DBA (holds corn) +0.91%. Corn's bar gapped +5.4% AT THE OPEN
# (436.75 -> 460.25) on 2.3x volume with the intraday rally only
# 460.25 -> 481.00 (+4.5%), the classic unadjusted Sep->Dec front-month
# roll shape in mid-August. The historical cell above is built on the
# same continuous series and inherits the same embedded roll returns, so
# BOTH the stat and the idea are dead. Salvage: the catch itself is the
# post (process/discipline), with the hedged wording that this is the
# boring roll explanation, not a proven roll date.
