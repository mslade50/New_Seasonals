"""Drill 05 — yen up hard WHILE US yields sit at 52-week highs. Not in the sweep.

Friday: JPY=X -2.05% (yen strongest move on the tape) and yet ^TNX closed 0.25%
off its own 52-week HIGH, with TLT/IEF/LQD sitting 1.44/0.44/0.25% off 52-week
LOWS. The engine has no trigger for this: P9 covers stocks+bonds and dollar+gold,
not rates+yen. Flagged in the cell map as an inventory gap.

Why it is odd: USDJPY is the most rate-differential-driven major there is. Yen
strength normally arrives when US yields FALL. Yen strength into US yields at
the highs means the move is not coming from the US leg, which historically
points at a domestic Japanese repricing or a carry unwind.

Cell: session where USDJPY fell >= 1% AND the US 10y yield ROSE, with ^TNX
already in the top decile of its trailing year. Then loosen if that is too thin.
Anchor is the printing session, h1 lag=0.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    close_panel, fwd_ret, pct_rank, summarize, era_split, cluster_note,
    sign_test, declusters, local_control, show,
)

TK = ["JPY=X", "^TNX", "^GSPC", "SPY", "TLT", "EWJ", "^N225"]
raw = close_panel(TK)
nyse = raw["^GSPC"].dropna().index
px = raw.reindex(nyse).dropna(subset=["JPY=X", "^TNX"], how="any")
print(f"panel {px.index[0].date()} .. {px.index[-1].date()}  n={len(px)}")

jpy_ret = px["JPY=X"].pct_change()
tnx_ret = px["^TNX"].pct_change()
# the LEVEL percentile (pct_rank would give the 1d-RETURN percentile)
tnx_rank = px["^TNX"].rolling(252).rank(pct=True) * 100.0

last = px.index[-1]
print(f"\nlatest {last.date()}: USDJPY {100 * jpy_ret.loc[last]:+.2f}%, "
      f"^TNX {100 * tnx_ret.loc[last]:+.2f}%, ^TNX 252d pctile {tnx_rank.loc[last]:.1f}")
print(f"  ^TNX close {px['^TNX'].loc[last]:.3f}, "
      f"52w high {px['^TNX'].iloc[-252:].max():.3f}")

# tiers, loosest last
CELLS = {
    "USDJPY <= -1% AND 10y yield UP AND yield in top decile":
        (jpy_ret <= -0.01) & (tnx_ret > 0) & (tnx_rank >= 90),
    "USDJPY <= -1% AND 10y yield UP":
        (jpy_ret <= -0.01) & (tnx_ret > 0),
    "USDJPY <= -1% (any yield direction)":
        (jpy_ret <= -0.01),
    "USDJPY <= -1% AND 10y yield DOWN (the normal case)":
        (jpy_ret <= -0.01) & (tnx_ret < 0),
}

print("\n=== how rare is the divergence ===")
for name, m in CELLS.items():
    print(f"  {m.sum():5d} sessions  {name}")

f1j, f5j = fwd_ret(px["JPY=X"], 1), fwd_ret(px["JPY=X"], 5)
f1s, f5s = fwd_ret(px["^GSPC"], 1), fwd_ret(px["^GSPC"], 5)

for name, m in CELLS.items():
    dates_all = px.index[m.fillna(False)]
    dates = declusters(dates_all, 5, px.index)
    if len(dates) < 5:
        print(f"\n--- {name}: only {len(dates)} episodes, too thin ---")
        continue
    print(f"\n--- {name} ---")
    print(f"    {len(dates_all)} raw -> {len(dates)} episodes")
    ctrl = local_control(px.index, dates, 126)
    show([summarize(f1j.reindex(dates).dropna().values, "USDJPY h1"),
          summarize(f5j.reindex(dates).dropna().values, "USDJPY h5"),
          summarize(f1j.reindex(ctrl).dropna().values, "USDJPY h1 local ctrl"),
          summarize(f1s.reindex(dates).dropna().values, "^GSPC h1"),
          summarize(f5s.reindex(dates).dropna().values, "^GSPC h5"),
          summarize(f1s.reindex(ctrl).dropna().values, "^GSPC h1 local ctrl")],
         name)
    v = f1j.reindex(dates).dropna()
    w, n = int((v.values > 0).sum()), len(v)
    print(f"    USDJPY h1 record {w}-{n - w} up, sign p(up) {sign_test(w, n):.4f}")
    for e in era_split(v.index, v.values):
        print(f"      era {e['label']}: n={e['n']} mean {e['mean_pct']:+.3f}% "
              f"hit {e['hit']:.1f}")
    print(f"      conc: {cluster_note(v.index, v.values)}")
    if len(dates) <= 20:
        print(f"      episodes: {[str(d.date()) for d in dates]}")

# Japanese equities are the transmission channel worth naming
print("\n=== ^N225 / EWJ after the tight divergence cell ===")
tight = px.index[CELLS["USDJPY <= -1% AND 10y yield UP AND yield in top decile"]
                 .fillna(False)]
tight_ep = declusters(tight, 5, px.index)
for t in ("^N225", "EWJ"):
    s = px[t].dropna()
    f1 = fwd_ret(s, 1)
    show([summarize(f1.reindex(tight_ep).dropna().values, f"{t} h1 after divergence"),
          summarize(f1.dropna().values, f"{t} all days")], t)
