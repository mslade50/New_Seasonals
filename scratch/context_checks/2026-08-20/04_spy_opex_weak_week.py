"""The opex session itself, conditioned on the week that walked into it.

Bare cell E:opex|SPY|k1 is nothing: n=319, -0.071%, 162-156, t=-1.24. SPY enters
tomorrow's opex with a 5d return of -1.96%, the 9.5th percentile of its own year.
Does a down week into opex change the opex bar, and does the following week
recover it? Controls: the instrument's own all-day drift and the +/-126td local
neighbourhood of the same anchors.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import close_panel, cluster_note, load_events, local_control, sign_test, summarize  # noqa


def report(label, v):
    v = np.asarray(v)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        print(f"  {label:<46} n=0")
        return
    st = summarize(v, label)
    up = int((v > 0).sum())
    print(
        f"  {label:<46} n={st['n']:<5} mean={st['mean_pct']:+.3f}%  med={st['median_pct']:+.3f}%"
        f"  {up}-{st['n'] - up}  hit={st['hit']:.1f}%  t={st['t']:+.2f}  sign_p={sign_test(up, st['n']):.4f}"
    )


for tkr in ("SPY", "QQQ"):
    px = close_panel([tkr])[tkr].dropna()
    idx = px.index
    opex = set(pd.to_datetime(load_events(["opex"])["date"]).dt.normalize())
    nxt = list(idx[1:]) + [pd.NaT]
    next_is_opex = np.array([pd.notna(d) and d.normalize() in opex for d in nxt])

    fwd1 = px.pct_change().shift(-1).values          # the opex bar
    fwd6 = (px.shift(-6) / px - 1.0).values          # opex bar + the week after it
    ret5 = px.pct_change(5).values                   # the week into the anchor
    # trailing-252 percentile of the 5d return, matching the engine's rank_5d
    rank5 = px.pct_change(5).rolling(252).apply(lambda w: (w[-1] > w[:-1]).mean() * 100, raw=True).values

    ok = next_is_opex & ~np.isnan(fwd1) & ~np.isnan(rank5)
    live_rank = rank5[-1]
    print(f"\n=== {tkr}  {idx[0].date()} to {idx[-1].date()} ===")
    print(f"  live: 5d {ret5[-1] * 100:+.2f}%, 5d rank {live_rank:.1f} of its own year")

    report("all opex sessions", fwd1[ok])
    report("opex after a down week", fwd1[ok & (ret5 < 0)])
    report("opex after 5d rank < 20", fwd1[ok & (rank5 < 20)])
    report("opex after 5d rank < 10", fwd1[ok & (rank5 < 10)])
    report("opex after an up week", fwd1[ok & (ret5 >= 0)])
    report("opex after 5d rank > 80", fwd1[ok & (rank5 > 80)])
    print("  controls:")
    report("  all sessions (own drift)", px.pct_change().shift(-1).values[~np.isnan(fwd1)])
    lc = local_control(idx, idx[ok & (rank5 < 20)], win=126)
    report("  local +/-126td around those anchors", px.pct_change().shift(-1).reindex(lc).values)

    print("  follow-on, opex bar plus the week after (h6):")
    report("  opex after 5d rank < 20, h6", fwd6[ok & (rank5 < 20)])
    report("  all opex, h6", fwd6[ok])

    cell = ok & (rank5 < 20)
    print(f"  concentration:", cluster_note(idx[cell], fwd1[cell]))
    if tkr == "SPY":
        print("  most recent 10 of that cell:")
        for dt, v in list(zip(idx[cell], fwd1[cell] * 100))[-10:]:
            print(f"    {dt.date()}  rank5={rank5[list(idx).index(dt)]:.0f}  opex {v:+.2f}%")

# --- addendum: the h6 cell is the one worth publishing, so grade it properly ---
print("\n=== h6 cell diligence (opex bar + the week after, 5d rank < 20 into it) ===")
from pitch_lab import era_split  # noqa

for tkr in ("SPY", "QQQ"):
    px = close_panel([tkr])[tkr].dropna()
    idx = px.index
    opex = set(pd.to_datetime(load_events(["opex"])["date"]).dt.normalize())
    nxt = list(idx[1:]) + [pd.NaT]
    next_is_opex = np.array([pd.notna(d) and d.normalize() in opex for d in nxt])
    fwd6 = (px.shift(-6) / px - 1.0).values
    rank5 = px.pct_change(5).rolling(252).apply(lambda w: (w[-1] > w[:-1]).mean() * 100, raw=True).values
    cell = next_is_opex & ~np.isnan(fwd6) & ~np.isnan(rank5) & (rank5 < 20)
    v, d = fwd6[cell], idx[cell]
    print(f"\n  {tkr}: n={len(v)}")
    for e in era_split(d, v):
        print(f"    {e['label']:<9} n={e['n']:<3} mean={e['mean_pct']:+.3f}%  hit={e['hit']:.1f}%  t={e['t']:+.2f}")
    print("    concentration:", cluster_note(d, v))
    # drop the two biggest and re-summarize
    order = np.argsort(v)[::-1]
    keep = np.ones(len(v), dtype=bool)
    keep[order[:2]] = False
    st = summarize(v[keep], "drop-best-2")
    print(f"    drop best 2: n={st['n']} mean={st['mean_pct']:+.3f}% hit={st['hit']:.1f}% t={st['t']:+.2f}")
    print("    last 8 episodes:")
    for dt, x in list(zip(d, v * 100))[-8:]:
        print(f"      {dt.date()} -> {x:+.2f}%")
