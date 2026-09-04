"""Tonight's context brief scored the five sessions after August expiration
at QQQ +1.11% on 17-9 (n=26, t=2.34, first of twelve months) and SPY +0.60%
on 18-8, against September's 7-19. The brief's cell runs from the expiration
CLOSE. A post drafted Sunday can only enter MOO on the Monday, so the
tradeable slice is Monday open -> the fifth close (which this year is the
Jackson Hole session). If the weekend gap carries the move, the idea does not
ship and the stat runs alone.

Cells per ticker, anchored on each August opex close:
  - cc5: opex close -> +5 closes (recheck of the brief)
  - gap: opex close -> Monday open (what MOO forfeits)
  - moo: Monday open -> +5th close from the opex close (what MOO captures)
Controls: same legs after every NON-August opex, and every Monday open ->
+4 closes as the unconditional MOO drift. Era split at 2018, worst print,
and the 2-episode concentration share.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import pitch_lab as pl  # noqa: E402

ev = pl.load_events(["opex"])
opex = pd.DatetimeIndex(sorted(ev["date"].unique()))
px = pl.load_prices(["QQQ", "SPY"])


def legs(df, anchors, h=5):
    close = df["Close"].astype(float)
    opn = df["Open"].astype(float).reindex(close.index)
    idx = close.index
    rows = []
    for d in anchors:
        pos = idx.searchsorted(d)
        if pos >= len(idx) or idx[pos] != d or pos + h >= len(idx):
            continue
        c0, o1, ch = close.iloc[pos], opn.iloc[pos + 1], close.iloc[pos + h]
        rows.append((idx[pos], ch / c0 - 1, o1 / c0 - 1, ch / o1 - 1))
    return pd.DataFrame(rows, columns=["anchor", "cc5", "gap", "moo"]).set_index("anchor")


def line(v, label):
    v = v.dropna()
    up = int((v > 0).sum())
    s = pl.summarize(v.values, label)
    t = s.get("t")
    ts = f"{t:+.2f}" if t is not None and not np.isnan(t) else "  n/a"
    print(f"  {label:34s} n={s['n']:3d} mean={s['mean_pct']:+.2f}% med={s['median_pct']:+.2f}%"
          f" {up}-{s['n']-up} hit={s['hit']:.0f}% t={ts} p={pl.sign_test(up, s['n']):.4f}"
          f" worst={100*v.min():+.1f}%")


for tkr, df in px.items():
    idx = df.index
    hist = opex[(opex >= idx[0]) & (opex <= idx[-1])]
    aug = pd.DatetimeIndex([d for d in hist if d.month == 8])
    oth = pd.DatetimeIndex([d for d in hist if d.month != 8])
    A, O = legs(df, aug), legs(df, oth)
    print("=" * 78)
    print(f"{tkr}: August opex close -> +5 closes, n_anchor={len(A)}")
    for col in ("cc5", "gap", "moo"):
        line(A[col], f"AUG {col}")
        line(O[col], f"other months {col}")
    close = df["Close"].astype(float)
    opn = df["Open"].astype(float).reindex(close.index)
    mon = (close.index.weekday == 0)
    base = (close.shift(-4) / opn - 1)[mon]
    line(base, "every Monday open -> +4 closes")
    cut = pd.Timestamp("2018-01-01")
    line(A.loc[A.index < cut, "moo"], "AUG moo pre-2018")
    line(A.loc[A.index >= cut, "moo"], "AUG moo 2018+")
    mid = A[A.index.year % 4 == 2]
    line(mid["moo"], "AUG moo midterm years")
    print("  concentration:", pl.cluster_note(A.index, A["moo"].values, k=2))
    print("  per-year moo:", ", ".join(f"{d.year}:{100*x:+.1f}" for d, x in A["moo"].items()))
