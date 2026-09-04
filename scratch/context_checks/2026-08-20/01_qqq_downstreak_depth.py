"""QQQ 5+ consecutive down closes: does the bounce depend on how deep the streak was?

Live state 2026-08-20: QQQ has closed down 5 sessions running, cumulative -2.89%,
worst single day -1.69%. The sweep's pooled cell (n=96, h1 +1.20%, t=4.08) has a
best outcome of +12.2%, which smells like 2008-style flushes carrying the mean.
This splits the cell by streak depth and checks concentration and era.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import close_panel, cluster_note, era_split, sign_test, summarize  # noqa


def streak_state(s: pd.Series) -> pd.DataFrame:
    """Run length of consecutive down closes and the cumulative move over that run."""
    r = s.pct_change()
    down = r < 0
    run = np.zeros(len(s), dtype=int)
    for i in range(1, len(s)):
        run[i] = run[i - 1] + 1 if down.iloc[i] else 0
    depth = np.full(len(s), np.nan)
    for i in range(len(s)):
        k = run[i]
        if k >= 1 and i - k >= 0:
            depth[i] = s.iloc[i] / s.iloc[i - k] - 1.0
    return pd.DataFrame({"run": run, "depth": depth}, index=s.index)


def report(label, dates, fwd):
    if len(dates) == 0:
        print(f"  {label:<34} n=0")
        return
    v = np.asarray(fwd)  # FRACTIONS in, percent out (pitch_lab convention)
    st = summarize(v, label)
    up = int((v > 0).sum())
    print(
        f"  {label:<34} n={st['n']:<4} mean={st['mean_pct']:+.3f}%  med={st['median_pct']:+.3f}%"
        f"  {up}-{st['n'] - up}  hit={st['hit']:.1f}%  t={st['t']:+.2f}"
        f"  sign_p={sign_test(up, st['n']):.4f}  worst={st['worst_pct']:+.2f}%  best={st['best_pct']:+.2f}%"
    )


for tkr in ("QQQ", "^NDX"):
    px = close_panel([tkr])[tkr].dropna()
    ss = streak_state(px)
    fwd1 = px.pct_change().shift(-1)  # lag=0 close-to-close, h1 = next session

    fired = ss.index[(ss["run"] >= 5) & fwd1.reindex(ss.index).notna()]
    all_v = fwd1.loc[fired].values

    print(f"\n=== {tkr}  ({px.index[0].date()} to {px.index[-1].date()}) ===")
    print(f"  live: run={int(ss['run'].iloc[-1])}, depth={ss['depth'].iloc[-1] * 100:+.2f}%")
    report("all 5+ down streaks", fired, all_v)

    d = ss.loc[fired, "depth"] * 100
    shallow = fired[d >= -4.0]
    deep = fired[d < -4.0]
    report("shallow streak (>= -4%)", shallow, fwd1.loc[shallow].values)
    report("deep streak (< -4%)", deep, fwd1.loc[deep].values)

    # the live cell is tighter than that: exactly 5 down, shallow
    live = fired[(ss.loc[fired, "run"] == 5) & (d >= -4.0)]
    report("exactly 5 down and >= -4%", live, fwd1.loc[live].values)

    if tkr == "QQQ":
        print("  era split (all 5+):")
        for e in era_split(fired, all_v):
            print(f"    {e}")
        print("  concentration (all 5+):", cluster_note(fired, all_v))
        print("  concentration (shallow):", cluster_note(shallow, fwd1.loc[shallow].values))
        print("  shallow dates + h1:")
        for dt, v in zip(shallow, fwd1.loc[shallow].values * 100):
            print(f"    {dt.date()}  depth={ss.loc[dt, 'depth'] * 100:+.2f}%  h1={v:+.2f}%")
