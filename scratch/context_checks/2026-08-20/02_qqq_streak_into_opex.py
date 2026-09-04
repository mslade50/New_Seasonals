"""The crossing: a QQQ 5+ down-close streak whose NEXT session is a monthly opex.

That is exactly tomorrow. Two search modes crossed, event and price state, which
is the thing neither lane can see on its own. Also asks the reverse question the
brief has to answer honestly: is the opex session itself different from any other
next session after a streak?
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import close_panel, cluster_note, load_events, sign_test, summarize  # noqa


def down_run(s: pd.Series) -> np.ndarray:
    r = s.pct_change()
    down = (r < 0).values
    run = np.zeros(len(s), dtype=int)
    for i in range(1, len(s)):
        run[i] = run[i - 1] + 1 if down[i] else 0
    return run


def report(label, v):
    v = np.asarray(v)
    if len(v) == 0:
        print(f"  {label:<40} n=0")
        return
    st = summarize(v, label)
    up = int((v > 0).sum())
    print(
        f"  {label:<40} n={st['n']:<4} mean={st['mean_pct']:+.3f}%  med={st['median_pct']:+.3f}%"
        f"  {up}-{st['n'] - up}  hit={st['hit']:.1f}%  t={st['t']:+.2f}"
        f"  sign_p={sign_test(up, st['n']):.4f}"
    )


px = close_panel(["QQQ"])["QQQ"].dropna()
run = down_run(px)
fwd1 = px.pct_change().shift(-1)

opex = pd.to_datetime(load_events(["opex"])["date"])
opex_set = set(opex.dt.normalize())

# anchor = the session before an opex, so h1 is the opex bar itself
idx = px.index
nxt = pd.Series(index=idx, dtype="datetime64[ns]")
nxt.iloc[:-1] = idx[1:]
next_is_opex = nxt.map(lambda d: (pd.notna(d)) and (d.normalize() in opex_set)).values

streak = (run >= 5) & fwd1.notna().values
print(f"QQQ {idx[0].date()} to {idx[-1].date()}, opex dates in calendar: {len(opex_set)}")
print(f"live: run={int(run[-1])}, next session is opex = True (2026-08-21)\n")

report("5+ down streak, any next session", fwd1.values[streak])
cross = streak & next_is_opex
report("5+ down streak, next session = opex", fwd1.values[cross])
report("5+ down streak, next session not opex", fwd1.values[streak & ~next_is_opex])
print()
report("opex session, no streak into it", fwd1.values[next_is_opex & ~streak & fwd1.notna().values])
report("opex session, all", fwd1.values[next_is_opex & fwd1.notna().values])

cd = idx[cross]
print("\n  crossing episodes (anchor date -> opex-session move):")
for dt, v in zip(cd, fwd1.values[cross] * 100):
    print(f"    {dt.date()}  run={run[list(idx).index(dt)]}  opex h1={v:+.2f}%")
print("  concentration:", cluster_note(cd, fwd1.values[cross]))

# how much of the streak cell's edge is just "the bounce", regardless of opex
base = fwd1.values[streak]
print(
    f"\n  difference in means (cross minus non-cross): "
    f"{np.nanmean(fwd1.values[cross]) * 100 - np.nanmean(fwd1.values[streak & ~next_is_opex]) * 100:+.3f} pp"
)
print(f"  streak cell mean {np.nanmean(base) * 100:+.3f}%, n={int(streak.sum())}")
