"""The quad-witching session itself, September only.

Tomorrow (2026-09-18) is September quad witching. The engine's pooled cell
(all four witchings, n=106) reads SPY -0.175% at t -1.76. Pooling March, June,
September and December hides whichever month carries it. Split by month, then
by cycle phase, then control against all Fridays and against the same month's
non-witching sessions.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    close_panel, load_events, fwd_ret, summarize, show, sign_test, cluster_note,
)

TICKERS = ["SPY", "QQQ", "IWM", "^GSPC", "^VIX"]
px = close_panel(TICKERS)
idx = px.index

qw = load_events(["quad_witching"])["date"]
qw = pd.DatetimeIndex([d for d in qw if idx[0] <= d <= idx[-1]])
# the witching session must be a real bar
qw = pd.DatetimeIndex([d for d in qw if d in set(idx)])
print(f"quad witchings on the tape: {len(qw)}  {qw[0].date()} .. {qw[-1].date()}")

for tkr in TICKERS:
    s = px[tkr].dropna()
    r1 = (s / s.shift(1) - 1.0)          # the session's own close-to-close move
    live = pd.DatetimeIndex(qw).intersection(r1.dropna().index)

    rows = []
    allq = r1.loc[live].values
    rows.append(summarize(allq, "all quad witchings"))
    for m, name in [(3, "March"), (6, "June"), (9, "September"), (12, "December")]:
        d = live[live.month == m]
        rows.append(summarize(r1.loc[d].values, name))
    # controls
    fri = r1.dropna()
    fri_all = fri[fri.index.weekday == 4]
    rows.append(summarize(fri_all.values, "CTL all Fridays"))
    sep_fri = fri_all[fri_all.index.month == 9]
    rows.append(summarize(sep_fri.values, "CTL September Fridays"))
    sep_nonqw = fri[(fri.index.month == 9) & (~fri.index.isin(live))]
    rows.append(summarize(sep_nonqw.values, "CTL September non-witching sessions"))
    rows.append(summarize(fri.values, "CTL all days"))
    show(rows, f"{tkr}: the witching session itself (close-to-close)")

    sep = live[live.month == 9]
    v = r1.loc[sep]
    up = int((v > 0).sum())
    n = len(v)
    print(f"  {tkr} September witching record: {up}-{n - up} up, "
          f"sign p(down) = {sign_test(n - up, n):.4f}, "
          f"sign p(up) = {sign_test(up, n):.4f}")
    print(f"  {tkr} cycle split, September witching:")
    for phase, mod in [("midterm", 2), ("pre-election", 3), ("election", 0), ("post-election", 1)]:
        d = sep[sep.year % 4 == mod]
        r = summarize(r1.loc[d].values, f"    {phase}")
        if r["n"]:
            k = int((r1.loc[d] > 0).sum())
            print(f"    {phase:<14} n={r['n']:<3} mean={r['mean_pct']:+.3f}%  "
                  f"med={r['median_pct']:+.3f}%  {k}-{r['n'] - k} up  "
                  f"sign p(down)={sign_test(r['n'] - k, r['n']):.4f}")
    print(f"  {tkr} era split, September witching:")
    for lab, m in [("pre-2018", sep < pd.Timestamp("2018-01-01")),
                   ("2018+", sep >= pd.Timestamp("2018-01-01"))]:
        r = summarize(r1.loc[sep[m]].values, lab)
        if r["n"]:
            k = int((r1.loc[sep[m]] > 0).sum())
            print(f"    {lab:<10} n={r['n']:<3} mean={r['mean_pct']:+.3f}%  {k}-{r['n'] - k} up")
    print(f"  {tkr} concentration: {cluster_note(sep, r1.loc[sep].values, k=2)}")
    print()

# the specific conjunction: September witching two sessions after an FOMC decision
fomc = pd.DatetimeIndex(load_events(["fomc_decision"])["date"])
pos = pd.Series(range(len(idx)), index=idx)
print("=== September witching that lands 2 sessions after an FOMC decision ===")
for tkr in ["SPY", "^GSPC", "QQQ", "IWM"]:
    s = px[tkr].dropna()
    r1 = s / s.shift(1) - 1.0
    sep = pd.DatetimeIndex([d for d in qw if d.month == 9]).intersection(r1.dropna().index)
    hit, miss = [], []
    for d in sep:
        p = int(idx.searchsorted(d))
        prior2 = idx[max(0, p - 2)]
        if prior2 in set(fomc):
            hit.append(d)
        else:
            miss.append(d)
    hit, miss = pd.DatetimeIndex(hit), pd.DatetimeIndex(miss)
    rows = [summarize(r1.loc[hit].values, "FOMC 2 sessions earlier"),
            summarize(r1.loc[miss].values, "no FOMC")]
    show(rows, tkr)
    if len(hit):
        k = int((r1.loc[hit] > 0).sum())
        print(f"  dates: {[str(d.date()) for d in hit]}")
        print(f"  record {k}-{len(hit) - k} up, sign p(down) = {sign_test(len(hit) - k, len(hit)):.4f}")
