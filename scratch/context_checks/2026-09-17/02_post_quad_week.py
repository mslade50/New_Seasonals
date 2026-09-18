"""The 5 sessions after September quad witching.

The engine's seasonal_doy cell smears +/-2 calendar days around Sep 18 and
reads QQQ h5 -1.12%, 18-8 down. That is the famous "week after September opex
is the worst week of the year" claim arriving through the wrong door. Re-anchor
it on the actual September quad-witching close, which is a clean event, and
control it against the other three witchings and against all 5-session windows.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    close_panel, load_events, summarize, show, sign_test, cluster_note,
)

TICKERS = ["SPY", "QQQ", "IWM", "^GSPC", "^VIX"]
px = close_panel(TICKERS)
idx = px.index
qw = load_events(["quad_witching"])["date"]
qw = pd.DatetimeIndex([d for d in qw if d in set(idx)])

H = 5


def fwd(s, h):
    return s.shift(-h) / s - 1.0


for tkr in TICKERS:
    s = px[tkr].dropna()
    f = fwd(s, H)
    live = pd.DatetimeIndex(qw).intersection(f.dropna().index)

    rows = [summarize(f.loc[live].values, "all witchings")]
    for m, name in [(3, "March"), (6, "June"), (9, "September"), (12, "December")]:
        rows.append(summarize(f.loc[live[live.month == m]].values, name))
    base = f.dropna()
    rows.append(summarize(base.values, "CTL every 5-session window"))
    rows.append(summarize(base[base.index.month == 9].values, "CTL every September window"))
    show(rows, f"{tkr}: 5 sessions AFTER the witching close")

    sep = live[live.month == 9]
    v = f.loc[sep]
    k = int((v > 0).sum())
    n = len(v)
    print(f"  {tkr} September post-witching week: {k}-{n - k} up, "
          f"mean {100 * v.mean():+.2f}%, median {100 * v.median():+.2f}%, "
          f"sign p(down) = {sign_test(n - k, n):.4f}")
    print(f"  worst {100 * v.min():+.2f}% ({v.idxmin().date()}), "
          f"best {100 * v.max():+.2f}% ({v.idxmax().date()})")
    for lab, m in [("pre-2018", sep < pd.Timestamp("2018-01-01")),
                   ("2018+", sep >= pd.Timestamp("2018-01-01"))]:
        vv = f.loc[sep[m]]
        kk = int((vv > 0).sum())
        print(f"    {lab:<10} n={len(vv):<3} mean={100 * vv.mean():+.3f}%  {kk}-{len(vv) - kk} up")
    for phase, mod in [("midterm", 2), ("pre-election", 3), ("election", 0), ("post-election", 1)]:
        vv = f.loc[sep[sep.year % 4 == mod]]
        kk = int((vv > 0).sum())
        print(f"    {phase:<14} n={len(vv):<3} mean={100 * vv.mean():+.3f}%  "
              f"med={100 * vv.median():+.3f}%  {kk}-{len(vv) - kk} up  "
              f"sign p(down)={sign_test(len(vv) - kk, len(vv)):.4f}")
    print(f"  concentration: {cluster_note(sep, v.values, k=2)}")
    print(f"  year-by-year: {dict((d.year, round(100 * x, 2)) for d, x in v.items())}")
    print()

# horizon shape: where in the post-witching stretch does it sit?
print("=== SPY / QQQ / IWM: horizon scan from the September witching close ===")
for tkr in ["SPY", "QQQ", "IWM"]:
    s = px[tkr].dropna()
    sep_all = pd.DatetimeIndex([d for d in qw if d.month == 9])
    out = []
    for h in (1, 2, 3, 5, 7, 10, 15, 21):
        f = fwd(s, h)
        live = sep_all.intersection(f.dropna().index)
        r = summarize(f.loc[live].values, f"h={h}")
        base = f.dropna()
        r["ctl_pct"] = round(100 * base.mean(), 3)
        r["edge_pct"] = round(r["mean_pct"] - 100 * base.mean(), 3)
        k = int((f.loc[live] > 0).sum())
        r["record"] = f"{k}-{r['n'] - k}"
        out.append(r)
    show(out, tkr)

# does the IWM-vs-SPY spread widen in that week?
print("\n=== IWM minus SPY over the 5 sessions after September witching ===")
fi, fs = fwd(px["IWM"].dropna(), H), fwd(px["SPY"].dropna(), H)
sp = (fi - fs).dropna()
sep = pd.DatetimeIndex([d for d in qw if d.month == 9]).intersection(sp.index)
k = int((sp.loc[sep] > 0).sum())
show([summarize(sp.loc[sep].values, "September witching"),
      summarize(sp.loc[pd.DatetimeIndex(qw).intersection(sp.index)].values, "all witchings"),
      summarize(sp.values, "CTL every window")], "IWM - SPY, 5 sessions")
print(f"  September record: IWM beat SPY {k} of {len(sep)}, "
      f"sign p(IWM lags) = {sign_test(len(sep) - k, len(sep)):.4f}")
