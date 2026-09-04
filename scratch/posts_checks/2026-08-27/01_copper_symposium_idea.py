"""Idea check: long copper (HG=F) across the Jackson Hole symposium session.

Tonight (Thu 2026-08-27) is symposium eve. The context brief's event lane
found copper the strongest of the set on the symposium session itself
(prior close -> JH close: 19 of 25, +0.82%, and all six midterm sessions up).
That cell is anchored on TONIGHT'S close, so the tradeable shapes from here
are open-anchored. The post grammar needs time_td >= 1, so the shortest legal
version is:

    entry: MOO Fri 08-28 (the symposium session)
    exit:  MOC Mon 08-31 (JH+1, which in 2026 is also August's last session)

Decomposition:
  A. prior close -> JH close  (the brief's cell, replicated lag-0)
  B. JH open  -> JH close     (the part an MOO entry actually captures on the day)
  C. JH close -> JH+1 close   (the overnight + Monday leg the time stop forces)
  D. JH open  -> JH+1 close   (the whole trade as graded)
  E. the same four shapes for IWM, the brief's headline vehicle, for contrast
Kill attempts on D: all-days and local (+/-126td) controls, era split at
2018, midterm split, concentration, the JH+1 session on its own, and the
overnight gap into the open (does the cell get paid before the open?).
Also Wilder-14 ATR and ref_close for the idea spec.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    cluster_note, era_split, load_events, load_prices, local_control,
    sign_test, summarize, wilder_atr,
)

JH = pd.to_datetime(sorted(load_events(["jackson_hole"])["date"].unique()))
JH = JH[JH.year <= 2025]
print(f"symposium anchors: {len(JH)}  {JH[0].date()} .. {JH[-1].date()}")

px = load_prices(["HG=F", "IWM", "SPY"])


def rec(name: str, r: pd.Series, base: pd.Series | None = None) -> None:
    st = summarize(r.values)
    nup = int((r > 0).sum())
    extra = ""
    if base is not None:
        extra = f"| all-days {100*base.mean():+.3f}%  "
    print(f"  {name:<30} n={st['n']:<3} mean={st['mean_pct']:+.3f}%  med={st['median_pct']:+.3f}%  "
          f"{nup}-{len(r)-nup}  t={st['t']:+.2f}  sign_p={sign_test(nup, len(r)):.4f}  {extra}"
          f"worst {st['worst_pct']:+.2f}% ({r.idxmin().year})  best {st['best_pct']:+.2f}% ({r.idxmax().year})")


def shapes(tkr: str) -> dict[str, pd.Series]:
    f = px[tkr]
    idx = f.index
    pos = pd.Series(range(len(idx)), index=idx)
    rows = {"A prior close->JH close": [], "B JH open->JH close": [],
            "C JH close->JH+1 close": [], "D JH open->JH+1 close": [],
            "gap prior close->JH open": [], "JH+1 session itself": []}
    dates = []
    for d in JH:
        p = pos.get(d)
        if p is None or p + 1 >= len(idx) or p == 0:
            continue
        c0, o1, c1, o2, c2 = (f["Close"].iloc[p-1], f["Open"].iloc[p], f["Close"].iloc[p],
                              f["Open"].iloc[p+1], f["Close"].iloc[p+1])
        if not np.isfinite([c0, o1, c1, c2]).all() or o1 <= 0:
            continue
        dates.append(d)
        rows["A prior close->JH close"].append(c1 / c0 - 1)
        rows["B JH open->JH close"].append(c1 / o1 - 1)
        rows["C JH close->JH+1 close"].append(c2 / c1 - 1)
        rows["D JH open->JH+1 close"].append(c2 / o1 - 1)
        rows["gap prior close->JH open"].append(o1 / c0 - 1)
        rows["JH+1 session itself"].append(c2 / o2 - 1 if o2 > 0 else np.nan)
    return {k: pd.Series(v, index=pd.DatetimeIndex(dates)) for k, v in rows.items()}


def baseline_D(tkr: str) -> pd.Series:
    f = px[tkr]
    return (f["Close"].shift(-1) / f["Open"] - 1).dropna()


def splits(r: pd.Series) -> None:
    v = r.values
    print("    era:", [(e["label"], e["n"], round(e.get("mean_pct", np.nan), 3),
                        round(e.get("hit", np.nan), 1)) for e in era_split(r.index, v)])
    print("    concentration:", cluster_note(r.index, v))
    for lab, sub in (("midterm", r[[d.year % 4 == 2 for d in r.index]]),
                     ("non-midterm", r[[d.year % 4 != 2 for d in r.index]])):
        if len(sub):
            su = summarize(sub.values)
            nu = int((sub > 0).sum())
            print(f"    {lab:<12} n={su['n']:<3} mean={su['mean_pct']:+.3f}%  {nu}-{len(sub)-nu}  "
                  f"sign_p={sign_test(nu, len(sub)):.4f}  worst={su['worst_pct']:+.2f}% ({sub.idxmin().year})")
    print("    by year:", [(d.year, round(100 * x, 2)) for d, x in r.items()])


for tkr in ("HG=F", "IWM", "SPY"):
    f = px[tkr]
    atr = pd.Series(wilder_atr(f["High"], f["Low"], f["Close"]), index=f.index)
    print(f"\n=== {tkr}: close {f['Close'].iloc[-1]:.4f}  Wilder-14 ATR {atr.iloc[-1]:.4f} "
          f"({100*atr.iloc[-1]/f['Close'].iloc[-1]:.2f}%)  bar {f.index[-1].date()} ===")
    sh = shapes(tkr)
    base = baseline_D(tkr)
    for k, r in sh.items():
        rec(k, r.dropna(), base if k.startswith("D") else None)
    d = sh["D JH open->JH+1 close"]
    loc = base.reindex(local_control(base.index, d.index, 126)).dropna()
    print(f"  local control (+/-126td, ex-trigger) for D: {100*loc.mean():+.3f}%  n={len(loc)}")
    print("  splits on D:")
    splits(d)
    print("  splits on B (the on-the-day leg):")
    splits(sh["B JH open->JH close"])
    # how much of A is already gone by the open?
    a, g = sh["A prior close->JH close"], sh["gap prior close->JH open"]
    print(f"  gap share of A: mean gap {100*g.mean():+.3f}% of mean A {100*a.mean():+.3f}%")
