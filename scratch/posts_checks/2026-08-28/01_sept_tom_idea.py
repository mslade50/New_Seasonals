"""Idea check: the turn of the month into September, in pitch convention.

Tonight (Fri 2026-08-28) is the second-to-last session of August. Monday
08-31 is the last. So a signal tonight, entry MOC next session, means the
entry is the MONTH-END close and the hold is the first h sessions of the new
month: the classic turn-of-month window, minus the last-day rung the 08-24
pitch killed in bonds.

    signal: dist-to-month-end == 1 (tonight)
    entry:  MOC on ME-1 (Mon 08-31)
    exit:   MOC h sessions later (h=3 -> Thu 09-03, h=4 -> Fri 09-04 = NFP)

Vehicles SPY, QQQ, IWM. Kill attempts: all-days and local (+/-126td)
controls, era split at 2018, midterm years, SEPTEMBER only (the month
everyone will be talking about this weekend), the h ladder, concentration,
and a placebo offset ladder on the signal (signal at ME-2, ME-3, ME-4 with the
same h) so the anchor has to beat its own neighbours.

Second block: the same cell with entry at the OPEN of the first session
(MOO Tue 09-01), for the reader who would rather not hold the month-end close.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    cluster_note, era_split, fwd_lag, load_prices, local_control, sign_test,
    summarize, wilder_atr,
)

px = load_prices(["SPY", "QQQ", "IWM"])
closes = {k: v["Close"].dropna() for k, v in px.items()}
for k, f in px.items():
    atr = pd.Series(wilder_atr(f["High"], f["Low"], f["Close"]), index=f.index)
    print(f"tonight {k}: close {closes[k].iloc[-1]:.2f} | Wilder-14 ATR {atr.iloc[-1]:.4f} "
          f"({atr.iloc[-1] / closes[k].iloc[-1] * 100:.2f}%) | bar {f.index[-1].date()}")


def dist_to_month_end(idx: pd.DatetimeIndex) -> pd.Series:
    ser = pd.Series(np.arange(len(idx)), index=idx)
    ym = idx.to_period("M")
    last_pos = ser.groupby(ym).transform("max")
    d = last_pos - ser
    d[ym == ym[-1]] = np.nan
    return d


def block(name: str, r: pd.Series, s: pd.Series, h: int) -> dict:
    v = r.values
    st = summarize(v)
    nup = int((r > 0).sum())
    allr = fwd_lag(s, h, 1).dropna()
    loc = allr.reindex(local_control(s.index, r.index, 126)).dropna()
    print(f"  {name:<30} n={st['n']:<4} mean={st['mean_pct']:+.3f}%  med={st['median_pct']:+.3f}%  "
          f"{nup}-{len(r)-nup} ({st['hit']:.1f}%)  t={st['t']:+.2f}  sign_p={sign_test(nup, len(r)):.4f}  "
          f"| all {100*allr.mean():+.3f}%  local {100*loc.mean():+.3f}%  "
          f"| worst {st['worst_pct']:+.2f}% ({r.idxmin().date()})")
    return st


def splits(r: pd.Series) -> None:
    v = r.values
    print("    era:", [(e["label"], e["n"], round(e.get("mean_pct", np.nan), 3),
                        round(e.get("hit", np.nan), 1), round(e.get("t", np.nan), 2))
                       for e in era_split(r.index, v)])
    print("    concentration:", cluster_note(r.index, v))
    for lab, sub in (("midterm", r[[d.year % 4 == 2 for d in r.index]]),
                     ("non-midterm", r[[d.year % 4 != 2 for d in r.index]]),
                     ("august signal (sept TOM)", r[[d.month == 8 for d in r.index]]),
                     ("other 11 months", r[[d.month != 8 for d in r.index]])):
        if len(sub):
            su = summarize(sub.values)
            nu = int((sub > 0).sum())
            print(f"    {lab:<26} n={su['n']:<4} mean={su['mean_pct']:+.3f}%  med={su['median_pct']:+.3f}%  "
                  f"{nu}-{len(sub)-nu}  sign_p={sign_test(nu, len(sub)):.4f}  t={su['t']:+.2f}  worst={su['worst_pct']:+.2f}%")
    aug = r[[d.month == 8 for d in r.index]]
    print("    sept TOM by year:", [(d.year, round(100 * x, 2)) for d, x in aug.items()])
    am = aug[[d.year % 4 == 2 for d in aug.index]]
    print(f"    sept TOM midterms: n={len(am)} mean={100*am.mean():+.3f}% "
          f"{int((am>0).sum())}-{int((am<=0).sum())}  {[(d.year, round(100*x,2)) for d,x in am.items()]}")


print("\n" + "=" * 96)
print("SECTION 1: signal = ME-2 (tonight); entry MOC ME-1 (month-end close); exit MOC h later")
print("=" * 96)
for tkr in ("SPY", "QQQ", "IWM"):
    s = closes[tkr]
    dist = dist_to_month_end(s.index)
    sig = s.index[(dist == 1).values]
    print(f"\n--- {tkr}  signals {len(sig)}  {sig[0].date()} .. {sig[-1].date()} ---")
    for h in (1, 2, 3, 4, 5):
        r = fwd_lag(s, h, 1).reindex(sig).dropna()
        block(f"h={h}", r, s, h)
    for h in (3, 4):
        print(f"  splits for h={h}:")
        splits(fwd_lag(s, h, 1).reindex(sig).dropna())

print("\n" + "=" * 96)
print("SECTION 2: placebo offset ladder, SPY, h=3: signal at ME-k, entry MOC ME-(k-1)")
print("=" * 96)
s = closes["SPY"]
dist = dist_to_month_end(s.index)
for k in (1, 2, 3, 4, 5, 6):
    sig = s.index[(dist == k).values]
    r = fwd_lag(s, 3, 1).reindex(sig).dropna()
    block(f"signal ME-{k+1} (entry ME-{k})", r, s, 3)
# and the post-month-start rungs: signal on first session, second session...
first = s.index[(dist.shift(1) == 0).values]
for j in range(0, 3):
    sig = pd.DatetimeIndex([s.index[min(len(s)-1, s.index.get_loc(d) + j)] for d in first])
    r = fwd_lag(s, 3, 1).reindex(sig).dropna()
    block(f"signal new-month day {j+1}", r, s, 3)

print("\n" + "=" * 96)
print("SECTION 3: MOO variant. entry at OPEN of first session of the month, exit MOC h-1 sessions later")
print("=" * 96)
for tkr in ("SPY", "QQQ", "IWM"):
    f = px[tkr]
    o, c = f["Open"], f["Close"]
    dist = dist_to_month_end(c.index)
    firstpos = np.where((dist.shift(1) == 0).values)[0]
    for h in (2, 3, 4):
        vals, dates = [], []
        for p in firstpos:
            if p + h - 1 < len(c):
                vals.append(c.iloc[p + h - 1] / o.iloc[p] - 1)
                dates.append(c.index[p])
        r = pd.Series(vals, index=pd.DatetimeIndex(dates))
        st = summarize(r.values)
        nup = int((r > 0).sum())
        aug = r[[d.month == 9 for d in r.index]]
        sa = summarize(aug.values)
        print(f"  {tkr} MOO day1 -> MOC day{h}: n={st['n']} mean={st['mean_pct']:+.3f}% {nup}-{len(r)-nup} t={st['t']:+.2f}"
              f"  | september n={sa['n']} mean={sa['mean_pct']:+.3f}% {int((aug>0).sum())}-{int((aug<=0).sum())}")
