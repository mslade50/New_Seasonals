"""The Aug-19 seasonal window in midterm years.

The sweep reports QQQ 6-0 and IWM 6-0 up on the matching trading day of year
in the six prior midterm years, sign p 0.0156 each, means +0.465% and +0.839%.
N=6 is anecdote tier and cannot headline. Before it can be printed at all it
has to survive three questions: are the six years independent or one repeated
regime, does the wider all-years cell agree, and is the +/-2 day matching
window doing the work.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import close_panel, summarize, show, sign_test  # noqa: E402

TKRS = ["SPY", "QQQ", "IWM", "^GSPC"]
px = close_panel(TKRS)
idx = px.index
TARGET = pd.Timestamp("2026-08-19")


def td_of_year(dates):
    s = pd.Series(1, index=dates)
    return s.groupby(dates.year).cumsum().values


tdy = pd.Series(td_of_year(idx), index=idx)
target_td = None
# the trading-day-of-year the NEXT session will occupy
this_yr = idx[idx.year == 2026]
target_td = int(tdy.loc[this_yr[-1]]) + 1
print(f"next session 2026-08-19 is trading day {target_td} of 2026")

print("\n" + "=" * 74)
print("A. anchor = the session before the matching td, h1 = the matching session")
print("=" * 74)
rows = []
for y in sorted(set(idx.year)):
    if y == 2026:
        continue
    yr_days = idx[idx.year == y]
    if len(yr_days) < target_td:
        continue
    anchor = yr_days[target_td - 2]  # session before the matching one
    p = idx.get_loc(anchor)
    if p + 5 >= len(idx):
        continue
    rec = {"year": y, "anchor": anchor.date(),
           "matched": idx[p + 1].date(), "midterm": y % 4 == 2}
    for t in TKRS:
        v = px[t].values
        rec[f"{t}_h1"] = v[p + 1] / v[p] - 1.0
        rec[f"{t}_h5"] = v[p + 5] / v[p] - 1.0
    rows.append(rec)
R = pd.DataFrame(rows)
print(R[["year", "anchor", "matched", "midterm"]].tail(8).to_string(index=False))

print("\n" + "=" * 74)
print("B. all years vs midterm years, h1 and h5")
print("=" * 74)
for t in TKRS:
    out = []
    for h in ["h1", "h5"]:
        col = R[f"{t}_{h}"].dropna()
        for name, m in [("all years", np.ones(len(R), bool)),
                        ("MIDTERM only", R["midterm"].values),
                        ("non-midterm", ~R["midterm"].values)]:
            v = R[f"{t}_{h}"][m].dropna().values
            if len(v) == 0:
                continue
            r = summarize(v, f"{h} {name}")
            r["rec"] = f"{int((v > 0).sum())}-{int((v <= 0).sum())}"
            r["sign_p"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
            out.append(r)
    show(out, f"{t}: matching trading day of year")

print("\n" + "=" * 74)
print("C. WHICH six midterm years, and what each one did (the independence test)")
print("=" * 74)
mt = R[R["midterm"]]
print(mt[["year", "matched"] + [f"{t}_h1" for t in ["QQQ", "IWM", "SPY"]]]
      .assign(**{f"{t}_h1": lambda d, t=t: (100 * d[f"{t}_h1"]).round(2)
                 for t in ["QQQ", "IWM", "SPY"]}).to_string(index=False))
print("\n  the six midterm years span:", mt["year"].min(), "to", mt["year"].max())

print("\n" + "=" * 74)
print("D. is the +/-2 matching window load-bearing? walk the anchor +/-3 sessions")
print("=" * 74)
for t in ["QQQ", "IWM"]:
    out = []
    for shift in range(-3, 4):
        vals = []
        for y in sorted(set(idx.year)):
            if y == 2026 or y % 4 != 2:
                continue
            yr_days = idx[idx.year == y]
            if len(yr_days) < target_td:
                continue
            anchor = yr_days[target_td - 2]
            p = idx.get_loc(anchor) + shift
            if p < 0 or p + 1 >= len(idx):
                continue
            v = px[t].values
            vals.append(v[p + 1] / v[p] - 1.0)
        vals = np.array(vals)
        r = summarize(vals, f"anchor {shift:+d} td")
        r["rec"] = f"{int((vals > 0).sum())}-{int((vals <= 0).sum())}"
        out.append(r)
    show(out, f"{t}: midterm years, anchor walked around the matching day")

print("\n" + "=" * 74)
print("E. control: every August session in midterm years")
print("=" * 74)
for t in ["QQQ", "IWM", "SPY"]:
    r1 = px[t].pct_change(fill_method=None)
    m = (idx.month == 8) & (idx.year % 4 == 2) & (idx.year != 2026)
    v = r1[m].dropna().values
    out = [summarize(v, f"{t} all August sessions, midterm years")]
    v2 = r1.dropna().values
    out.append(summarize(v2, f"{t} all days"))
    show(out, f"{t}: what a random midterm-August session pays")
