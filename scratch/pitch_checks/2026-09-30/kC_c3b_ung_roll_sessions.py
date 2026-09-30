"""C3 mechanism check: locate UNG's October roll and decompose the short BY SESSION.

UNG rolls the front NG contract into the next over four business days starting about
two weeks before the front contract's last trade (NG last trade = 3 bd before the 1st
of the delivery month). In October the held contract is November (expiry ~Oct 28), so
the roll is ~Oct 13-19, sessions ~9-13 after the Sep ME close. If the winter contango
is the mechanism, the short's excess should sit in the roll/pre-roll sessions, and the
UNG-minus-NG=F wedge should be widest in October.
Also: short from each ME close THROUGH the next roll's last day (the window the
mechanism actually names), October vs the other eleven months.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

P = load_prices(["UNG", "NG=F"])
u = P["UNG"]["Close"].dropna()
ng = P["NG=F"]["Close"].dropna()
ng = ng[ng > 0]
idx = u.index
LAST = idx[-1]


def ng_expiry(year: int, month: int) -> pd.Timestamp:
    return pd.Timestamp(year=year, month=month, day=1) - pd.offsets.BDay(3)


def next_session(d: pd.Timestamp) -> int:
    return int(idx.searchsorted(d))


# for each calendar month M, the held contract is delivery M+1; its roll starts ~2 weeks pre-expiry
roll_days = set()
roll_by_month = {}
for y in range(2007, 2027):
    for m in range(1, 13):
        dy, dm = (y, m + 1) if m < 12 else (y + 1, 1)
        exp = ng_expiry(dy, dm)
        p0 = next_session(exp - pd.Timedelta(days=14))
        if p0 + 4 > len(idx):
            continue
        days = idx[p0:p0 + 4]
        roll_by_month[(y, m)] = days
        roll_days.update(days)
print("Oct roll windows located (first/last day):")
for y in (2022, 2023, 2024, 2025, 2026):
    d = roll_by_month.get((y, 10))
    if d is not None:
        print(f"  {y}: {d[0].date()} .. {d[-1].date()}  (sessions after Sep ME: "
              f"{idx.get_loc(d[0]) - idx[(idx.year == y) & (idx.month == 9)].size - idx[idx < pd.Timestamp(y, 9, 1)].size + 1}"
              f"..{idx.get_loc(d[-1]) - idx[(idx.year == y) & (idx.month == 9)].size - idx[idx < pd.Timestamp(y, 9, 1)].size + 1})")
print("  2026 Oct roll (projected): Nov contract last trade", ng_expiry(2026, 11).date(),
      "-> roll ~", (ng_expiry(2026, 11) - pd.Timedelta(days=14)).date(), "+4 bd (sessions ~10-13 from 09-30)")

# daily returns
ru = u.pct_change()
rn = ng.pct_change().reindex(idx)
exp_all = pd.DatetimeIndex([ng_expiry(y, m) for y in range(2007, 2028) for m in range(1, 13)])
# NG=F seam day = first session AFTER an expiry (continuous series switches)
seam = pd.Series(False, index=idx)
for e in exp_all:
    p = idx.searchsorted(e + pd.Timedelta(days=1))
    if p < len(idx):
        seam.iloc[p] = True
    p = idx.searchsorted(e)
    if p < len(idx):
        seam.iloc[p] = True
wedge = (ru - rn)
ok = rn.notna() & ~seam
isroll = pd.Series(idx.isin(list(roll_days)), index=idx)
# pre-roll = the 5 sessions before each roll start
pre = pd.Series(False, index=idx)
for days in roll_by_month.values():
    p0 = idx.get_loc(days[0])
    pre.iloc[max(0, p0 - 5):p0] = True

rows = []
for lbl, mm in (("October", idx.month == 10), ("other months", idx.month != 10),
                ("September", idx.month == 9), ("November", idx.month == 11),
                ("December", idx.month == 12)):
    mm = pd.Series(mm, index=idx)
    for seg, s in (("roll days", isroll), ("pre-roll 5d", pre & ~isroll), ("rest", ~isroll & ~pre)):
        m = mm & s & ru.notna()
        mw = m & ok
        rows.append({"month": lbl, "segment": seg, "n_sess": int(m.sum()),
                     "short_UNG_bp": -1e4 * ru[m].mean(),
                     "short_NG_bp": -1e4 * rn[mw].mean(),
                     "wedge_UNG_minus_NG_bp": 1e4 * wedge[mw].mean()})
show(rows, "per-session means: SHORT UNG, SHORT NG=F (seam days dropped), and the UNG-NG=F wedge")

# the mechanism's own window: ME close -> last day of the next month's roll
rows = []
out = {}
MEs = pd.DatetimeIndex(pd.Series(idx, index=idx).groupby([idx.year, idx.month]).max().values)
MEs = MEs[MEs < LAST]
for a in MEs:
    y, m = (a.year, a.month + 1) if a.month < 12 else (a.year + 1, 1)
    d = roll_by_month.get((y, m))
    if d is None:
        continue
    out[a] = -(u.loc[d[-1]] / u.loc[a] - 1.0)
thr = pd.Series(out)
drift = {}
for a, v in thr.items():
    y, m = (a.year, a.month + 1) if a.month < 12 else (a.year + 1, 1)
    L = idx.get_loc(roll_by_month[(y, m)][-1]) - idx.get_loc(a)
    drift[a] = L
L = pd.Series(drift)
for m in range(1, 13):
    wm = (thr.index.month % 12) + 1
    v = thr[wm == m]
    w = int((v > 0).sum())
    rows.append({"window_month": m, "n": len(v), "mean_len_td": L[wm == m].mean(),
                 "short_to_roll_end_pct": 100 * v.mean(), "median": 100 * v.median(),
                 "rec": f"{w}-{len(v)-w}", "sign_p": sign_test(w, len(v))})
df = pd.DataFrame(rows)
df["rank"] = df["short_to_roll_end_pct"].rank(ascending=False).astype(int)
print("\n=== SHORT UNG from ME close THROUGH the next month's roll end, by window month ===")
print(df.round(3).to_string(index=False))
o = thr[thr.index.month == 9]
x = thr[thr.index.month != 9]
dr13 = -(u.shift(-13) / u - 1.0).dropna().mean()
print(f"  Oct {100*o.mean():+.3f}% vs other months {100*x.mean():+.3f}%, own all-days 13td drift (short) {100*dr13:+.3f}%")
print(f"  Oct pre-2018 {100*o[o.index<'2018'].mean():+.3f}% | 2018+ {100*o[o.index>='2018'].mean():+.3f}%  "
      f"| midterm {100*o[o.index.year%4==2].mean():+.3f}%")
print("  Oct by year:", ", ".join(f"{d.year}:{100*v:+.1f}" for d, v in o.items()))
