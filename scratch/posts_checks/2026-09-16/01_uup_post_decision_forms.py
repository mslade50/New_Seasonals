"""Idea check (2026-09-16): long the dollar (UUP) the session after an FOMC decision,
in the forms a reader can actually trade tomorrow.

Tonight's brief cell is h1 FROM the decision close (13-2 after a decision that sank the S&P
and lifted DXY 0.5%+). That close is gone, so the post needs the lag-1 forms:
  MOO  D+1 open  -> D+1+h close      (execute_on = 2026-09-17 open)
  MOC  D+1 close -> D+1+h close      (execute_on = 2026-09-17 close)
Both are measured over ALL scheduled decisions (the parent, N large) and over the
conjunction sub-cell (S&P down on decision day AND DXY >= +0.5%), on UUP (ETF, 2007+)
and on DX-Y.NYB (cash index, 2000+) so the ETF's short history is cross-checked.
Controls: every non-decision Wednesday anchor (same weekday, same MOO/MOC leg).
Era split at 2018, drop-2-best-years haircut, sign test on the record.
Also the natgas Thursday-before-opex cell in its MOO form (open->close tomorrow),
as the second candidate.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa: F401,F403

import numpy as np
import pandas as pd

ASOF = pd.Timestamp("2026-09-16")
ERA = "2018-01-01"
HS = (1, 2, 3, 5)
TK = ["UUP", "DX-Y.NYB", "^GSPC", "NG=F", "SPY"]

px = load_prices(TK)
nyse = px["SPY"]["Close"].dropna().index
nyse = nyse[nyse <= ASOF]
C = {t: px[t]["Close"].astype(float).reindex(nyse) for t in TK}
O = {t: px[t]["Open"].astype(float).reindex(nyse) for t in TK}

ev = load_events(["fomc_decision"])
dec = pd.DatetimeIndex([d for d in pd.to_datetime(ev["date"].unique()) if d in nyse and d < ASOF])
print("decisions:", len(dec), dec[0].date(), "->", dec[-1].date())

spx_dd = (C["^GSPC"] / C["^GSPC"].shift(1) - 1).reindex(dec)
dxy_dd = (C["DX-Y.NYB"] / C["DX-Y.NYB"].shift(1) - 1).reindex(dec)
conj = dec[((spx_dd < 0) & (dxy_dd >= 0.005)).values]
print("conjunction (S&P down, DXY >= +0.5%):", len(conj), [str(x.date()) for x in conj])
print("tonight: S&P %.2f%%  DXY %.2f%%" % (100 * (C["^GSPC"].iloc[-1] / C["^GSPC"].iloc[-2] - 1),
                                            100 * (C["DX-Y.NYB"].iloc[-1] / C["DX-Y.NYB"].iloc[-2] - 1)))


def leg(tk, form, h):
    c, o = C[tk], O[tk]
    if form == "MOC":
        return c.shift(-(1 + h)) / c.shift(-1) - 1
    if form == "MOO":
        return c.shift(-(1 + h - 1)) / o.shift(-1) - 1
    if form == "H1_FROM_CLOSE":
        return c.shift(-h) / c - 1
    raise ValueError(form)


def rec(v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    return int((v > 0).sum()), int((v < 0).sum()), len(v)


def line(label, s, dates, ctrl=None):
    v = s.reindex(dates).dropna()
    if len(v) == 0:
        print(f"  {label}: n=0")
        return v
    up, dn, n = rec(v.values)
    sm = summarize(v.values)
    extra = ""
    if ctrl is not None:
        cv = ctrl.dropna()
        cu, cd, cn = rec(cv.values)
        extra = f" | ctrl mean {100*cv.mean():+.3f}% hit {100*(cv>0).mean():.1f}% ({cu}-{cd})"
    print(f"  {label}: n={n} {up}-{dn} mean {sm['mean_pct']:+.3f}% med {sm['median_pct']:+.3f}% "
          f"t {sm['t']:+.2f} signp {sign_test(up, n):.4f} worst {sm['worst_pct']:+.2f}% ({v.idxmin().date()}) "
          f"best {sm['best_pct']:+.2f}%{extra}")
    return v


def ex_top2_years(v):
    by = v.groupby(v.index.year).sum().sort_values(ascending=False)
    drop = set(by.head(2).index)
    keep = v[~v.index.year.isin(drop)]
    up, dn, n = rec(keep.values)
    return f"drop {sorted(drop)}: n={n} {up}-{dn} mean {100*keep.mean():+.3f}%"


wed = nyse[(nyse.weekday == 2) & (~nyse.isin(dec))]

for tk in ["UUP", "DX-Y.NYB"]:
    print(f"\n================ {tk} ================")
    for form in ["H1_FROM_CLOSE", "MOO", "MOC"]:
        print(f"-- form {form}")
        for h in HS:
            s = leg(tk, form, h)
            print(f" h={h}")
            v_all = line("all decisions", s, dec, ctrl=s.reindex(wed))
            if len(v_all):
                pre, post = v_all[v_all.index < ERA], v_all[v_all.index >= ERA]
                print(f"     era: pre-2018 {rec(pre.values)[:2]} {100*pre.mean():+.3f}% | 2018+ {rec(post.values)[:2]} {100*post.mean():+.3f}%")
                print(f"     haircut: {ex_top2_years(v_all)}")
                mid = v_all[v_all.index.year % 4 == 2]
                print(f"     midterm years: {rec(mid.values)[:2]} {100*mid.mean():+.3f}%")
            v_c = line("S&P down & DXY>=+0.5%", s, conj, ctrl=s.reindex(wed))
            if len(v_c) and h == 1:
                print("     episodes:", [(str(d.date()), round(100 * x, 2)) for d, x in v_c.items()])

# the second candidate: natgas the Thursday before monthly opex, open->close tomorrow
print("\n================ NG=F Thursday-before-opex, MOO->MOC same session ================")
opx = load_events(["opex"])
opx_d = pd.DatetimeIndex([d for d in pd.to_datetime(opx["date"].unique()) if d in nyse and d <= ASOF + pd.Timedelta(days=5)])
thu = []
for d in opx_d:
    i = nyse.searchsorted(d)
    if i - 1 >= 0 and nyse[i - 1] < ASOF and nyse[i - 1].weekday() == 3:
        thu.append(nyse[i - 1])
thu = pd.DatetimeIndex(thu)
print("thursdays before opex:", len(thu))
ng_oc = C["NG=F"] / O["NG=F"] - 1
ng_cc = C["NG=F"] / C["NG=F"].shift(1) - 1
other_thu = nyse[(nyse.weekday == 3) & (~nyse.isin(thu))]
v = line("close->close (the brief's cell)", ng_cc, thu, ctrl=ng_cc.reindex(other_thu))
print("   ", ex_top2_years(-v) if len(v) else "")
v = line("open->close (MOO short form)", ng_oc, thu, ctrl=ng_oc.reindex(other_thu))
if len(v):
    pre, post = v[v.index < ERA], v[v.index >= ERA]
    print(f"     era: pre-2018 {rec(pre.values)[:2]} {100*pre.mean():+.3f}% | 2018+ {rec(post.values)[:2]} {100*post.mean():+.3f}%")
    print("     haircut (short side, drop 2 best years for the short):", ex_top2_years(-v))
    s22 = v[v.index >= "2022-01-01"]
    print(f"     2022+: {rec(s22.values)[:2]} {100*s22.mean():+.3f}%")

# frozen levels for the idea spec
for tk in ["UUP", "NG=F"]:
    df = px[tk][px[tk].index <= ASOF]
    hi, lo, cl = df["High"].astype(float), df["Low"].astype(float), df["Close"].astype(float)
    tr = pd.concat([hi - lo, (hi - cl.shift(1)).abs(), (lo - cl.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    print(f"\nFROZEN {tk}: close {cl.iloc[-1]:.4f} on {cl.index[-1].date()}  wilder14 ATR {atr.iloc[-1]:.4f} ({100*atr.iloc[-1]/cl.iloc[-1]:.2f}%)")
