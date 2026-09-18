"""c10 follow-up: the one row that looked like the mechanism's era prediction.

kB_c10_r1.py: ^HSI's own-calendar run-in into Sep 30 goes 2-8 (-2.030%) in
2015+ ex-2024 against 8-7 (+0.289%) in 2000-2014. Is that China, or EM beta?
Residual the HSI window against EEM over the SAME calendar dates (EEM close on
or before each HSI date), at a trailing-252 weekly-return beta (weekly to
dodge the HK/US close asynchrony). Also shorter run-ins on FXI (RI3, RI5).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

P = load_prices(["^HSI", "EEM", "FXI", "SPY"])
HSI = P["^HSI"]["Close"].dropna()
EEM = P["EEM"]["Close"].dropna()
FXI = P["FXI"]["Close"].dropna()
hcal = HSI.index


def asof(s, d):
    return s.loc[:d].iloc[-1] if len(s.loc[:d]) else np.nan


def wbeta(y, x, end, n=52):
    wy = y.loc[:end].resample("W-FRI").last().pct_change().dropna().iloc[-n:]
    wx = x.loc[:end].resample("W-FRI").last().pct_change().dropna().iloc[-n:]
    j = wy.index.intersection(wx.index)
    if len(j) < 30:
        return np.nan
    return np.cov(wy[j], wx[j])[0, 1] / wx[j].var()


rows = []
for y in range(2004, 2026):
    a = int(hcal.searchsorted(pd.Timestamp(y, 9, 30), side="right")) - 1
    d0, d1 = hcal[a - 8], hcal[a]
    h = HSI.iloc[a] / HSI.iloc[a - 8] - 1
    e = asof(EEM, d1) / asof(EEM, d0) - 1
    b = wbeta(HSI, EEM, d0)
    rows.append({"year": y, "d0": d0.date(), "d1": d1.date(), "HSI": h, "EEM": e, "beta": b,
                 "RES": h - b * e})
T = pd.DataFrame(rows).set_index("year")
print((T.assign(HSI=100 * T.HSI, EEM=100 * T.EEM, RES=100 * T.RES)).round(2).to_string())


def st(v, label):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    r = summarize(v, label)
    if r["n"]:
        w = int((v > 0).sum())
        r["rec"] = f"{w}-{len(v)-w}"
        r["p_dn"] = round(sign_test(len(v) - w, len(v)), 4)
    return r


m15 = T.index >= 2015
ex = m15 & (T.index != 2024)
show([st(T.RES[T.index <= 2014], "HSI resid 2004-2014"), st(T.RES[m15], "HSI resid 2015+"),
      st(T.RES[ex], "HSI resid 2015+ ex2024"), st(T.HSI[ex], "HSI raw 2015+ ex2024"),
      st(T.EEM[ex], "EEM same dates 2015+ ex2024")], "HSI run-in vs EEM, same calendar dates")

# FXI shorter run-ins into A (US calendar)
cal = P["SPY"].index
F = FXI.reindex(cal)
E = EEM.reindex(cal)
rows = []
for k in (3, 5, 8):
    for era, yrs in (("2005-2014", range(2005, 2015)), ("2015+ ex2024", [y for y in range(2015, 2026) if y != 2024])):
        v, r_ = [], []
        for y in yrs:
            a = int(cal.searchsorted(pd.Timestamp(y, 9, 30), side="right")) - 1
            f = F.iloc[a] / F.iloc[a - k] - 1
            e = E.iloc[a] / E.iloc[a - k] - 1
            v.append(f)
            r_.append(f - e)
        rr = st(v, f"FXI RI{k} {era}")
        rr["minus_EEM_pct"] = round(100 * np.mean(r_), 3)
        rows.append(rr)
show(rows, "FXI shorter run-ins (minus_EEM = beta-1 residual)")
