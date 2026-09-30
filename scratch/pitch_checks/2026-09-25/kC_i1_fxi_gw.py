"""I1 round 1: long FXI from QE-3 across China's Golden Week to the first
post-reopen close. The INVERSION of the 2026-09-18 short (kB_c10_r1.py), so it
owes the flip charge.

Window: A = last US session <= Sep 30. Entry close A-3 (2026: 09-25). Exit close
of the first US session on/after the mainland reopen date (2026: 10-08, 9 sessions).
Mainland reopen dates are a HAND TABLE from SSE holiday notices (memory; no China
mainland series in master_prices). A uniform exit (A+6) is run as a check on it.

Pre-specified mechanism: National Day policy-announcement season plus the absence
of southbound selling during the Stock Connect suspension leave HK-listed China
names to US/HK holders who add into the reopening. Southbound exists from
2014-11-17, so the suspension leg predicts 2015+ >> 2005-2014.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

REOPEN = {2005: "2005-10-10", 2006: "2006-10-09", 2007: "2007-10-08", 2008: "2008-10-06",
          2009: "2009-10-09", 2010: "2010-10-08", 2011: "2011-10-10", 2012: "2012-10-08",
          2013: "2013-10-08", 2014: "2014-10-08", 2015: "2015-10-08", 2016: "2016-10-10",
          2017: "2017-10-09", 2018: "2018-10-08", 2019: "2019-10-08", 2020: "2020-10-09",
          2021: "2021-10-08", 2022: "2022-10-10", 2023: "2023-10-09", 2024: "2024-10-08",
          2025: "2025-10-09", 2026: "2026-10-08"}

TK = ["FXI", "EEM", "KWEB", "SPY", "^HSI"]
P = load_prices(TK)
cal = P["SPY"].index
C = pd.DataFrame({t: P[t]["Close"] for t in TK if t != "^HSI"}).reindex(cal)
R = C.pct_change(fill_method=None)
HSI = P["^HSI"]["Close"].dropna()
hcal = HSI.index


def st(v, label):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    r = summarize(v, label)
    if r["n"]:
        w = int((v > 0).sum())
        r["rec"] = f"{w}-{len(v)-w}"
        r["p_up"] = round(sign_test(w, len(v)), 4)
        if len(v) >= 3:
            r["boot_p"] = round(bootstrap_p_le0(v), 3)
    return r


def beta_at(t, b, p, n=252):
    y, x = R[t].iloc[max(0, p - n):p], R[b].iloc[max(0, p - n):p]
    m = y.notna() & x.notna()
    if m.sum() < 120:
        return np.nan
    return np.cov(y[m], x[m])[0, 1] / x[m].var()


def w(t, p0, p1):
    if p0 < 0 or p1 >= len(cal):
        return np.nan
    a, b = C[t].iloc[p0], C[t].iloc[p1]
    return b / a - 1 if (a == a and b == b) else np.nan


def last_le(idx, d):
    return int(idx.searchsorted(pd.Timestamp(d), side="right")) - 1


rows = []
for y, ro in REOPEN.items():
    a = last_le(cal, f"{y}-09-30")
    x = int(cal.searchsorted(pd.Timestamp(ro)))
    e = a - 3
    if y == 2026 or np.isnan(C["FXI"].iloc[e]):
        continue
    b = beta_at("FXI", "EEM", e)
    f, m = w("FXI", e, x), w("EEM", e, x)
    rows.append({"year": y, "entry": cal[e].date(), "exit": cal[x].date(), "h": x - e, "beta": b,
                 "FXI": f, "EEM": m, "RES": f - b * m, "FXI_pre": w("FXI", e, a), "FXI_post": w("FXI", a, x),
                 "FXI_A6": w("FXI", e, a + 6), "KWEB": w("KWEB", e, x), "SPY": w("SPY", e, x)})
G = pd.DataFrame(rows).set_index("year")
pd.set_option("display.width", 250)
D = G.copy()
for c in ["FXI", "EEM", "RES", "FXI_pre", "FXI_post", "FXI_A6", "KWEB", "SPY"]:
    D[c] = (100 * D[c]).round(2)
D["beta"] = D["beta"].round(2)
print("1. per-year window QE-3 close -> first US close on/after mainland reopen (pct)")
print(D.to_string())

own = {h: fwd_ret(C["FXI"], h).dropna() for h in (6, 7, 8, 9, 10)}
own_res = []
cells = []
for era, m in (("all", G.index >= 0), ("2005-2014", G.index <= 2014), ("2015+", G.index >= 2015),
               ("all ex2024", G.index != 2024), ("2015+ ex2024", (G.index >= 2015) & (G.index != 2024)),
               ("2018+", G.index >= 2018)):
    for c in ("FXI", "EEM", "RES", "FXI_pre", "FXI_post", "FXI_A6", "KWEB"):
        cells.append(st(G.loc[m, c].values, f"{c} {era}"))
show(cells, "2. cells (FXI raw, EEM, FXI residual vs beta-EEM; pre = A-3..A, post = A..reopen)")
hm = int(round(G.h.mean()))
print(f"\n  mean window length {G.h.mean():.2f} sessions; FXI own drift over {hm} sessions, all days "
      f"{100*own[hm].mean():+.3f}% (hit {100*(own[hm] > 0).mean():.1f}%), 2015+ "
      f"{100*own[hm][own[hm].index >= '2015-01-01'].mean():+.3f}%")

# local control: every same-length window starting within +/-126 td of each entry, ex the window itself
lc = []
for y, r in G.iterrows():
    e = cal.get_loc(pd.Timestamp(r.entry))
    h = int(r.h)
    for q in range(max(0, e - 126), min(len(cal) - h - 1, e + 127)):
        if abs(q - e) <= h:
            continue
        lc.append({"year": y, "v": w("FXI", q, q + h)})
LC = pd.DataFrame(lc)
print(f"  local +/-126td control (same length, ex window): FXI {100*LC.v.mean():+.3f}% "
      f"(hit {100*(LC.v > 0).mean():.1f}%, N={len(LC)}); per-year local mean vs window:")
lm = LC.groupby("year").v.mean()
print("   ", " ".join(f"{y}:{100*(G.FXI[y]-lm[y]):+.1f}" for y in G.index))
ex = (G.FXI - lm).values
print(f"  window minus its own local mean: {st(ex, 'excess')}")

# 3. placebo offset ladder, whole window shifted k sessions
print("\n3. placebo offset ladder (entry e+k, exit x+k), FXI raw and residual")
for era, yrs in (("all", list(G.index)), ("2015+", [y for y in G.index if y >= 2015]),
                 ("all ex2024", [y for y in G.index if y != 2024]),
                 ("2015+ ex2024", [y for y in G.index if y >= 2015 and y != 2024])):
    for pre in ("FXI", "RES"):
        res = []
        for k in range(-10, 11):
            v = []
            for y in yrs:
                e = cal.get_loc(pd.Timestamp(G.entry[y])) + k
                x = cal.get_loc(pd.Timestamp(G.exit[y])) + k
                f = w("FXI", e, x)
                if pre == "RES":
                    f = f - beta_at("FXI", "EEM", e) * w("EEM", e, x)
                v.append(f)
            res.append((k, 100 * np.nanmean(v)))
        m0 = dict(res)[0]
        rk = 1 + sum(1 for _, m in res if m > m0)
        print(f"  {pre} {era:13s}: k=0 {m0:+.2f}%  rank {rk} of 21 from the TOP; ladder ex-0 mean "
              f"{np.mean([m for k, m in res if k != 0]):+.2f}%  | " +
              " ".join(f"{k:+d}:{m:+.1f}" for k, m in res))

# 4. the same shape at the other quarter-ends (QE-3 -> QE+6): is it a quarter-turn effect?
print("\n4. same shape at other quarter ends (entry QE-3, exit QE+6), FXI, 2005-2025")
qrows = []
for mo in (3, 6, 9, 12):
    v, rv = [], []
    for y in range(2005, 2026):
        a = last_le(cal, pd.Timestamp(y, mo, 1) + pd.offsets.MonthEnd(0))
        f = w("FXI", a - 3, a + 6)
        if f != f:
            continue
        v.append(f)
        rv.append(f - beta_at("FXI", "EEM", a - 3) * w("EEM", a - 3, a + 6))
    r = st(v, f"QE month {mo:2d}")
    r["res_pct"] = round(100 * np.nanmean(rv), 3)
    qrows.append(r)
show(qrows)

# 5. ^HSI on its own calendar (2000+): last HK session <= Sep 30 minus 3 -> first HK session >= reopen
print("\n5. ^HSI own calendar, entry = 3 HK sessions before the last HK session <= Sep 30")
REOPEN_EARLY = {2000: "2000-10-09", 2001: "2001-10-08", 2002: "2002-10-08", 2003: "2003-10-08",
                2004: "2004-10-08"}
hr = []
for y, ro in {**REOPEN_EARLY, **REOPEN}.items():
    if y == 2026:
        continue
    a = last_le(hcal, f"{y}-09-30")
    x = int(hcal.searchsorted(pd.Timestamp(ro)))
    hr.append({"year": y, "v": HSI.iloc[x] / HSI.iloc[a - 3] - 1, "h": x - a + 3})
H = pd.DataFrame(hr).set_index("year")
print("   ", " ".join(f"{y}:{100*v:+.1f}" for y, v in H.v.items()))
hd = (HSI.shift(-int(round(H.h.mean()))) / HSI - 1).dropna()
show([st(H.v.values, "HSI all 2000-2025"), st(H.v[H.index <= 2014].values, "HSI 2000-2014"),
      st(H.v[H.index >= 2015].values, "HSI 2015+"), st(H.v[(H.index >= 2015) & (H.index != 2024)].values,
                                                       "HSI 2015+ ex2024"),
      st(H.v[H.index != 2024].values, "HSI all ex2024"),
      {"label": f"HSI own {int(round(H.h.mean()))}-session drift", "mean_pct": 100 * hd.mean(), "n": len(hd)}])

print("\nconcentration FXI all:", cluster_note(pd.DatetimeIndex([pd.Timestamp(e) for e in G.entry]),
                                             G.FXI.values))
print("concentration RES all:", cluster_note(pd.DatetimeIndex([pd.Timestamp(e) for e in G.entry]),
                                             G.RES.values))
live_e = len(cal) - 1  # 09-24 is the last bar; entry 09-25
print(f"\nLIVE: FXI 09-24 close {C['FXI'].iloc[-1]:.2f}; beta(FXI,EEM) trailing 252 {beta_at('FXI','EEM', live_e+1):.3f}")
fx = P["FXI"]
atr = wilder_atr(fx["High"], fx["Low"], fx["Close"], 14)
print(f"FXI Wilder-14 ATR {atr[-1]:.3f} = {100*atr[-1]/fx['Close'].iloc[-1]:.2f}%")
