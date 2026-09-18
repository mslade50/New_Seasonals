"""c10 KILL CHECK round 1 - FXI into and across the China Golden Week closure.

Pre-specified mechanism (surface map section 6): mainland China closes for
National Day from Oct 1 and Stock Connect SOUTHBOUND (mainland buyers of HK
stocks, which FXI holds) is suspended across it. Claims:
  (a) mainland investors de-risk into the holiday -> FXI WEAK over the run-in
      to the last pre-holiday session (A = last US session <= Sep 30)
  (b) the southbound bid vanishes during the suspension -> FXI WEAK across it
Southbound only exists since Nov 17 2014, so the mechanism PREDICTS a 2015+
effect absent/weaker in 2005-2014. That era split is the mechanism test.

Live: A = 2026-09-30; entry MOC today 09-18 = A-8 (signal D = 09-17 = A-9);
h=8 exits at A, h=10 exits at A+2 (10-02, NFP).

Windows (all close-to-close, lag-1 from D):
  RI8   entry A-8 -> exit A        (the pitchable run-in)
  RI10  entry A-8 -> exit A+2      (the h=10 limit)
  POST5 entry A   -> exit A+5      (across the closure; pitchable only on 09-30)
Controls: own drift; month-end run-ins in every other month (quarter-ends
separately, the QE confound); EEM on the same windows and the FXI residual at a
trailing-252d beta; placebo offset ladder k=-5..+5; 2024 with/without; ^HSI on
its own calendar back to 2000; Lunar New Year pooled from ^HSI's own closure gaps.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

TK = ["FXI", "EEM", "KWEB", "EWT", "EWY", "SPY", "^HSI"]
P = load_prices(TK)
cal = P["SPY"].index
C = pd.DataFrame({t: P[t]["Close"] for t in TK if t != "^HSI"}).reindex(cal)
HSI = P["^HSI"]["Close"].dropna()
hcal = HSI.index
pos = pd.Series(range(len(cal)), index=cal)
R = C.pct_change()


def st(v, label):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    r = summarize(v, label)
    if r["n"]:
        w = int((v > 0).sum())
        r["rec"] = f"{w}-{len(v)-w}"
        r["p_up"] = round(sign_test(w, len(v)), 4)
        r["p_dn"] = round(sign_test(len(v) - w, len(v)), 4)
    return r


def last_session_le(idx, d):
    loc = int(idx.searchsorted(d, side="right")) - 1
    return loc


def win(t, a, lo, hi):
    """close-to-close return of ticker t from pos a+lo to a+hi (cal positions)."""
    if a + lo < 0 or a + hi >= len(cal):
        return np.nan
    x0, x1 = C[t].iloc[a + lo], C[t].iloc[a + hi]
    return x1 / x0 - 1 if (x0 == x0 and x1 == x1) else np.nan


def beta_at(t, b, p, n=252):
    y = R[t].iloc[max(0, p - n):p]
    x = R[b].iloc[max(0, p - n):p]
    m = y.notna() & x.notna()
    if m.sum() < 120:
        return np.nan
    return np.cov(y[m], x[m])[0, 1] / x[m].var()


YEARS = range(2000, 2026)
GW = {}
for y in YEARS:
    a = last_session_le(cal, pd.Timestamp(y, 9, 30))
    GW[y] = a

# ------------------------------------------------------------ 1. Golden Week windows, FXI/EEM/resid
rows = []
for y, a in GW.items():
    if np.isnan(C["FXI"].iloc[a - 8]) if a - 8 >= 0 else True:
        continue
    bF = beta_at("FXI", "EEM", a - 8)
    r = {"year": y, "A": cal[a].date()}
    for name, lo, hi in (("RI8", -8, 0), ("RI10", -8, 2), ("POST5", 0, 5)):
        f, e = win("FXI", a, lo, hi), win("EEM", a, lo, hi)
        r[f"FXI_{name}"] = f
        r[f"EEM_{name}"] = e
        r[f"RES_{name}"] = f - bF * e if bF == bF else np.nan
    r["beta"] = bF
    r["KWEB_RI8"] = win("KWEB", a, -8, 0)
    r["EWT_RI8"] = win("EWT", a, -8, 0)
    r["EWY_RI8"] = win("EWY", a, -8, 0)
    rows.append(r)
G = pd.DataFrame(rows).set_index("year")
pd.set_option("display.width", 250)
print("1. per-year Golden Week windows (pct)")
cols = ["A", "beta", "FXI_RI8", "EEM_RI8", "RES_RI8", "FXI_RI10", "RES_RI10", "FXI_POST5", "EEM_POST5",
        "RES_POST5", "KWEB_RI8", "EWT_RI8", "EWY_RI8"]
disp = G[cols].copy()
for c in cols[2:]:
    disp[c] = (100 * disp[c]).round(2)
disp["beta"] = disp["beta"].round(2)
print(disp.to_string())

drift = {h: fwd_lag(C["FXI"], h, 1).dropna() for h in (5, 8, 10)}
out = []
for name, h in (("RI8", 8), ("RI10", 10), ("POST5", 5)):
    for era, m in (("all", G.index >= 0), ("2005-2014", G.index <= 2014), ("2015+", G.index >= 2015),
                   ("2015+ ex2024", (G.index >= 2015) & (G.index != 2024)), ("all ex2024", G.index != 2024)):
        for pre in ("FXI", "EEM", "RES"):
            out.append(st(G.loc[m, f"{pre}_{name}"].values, f"{pre} {name} {era}"))
    out.append({"label": f"FXI own drift h={h} all days", "mean_pct": 100 * drift[h].mean(), "n": len(drift[h])})
show(out, "2. Golden Week cells (p_up / p_dn = one-sided sign tests for each direction)")

# ------------------------------------------------------------ 3. offset ladder (FXI RI8 and RES RI8)
print("\n3. placebo offset ladder: whole window shifted k sessions (entry A-8+k, exit A+k)")
for era, yrs in (("all", list(G.index)), ("2015+", [y for y in G.index if y >= 2015]),
                 ("2015+ ex2024", [y for y in G.index if y >= 2015 and y != 2024])):
    for pre in ("FXI", "RES"):
        res = []
        for k in range(-5, 6):
            v = []
            for y in yrs:
                a = GW[y] + k
                f, e = win("FXI", a, -8, 0), win("EEM", a, -8, 0)
                bF = beta_at("FXI", "EEM", a - 8)
                v.append(f if pre == "FXI" else f - bF * e)
            res.append((k, 100 * np.nanmean(v)))
        m0 = res[5][1]
        rk_lo = 1 + sum(1 for _, m in res if m < m0)
        print(f"  {pre} {era:13s}: " + " ".join(f"k{k:+d}:{m:+.2f}" for k, m in res)
              + f"  -> k=0 rank {rk_lo} of 11 from the MOST NEGATIVE")

# ------------------------------------------------------------ 4. month-end run-ins other months (QE confound)
print("\n4. same RI8 window ending on the last session of EVERY month, FXI and residual, by month (2005+)")
rows = []
for m in range(1, 13):
    fv, rv = [], []
    for y in range(2005, 2026):
        if y == 2026:
            continue
        me = pd.Timestamp(y, m, 1) + pd.offsets.MonthEnd(0)
        a = last_session_le(cal, me)
        if a - 8 < 0 or np.isnan(C["FXI"].iloc[a - 8]):
            continue
        f, e = win("FXI", a, -8, 0), win("EEM", a, -8, 0)
        bF = beta_at("FXI", "EEM", a - 8)
        fv.append(f)
        rv.append(f - bF * e)
    r = st(fv, f"month {m:2d} FXI RI8")
    r["res_pct"] = round(100 * np.nanmean(rv), 3)
    r["res_rec"] = f"{int((np.array(rv) > 0).sum())}-{int((np.array(rv) <= 0).sum())}"
    rows.append(r)
show(rows)

# ------------------------------------------------------------ 5. ^HSI own calendar back to 2000
print("\n5. ^HSI on its own calendar: last HK session <= Sep 29 (Oct 1 is a HK holiday too)")
hrows = []
for y in range(2000, 2026):
    a = last_session_le(hcal, pd.Timestamp(y, 9, 30))
    if a - 8 < 0 or a + 5 >= len(hcal):
        continue
    hrows.append({"year": y, "A": hcal[a].date(),
                  "RI8": HSI.iloc[a] / HSI.iloc[a - 8] - 1,
                  "POST5": HSI.iloc[a + 5] / HSI.iloc[a] - 1})
H = pd.DataFrame(hrows).set_index("year")
hd8 = (HSI.shift(-8) / HSI - 1).dropna()
show([st(H.RI8.values, "HSI RI8 all 2000-2025"), st(H.loc[H.index <= 2014, "RI8"].values, "HSI RI8 2000-2014"),
      st(H.loc[H.index >= 2015, "RI8"].values, "HSI RI8 2015+"),
      st(H.loc[(H.index >= 2015) & (H.index != 2024), "RI8"].values, "HSI RI8 2015+ ex2024"),
      st(H.POST5.values, "HSI POST5 all"), st(H.loc[H.index >= 2015, "POST5"].values, "HSI POST5 2015+"),
      {"label": "HSI own 8-session drift", "mean_pct": 100 * hd8.mean(), "n": len(hd8)}])

# ------------------------------------------------------------ 6. Lunar New Year from HSI gaps
print("\n6. Lunar New Year closures detected from ^HSI gaps (>=2 consecutive missing weekdays, Jan 15-Feb 28)")
wd = pd.bdate_range("2000-01-01", hcal[-1])
missing = wd.difference(hcal)
lny = []
grp = []
for d in missing:
    if not ((d.month == 1 and d.day >= 15) or d.month == 2):
        continue
    if grp and (d - grp[-1]).days <= 3:
        grp.append(d)
    else:
        if len(grp) >= 2:
            lny.append(grp)
        grp = [d]
if len(grp) >= 2:
    lny.append(grp)
lrows = []
for g in lny:
    first = g[0]
    a = int(cal.searchsorted(first)) - 1  # last US session strictly before the first HK holiday
    if a - 8 < 0 or a + 5 >= len(cal) or np.isnan(C["FXI"].iloc[a - 8]):
        print(f"  {first.date()} (gap {len(g)} wd) -> FXI n/a")
        continue
    bF = beta_at("FXI", "EEM", a - 8)
    f, e = win("FXI", a, -8, 0), win("EEM", a, -8, 0)
    lrows.append({"first_gap": first.date(), "gap_wd": len(g), "A": cal[a].date(), "FXI_RI8": f,
                  "RES_RI8": f - bF * e, "FXI_POST5": win("FXI", a, 0, 5)})
    print(f"  HK closed {', '.join(str(x.date()) for x in g)} -> A {cal[a].date()} "
          f"FXI RI8 {100*f:+.2f}% res {100*(f-bF*e):+.2f}%")
L = pd.DataFrame(lrows)
L["year"] = [d.year for d in L.first_gap]
show([st(L.FXI_RI8.values, "LNY FXI RI8 all"), st(L.RES_RI8.values, "LNY RES RI8 all"),
      st(L.loc[L.year >= 2015, "FXI_RI8"].values, "LNY FXI RI8 2015+"),
      st(L.loc[L.year >= 2015, "RES_RI8"].values, "LNY RES RI8 2015+"),
      st(L.loc[L.year <= 2014, "FXI_RI8"].values, "LNY FXI RI8 2005-2014"),
      st(L.FXI_POST5.values, "LNY FXI POST5 all")])

# pooled GW + LNY residual, 2015+
pool = np.concatenate([G.loc[G.index >= 2015, "RES_RI8"].values, L.loc[L.year >= 2015, "RES_RI8"].values])
show([st(pool, "POOLED GW+LNY residual RI8 2015+")])

# ------------------------------------------------------------ live
d = pd.Timestamp("2026-09-17")
print(f"\nLIVE 2026-09-17: FXI {C['FXI'][d]:.2f}, beta(FXI,EEM) trailing 252 at 09-18 entry "
      f"{beta_at('FXI','EEM', pos[pd.Timestamp('2026-09-18')] if pd.Timestamp('2026-09-18') in pos else len(cal)):.3f}")
