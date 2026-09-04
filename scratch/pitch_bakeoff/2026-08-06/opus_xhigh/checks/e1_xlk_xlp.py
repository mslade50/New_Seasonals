"""E1 - Tech-over-defensive 5-day blowout (relative_value).

Cell: XLK 5d return minus XLP 5d return >= trailing-756d 97.5th pctile
      (and a fixed >= +10pp variant). Same for XLK minus SPY.
Measure: forward LONG-XLK / SHORT-XLP spread at 3/5/10 sessions on BOTH the
signal-close basis and the executable MOO basis (enter next open, exit MOC).

Adversarial brief: sign is unknown a priori; report whatever the data says.
Kill vectors: era concentration (2000-02 dot-com unwind, 2020 COVID),
episode clustering, SPY beta contamination, LOYO, XLK-below-52w-high split.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import _common as C

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

HORIZONS = [3, 5, 10]


# ----------------------------------------------------------------- helpers
def hdr(s: str) -> None:
    print("\n" + "=" * 100)
    print(s)
    print("=" * 100)


def episodes(dates: pd.DatetimeIndex, vals: np.ndarray, gap_td: int):
    d = pd.DatetimeIndex(dates)
    v = np.asarray(vals, dtype=float)
    if len(d) == 0:
        return d, v
    m = C.declusterize(d, gap_td=gap_td).astype(bool)
    return d[m], v[m]


def loyo_floor(dates, vals):
    d = pd.DatetimeIndex(dates)
    v = np.asarray(vals, dtype=float)
    ok = np.isfinite(v)
    d, v = d[ok], v[ok]
    yrs = sorted(set(d.year))
    rows = []
    for y in yrs:
        m = d.year != y
        if m.sum() < 3:
            continue
        rows.append((y, int(m.sum()), round(float(v[m].mean()), 3),
                     round(C.tstat(v[m]), 2)))
    if not rows:
        return float("nan"), rows
    return min(r[3] for r in rows), rows


def per_year(dates, vals):
    s = pd.Series(np.asarray(vals, float), index=pd.DatetimeIndex(dates)).dropna()
    if s.empty:
        return pd.DataFrame()
    g = s.groupby(s.index.year)
    return pd.DataFrame({"n": g.size(), "avg": g.mean().round(3),
                         "sum": g.sum().round(2), "worst": g.min().round(2)})


def full_report(label, dates, vals, baseline, do_loyo=True, do_year=True):
    d = pd.DatetimeIndex(dates)
    v = np.asarray(vals, dtype=float)
    ok = np.isfinite(v)
    d, v = d[ok], v[ok]
    rows = [C.describe(label + " | signal-days", v, baseline)]
    for g in (10, 21):
        ed, ev = episodes(d, v, g)
        rows.append(C.describe(f"{label} | episodes gap{g}", ev, baseline))
    C.show(rows)
    if len(d) == 0:
        return
    ed10, ev10 = episodes(d, v, 10)
    if do_loyo and len(ed10) >= 6:
        floor, tbl = loyo_floor(ed10, ev10)
        print(f"  LOYO (episodes gap10) floor t = {floor}")
        print("   " + " ".join(f"{y}:{t}" for y, _, _, t in tbl))
    print("  era split (signal-days, cut 2018):")
    C.show(C.era_split(d, v))
    if do_year:
        py = per_year(d, v)
        if not py.empty:
            print("  per-year (signal-days):")
            print(py.to_string())


# ----------------------------------------------------------------- data
hdr("E1.0  DATA")
px = C.load(["XLK", "XLP", "XLU", "SPY", "QQQ"])
for t, df in px.items():
    print(f"  {t:5s} {df.index.min().date()} .. {df.index.max().date()}  n={len(df)}")

idx = px["XLK"].index.intersection(px["XLP"].index).intersection(px["SPY"].index)
xlk = px["XLK"].reindex(idx)
xlp = px["XLP"].reindex(idx)
xlu = px["XLU"].reindex(idx)
spy = px["SPY"].reindex(idx)
print(f"  common calendar: {len(idx)} bars {idx.min().date()} .. {idx.max().date()}")

r5_xlk = C.ret(xlk["Close"], 5)
r5_xlp = C.ret(xlp["Close"], 5)
r5_spy = C.ret(spy["Close"], 5)
r5_xlu = C.ret(xlu["Close"], 5)

spread_kp = r5_xlk - r5_xlp          # today +13.93
spread_ks = r5_xlk - r5_spy          # today +6.08

# 52w-high distance for XLK
xlk_52h = xlk["Close"].rolling(252, min_periods=120).max()
xlk_below = (xlk["Close"] / xlk_52h - 1.0) * 100.0   # today -6.09

print(f"\n  TODAY (2026-08-05 close):")
print(f"    XLK 5d {r5_xlk.iloc[-1]:+.2f}%  XLP 5d {r5_xlp.iloc[-1]:+.2f}%  "
      f"XLU 5d {r5_xlu.iloc[-1]:+.2f}%  SPY 5d {r5_spy.iloc[-1]:+.2f}%")
print(f"    XLK-XLP 5d spread {spread_kp.iloc[-1]:+.2f}pp | "
      f"XLK-SPY {spread_ks.iloc[-1]:+.2f}pp")
print(f"    XLK vs its 252d closing high: {xlk_below.iloc[-1]:+.2f}%")

# ----------------------------------------------------------------- forward returns
def leg_fwd_close(df, k):
    return C.fwd(df["Close"], k)


def leg_fwd_moo(df, k):
    return C.fwd_from_next_open(df, k)


FWD = {}
for k in HORIZONS:
    FWD[("close", k)] = {
        "kp": leg_fwd_close(xlk, k) - leg_fwd_close(xlp, k),
        "ks": leg_fwd_close(xlk, k) - leg_fwd_close(spy, k),
        "ku": leg_fwd_close(xlk, k) - leg_fwd_close(xlu, k),
        "spy": leg_fwd_close(spy, k),
        "xlk": leg_fwd_close(xlk, k),
        "xlp": leg_fwd_close(xlp, k),
    }
    FWD[("moo", k)] = {
        "kp": leg_fwd_moo(xlk, k) - leg_fwd_moo(xlp, k),
        "ks": leg_fwd_moo(xlk, k) - leg_fwd_moo(spy, k),
        "ku": leg_fwd_moo(xlk, k) - leg_fwd_moo(xlu, k),
        "spy": leg_fwd_moo(spy, k),
        "xlk": leg_fwd_moo(xlk, k),
        "xlp": leg_fwd_moo(xlp, k),
    }

# ----------------------------------------------------------------- triggers
thr975 = spread_kp.rolling(756, min_periods=252).quantile(0.975).shift(1)
thr975_ks = spread_ks.rolling(756, min_periods=252).quantile(0.975).shift(1)

trig = {
    "A_kp_pct975": (spread_kp >= thr975) & thr975.notna(),
    "B_kp_ge10pp": (spread_kp >= 10.0),
    "C_ks_pct975": (spread_ks >= thr975_ks) & thr975_ks.notna(),
    "D_ks_ge6pp":  (spread_ks >= 6.0),
}

hdr("E1.1  TRIGGER SANITY - does today fire, and how often historically?")
for name, m in trig.items():
    m = m.fillna(False)
    print(f"  {name:14s} n_days={int(m.sum()):5d}  rate={m.mean()*100:5.2f}%  "
          f"fires_today={bool(m.iloc[-1])}")
print(f"  trailing-756d 97.5 pctile of XLK-XLP spread today = {thr975.iloc[-1]:.2f}pp "
      f"(actual {spread_kp.iloc[-1]:.2f}pp)")
print(f"  trailing-756d 97.5 pctile of XLK-SPY spread today = {thr975_ks.iloc[-1]:.2f}pp "
      f"(actual {spread_ks.iloc[-1]:.2f}pp)")

# ----------------------------------------------------------------- main tables
hdr("E1.2  MAIN GRID - forward spread returns, MOO basis LEADS")
rows = []
for tname, m in trig.items():
    m = m.fillna(False)
    dts = idx[m.values]
    for basis in ("moo", "close"):
        for k in HORIZONS:
            for leg, legname in (("kp", "LONG XLK / SHORT XLP"),
                                 ("ks", "LONG XLK / SHORT SPY")):
                if tname.startswith(("A", "B")) and leg != "kp":
                    continue
                if tname.startswith(("C", "D")) and leg != "ks":
                    continue
                f = FWD[(basis, k)][leg]
                v = f.reindex(dts).to_numpy()
                base = f.dropna().to_numpy()
                d = C.describe(f"{tname} {basis} k={k} {legname}", v, base)
                ed10, ev10 = episodes(dts, v, 10)
                ed21, ev21 = episodes(dts, v, 21)
                d["ep10_n"] = int(np.isfinite(ev10).sum())
                d["ep10_t"] = round(C.tstat(ev10), 2)
                d["ep21_n"] = int(np.isfinite(ev21).sum())
                d["ep21_t"] = round(C.tstat(ev21), 2)
                rows.append(d)
C.show(rows)

# ----------------------------------------------------------------- deep dive on the best
hdr("E1.3  DEEP DIVE - trigger A (XLK-XLP >= trailing 97.5 pctile), MOO basis")
mA = trig["A_kp_pct975"].fillna(False)
dtsA = idx[mA.values]
print(f"  signal days: {len(dtsA)}  first {dtsA.min().date()} last {dtsA.max().date()}")
for k in HORIZONS:
    print(f"\n  --- horizon k={k} sessions, LONG XLK / SHORT XLP, MOO basis ---")
    f = FWD[("moo", k)]["kp"]
    full_report(f"A k={k}", dtsA, f.reindex(dtsA).to_numpy(), f.dropna().to_numpy())

hdr("E1.4  DEEP DIVE - trigger B (XLK-XLP >= +10pp fixed), MOO basis")
mB = trig["B_kp_ge10pp"].fillna(False)
dtsB = idx[mB.values]
print(f"  signal days: {len(dtsB)}")
if len(dtsB):
    print("  dates:", ", ".join(str(d.date()) for d in dtsB[:80]),
          "..." if len(dtsB) > 80 else "")
for k in HORIZONS:
    print(f"\n  --- horizon k={k}, LONG XLK / SHORT XLP, MOO basis ---")
    f = FWD[("moo", k)]["kp"]
    full_report(f"B k={k}", dtsB, f.reindex(dtsB).to_numpy(), f.dropna().to_numpy())

# ----------------------------------------------------------------- era concentration
hdr("E1.5  ERA CONCENTRATION - is the whole cell 2000-2002 or 2020?")
for tname in ("A_kp_pct975", "B_kp_ge10pp"):
    m = trig[tname].fillna(False)
    dts = idx[m.values]
    print(f"\n  {tname}: signal-day counts by year")
    cnt = pd.Series(1, index=dts).groupby(dts.year).sum()
    print("   " + "  ".join(f"{y}:{int(n)}" for y, n in cnt.items()))
    for k in (5,):
        f = FWD[("moo", k)]["kp"]
        v = f.reindex(dts)
        subsets = {
            "ALL": v,
            "excl 2000-2002": v[(v.index.year < 2000) | (v.index.year > 2002)],
            "excl 2020": v[v.index.year != 2020],
            "excl 2000-02 AND 2020": v[((v.index.year < 2000) | (v.index.year > 2002))
                                       & (v.index.year != 2020)],
            "2003-2017": v[(v.index.year >= 2003) & (v.index.year <= 2017)],
            "2018+": v[v.index.year >= 2018],
        }
        rr = []
        for nm, s in subsets.items():
            s = s.dropna()
            d = C.describe(f"{tname} k={k} {nm}", s.to_numpy())
            ed, ev = episodes(s.index, s.to_numpy(), 10)
            d["ep10_n"] = len(ev)
            d["ep10_t"] = round(C.tstat(ev), 2)
            rr.append(d)
        C.show(rr)

# ----------------------------------------------------------------- beta
hdr("E1.6  BETA - the XLK/XLP pair is NOT market neutral")
dr_xlk = xlk["Close"].pct_change()
dr_xlp = xlp["Close"].pct_change()
dr_spy = spy["Close"].pct_change()
cov_kp = dr_xlk.rolling(252).cov(dr_spy)
cov_pp = dr_xlp.rolling(252).cov(dr_spy)
var_s = dr_spy.rolling(252).var()
beta_xlk = cov_kp / var_s
beta_xlp = cov_pp / var_s
net_beta = beta_xlk - beta_xlp
print(f"  today: beta(XLK)={beta_xlk.iloc[-1]:.2f}  beta(XLP)={beta_xlp.iloc[-1]:.2f}  "
      f"NET beta of $1/$1 spread = {net_beta.iloc[-1]:.2f}")

for tname in ("A_kp_pct975", "B_kp_ge10pp"):
    m = trig[tname].fillna(False)
    dts = idx[m.values]
    print(f"\n  {tname}: mean ex-ante net beta in cell = "
          f"{net_beta.reindex(dts).mean():.2f}")
    for k in HORIZONS:
        f = FWD[("moo", k)]["kp"].reindex(dts)
        fs = FWD[("moo", k)]["spy"].reindex(dts)
        nb = net_beta.reindex(dts)
        hedged = f - nb * fs
        ok = np.isfinite(f) & np.isfinite(fs)
        # in-cell regression alpha
        if ok.sum() > 5:
            X = np.column_stack([np.ones(ok.sum()), fs[ok].to_numpy()])
            y = f[ok].to_numpy()
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            resid = y - X @ coef
            dof = len(y) - 2
            se = np.sqrt((resid @ resid) / dof * np.linalg.inv(X.T @ X)[0, 0])
            a_t = coef[0] / se
        else:
            coef, a_t = [np.nan, np.nan], np.nan
        rr = [C.describe(f"{tname} k={k} RAW spread", f.to_numpy()),
              C.describe(f"{tname} k={k} EX-ANTE beta-hedged", hedged.to_numpy())]
        C.show(rr)
        print(f"    in-cell regression: alpha={coef[0]:+.3f}pp  t(alpha)={a_t:+.2f}  "
              f"realized beta to SPY={coef[1]:+.2f}")

# ----------------------------------------------------------------- 52w high split
hdr("E1.7  XLK BELOW 52w HIGH vs AT IT  (today XLK is -6.09% below)")
for tname in ("A_kp_pct975", "B_kp_ge10pp"):
    m = trig[tname].fillna(False)
    dts = idx[m.values]
    below = xlk_below.reindex(dts)
    for k in HORIZONS:
        f = FWD[("moo", k)]["kp"].reindex(dts)
        cells = {
            "XLK <= -3% below 52wh": f[below <= -3.0],
            "XLK -3%..0 (at highs)": f[below > -3.0],
            "XLK <= -5% below 52wh": f[below <= -5.0],
        }
        rr = []
        for nm, s in cells.items():
            s = s.dropna()
            d = C.describe(f"{tname} k={k} {nm}", s.to_numpy())
            ed, ev = episodes(s.index, s.to_numpy(), 10)
            d["ep10_n"] = len(ev)
            d["ep10_t"] = round(C.tstat(ev), 2)
            rr.append(d)
        C.show(rr)

# ----------------------------------------------------------------- today's nearest analogues
hdr("E1.8  NEAREST HISTORICAL ANALOGUES to today (spread>=10pp AND XLK<=-3% below 52wh)")
mask = (spread_kp >= 10.0) & (xlk_below <= -3.0)
dts = idx[mask.fillna(False).values]
print(f"  n={len(dts)}")
if len(dts):
    tab = pd.DataFrame({
        "spread_5d": spread_kp.reindex(dts).round(2),
        "xlk_below_52h": xlk_below.reindex(dts).round(2),
        "fwd3_moo": FWD[("moo", 3)]["kp"].reindex(dts).round(2),
        "fwd5_moo": FWD[("moo", 5)]["kp"].reindex(dts).round(2),
        "fwd10_moo": FWD[("moo", 10)]["kp"].reindex(dts).round(2),
        "spy_fwd5_moo": FWD[("moo", 5)]["spy"].reindex(dts).round(2),
    })
    print(tab.to_string())
    ed, ev = episodes(dts, FWD[("moo", 5)]["kp"].reindex(dts).to_numpy(), 10)
    print(f"\n  episodes gap10 n={len(ev)} avg={np.nanmean(ev):+.3f} t={C.tstat(ev):+.2f}")

# ----------------------------------------------------------------- XLK vs XLU too
hdr("E1.9  ROBUSTNESS - substitute XLU for XLP as the defensive leg")
r5_u = C.ret(xlu["Close"], 5)
sp_ku = r5_xlk - r5_u
thr_ku = sp_ku.rolling(756, min_periods=252).quantile(0.975).shift(1)
m = ((sp_ku >= thr_ku) & thr_ku.notna()).fillna(False)
dts = idx[m.values]
print(f"  today XLK-XLU 5d spread = {sp_ku.iloc[-1]:+.2f}pp  thr={thr_ku.iloc[-1]:.2f}  "
      f"fires={bool(m.iloc[-1])}   n_days={len(dts)}")
rr = []
for k in HORIZONS:
    f = FWD[("moo", k)]["ku"]
    v = f.reindex(dts).to_numpy()
    d = C.describe(f"XLK-XLU pct975 k={k} MOO", v, f.dropna().to_numpy())
    ed, ev = episodes(dts, v, 10)
    d["ep10_n"] = len(ev)
    d["ep10_t"] = round(C.tstat(ev), 2)
    rr.append(d)
C.show(rr)

# ----------------------------------------------------------------- leg decomposition
hdr("E1.10  LEG DECOMPOSITION - which leg carries it? (trigger A, MOO)")
mA = trig["A_kp_pct975"].fillna(False)
dtsA = idx[mA.values]
rr = []
for k in HORIZONS:
    for leg in ("xlk", "xlp", "spy"):
        f = FWD[("moo", k)][leg]
        rr.append(C.describe(f"A k={k} {leg.upper()} outright",
                             f.reindex(dtsA).to_numpy(), f.dropna().to_numpy()))
C.show(rr)

hdr("E1.END")
