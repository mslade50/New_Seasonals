"""Posts check (2026-09-22): candidate cells for Tuesday night's queue.

Run date is Tuesday 2026-09-22 and that IS the freshest bar, so every price
cell anchors on the 09-22 close and the tradeable number is lag=1 (enter MOC
Wednesday 09-23, exit MOC h sessions later). lag=0 is printed alongside as the
naive contrast.

A. IDEA candidate, long XLU: XLU z10 <= -2.0 while SPY closes within 1% of its
   252-session closing high. Declustered 10.
B. ^VIX down >= 4% on a flat/down ^GSPC day with ^VIX < 16. Declustered 5.
C. QQQ 5-session return >= +6% while closing AT its 252-session high.
D. SPY minus TLT into quarter end, conditioned on a >= 8pp QTD gap.
E. The Wednesday after September quad witching, split by FOMC-on-that-day.
F. USO 5-session return <= -10% while >= 1.20x its 200-session SMA.
G. Open-idea marks (prices only).
H. Frozen parameters at the 2026-09-22 close (Wilder-14 ATR).

z10 here is the build_pitch_state._metrics_for definition (10-session return
over the 21-session daily-return sd scaled by sqrt(10)), NOT pitch_lab.zscore.

CACHE BOUND: master_prices.parquet starts 2000-01-03. TLT starts 2002
(so cell D runs from Q4 2002 and cell E's TLT column from 2002), USO starts
2006 (cell F, and its 200-SMA needs 200 more sessions of warm-up).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pitch_lab import (  # noqa: E402
    declusters, fwd_lag, fwd_ret, load_events, load_prices, local_control,
    rolling_on_valid, sign_test, summarize, wilder_atr,
)

ROOT = Path(__file__).resolve().parents[3]
ASOF = pd.Timestamp("2026-09-22")
ERA = "2018-01-01"
TK = ["XLU", "SPY", "QQQ", "TLT", "USO", "^VIX", "^GSPC", "UUP", "IWM", "XLE"]

raw_px = load_prices(TK)
nyse = raw_px["SPY"]["Close"].dropna().index
nyse = pd.DatetimeIndex(nyse[nyse <= ASOF])
C = {t: raw_px[t]["Close"].astype(float).reindex(nyse) for t in TK}
POS = pd.Series(range(len(nyse)), index=nyse)

print(f"NYSE calendar (SPY): {nyse[0].date()} .. {nyse[-1].date()}  n={len(nyse)}")
print("ASOF (anchor bar) =", ASOF.date(), "weekday", ASOF.day_name())
for t in TK:
    v = C[t].dropna()
    if len(v) == 0:
        print(f"  {t}: NO HISTORY in cache")
        continue
    print(f"  {t}: {v.index[0].date()} .. {v.index[-1].date()}  n={len(v)}  "
          f"last {v.iloc[-1]:.4f}")
print("\nNOTE: cache begins 2000-01-03; TLT from 2002, USO from 2006.")


# ---------------------------------------------------------------------------
# helpers (same shapes as 2026-09-21/01_cells.py)
# ---------------------------------------------------------------------------
SIGN_TEST_MAX_N = 2000  # the exact-rational p=0.5 path costs ~9s at n=6700


def stp(wins, n):
    """sign_test, or a normal approximation once the exact path gets slow."""
    if n <= SIGN_TEST_MAX_N:
        return f"{sign_test(wins, n):.4f}"
    if n <= 0:
        return "n/a"
    z = (wins - 0.5 - n / 2.0) / np.sqrt(n / 4.0)
    from math import erfc
    return f"{0.5 * erfc(z / np.sqrt(2)):.4f}~"  # ~ = normal approximation


def rec(v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    return int((v > 0).sum()), int((v < 0).sum()), len(v)


def line(label, s, dates, ctrls=()):
    v = s.reindex(pd.DatetimeIndex(dates)).dropna()
    if len(v) == 0:
        print(f"  {label}: n=0")
        return v
    up, dn, n = rec(v.values)
    sm = summarize(v.values)
    t = sm["t"]
    tt = f"{t:+.2f}" if t is not None and np.isfinite(t) else "n/a"
    print(f"  {label}: n={n} {up}-{dn} mean {sm['mean_pct']:+.3f}% "
          f"med {sm['median_pct']:+.3f}% hit {sm['hit']:.1f}% t {tt} "
          f"p_up {stp(up, n)} p_dn {stp(dn, n)} "
          f"worst {sm['worst_pct']:+.2f}% ({v.idxmin().date()}) "
          f"best {sm['best_pct']:+.2f}% ({v.idxmax().date()})")
    for cl, cs in ctrls:
        cv = pd.Series(cs).dropna()
        if len(cv) == 0:
            print(f"      {cl}: n=0")
            continue
        cu, cd, cn = rec(cv.values)
        print(f"      {cl}: n={cn} {cu}-{cd} mean {100*cv.mean():+.3f}% "
              f"med {100*float(np.median(cv.values)):+.3f}% "
              f"hit {100*(cv > 0).mean():.1f}% "
              f"p_up {stp(cu, cn)}")
    return v


def era(label, v):
    a, b = v[v.index < ERA], v[v.index >= ERA]

    def part(x):
        if len(x) == 0:
            return "n=0"
        u, d, n = rec(x.values)
        return (f"n={n} {u}-{d} mean {100*x.mean():+.3f}% "
                f"med {100*float(np.median(x.values)):+.3f}% "
                f"hit {100*(x > 0).mean():.1f}%")
    print(f"   {label} era: pre-2018 [{part(a)}] | 2018+ [{part(b)}]")


def episodes(dates, series):
    cols = list(series)
    print("   episode      " + "".join(f"{c:>12}" for c in cols))
    for d in dates:
        cells = []
        for c in cols:
            x = series[c].get(d, np.nan)
            cells.append("        n/a" if (x is None or
                                           (isinstance(x, float) and np.isnan(x)))
                         else f"{100*x:+11.2f}")
        print(f"   {d.date()}  " + "".join(f"{c:>12}" for c in cells))


def daily_ret(t):
    v = C[t].dropna()
    return (v / v.shift(1) - 1.0).reindex(nyse)


def nret(t, n):
    v = C[t].dropna()
    return (v / v.shift(n) - 1.0).reindex(nyse)


def sma(t, n):
    return rolling_on_valid(C[t], lambda x: x.rolling(n).mean())


def roll_max(t, n):
    return rolling_on_valid(C[t], lambda x: x.rolling(n).max())


def mask_dates(m):
    m = m.reindex(nyse)
    return pd.DatetimeIndex(nyse[m.fillna(False).to_numpy(dtype=bool)])


def z10_state(t):
    """build_pitch_state._metrics_for z10: ret10 / (sd21(daily) * sqrt(10))."""
    v = C[t].dropna()
    r10 = v.pct_change(10)
    vol21 = v.pct_change().rolling(21).std()
    return (r10 / (vol21 * np.sqrt(10))).reindex(nyse)


def cell_block(name, tkr, trig, comp, horizons, lags=(1, 0), extra_ctrl=None):
    loc = local_control(nyse, trig, 126)
    out = {}
    for h in horizons:
        print(f"\n  --- {name} {tkr} h={h} ---")
        for lag in lags:
            f = fwd_lag(C[tkr], h, lag) if lag else fwd_ret(C[tkr], h)
            ctr = [("ctrl all sessions", f.dropna()),
                   ("ctrl local +/-126td", f.reindex(loc).dropna())]
            if comp is not None:
                ctr.append(("ctrl complement", f.reindex(comp).dropna()))
            if extra_ctrl:
                for lab, idx in extra_ctrl:
                    ctr.append((lab, f.reindex(idx).dropna()))
            v = line(f"{tkr} h={h} lag={lag}", f, trig, ctr)
            if len(v) and lag == 1:
                era(f"{tkr} h={h} lag=1", v)
            if len(v) and lag == 0 and 1 not in lags:
                era(f"{tkr} h={h} lag=0", v)
            out[(h, lag)] = v
    return out


# ===========================================================================
# A. XLU z10 <= -2 while SPY within 1% of its 252d closing high
# ===========================================================================
print("\n" + "=" * 78)
print("=== A. XLU z10 <= -2.0 AND SPY close within 1% of its 252d closing high ===")
xz = z10_state("XLU")
sz = z10_state("SPY")
s_dist = C["SPY"] / roll_max("SPY", 252) - 1.0
print(f"  TODAY {ASOF.date()}: XLU z10 {xz.iloc[-1]:+.3f} (state says -2.26) | "
      f"SPY z10 {sz.iloc[-1]:+.3f} (state says +0.59) | "
      f"SPY {100*s_dist.iloc[-1]:+.2f}% from its 252d closing high")
mA = (xz <= -2.0) & (s_dist >= -0.01)
mA_c = (xz <= -2.0) & (s_dist < -0.01)
mA_u = (xz <= -2.0)
print(f"  today qualifies: XLU z10<=-2 {bool(mA_u.iloc[-1])} | SPY within 1% "
      f"{bool((s_dist >= -0.01).iloc[-1])} | CELL {bool(mA.iloc[-1])}")
rawA = mask_dates(mA)
trigA = declusters(rawA, 10, nyse)
trigA_c = declusters(mask_dates(mA_c), 10, nyse)
trigA_u = declusters(mask_dates(mA_u), 10, nyse)
print(f"  raw {len(rawA)} -> declustered(10) {len(trigA)} | complement "
      f"declustered {len(trigA_c)} | unconditional z10<=-2 declustered "
      f"{len(trigA_u)}")
print("  trigger dates: " + ", ".join(str(d.date()) for d in trigA))
cell_block("A", "XLU", trigA, trigA_c, (1, 5, 10, 21))

print("\n  --- XLU minus SPY spread, lag=1 ---")
locA = local_control(nyse, trigA, 126)
for h in (5, 10):
    sp = fwd_lag(C["XLU"], h, 1) - fwd_lag(C["SPY"], h, 1)
    v = line(f"XLU-SPY h={h} lag=1", sp, trigA,
             [("ctrl all sessions", sp.dropna()),
              ("ctrl local +/-126td", sp.reindex(locA).dropna()),
              ("ctrl complement", sp.reindex(trigA_c).dropna())])
    if len(v):
        era(f"XLU-SPY h={h}", v)
    s1 = fwd_lag(C["SPY"], h, 1)
    line(f"SPY alone h={h} lag=1 (same anchors)", s1, trigA,
         [("ctrl all sessions", s1.dropna())])

print("\n  --- unconditional XLU z10 <= -2.0 (declustered 10) ---")
for h in (5, 10):
    for lag in (1, 0):
        f = fwd_lag(C["XLU"], h, lag) if lag else fwd_ret(C["XLU"], h)
        v = line(f"XLU uncond h={h} lag={lag}", f, trigA_u,
                 [("ctrl all sessions", f.dropna())])
        if len(v) and lag == 1:
            era(f"XLU uncond h={h}", v)

print("\n  episode table, lag=1 (%):")
serA = {f"XLU_h{h}": fwd_lag(C["XLU"], h, 1) for h in (1, 5, 10, 21)}
serA["XLU-SPY_5"] = fwd_lag(C["XLU"], 5, 1) - fwd_lag(C["SPY"], 5, 1)
serA["XLU-SPY_10"] = fwd_lag(C["XLU"], 10, 1) - fwd_lag(C["SPY"], 10, 1)
episodes(trigA, serA)
print("  z10 / SPY dist on each episode:")
for d in trigA:
    print(f"   {d.date()}  XLU z10 {xz[d]:+.2f}  SPY dist {100*s_dist[d]:+.2f}%")

# ===========================================================================
# B. VIX crush on a flat/down S&P day with VIX < 16
# ===========================================================================
print("\n" + "=" * 78)
print("=== B. ^VIX 1d <= -4% AND ^GSPC 1d <= 0 AND ^VIX close < 16 ===")
vr1 = daily_ret("^VIX")
gr1 = daily_ret("^GSPC")
print(f"  TODAY {ASOF.date()}: ^VIX {100*vr1.iloc[-1]:+.2f}% close "
      f"{C['^VIX'].iloc[-1]:.2f} | ^GSPC {100*gr1.iloc[-1]:+.2f}%")
for thr in (-0.04, -0.03):
    print(f"\n  ##### threshold ^VIX 1d <= {100*thr:.1f}% #####")
    mB = (vr1 <= thr) & (gr1 <= 0.0) & (C["^VIX"] < 16)
    mB_c = (vr1 <= thr) & (gr1 <= 0.0) & (C["^VIX"] >= 16)
    print(f"  today qualifies: {bool(mB.iloc[-1])}")
    rawB = mask_dates(mB)
    trigB = declusters(rawB, 5, nyse)
    trigB_c = declusters(mask_dates(mB_c), 5, nyse)
    print(f"  raw {len(rawB)} -> declustered(5) {len(trigB)} | complement "
          f"(VIX >= 16) declustered {len(trigB_c)}")
    print("  trigger dates: " + ", ".join(str(d.date()) for d in trigB))
    resB = cell_block("B", "SPY", trigB, trigB_c, (1, 5, 10), lags=(0, 1))
    for lag in (0, 1):
        v = resB.get((5, lag))
        if v is not None and len(v):
            if lag == 0:
                era("SPY h=5 lag=0", v)
            tot = v.sum()
            top = v.reindex(v.abs().sort_values(ascending=False).index[:2])
            print(f"   SPY h=5 lag={lag}: sum of returns {100*tot:+.2f}pp; two "
                  f"largest-magnitude episodes: " + ", ".join(
                      f"{d.date()} {100*x:+.2f}%" for d, x in top.items()) +
                  f" -> {100*top.sum():+.2f}pp = "
                  f"{100*top.sum()/tot if tot else np.nan:.1f}% of the total; "
                  f"ex-those two: n={len(v)-2} mean "
                  f"{100*v.drop(top.index).mean():+.3f}% "
                  f"{rec(v.drop(top.index).values)[0]}-"
                  f"{rec(v.drop(top.index).values)[1]}")
    locB = local_control(nyse, trigB, 126)
    fv = fwd_ret(C["^VIX"], 5)
    v = line("^VIX h=5 lag=0", fv, trigB,
             [("ctrl all sessions", fv.dropna()),
              ("ctrl local +/-126td", fv.reindex(locB).dropna()),
              ("ctrl all VIX<16 sessions",
               fv.reindex(mask_dates(C["^VIX"] < 16)).dropna()),
              ("ctrl complement", fv.reindex(trigB_c).dropna())])
    if len(v):
        era("^VIX h=5 lag=0", v)
    if len(trigB) <= 40:
        print("\n  episode table (%):")
        serB = {"VIX_1d": vr1, "SPX_1d": gr1}
        serB |= {f"SPY_h{h}L0": fwd_ret(C["SPY"], h) for h in (1, 5, 10)}
        serB |= {f"SPY_h{h}L1": fwd_lag(C["SPY"], h, 1) for h in (1, 5, 10)}
        serB["VIX_h5L0"] = fwd_ret(C["^VIX"], 5)
        episodes(trigB, serB)
        print("  VIX close on each: " + ", ".join(
            f"{d.date()} {C['^VIX'][d]:.2f}" for d in trigB))

# ===========================================================================
# C. QQQ 5d >= +6% at a 252d closing high
# ===========================================================================
print("\n" + "=" * 78)
print("=== C. QQQ 5-session return >= +6% AND close at its 252d closing high ===")
q5 = nret("QQQ", 5)
q_dist = C["QQQ"] / roll_max("QQQ", 252) - 1.0
print(f"  TODAY {ASOF.date()}: QQQ 5d {100*q5.iloc[-1]:+.2f}% (state says +6.20) | "
      f"{100*q_dist.iloc[-1]:+.3f}% from 252d closing high")
for thr in (0.06, 0.05):
    print(f"\n  ##### QQQ 5d >= +{100*thr:.0f}% at a high #####")
    mC = (q5 >= thr) & (q_dist >= -0.001)
    mC_c = (q5 >= thr) & (q_dist < -0.001)
    print(f"  today qualifies: 5d {bool((q5 >= thr).iloc[-1])} | at high "
          f"{bool((q_dist >= -0.001).iloc[-1])} | CELL {bool(mC.iloc[-1])}")
    rawC = mask_dates(mC)
    trigC = declusters(rawC, 10, nyse)
    trigC_c = declusters(mask_dates(mC_c), 10, nyse)
    print(f"  raw {len(rawC)} -> declustered(10) {len(trigC)} | complement "
          f"(5d >= thr, NOT at high) declustered {len(trigC_c)}")
    print("  trigger dates: " + ", ".join(str(d.date()) for d in trigC))
    cell_block("C", "QQQ", trigC, trigC_c, (1, 5, 10, 21))
    if len(trigC) <= 40:
        print("\n  episode table (%):")
        serC = {"QQQ_5d": q5}
        serC |= {f"h{h}L1": fwd_lag(C["QQQ"], h, 1) for h in (1, 5, 10, 21)}
        serC |= {f"h{h}L0": fwd_ret(C["QQQ"], h) for h in (5, 10, 21)}
        episodes(trigC, serC)

# ===========================================================================
# D. SPY minus TLT into quarter end
# ===========================================================================
print("\n" + "=" * 78)
print("=== D. SPY-TLT into quarter end (anchor = 6 sessions remaining), 2002+ ===")
fut = pd.bdate_range("2026-09-22", "2026-09-30")
print("  forward sessions after 09-22: " +
      ", ".join(str(d.date()) for d in fut[1:]) +
      f"  -> {len(fut) - 1} remaining (no NYSE holidays in this span)")

mk = pd.DataFrame({"pos": range(len(nyse))}, index=nyse)
mk["y"] = nyse.year
mk["q"] = nyse.quarter
last_pos = mk.groupby(["y", "q"])["pos"].max()
cur_key = (ASOF.year, ASOF.quarter)
prior_last = last_pos.shift(1)
rowsD = []
for (y, q), p in last_pos.items():
    pl = prior_last.get((y, q))
    if pl is None or np.isnan(pl):
        continue
    pl = int(pl)
    if (y, q) == cur_key:
        a = int(POS[ASOF])
        end = None
    else:
        a = p - 6
        end = nyse[p]
    d0, da = nyse[pl], nyse[a]
    s0, sa = C["SPY"].get(d0, np.nan), C["SPY"].get(da, np.nan)
    t0, ta = C["TLT"].get(d0, np.nan), C["TLT"].get(da, np.nan)
    if np.isnan(t0) or np.isnan(ta):
        continue
    gap = (sa / s0 - 1.0) - (ta / t0 - 1.0)
    r = {"y": y, "q": q, "anchor": da, "end": end, "gap": gap,
         "spy_qtd": sa / s0 - 1.0, "tlt_qtd": ta / t0 - 1.0}
    if end is not None:
        se, te = C["SPY"][end], C["TLT"][end]
        s1, t1 = C["SPY"][nyse[a + 1]], C["TLT"][nyse[a + 1]]
        r.update(spy6=se / sa - 1.0, tlt6=te / ta - 1.0,
                 spy5=se / s1 - 1.0, tlt5=te / t1 - 1.0)
        r["sp6"] = r["spy6"] - r["tlt6"]
        r["sp5"] = r["spy5"] - r["tlt5"]
    rowsD.append(r)
D = pd.DataFrame(rowsD)
today = D[D["end"].isna()]
Dh = D[D["end"].notna()].set_index("anchor")
if len(today):
    t = today.iloc[0]
    print(f"  TODAY anchor {t['anchor'].date()}: SPY QTD {100*t['spy_qtd']:+.2f}% "
          f"TLT QTD {100*t['tlt_qtd']:+.2f}%  gap {100*t['gap']:+.2f}pp "
          f"(brief says 8.13) qualifies {bool(t['gap'] >= 0.08)}")
    print(f"  today is {len(fut) - 1} sessions before 2026-09-30: "
          f"{len(fut) - 1 == 6}")
print(f"  historical quarters with TLT: n={len(Dh)} "
      f"({Dh.index[0].date()} .. {Dh.index[-1].date()})")
condD = Dh[Dh["gap"] >= 0.08]
compD = Dh[Dh["gap"] < 0.08]
print(f"  gap >= 8pp: n={len(condD)} | complement: n={len(compD)}")
for col, lab in (("sp6", "SPY-TLT lag0 h6"), ("sp5", "SPY-TLT lag1 h5"),
                 ("spy6", "SPY lag0 h6"), ("spy5", "SPY lag1 h5"),
                 ("tlt6", "TLT lag0 h6"), ("tlt5", "TLT lag1 h5")):
    v = line(lab, Dh[col], condD.index,
             [("ctrl all quarter-ends", Dh[col]),
              ("ctrl complement (<8pp)", compD[col])])
    if len(v):
        era(lab, v)
print("\n  episode table (gap >= 8pp), % :")
episodes(condD.index, {c: condD[c] for c in
                       ("gap", "sp6", "sp5", "spy6", "spy5", "tlt6", "tlt5")})
print("\n  all quarters sorted by gap (top 12) for context:")
for d, r in Dh.sort_values("gap", ascending=False).head(12).iterrows():
    print(f"   {d.date()} Q{r['q']} gap {100*r['gap']:+.2f}pp  "
          f"sp6 {100*r['sp6']:+.2f}  sp5 {100*r['sp5']:+.2f}")

# ===========================================================================
# E. The Wednesday after September quad witching
# ===========================================================================
print("\n" + "=" * 78)
print("=== E. Wednesday after the September quad-witching Friday, 2000-2025 ===")
ev = load_events(["fomc_decision"])
print(f"  events columns: {list(ev.columns)}  n={len(ev)}")
fomc = set(pd.DatetimeIndex(ev["date"]).normalize())
gE = daily_ret("^GSPC")
tE = daily_ret("TLT")
rowsE = []
for y in range(2000, 2027):
    fri = [d for d in pd.date_range(f"{y}-09-01", f"{y}-09-30")
           if d.weekday() == 4]
    qw = fri[2]
    wed = qw + pd.Timedelta(days=5)
    is_sess = wed in POS.index
    if y == 2026:
        print(f"  2026: quad Fri {qw.date()}, Wednesday {wed.date()} "
              f"(tomorrow; FOMC that day: {wed in fomc})")
        continue
    if not is_sess:
        print(f"  {y}: Wednesday {wed.date()} is not an NYSE session, skipped")
        continue
    p = int(POS[wed])
    tue = nyse[p - 1]
    mlast = nyse[(nyse.year == y) & (nyse.month == 9)][-1]
    rowsE.append({"year": y, "qw": qw, "tue": tue, "wed": wed,
                  "fomc": wed in fomc,
                  "spx_wed": gE[wed], "tlt_wed": tE.get(wed, np.nan),
                  "spy_rest": C["SPY"][mlast] / C["SPY"][wed] - 1.0,
                  "n_rest": int(POS[mlast]) - p, "mlast": mlast})
E = pd.DataFrame(rowsE).set_index("wed")
print("  year  quad-Fri     Tue(anchor)  Wed         FOMC  ^GSPC Wed  TLT Wed"
      "   SPY Wed->Sep-end (sessions)")
for d, r in E.iterrows():
    def fm(x):
        return "    n/a" if (x is None or np.isnan(x)) else f"{100*x:+7.2f}"
    print(f"  {r['year']}  {r['qw'].date()}  {r['tue'].date()}   {d.date()}  "
          f"{'Y' if r['fomc'] else '-':>4}  {fm(r['spx_wed'])}   "
          f"{fm(r['tlt_wed'])}   {fm(r['spy_rest'])} ({r['n_rest']}, to "
          f"{r['mlast'].date()})")
for lab, sub in (("ALL", E), ("FOMC-Wed", E[E["fomc"]]),
                 ("non-FOMC Wed", E[~E["fomc"]])):
    for col, nm in (("spx_wed", "^GSPC Wed (h1 lag0 from Tue)"),
                    ("tlt_wed", "TLT Wed"),
                    ("spy_rest", "SPY Wed close -> Sep last close")):
        v = line(f"{lab:<13} {nm}", sub[col], sub.index)
        if len(v) and lab != "FOMC-Wed":
            era(f"{lab} {col}", v)
print("  ctrl all sessions: ^GSPC 1d " + (
    lambda v: f"n={len(v)} mean {100*v.mean():+.3f}% hit {100*(v > 0).mean():.1f}%")(
    gE.dropna()) + " | TLT 1d " + (
    lambda v: f"n={len(v)} mean {100*v.mean():+.3f}% hit {100*(v > 0).mean():.1f}%")(
    tE.dropna()))
wedall = nyse[nyse.weekday == 2]
line("ctrl all Wednesdays ^GSPC", gE, wedall)
sepwed = wedall[wedall.month == 9]
line("ctrl all September Wednesdays ^GSPC", gE, sepwed)

# ===========================================================================
# F. USO 5d <= -10% while >= 1.20x SMA200
# ===========================================================================
print("\n" + "=" * 78)
print("=== F. USO 5-session return <= -10% AND close >= 1.20 x SMA200 ===")
u5 = nret("USO", 5)
u_ratio = C["USO"] / sma("USO", 200)
print(f"  TODAY {ASOF.date()}: USO 5d {100*u5.iloc[-1]:+.2f}% | close/SMA200 "
      f"{u_ratio.iloc[-1]:.4f}")
for lab, gate in (("strict >= 1.20x", u_ratio >= 1.20),
                  ("loose > 1.00x", u_ratio > 1.0)):
    print(f"\n  ##### {lab} #####")
    mF = (u5 <= -0.10) & gate
    mF_c = (u5 <= -0.10) & ~gate & u_ratio.notna()
    print(f"  today qualifies: 5d<=-10% {bool((u5 <= -0.10).iloc[-1])} | gate "
          f"{bool(gate.iloc[-1])} | CELL {bool(mF.iloc[-1])}")
    rawF = mask_dates(mF)
    trigF = declusters(rawF, 10, nyse)
    trigF_c = declusters(mask_dates(mF_c), 10, nyse)
    print(f"  raw {len(rawF)} -> declustered(10) {len(trigF)} | complement "
          f"(5d<=-10%, gate fails) declustered {len(trigF_c)} | episodes in "
          f"2026: {int((trigF.year == 2026).sum())} "
          f"({', '.join(str(d.date()) for d in trigF if d.year == 2026)})")
    print("  trigger dates: " + ", ".join(str(d.date()) for d in trigF))
    cell_block("F", "USO", trigF, trigF_c, (5, 10))
    if len(trigF) <= 40:
        print("\n  episode table (%):")
        serF = {"USO_5d": u5, "ratio-1": u_ratio - 1.0}
        serF |= {f"h{h}L1": fwd_lag(C["USO"], h, 1) for h in (5, 10)}
        serF |= {f"h{h}L0": fwd_ret(C["USO"], h) for h in (5, 10)}
        episodes(trigF, serF)

# ===========================================================================
# G. Open-idea marks (prices only)
# ===========================================================================
print("\n" + "=" * 78)
print("=== G. Open-idea marks (ADJUSTED bars; master_prices stores adjusted "
      "OHLCV) ===")
fwd_cal = pd.DatetimeIndex(list(nyse) + list(pd.bdate_range(
    ASOF + pd.Timedelta(days=1), "2026-10-09")))  # no NYSE holidays in span


def add_sessions(d, n):
    p = list(fwd_cal).index(pd.Timestamp(d))
    return fwd_cal[p + n]


for qd, iid in (("2026-09-16", "x20260916-1"), ("2026-09-17", "x20260917-1"),
                ("2026-09-21", "x20260921-1")):
    d = json.loads((ROOT / "content" / "queue" / f"{qd}.json").read_text(
        encoding="utf-8"))
    it = next(x for x in d["drafts"] if x.get("id") == iid)
    idea = it["idea"]
    tk, side, et = idea["ticker"], idea["side"], idea["entry"]["type"]
    ex_on, td = idea["execute_on"], int(idea["time_td"])
    atr, refc = float(idea["atr"]), float(idea["ref_close"])
    f = raw_px[tk]
    field = "Open" if et == "MOO" else "Close"
    try:
        p0 = float(f.loc[pd.Timestamp(ex_on), field])
    except KeyError:
        print(f"  {iid}: missing {field} bar on {ex_on}")
        continue
    p1 = float(f.loc[ASOF, "Close"])
    sgn = 1.0 if side == "long" else -1.0
    pct = sgn * (p1 / p0 - 1.0)
    atrm = sgn * (p1 - p0) / atr
    exit_d = add_sessions(ex_on, td)
    print(f"  {iid} {tk} {side.upper()} entry {json.dumps(idea['entry'])} on "
          f"{ex_on}, time_td {td}, stop {idea['stop_atr']} target "
          f"{idea['target_atr']}")
    print(f"     frozen atr {atr} ref_close {refc} | entry price ({field} "
          f"{ex_on}) {p0:.4f} | close {ASOF.date()} {p1:.4f} | in-favour "
          f"{100*pct:+.3f}% = {atrm:+.2f} ATR | exit MOC {exit_d.date()} "
          f"({td} sessions after {ex_on})")

# ===========================================================================
# H. Frozen parameters as of the 2026-09-22 close
# ===========================================================================
print("\n" + "=" * 78)
print("=== H. Frozen levels at the 2026-09-22 close (Wilder-14 ATR) ===")
for t in ("XLU", "SPY", "QQQ", "USO", "^VIX"):
    f = raw_px.get(t)
    if f is None or len(f) == 0:
        print(f"  {t}: NO HISTORY")
        continue
    f = f[f.index <= ASOF].dropna(subset=["High", "Low", "Close"])
    a = float(np.asarray(wilder_atr(f["High"].to_numpy(), f["Low"].to_numpy(),
                                    f["Close"].to_numpy()), dtype=float)[-1])
    c = float(f["Close"].iloc[-1])
    print(f"  {t}: close {c:.4f}  Wilder-14 ATR {a:.4f}  "
          f"({100*a/c:.2f}% of price)  bar {f.index[-1].date()}")

print("\ndone.")
