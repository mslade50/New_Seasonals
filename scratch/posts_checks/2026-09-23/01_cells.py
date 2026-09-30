"""Posts check (2026-09-23): candidate cells for Wednesday night's queue.

Run date is Wednesday 2026-09-23 and that IS the freshest bar, so every price
cell anchors on the 09-23 close and the tradeable number is lag=1 (enter MOC
Thursday 09-24, exit MOC h sessions later). lag=0 is printed alongside as the
naive contrast.

A. IDEA candidate, long SPY: ^MOVE 1d >= +12% while ^VIX < 20. Declustered 5.
B. IDEA candidate, long TLT/IEF: ^TNX +12bp on the day AND at a 252-session
   closing high. Declustered 5, 2003+.
C. IWM minus SPY after the cell-B trigger (reproduce the brief's 3-17).
D. GDX 1d <= -4% while GLD >= 15% below its 252d closing high. Declustered 5.
E. TLT and IEF both at 252-session closing lows. Declustered 10.
F. ^GSPC on the Thursday after September quad witching, all years + midterms.
G. Open-idea marks (prices only).
H. Frozen levels at the 2026-09-23 close (Wilder-14 ATR).

^TNX is the 10y yield in percent, so +12bp = +0.12 change in level. Daily
changes and rolling extremes are computed on each ticker's VALID bars
(rolling_on_valid), then reindexed to the SPY calendar.

CACHE BOUND: master_prices.parquet starts 2000-01-03. TLT and IEF from 2002,
GLD from 2004, GDX from 2006; ^MOVE and ^TNX print their own first bar below.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pitch_lab import (  # noqa: E402
    declusters, fwd_lag, fwd_ret, load_prices, local_control,
    rolling_on_valid, sign_test, summarize, wilder_atr,
)

ROOT = Path(__file__).resolve().parents[3]
ASOF = pd.Timestamp("2026-09-23")
ERA = "2018-01-01"
TK = ["SPY", "TLT", "IEF", "IWM", "GDX", "GLD", "QQQ", "UUP", "XLE",
      "^MOVE", "^TNX", "^VIX", "^GSPC"]

raw_px = load_prices(TK)
nyse = raw_px["SPY"]["Close"].dropna().index
print(f"SPY last bar in cache: {nyse[-1].date()} (ASOF {ASOF.date()}, "
      f"match {nyse[-1] == ASOF})")
nyse = pd.DatetimeIndex(nyse[nyse <= ASOF])
C = {t: raw_px[t]["Close"].astype(float).reindex(nyse) for t in TK}
POS = pd.Series(range(len(nyse)), index=nyse)

print(f"NYSE calendar (SPY): {nyse[0].date()} .. {nyse[-1].date()}  n={len(nyse)}")
print("ASOF (anchor bar) =", ASOF.date(), "weekday", ASOF.day_name())
for t in TK:
    v = raw_px[t]["Close"].dropna()
    v = v[v.index <= ASOF]
    if len(v) == 0:
        print(f"  {t}: NO HISTORY in cache")
        continue
    r1 = v.iloc[-1] / v.iloc[-2] - 1.0
    hi = v.rolling(252).max().iloc[-1]
    lo = v.rolling(252).min().iloc[-1]
    print(f"  {t}: {v.index[0].date()} .. {v.index[-1].date()}  n={len(v)}  "
          f"last {v.iloc[-1]:.4f}  1d {100*r1:+.2f}% (chg {v.iloc[-1]-v.iloc[-2]:+.4f})"
          f"  vs 252hi {100*(v.iloc[-1]/hi-1):+.2f}%  vs 252lo "
          f"{100*(v.iloc[-1]/lo-1):+.2f}%")
print("\nNOTE: cache begins 2000-01-03; TLT/IEF from 2002, GLD 2004, GDX 2006.")


# ---------------------------------------------------------------------------
# helpers (same shapes as 2026-09-22/01_cells.py)
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


def episodes(dates, series, raw=()):
    """series values printed as %, `raw` column names printed as levels."""
    cols = list(series)
    print("   episode     " + "".join(f"{c:>12}" for c in cols))
    for d in dates:
        cells = []
        for c in cols:
            x = series[c].get(d, np.nan)
            if x is None or (isinstance(x, float) and np.isnan(x)):
                cells.append("n/a")
            elif c in raw:
                cells.append(f"{x:.3f}")
            else:
                cells.append(f"{100*x:+.2f}")
        print(f"   {d.date()}  " + "".join(f"{c:>12}" for c in cells))


def daily_ret(t):
    v = C[t].dropna()
    return (v / v.shift(1) - 1.0).reindex(nyse)


def daily_chg(t):
    v = C[t].dropna()
    return (v - v.shift(1)).reindex(nyse)


def roll_max(t, n):
    return rolling_on_valid(C[t], lambda x: x.rolling(n).max())


def roll_min(t, n):
    return rolling_on_valid(C[t], lambda x: x.rolling(n).min())


def mask_dates(m):
    m = m.reindex(nyse)
    return pd.DatetimeIndex(nyse[m.fillna(False).to_numpy(dtype=bool)])


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


def subset_lines(label, tkr, trig_sub, horizons, lags=(1, 0)):
    for h in horizons:
        for lag in lags:
            f = fwd_lag(C[tkr], h, lag) if lag else fwd_ret(C[tkr], h)
            v = line(f"{label} {tkr} h={h} lag={lag}", f, trig_sub)
            if len(v) and lag == 1:
                era(f"{label} {tkr} h={h} lag=1", v)


def first_bar(t):
    v = C[t].dropna()
    return v.index[0] if len(v) else None


# shared series
move1 = daily_ret("^MOVE")
vix = C["^VIX"]
tnx_chg = daily_chg("^TNX")
tnx_hi = C["^TNX"] >= roll_max("^TNX", 252) - 1e-9
spy_dist = C["SPY"] / roll_max("SPY", 252) - 1.0

# ===========================================================================
# A. MOVE one-day jump >= +12% with VIX < 20 -> long SPY
# ===========================================================================
print("\n" + "=" * 78)
print("=== A. ^MOVE 1d >= +12% AND ^VIX close < 20 -> SPY forward ===")
print(f"  ^MOVE first bar {first_bar('^MOVE').date()} | ^VIX first bar "
      f"{first_bar('^VIX').date()}")
print(f"  TODAY {ASOF.date()}: ^MOVE {C['^MOVE'].iloc[-1]:.2f} "
      f"({100*move1.iloc[-1]:+.2f}%) | ^VIX {vix.iloc[-1]:.2f} | SPY "
      f"{100*spy_dist.iloc[-1]:+.2f}% from 252d closing high")
mA = (move1 >= 0.12) & (vix < 20)
mA_c = (move1 >= 0.12) & (vix >= 20)
print(f"  today qualifies: MOVE>=+12% {bool((move1 >= 0.12).iloc[-1])} | "
      f"VIX<20 {bool((vix < 20).iloc[-1])} | CELL {bool(mA.iloc[-1])}")
rawA = mask_dates(mA)
trigA = declusters(rawA, 5, nyse)
trigA_c = declusters(mask_dates(mA_c), 5, nyse)
print(f"  raw {len(rawA)} -> declustered(5) {len(trigA)} | complement "
      f"(VIX >= 20) declustered {len(trigA_c)}")
print("  trigger dates: " + ", ".join(str(d.date()) for d in trigA))
cell_block("A", "SPY", trigA, trigA_c, (1, 2, 5))
trigA_hi = pd.DatetimeIndex([d for d in trigA if spy_dist.get(d, np.nan) >= -0.02])
trigA_nhi = pd.DatetimeIndex([d for d in trigA if spy_dist.get(d, np.nan) < -0.02])
print(f"\n  --- subset: SPY within 2% of its 252d closing high on the trigger "
      f"day (n={len(trigA_hi)}; not-near-high n={len(trigA_nhi)}) ---")
subset_lines("A near-high", "SPY", trigA_hi, (1, 2, 5))
subset_lines("A NOT near-high", "SPY", trigA_nhi, (1, 2, 5), lags=(1,))
print("\n  episode table (%; MOVE_1d in %, VIX and SPYdist as printed):")
serA = {"MOVE_1d": move1, "VIX": vix, "SPYdist": spy_dist,
        "SPY_h1L1": fwd_lag(C["SPY"], 1, 1),
        "SPY_h2L1": fwd_lag(C["SPY"], 2, 1),
        "SPY_h5L1": fwd_lag(C["SPY"], 5, 1),
        "SPY_h2L0": fwd_ret(C["SPY"], 2),
        "SPY_h5L0": fwd_ret(C["SPY"], 5)}
episodes(trigA, serA, raw=("VIX",))

# ===========================================================================
# B. TNX +12bp at a 252d closing high -> long TLT / IEF
# ===========================================================================
print("\n" + "=" * 78)
print("=== B. ^TNX 1d change >= +0.12 (12bp) AND ^TNX at 252d closing high, "
      "2003+ -> TLT / IEF forward ===")
print(f"  ^TNX first bar {first_bar('^TNX').date()} | TLT first "
      f"{first_bar('TLT').date()} | IEF first {first_bar('IEF').date()}")
print(f"  TODAY {ASOF.date()}: ^TNX {C['^TNX'].iloc[-1]:.3f} chg "
      f"{tnx_chg.iloc[-1]:+.4f} ({100*tnx_chg.iloc[-1]:+.1f}bp) | at 252d high "
      f"{bool(tnx_hi.iloc[-1])} | ^MOVE 1d {100*move1.iloc[-1]:+.2f}%")
bigB = tnx_chg >= 0.12 - 1e-9
mB = bigB & tnx_hi
mB_c = bigB & ~tnx_hi & C["^TNX"].notna()
print(f"  today qualifies: +12bp {bool(bigB.iloc[-1])} | at high "
      f"{bool(tnx_hi.iloc[-1])} | CELL {bool(mB.iloc[-1])}")
rawB = mask_dates(mB)
rawB03 = rawB[rawB >= "2003-01-01"]
trigB = declusters(rawB03, 5, nyse)
trigB_c = declusters(mask_dates(mB_c & (pd.Series(nyse.year, index=nyse) >= 2003)),
                     5, nyse)
print(f"  raw (2000+) {len(rawB)} | raw 2003+ {len(rawB03)} -> declustered(5) "
      f"{len(trigB)} | complement (+12bp NOT at high, 2003+) declustered "
      f"{len(trigB_c)}")
print("  trigger dates: " + ", ".join(str(d.date()) for d in trigB))
for tk in ("TLT", "IEF"):
    cell_block("B", tk, trigB, trigB_c, (1, 2, 5, 10))
trigB_mv = pd.DatetimeIndex([d for d in trigB if move1.get(d, np.nan) >= 0.12])
trigB_nmv = pd.DatetimeIndex([d for d in trigB if move1.get(d, np.nan) < 0.12])
print(f"\n  --- sub-split: ^MOVE also >= +12% that day (n={len(trigB_mv)}; "
      f"MOVE < +12% n={len(trigB_nmv)}; MOVE n/a "
      f"{len(trigB) - len(trigB_mv) - len(trigB_nmv)}) ---")
print("  MOVE>=12% dates: " + ", ".join(str(d.date()) for d in trigB_mv))
for tk in ("TLT", "IEF"):
    subset_lines("B MOVE>=12%", tk, trigB_mv, (1, 2, 5, 10))
    subset_lines("B MOVE<12%", tk, trigB_nmv, (1, 2, 5, 10), lags=(1,))
print("\n  episode table (%; TNX_chg as level change, TNX level raw):")
serB = {"TNX": C["^TNX"], "TNX_chg": tnx_chg, "MOVE_1d": move1}
serB |= {f"TLT_h{h}L1": fwd_lag(C["TLT"], h, 1) for h in (1, 2, 5, 10)}
serB |= {f"IEF_h{h}L1": fwd_lag(C["IEF"], h, 1) for h in (2, 5, 10)}
serB |= {f"TLT_h{h}L0": fwd_ret(C["TLT"], h) for h in (1, 5)}
episodes(trigB, serB, raw=("TNX", "TNX_chg"))

# ===========================================================================
# C. IWM minus SPY after the cell-B trigger
# ===========================================================================
print("\n" + "=" * 78)
print("=== C. IWM - SPY (close-to-close spread) after the cell-B trigger ===")
iwm_r1, spy_r1 = daily_ret("IWM"), daily_ret("SPY")
print(f"  TODAY {ASOF.date()}: IWM {100*iwm_r1.iloc[-1]:+.2f}% SPY "
      f"{100*spy_r1.iloc[-1]:+.2f}% spread {100*(iwm_r1.iloc[-1]-spy_r1.iloc[-1]):+.2f}pp")


def spread(h, lag):
    if lag == 0:
        return fwd_ret(C["IWM"], h) - fwd_ret(C["SPY"], h)
    return fwd_lag(C["IWM"], h, lag) - fwd_lag(C["SPY"], h, lag)


def c_block(label, trig):
    loc = local_control(nyse, trig, 126)
    for h, lag in ((1, 0), (1, 1), (2, 1), (5, 1)):
        sp = spread(h, lag)
        v = line(f"{label} IWM-SPY h={h} lag={lag}", sp, trig,
                 [("ctrl all sessions", sp.dropna()),
                  ("ctrl local +/-126td", sp.reindex(loc).dropna())])
        if len(v):
            era(f"{label} IWM-SPY h={h} lag={lag}", v)
    for h in (2, 5):
        f = fwd_lag(C["IWM"], h, 1)
        v = line(f"{label} IWM alone h={h} lag=1", f, trig,
                 [("ctrl all sessions", f.dropna()),
                  ("ctrl local +/-126td", f.reindex(loc).dropna())])
        if len(v):
            era(f"{label} IWM h={h} lag=1", v)


print("\n  ##### cell-B trigger as defined (2003+, SPY calendar, declustered 5)"
      " #####")
c_block("C", trigB)
trigB00 = declusters(rawB, 5, nyse)
print(f"\n  ##### brief-style reproduction: 2000+ (IWM from "
      f"{first_bar('IWM').date()}), declustered 5, n={len(trigB00)} #####")
print("  (brief 03_iwm_after_yield_jump.py: TNX calendar, bp = diff*100 >= 12,"
      " rolling(252, min_periods=200) max)")
c_block("C-2000+", trigB00)
extra = trigB00.difference(trigB)
print("  dates in 2000+ set not in 2003+ set: " +
      ", ".join(str(d.date()) for d in extra))
print("\n  episode table (2000+ set; %):")
serC = {"IWM_1d": iwm_r1, "SPY_1d": spy_r1,
        "sp_h1L0": spread(1, 0), "sp_h1L1": spread(1, 1),
        "sp_h2L1": spread(2, 1), "sp_h5L1": spread(5, 1),
        "IWM_h2L1": fwd_lag(C["IWM"], 2, 1), "IWM_h5L1": fwd_lag(C["IWM"], 5, 1)}
episodes(trigB00, serC)

# ===========================================================================
# D. GDX crash day while gold is well off its high
# ===========================================================================
print("\n" + "=" * 78)
print("=== D. GDX 1d <= -4% AND GLD close >= 15% below its 252d closing high ===")
gdx1 = daily_ret("GDX")
gld_dist = C["GLD"] / roll_max("GLD", 252) - 1.0
print(f"  GDX first {first_bar('GDX').date()} | GLD first {first_bar('GLD').date()}")
print(f"  TODAY {ASOF.date()}: GDX {100*gdx1.iloc[-1]:+.2f}% | GLD "
      f"{100*gld_dist.iloc[-1]:+.2f}% from its 252d closing high")
mD = (gdx1 <= -0.04) & (gld_dist <= -0.15)
mD_c = (gdx1 <= -0.04) & (gld_dist > -0.15)
print(f"  today qualifies: GDX<=-4% {bool((gdx1 <= -0.04).iloc[-1])} | GLD "
      f"<=-15% {bool((gld_dist <= -0.15).iloc[-1])} | CELL {bool(mD.iloc[-1])}")
rawD = mask_dates(mD)
trigD = declusters(rawD, 5, nyse)
trigD_c = declusters(mask_dates(mD_c), 5, nyse)
print(f"  raw {len(rawD)} -> declustered(5) {len(trigD)} | complement (GLD "
      f"within 15% of high) declustered {len(trigD_c)}")
print("  trigger dates: " + ", ".join(str(d.date()) for d in trigD))
cell_block("D", "GDX", trigD, trigD_c, (5, 10))
cell_block("D", "GLD", trigD, trigD_c, (5, 10))
print("\n  episode table (%):")
serD = {"GDX_1d": gdx1, "GLDdist": gld_dist}
serD |= {f"GDX_h{h}L1": fwd_lag(C["GDX"], h, 1) for h in (5, 10)}
serD |= {f"GDX_h{h}L0": fwd_ret(C["GDX"], h) for h in (5, 10)}
serD |= {f"GLD_h{h}L1": fwd_lag(C["GLD"], h, 1) for h in (5, 10)}
episodes(trigD, serD)

# ===========================================================================
# E. TLT and IEF both at 252d closing lows
# ===========================================================================
print("\n" + "=" * 78)
print("=== E. TLT AND IEF both at 252-session closing lows, declustered 10 ===")
tlt_lo = C["TLT"] <= roll_min("TLT", 252) + 1e-9
ief_lo = C["IEF"] <= roll_min("IEF", 252) + 1e-9
mE = tlt_lo & ief_lo
print(f"  TODAY {ASOF.date()}: TLT at low {bool(tlt_lo.iloc[-1])} | IEF at low "
      f"{bool(ief_lo.iloc[-1])} | CELL {bool(mE.iloc[-1])}")
rawE = mask_dates(mE)
trigE = declusters(rawE, 10, nyse)
print(f"  raw sessions {len(rawE)} -> declustered(10) {len(trigE)}")
r26 = rawE[rawE.year == 2026]
print(f"  sessions in 2026 with BOTH at 252d closing lows: {len(r26)} -> "
      + ", ".join(str(d.date()) for d in r26))
print(f"  declustered episodes in 2026: {int((trigE.year == 2026).sum())}")
print("  trigger dates: " + ", ".join(str(d.date()) for d in trigE))
locE = local_control(nyse, trigE, 126)
for tk in ("TLT", "SPY"):
    for h in (5, 21):
        f = fwd_lag(C[tk], h, 1)
        v = line(f"E {tk} h={h} lag=1", f, trigE,
                 [("ctrl all sessions", f.dropna()),
                  ("ctrl local +/-126td", f.reindex(locE).dropna())])
        if len(v):
            era(f"E {tk} h={h} lag=1", v)
        f0 = fwd_ret(C[tk], h)
        line(f"E {tk} h={h} lag=0", f0, trigE,
             [("ctrl local +/-126td", f0.reindex(locE).dropna())])
print("\n  episode table (%):")
serE = {"TLT_h5L1": fwd_lag(C["TLT"], 5, 1), "TLT_h21L1": fwd_lag(C["TLT"], 21, 1),
        "SPY_h5L1": fwd_lag(C["SPY"], 5, 1), "SPY_h21L1": fwd_lag(C["SPY"], 21, 1),
        "SPYdist": spy_dist}
episodes(trigE, serE)

# ===========================================================================
# F. Thursday after September quad witching
# ===========================================================================
print("\n" + "=" * 78)
print("=== F. ^GSPC close-to-close on the Thursday after the September "
      "quad-witching Friday, 2000-2025 ===")
gF = daily_ret("^GSPC")
rowsF = []
for y in range(2000, 2027):
    fri = [d for d in pd.date_range(f"{y}-09-01", f"{y}-09-30")
           if d.weekday() == 4]
    qw = fri[2]
    thu = qw + pd.Timedelta(days=6)
    if y == 2026:
        print(f"  2026: quad Fri {qw.date()}, Thursday {thu.date()} "
              f"(tomorrow; = 2026-09-24 {thu == pd.Timestamp('2026-09-24')})")
        continue
    if thu not in POS.index:
        print(f"  {y}: Thursday {thu.date()} is not an NYSE session, skipped")
        continue
    rowsF.append({"year": y, "qw": qw, "thu": thu, "spx": gF[thu],
                  "midterm": y % 4 == 2})
F = pd.DataFrame(rowsF).set_index("thu")
print("  year  quad-Fri     Thursday     midterm  ^GSPC Thu")
for d, r in F.iterrows():
    print(f"  {r['year']}  {r['qw'].date()}  {d.date()}   "
          f"{'Y' if r['midterm'] else '-':>6}   {100*r['spx']:+7.2f}%")
for lab, sub in (("ALL 2000-2025", F), ("midterm years", F[F["midterm"]]),
                 ("non-midterm", F[~F["midterm"]])):
    v = line(f"{lab:<14} ^GSPC Thu", sub["spx"], sub.index)
    if len(v) and lab != "midterm years":
        era(f"{lab}", v)
thu_all = nyse[(nyse.weekday == 3) & (nyse.year <= 2025)]
line("ctrl all Thursdays 2000-2025 ^GSPC", gF, thu_all)
line("ctrl all September Thursdays ^GSPC", gF, thu_all[thu_all.month == 9])
line("ctrl all sessions 2000-2025 ^GSPC", gF, nyse[nyse.year <= 2025])

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
    status = ("REALISED (exit MOC today)" if exit_d == ASOF else
              "open mark" if exit_d > ASOF else "PAST EXIT")
    print(f"  {iid} {tk} {side.upper()} entry {json.dumps(idea['entry'])} on "
          f"{ex_on}, time_td {td}, stop {idea['stop_atr']} target "
          f"{idea['target_atr']}")
    print(f"     frozen atr {atr} ref_close {refc} | entry price ({field} "
          f"{ex_on}) {p0:.4f} | close {ASOF.date()} {p1:.4f} | in-favour "
          f"{100*pct:+.3f}% = {atrm:+.2f} R(ATR) | exit MOC {exit_d.date()} "
          f"({td} sessions after {ex_on}) -> {status}")
    stp_a, tgt_a = idea.get("stop_atr"), idea.get("target_atr")
    fr = f[(f.index >= pd.Timestamp(ex_on)) & (f.index <= ASOF)]
    if len(fr):
        if side == "long":
            mae = (fr["Low"].min() - p0) / atr
            mfe = (fr["High"].max() - p0) / atr
        else:
            mae = (p0 - fr["High"].max()) / atr
            mfe = (p0 - fr["Low"].min()) / atr
        print(f"     path since entry (incl. entry-day bar): MAE {mae:+.2f} ATR "
              f"MFE {mfe:+.2f} ATR | stop_atr {stp_a} target_atr {tgt_a}")
        print("     closes: " + ", ".join(
            f"{i.date()} {c:.2f}" for i, c in fr["Close"].items()))

# ===========================================================================
# H. Frozen levels as of the 2026-09-23 close
# ===========================================================================
print("\n" + "=" * 78)
print("=== H. Frozen levels at the 2026-09-23 close (Wilder-14 ATR) ===")
for t in ("SPY", "TLT", "IEF", "IWM", "GDX", "QQQ"):
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
