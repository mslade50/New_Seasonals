"""Posts check (2026-09-21): candidate cells for Monday night's queue.

Run date is Monday 2026-09-21 and that IS the freshest bar, so every cell
anchors on the 09-21 close and the tradeable number is lag=1 (enter MOC
Tuesday 09-22, exit h sessions later). lag=0 is printed alongside as the
naive contrast wherever asked.

A. IDEA candidate. QQQ up >= 2.0% on the day while closing within 1% of its
   252-session closing high. Declustered 10 sessions.
B. IDEA candidate. XLE down <= -2.5% on the day while the close is >= 10%
   above its 200-session SMA. Declustered 10 sessions.
C. Calendar. The last 6 NYSE sessions of September (enter MOC at the close
   6 sessions before the month's last session, exit MOC on it).
D. The Monday after September quad witching, split by that Monday's sign,
   held to that Friday's close.
E. Rails descriptive. CSX + UNP + NSC 21-session return rank all <= 5 on the
   same session, novelty-filtered 21 sessions.
F. Verification. ^GSPC up >= 1.25% (and >= 1.5%) with ^VIX closing HIGHER.
G. Open-idea marks (prices only).
H. Frozen idea parameters as of the 2026-09-21 close.

CACHE BOUND: master_prices.parquet starts 2000-01-03 for every ticker used
here, so the requested SPY-from-1993 and ^NDX-from-1999 depth is NOT
available. Those cells run from 2000 and say so.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pitch_lab import (  # noqa: E402
    declusters, fwd_lag, fwd_ret, load_prices, local_control, pct_rank,
    rolling_on_valid, sign_test, summarize, wilder_atr,
)

ASOF = pd.Timestamp("2026-09-21")
ERA = "2018-01-01"
TK = ["QQQ", "XLE", "SPY", "IWM", "CSX", "UNP", "NSC", "XLI", "UUP",
      "^NDX", "^GSPC", "^VIX"]

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
print("\nNOTE: cache begins 2000-01-03. SPY-from-1993 and ^NDX-from-1999 are "
      "NOT available; those cells start in 2000.")


# ---------------------------------------------------------------------------
# helpers (same shapes as 2026-09-20/01_cells.py)
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


def novelty(trig, win=21):
    tset = set(pd.DatetimeIndex(trig))
    keep = []
    for d in sorted(tset):
        p = int(POS[d])
        if not any(x in tset for x in nyse[max(0, p - win):p]):
            keep.append(d)
    return pd.DatetimeIndex(keep)


def line(label, s, dates, ctrls=()):
    v = s.reindex(pd.DatetimeIndex(dates)).dropna()
    if len(v) == 0:
        print(f"  {label}: n=0")
        return v
    up, dn, n = rec(v.values)
    sm = summarize(v.values)
    print(f"  {label}: n={n} {up}-{dn} mean {sm['mean_pct']:+.3f}% "
          f"med {sm['median_pct']:+.3f}% hit {sm['hit']:.1f}% t {sm['t']:+.2f} "
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


def sma(t, n):
    return rolling_on_valid(C[t], lambda x: x.rolling(n).mean())


def roll_max(t, n):
    return rolling_on_valid(C[t], lambda x: x.rolling(n).max())


def mask_dates(m):
    m = m.reindex(nyse)
    return pd.DatetimeIndex(nyse[m.fillna(False).to_numpy(dtype=bool)])


# ===========================================================================
# A. QQQ +2% while within 1% of its 252-session closing high
# ===========================================================================
print("\n" + "=" * 78)
print("=== A. QQQ 1d return >= +2.0% AND close within 1% of 252d closing high ===")

qr1 = daily_ret("QQQ")
q_hi = roll_max("QQQ", 252)
q_dist = C["QQQ"] / q_hi - 1.0
print(f"  TODAY {ASOF.date()}: QQQ 1d {100*qr1.iloc[-1]:+.2f}% (state says +2.77) | "
      f"{100*q_dist.iloc[-1]:+.2f}% from its 252d closing high (state says ~-0.5)")
maskA = (qr1 >= 0.02) & (q_dist >= -0.01)
maskA_comp = (qr1 >= 0.02) & (q_dist < -0.01)
print(f"  today qualifies: up>=2% {bool((qr1 >= 0.02).iloc[-1])} | "
      f"within 1% of high {bool((q_dist >= -0.01).iloc[-1])} | "
      f"CELL {bool(maskA.iloc[-1])}")

rawA = mask_dates(maskA)
rawA_c = mask_dates(maskA_comp)
print(f"  raw trigger sessions: {len(rawA)} "
      f"({rawA[0].date()} .. {rawA[-1].date()})" if len(rawA) else "  none")
trigA = declusters(rawA, 10, nyse)
trigA_c = declusters(rawA_c, 10, nyse)
print(f"  after declusters(10, keep first): {len(trigA)}  "
      f"(complement: {len(rawA_c)} raw -> {len(trigA_c)} declustered)")
print("  trigger dates: " + ", ".join(str(d.date()) for d in trigA))
locA = local_control(nyse, trigA, 126)

for h in (1, 5, 10):
    print(f"\n  --- QQQ h={h} ---")
    f1 = fwd_lag(C["QQQ"], h, 1)
    f0 = fwd_ret(C["QQQ"], h)
    ctr = [("ctrl all sessions", f1.dropna()),
           ("ctrl local +/-126td", f1.reindex(locA).dropna()),
           ("ctrl complement (+2%, >1% below high)", f1.reindex(trigA_c).dropna())]
    v = line(f"QQQ h={h} lag=1", f1, trigA, ctr)
    if len(v):
        era(f"QQQ h={h} lag=1", v)
    ctr0 = [("ctrl all sessions", f0.dropna()),
            ("ctrl local +/-126td", f0.reindex(locA).dropna()),
            ("ctrl complement", f0.reindex(trigA_c).dropna())]
    line(f"QQQ h={h} lag=0", f0, trigA, ctr0)

print("\n  episode table, QQQ lag=1 (%):")
episodes(trigA, {f"QQQ_h{h}": fwd_lag(C["QQQ"], h, 1) for h in (1, 5, 10)})

# Monday-only subset
monA = pd.DatetimeIndex([d for d in trigA if d.weekday() == 0])
print(f"\n  --- Monday-only subset of the QQQ cell: n={len(monA)} ---")
print("  Monday triggers: " + ", ".join(str(d.date()) for d in monA))
for h in (1, 5, 10):
    f1 = fwd_lag(C["QQQ"], h, 1)
    line(f"QQQ Monday-only h={h} lag=1", f1, monA,
         [("ctrl all Mondays", f1.reindex(nyse[nyse.weekday == 0]).dropna())])

# ^NDX depth (cache-limited to 2000)
print("\n  --- identical cell on ^NDX (requested from 1999; cache starts 2000) ---")
nr1 = daily_ret("^NDX")
n_hi = roll_max("^NDX", 252)
n_dist = C["^NDX"] / n_hi - 1.0
maskN = (nr1 >= 0.02) & (n_dist >= -0.01)
maskN_c = (nr1 >= 0.02) & (n_dist < -0.01)
rawN = mask_dates(maskN)
trigN = declusters(rawN, 10, nyse)
trigN_c = declusters(mask_dates(maskN_c), 10, nyse)
print(f"  ^NDX raw {len(rawN)} -> declustered(10) {len(trigN)}  "
      f"(complement declustered {len(trigN_c)}); "
      f"today {100*nr1.iloc[-1]:+.2f}%, {100*n_dist.iloc[-1]:+.2f}% from high, "
      f"qualifies {bool(maskN.iloc[-1])}")
locN = local_control(nyse, trigN, 126)
for h in (1, 5, 10):
    f1 = fwd_lag(C["^NDX"], h, 1)
    f0 = fwd_ret(C["^NDX"], h)
    v = line(f"^NDX h={h} lag=1", f1, trigN,
             [("ctrl all sessions", f1.dropna()),
              ("ctrl local +/-126td", f1.reindex(locN).dropna()),
              ("ctrl complement", f1.reindex(trigN_c).dropna())])
    if len(v):
        era(f"^NDX h={h} lag=1", v)
    line(f"^NDX h={h} lag=0", f0, trigN, [("ctrl all sessions", f0.dropna())])

# ===========================================================================
# B. XLE -2.5% while >= 10% above its 200-session SMA
# ===========================================================================
print("\n" + "=" * 78)
print("=== B. XLE 1d return <= -2.5% while close >= 10% above its 200d SMA ===")
xr1 = daily_ret("XLE")
x_sma = sma("XLE", 200)
x_dist = C["XLE"] / x_sma - 1.0
print(f"  TODAY {ASOF.date()}: XLE 1d {100*xr1.iloc[-1]:+.2f}% (state says -2.88) | "
      f"{100*x_dist.iloc[-1]:+.2f}% vs SMA200 (state says +12.5)")
maskB = (xr1 <= -0.025) & (x_dist >= 0.10)
maskB_comp = (xr1 <= -0.025) & (x_dist < 0.10)
print(f"  today qualifies: down<=-2.5% {bool((xr1 <= -0.025).iloc[-1])} | "
      f">=10% above SMA200 {bool((x_dist >= 0.10).iloc[-1])} | "
      f"CELL {bool(maskB.iloc[-1])}")
rawB = mask_dates(maskB)
rawB_c = mask_dates(maskB_comp)
trigB = declusters(rawB, 10, nyse)
trigB_c = declusters(rawB_c, 10, nyse)
print(f"  raw {len(rawB)} -> declustered(10) {len(trigB)}   "
      f"(complement {len(rawB_c)} raw -> {len(trigB_c)} declustered)")
print("  trigger dates: " + ", ".join(str(d.date()) for d in trigB))
locB = local_control(nyse, trigB, 126)
for h in (5, 10, 21):
    print(f"\n  --- h={h} ---")
    f1 = fwd_lag(C["XLE"], h, 1)
    f0 = fwd_ret(C["XLE"], h)
    v = line(f"XLE h={h} lag=1", f1, trigB,
             [("ctrl all sessions", f1.dropna()),
              ("ctrl local +/-126td", f1.reindex(locB).dropna()),
              ("ctrl complement (-2.5%, <10% above SMA200)",
               f1.reindex(trigB_c).dropna())])
    if len(v):
        era(f"XLE h={h} lag=1", v)
    line(f"XLE h={h} lag=0", f0, trigB,
         [("ctrl all sessions", f0.dropna()),
          ("ctrl complement", f0.reindex(trigB_c).dropna())])
    s1 = fwd_lag(C["SPY"], h, 1)
    vs = line(f"SPY h={h} lag=1 (same anchors)", s1, trigB,
              [("ctrl all sessions", s1.dropna())])
    if len(vs):
        era(f"SPY h={h} lag=1", vs)
    sp = f1 - s1
    line(f"XLE-SPY spread h={h} lag=1", sp, trigB,
         [("ctrl all sessions", sp.dropna())])

print("\n  episode table, lag=1 (%):")
episodes(trigB, {f"XLE_h{h}": fwd_lag(C["XLE"], h, 1) for h in (5, 10, 21)} |
         {f"SPY_h{h}": fwd_lag(C["SPY"], h, 1) for h in (5, 10, 21)})

# ===========================================================================
# C. Last 6 NYSE sessions of September (quarter-end)
# ===========================================================================
print("\n" + "=" * 78)
print("=== C. 6-session window into the last session of the month ===")

# verify the 2026 analog on a forward NYSE calendar (no Sep 2026 holidays
# between 09-22 and 09-30, so plain business days are the session list)
fut = pd.bdate_range("2026-09-21", "2026-09-30")
print("  Sep 2026 sessions 09-21..09-30:", ", ".join(str(d.date()) for d in fut))
try:
    i22, i30 = list(fut).index(pd.Timestamp("2026-09-22")), \
        list(fut).index(pd.Timestamp("2026-09-30"))
    print(f"  2026-09-22 is {i30 - i22} sessions before 2026-09-30 "
          f"-> matches the requested 6-session window: {i30 - i22 == 6}")
except ValueError:
    print("  could not locate the 2026 analog dates")

mk = pd.DataFrame({"pos": range(len(nyse))}, index=nyse)
mk["y"] = nyse.year
mk["m"] = nyse.month
last_pos = mk.groupby(["y", "m"])["pos"].max()
# The calendar is truncated at ASOF, so the CURRENT month's "last session" is
# just today's bar. September 2026 is not over: drop it or the cell books a
# fake 2026 observation (entry 09-11 -> exit 09-21).
last_pos = last_pos.drop(index=(ASOF.year, ASOF.month), errors="ignore")
print(f"  (dropped the incomplete current month {ASOF.year}-{ASOF.month:02d}: "
      "its last session is only today's bar)")


def window_ret(ticker, gap):
    """close of the session `gap` before a month's last session -> that close."""
    c = C[ticker]
    rows = []
    for (y, m), p in last_pos.items():
        a = p - gap
        if a < 0:
            continue
        d_end, d_beg = nyse[p], nyse[a]
        ce, cb = c.get(d_end, np.nan), c.get(d_beg, np.nan)
        if np.isnan(ce) or np.isnan(cb):
            continue
        rows.append({"y": y, "m": m, "beg": d_beg, "end": d_end,
                     "ret": ce / cb - 1.0})
    return pd.DataFrame(rows)


for tkr in ("SPY", "IWM"):
    print(f"\n  --- {tkr}, 6-session window into month end ---")
    w = window_ret(tkr, 6)
    if w.empty:
        print(f"    {tkr}: no history")
        continue
    sep = w[w["m"] == 9].set_index("end")["ret"]
    oth = w[w["m"] != 9]["ret"]
    qtr = w[w["m"].isin([3, 6, 12])]["ret"]
    line(f"{tkr} SEPTEMBER last-6", sep, sep.index,
         [("ctrl every OTHER month", oth),
          ("ctrl Mar/Jun/Dec month-ends", qtr)])
    era(f"{tkr} September last-6", sep)
    print(f"    yearly September values ({tkr}):")
    for d, r in sep.items():
        print(f"      {d.year}  entry {w.set_index('end').loc[d, 'beg'].date()} "
              f"-> exit {d.date()}   {100*r:+7.2f}%")
    print(f"\n  --- {tkr}, last 3 sessions of September ---")
    w3 = window_ret(tkr, 3)
    sep3 = w3[w3["m"] == 9].set_index("end")["ret"]
    oth3 = w3[w3["m"] != 9]["ret"]
    line(f"{tkr} SEPTEMBER last-3", sep3, sep3.index,
         [("ctrl every OTHER month", oth3)])
    era(f"{tkr} September last-3", sep3)

# ===========================================================================
# D. Post-quad-witching September Monday follow-through
# ===========================================================================
print("\n" + "=" * 78)
print("=== D. The Monday after September quad witching (third Friday) ===")

spy_r1 = daily_ret("SPY")
iwm_r1 = daily_ret("IWM")
rows = []
for y in range(2000, 2026):
    fri = [d for d in pd.date_range(f"{y}-09-01", f"{y}-09-30")
           if d.weekday() == 4]
    if len(fri) < 3:
        continue
    qw = fri[2]
    nxt = nyse[nyse > qw]
    if len(nxt) == 0:
        continue
    mon = nxt[0]
    p = int(POS[mon])
    if p + 4 >= len(nyse):
        continue
    rows.append({
        "year": y, "qw": qw.date(), "mon": mon,
        "spy_mon": spy_r1.get(mon, np.nan),
        "iwm_mon": iwm_r1.get(mon, np.nan),
        "spy_h4": fwd_ret(C["SPY"], 4).get(mon, np.nan),
        "iwm_h4": fwd_ret(C["IWM"], 4).get(mon, np.nan),
    })
D = pd.DataFrame(rows).set_index("mon")
print("  year   quad-Fri    Monday      SPY Mon    IWM Mon    SPY Mon->Fri  "
      "IWM Mon->Fri")
for d, r in D.iterrows():
    def f(x):
        return "    n/a" if (x is None or np.isnan(x)) else f"{100*x:+7.2f}"
    print(f"  {int(r['year'])}   {r['qw']}  {d.date()}   {f(r['spy_mon'])}    "
          f"{f(r['iwm_mon'])}       {f(r['spy_h4'])}       {f(r['iwm_h4'])}")

up = D[D["spy_mon"] > 0]
dn = D[D["spy_mon"] <= 0]
print(f"\n  today's analog 2026-09-21: SPY {100*spy_r1.iloc[-1]:+.2f}% "
      f"(state says +1.55), IWM {100*iwm_r1.iloc[-1]:+.2f}% (state says +0.52)")
print(f"  up-Mondays n={len(up)}, down-Mondays n={len(dn)}")
big = up[up["spy_mon"] >= 0.01]
print(f"  up-Monday years with SPY >= +1.0%: {len(big)} of {len(up)}  "
      f"({', '.join(str(int(y)) for y in big['year'])})")
for lbl, sub in (("SPY-UP Monday", up), ("SPY-DOWN Monday", dn),
                 ("ALL", D), ("SPY-UP >= +1%", big)):
    for col, tk in (("spy_h4", "SPY"), ("iwm_h4", "IWM")):
        s = sub[col].dropna()
        if len(s) == 0:
            print(f"  {lbl} {tk} Mon->Fri: n=0")
            continue
        u, dd, n = rec(s.values)
        sm = summarize(s.values)
        print(f"  {lbl:<16} {tk} Mon->Fri (h=4, lag=0): n={n} {u}-{dd} "
              f"mean {sm['mean_pct']:+.3f}% med {sm['median_pct']:+.3f}% "
              f"hit {sm['hit']:.1f}% t {sm['t']:+.2f} "
              f"p_up {stp(u, n)} "
              f"worst {sm['worst_pct']:+.2f}% best {sm['best_pct']:+.2f}%")
print("  era split, SPY-UP Monday SPY Mon->Fri:")
era("    D up-Monday SPY", up["spy_h4"].dropna())

# ===========================================================================
# E. Rails: CSX + UNP + NSC 21d return rank <= 5 together
# ===========================================================================
print("\n" + "=" * 78)
print("=== E. CSX, UNP, NSC all with 21d-return trailing-252 rank <= 5 ===")
RAILS = ["CSX", "UNP", "NSC"]
ranks = {t: pct_rank(C[t], 21, 252) for t in RAILS}
r21 = {t: (C[t].dropna() / C[t].dropna().shift(21) - 1.0).reindex(nyse)
       for t in RAILS}
print(f"  TODAY {ASOF.date()} ranks: " + "  ".join(
    f"{t} {ranks[t].iloc[-1]:.2f}" for t in RAILS) +
    "   (tape says 0.4 / 0.8 / 0.8)")
print(f"  TODAY {ASOF.date()} 21-session returns: " + "  ".join(
    f"{t} {100*r21[t].iloc[-1]:+.2f}%" for t in RAILS))
maskE = None
for t in RAILS:
    m = ranks[t] <= 5.0
    maskE = m if maskE is None else (maskE & m)
print(f"  today qualifies: {bool(maskE.iloc[-1])}")
rawE = mask_dates(maskE)
print(f"  raw trigger sessions: {len(rawE)}")
trigE = novelty(rawE, 21)
print(f"  after 21-session NOVELTY filter: {len(trigE)}  "
      f"(declusters(21) would keep {len(declusters(rawE, 21, nyse))})")
print("  episode dates: " + ", ".join(str(d.date()) for d in trigE))

basket = {}
for h in (5, 21):
    legs = [fwd_lag(C[t], h, 1) for t in RAILS]
    basket[h] = sum(legs) / 3.0
locE = local_control(nyse, trigE, 126)
for h in (5, 21):
    print(f"\n  --- h={h}, lag=1 ---")
    v = line(f"3-rail EW basket h={h}", basket[h], trigE,
             [("ctrl all sessions", basket[h].dropna()),
              ("ctrl local +/-126td", basket[h].reindex(locE).dropna())])
    if len(v):
        era(f"rails h={h}", v)
    for t in RAILS + ["XLI", "SPY"]:
        f1 = fwd_lag(C[t], h, 1)
        line(f"{t} h={h}", f1, trigE, [("ctrl all sessions", f1.dropna())])
    sp = basket[h] - fwd_lag(C["SPY"], h, 1)
    line(f"rails-SPY spread h={h}", sp, trigE,
         [("ctrl all sessions", sp.dropna())])

print("\n  episode table (%, lag=1):")
ser = {"BASKET_5": basket[5], "BASKET_21": basket[21]}
for t in ("XLI", "SPY"):
    ser[f"{t}_5"] = fwd_lag(C[t], 5, 1)
    ser[f"{t}_21"] = fwd_lag(C[t], 21, 1)
episodes(trigE, ser)

print("\n  per-rail 21d returns on each episode date (%):")
episodes(trigE, {t: r21[t] for t in RAILS})

# ===========================================================================
# F. ^GSPC up big with ^VIX higher
# ===========================================================================
print("\n" + "=" * 78)
print("=== F. ^GSPC up >= 1.25% / 1.5% while ^VIX closed HIGHER ===")
gr1 = daily_ret("^GSPC")
vr1 = daily_ret("^VIX")
print(f"  TODAY {ASOF.date()}: ^GSPC {100*gr1.iloc[-1]:+.2f}% (state says +1.49) | "
      f"^VIX {100*vr1.iloc[-1]:+.2f}% close {C['^VIX'].iloc[-1]:.2f} "
      f"(state says +0.41)")
print("  (requested since 1999; cache starts 2000-01-03)")
for thr in (0.0125, 0.015):
    print(f"\n  --- threshold >= {100*thr:.2f}% ---")
    m = (gr1 >= thr) & (vr1 > 0)
    dd = mask_dates(m)
    print(f"  today qualifies: {bool(m.iloc[-1])}")
    if len(dd) == 0:
        print("  no sessions")
        continue
    mondays = pd.DatetimeIndex([d for d in dd if d.weekday() == 0])
    lowvix = pd.DatetimeIndex([d for d in dd if C["^VIX"].get(d, np.nan) < 20])
    print(f"  total count: {len(dd)}  ({dd[0].date()} .. {dd[-1].date()})")
    print(f"  of those, Mondays: {len(mondays)}  "
          f"({', '.join(str(d.date()) for d in mondays)})")
    print(f"  of those, VIX < 20 at that close: {len(lowvix)}")
    print("    VIX<20 dates: " + ", ".join(str(d.date()) for d in lowvix))
    both = pd.DatetimeIndex([d for d in lowvix if d.weekday() == 0])
    print(f"    of the VIX<20 subset, Mondays: {len(both)}  "
          f"({', '.join(str(d.date()) for d in both)})")
    g_next = fwd_ret(C["^GSPC"], 1)
    v_next = fwd_ret(C["^VIX"], 1)
    line("  VIX<20 subset: ^GSPC next session (lag=0, h=1)", g_next, lowvix,
         [("ctrl all sessions", g_next.dropna()),
          ("ctrl whole cell", g_next.reindex(dd).dropna())])
    line("  VIX<20 subset: ^VIX next session (lag=0, h=1)", v_next, lowvix,
         [("ctrl all sessions", v_next.dropna()),
          ("ctrl whole cell", v_next.reindex(dd).dropna())])
    ve = line("  whole cell: ^GSPC next session (lag=0, h=1)", g_next, dd,
              [("ctrl all sessions", g_next.dropna())])
    if len(ve):
        era("  F ^GSPC next", ve)

# ===========================================================================
# G. Open-idea marks (prices only)
# ===========================================================================
print("\n" + "=" * 78)
print("=== G. Open-idea marks (ADJUSTED bars: master_prices stores adjusted "
      "OHLCV; load_prices exposes no raw basis) ===")


def atr_at(ticker, d):
    f = raw_px.get(ticker)
    if f is None or len(f) == 0:
        return np.nan
    a = pd.Series(np.asarray(wilder_atr(f["High"], f["Low"], f["Close"]),
                             dtype=float), index=f.index)
    a = a[a.index <= pd.Timestamp(d)]
    return float(a.iloc[-1]) if len(a) else np.nan


def mark(ticker, d0, field0, d1, atr_date):
    f = raw_px.get(ticker)
    if f is None or len(f) == 0:
        print(f"  {ticker}: NO HISTORY")
        return
    try:
        p0 = float(f.loc[pd.Timestamp(d0), field0])
        p1 = float(f.loc[pd.Timestamp(d1), "Close"])
    except KeyError as e:
        print(f"  {ticker}: missing bar {e}")
        return
    a = atr_at(ticker, atr_date)
    pct = p1 / p0 - 1.0
    atru = (p1 - p0) / a if a and not np.isnan(a) else np.nan
    print(f"  {ticker}: {field0} {d0} = {p0:.4f}  ->  Close {d1} = {p1:.4f}   "
          f"{100*pct:+.3f}%   ATR14({atr_date}) = {a:.4f}   "
          f"move = {atru:+.2f} ATR")


mark("UUP", "2026-09-17", "Open", "2026-09-21", "2026-09-16")
mark("IWM", "2026-09-18", "Close", "2026-09-21", "2026-09-17")

# ===========================================================================
# H. Frozen idea parameters as of the 2026-09-21 close
# ===========================================================================
print("\n" + "=" * 78)
print("=== H. Frozen levels at the 2026-09-21 close (Wilder-14 ATR) ===")
for t in ("QQQ", "XLE", "SPY", "IWM"):
    f = raw_px.get(t)
    if f is None or len(f) == 0:
        print(f"  {t}: NO HISTORY")
        continue
    a = atr_at(t, ASOF)
    c = f["Close"][f["Close"].index <= ASOF]
    print(f"  {t}: close {c.iloc[-1]:.4f}  Wilder-14 ATR {a:.4f}  "
          f"({100*a/c.iloc[-1]:.2f}% of price)  bar {c.index[-1].date()}")

print("\ndone.")
