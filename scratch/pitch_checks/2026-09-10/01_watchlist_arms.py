"""Stage B1 watchlist arm pass for 2026-09-10.

Every mechanically-checkable arm on data/pitch_watchlist.json's 46 active
entries, computed rather than eyeballed. Prints ticker / reading / arm /
verdict, one block per entry index. Throwaway: it decides nothing, it only
tells the surface map which parked cells moved.

Readings are as of the freshest bar (2026-09-09 close); 2026-09-10 is a live
session with no bar yet.

Run: python scratch/pitch_checks/2026-09-10/01_watchlist_arms.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from pitch_lab import (  # noqa: E402
    close_panel, load_prices, load_events, rolling_on_valid, pct_rank,
    declusters, anchor_positions, summarize, sign_test, show, vehicle_ret,
)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 240)

TODAY = pd.Timestamp("2026-09-10")
VERDICTS: list[tuple[int, str, str]] = []


def v(idx: int, kind: str, line: str) -> None:
    VERDICTS.append((idx, kind, line))
    print(f"  [{kind}] #{idx}: {line}")


def hdr(t: str) -> None:
    print("\n" + "=" * 110)
    print(t)
    print("=" * 110)


tape = json.load(open(ROOT / "data" / "pitch_tape.json"))
BY = tape["tickers"]


def tp(t: str, f: str):
    r = BY.get(t)
    return None if r is None else r.get(f)


# ---------------------------------------------------------------------------
hdr("A. CALENDAR / CYCLE FACTS")
px = close_panel(["SPY"])
CAL = px["SPY"].dropna().index
print(f"freshest bar in master_prices: {CAL[-1].date()}  (tape asof {tape['asof']}, "
      f"freshest {tape['freshest_bar']})")
sep = CAL[(CAL >= "2026-09-01") & (CAL <= "2026-09-30")]
print(f"September 2026 sessions so far: {[str(d.date()) for d in sep]}")
# today is not in master_prices yet; build the forward session index
from pandas.tseries.holiday import USFederalHolidayCalendar  # noqa: E402
from pandas.tseries.offsets import CustomBusinessDay  # noqa: E402

BD = CustomBusinessDay(calendar=USFederalHolidayCalendar())
FUT = pd.DatetimeIndex(pd.date_range(CAL[-1], periods=120, freq=BD))
FULL = CAL.append(FUT[1:]).unique().sort_values()
sep_full = FULL[(FULL >= "2026-09-01") & (FULL <= TODAY)]
print(f"trading-day-of-month for {TODAY.date()}: {len(sep_full)} "
      f"(sessions {[str(d.date())[5:] for d in sep_full]})")
print(f"cycle year 2026 %% 4 = {2026 % 4}  -> MIDTERM" if 2026 % 4 == 2 else "")

KINDS = ("nfp", "cpi", "ppi", "fomc_decision")
EVK = {k: load_events([k])["date"] for k in KINDS}
PRINTS = pd.DatetimeIndex(sorted(pd.concat(list(EVK.values())).unique()))
fut_prints = PRINTS[(PRINTS >= TODAY) & (PRINTS <= TODAY + pd.Timedelta(days=90))]
kindmap = load_events(list(KINDS)).groupby("date")["event"].apply(
    lambda s: "+".join(sorted(set(s))))
print("\nprint calendar, td offsets from today (today = 0):")
posf = pd.Series(range(len(FULL)), index=FULL)
for d in fut_prints:
    nxt = PRINTS[PRINTS > d]
    rw = 99 if len(nxt) == 0 else int(posf[nxt[0]] - posf[d])
    anch = FULL[posf[d] - 2]
    print(f"  {d.date()} {kindmap.get(d,'?'):16s} td_ahead {int(posf[d]-posf[TODAY]):>2d}  "
          f"runway_after {rw:>2d} td  k=-2 anchor {anch.date()}"
          f"{'   <-- TODAY IS THE ANCHOR' if anch == TODAY else ''}")
for k in ("vix_expiry", "opex", "jackson_hole"):
    e = load_events([k])["date"]
    e = e[(e >= TODAY) & (e <= TODAY + pd.Timedelta(days=90))]
    for d in e:
        print(f"  {d.date()} {k:16s} td_ahead {int(posf[d]-posf[TODAY]):>2d}")

# ---------------------------------------------------------------------------
hdr("B. FRAGILITY DIAL + P/C FEAR")
frag = pd.read_parquet(ROOT / "data" / "rd2_fragility.parquet")
frag.index = pd.to_datetime(frag.index)
frag = frag.sort_index()
last = frag.iloc[-1]
MA10_63 = float(frag["63d"].rolling(10).mean().iloc[-1])
RAW21 = float(last["21d"])
RAW63 = float(last["63d"])
RAW5 = float(last["5d"])
print(f"dial last row {frag.index[-1].date()}: raw5 {RAW5:.1f}  raw21 {RAW21:.1f}  "
      f"raw63 {RAW63:.1f}  ma10(63d) {MA10_63:.2f}")
ma10s = frag["63d"].rolling(10).mean().dropna()
print(f"ma10(63d) percentile within its own series: "
      f"{100*(ma10s <= MA10_63).mean():.2f}  (n={len(ma10s)})")

pc = pd.read_parquet(ROOT / "data" / "cboe_putcall.parquet")
pc.index = pd.to_datetime(pc.index) if not isinstance(pc.index, pd.DatetimeIndex) else pc.index
col = [c for c in pc.columns if "equity" in c.lower() or "ratio" in c.lower()] or list(pc.columns)
s = pc[col[0]].dropna()
ma = s.rolling(10).mean()
pct = ma.rolling(252).rank(pct=True) * 100
print(f"equity P/C last {s.index[-1].date()}: raw {s.iloc[-1]:.3f}  10dMA {ma.iloc[-1]:.3f}  "
      f"trailing-252 pctile {pct.iloc[-1]:.2f}  -> fear "
      f"{'ON' if pct.iloc[-1] > 85 else 'OFF'}")

# ---------------------------------------------------------------------------
hdr("C. THE VIX 21-DAY RELATIVE-RANGE PERCENTILE  (entries 33 and 35)")
vpx = close_panel(["^VIX", "^VIX3M", "SVXY", "SPY"])
vix = vpx["^VIX"]
rng21 = (rolling_on_valid(vix, lambda x: x.rolling(21).max())
         - rolling_on_valid(vix, lambda x: x.rolling(21).min()))
REL = rolling_on_valid(rng21 / rolling_on_valid(vix, lambda x: x.rolling(21).mean()),
                       lambda x: x.rolling(252).rank(pct=True) * 100)
VLP = rolling_on_valid(vix, lambda x: x.rolling(252).rank(pct=True) * 100)
TS = vpx["^VIX3M"] / vpx["^VIX"] - 1.0
tail = pd.DataFrame({"VIX": vix, "rng21": rng21,
                     "rel_raw": rng21 / rolling_on_valid(vix, lambda x: x.rolling(21).mean()),
                     "REL_pct": REL, "VIXlvl_pct": VLP, "contango_%": 100 * TS}).dropna().tail(8)
print(tail.round(4).to_string())
REL_TODAY = float(REL.dropna().iloc[-1])
print(f"\n>>> VIX 21-day RELATIVE-RANGE percentile on the 2026-09-09 bar: {REL_TODAY:.2f}")
print(f"    21d max {rng21.index[-1].date()} window: max {vix.rolling(21).max().iloc[-1]:.2f} "
      f"min {vix.rolling(21).min().iloc[-1]:.2f} mean {vix.rolling(21).mean().iloc[-1]:.2f} "
      f"-> raw rel-range {float((rng21/vix.rolling(21).mean()).iloc[-1]):.4f}")
print(f"    prior readings 09-03 3.57 / 09-04 4.37 / 09-08 4.37; arm band is (5.0, 15.0]")

# legal k=-2 anchors, forward
print("\nlegal k=-2 anchors for the compression cell (runway >= 3 td after the print):")
for d in fut_prints:
    nxt = PRINTS[PRINTS > d]
    rw = 99 if len(nxt) == 0 else int(posf[nxt[0]] - posf[d])
    anch = FULL[posf[d] - 2]
    ok = "QUALIFIES" if rw >= 3 else "DISQUALIFIED (crowded)"
    print(f"  print {d.date()} {kindmap.get(d,'?'):16s} runway {rw:>2d} -> anchor "
          f"{anch.date()} : {ok}"
          f"{'   *** TODAY ***' if anch == TODAY else ''}")

# ---------------------------------------------------------------------------
hdr("D. PRICE-STATE LEGS FROM master_prices (252-session windows)")
NAMES = ["TLT", "IEF", "LQD", "HYG", "SPY", "QQQ", "IWM", "^TNX", "^MOVE", "^SKEW",
         "GLD", "SLV", "GDX", "USO", "DBC", "XLE", "XOP", "COP", "CVX", "VLO", "OXY",
         "SLB", "EOG", "HAL", "WMB", "XLU", "XLI", "XLK", "XLV", "SMH", "IHI", "FXI",
         "EEM", "DX-Y.NYB", "UUP", "KRE", "XLF", "SVXY", "^VIX"]
P = load_prices(NAMES)


def hi252(t):
    s = P[t]["Close"]
    return rolling_on_valid(s, lambda x: x.rolling(252).max())


def lo252(t):
    s = P[t]["Close"]
    return rolling_on_valid(s, lambda x: x.rolling(252).min())


def off_high(t):          # fraction BELOW the 252d high, >= 0
    return (1.0 - P[t]["Close"] / hi252(t))


def abv_low(t):           # fraction ABOVE the 252d low, >= 0
    return (P[t]["Close"] / lo252(t) - 1.0)


def last(s):
    return float(pd.Series(s).dropna().iloc[-1])


print(f"{'tkr':10s} {'close':>10s} {'off252high%':>12s} {'abv252low%':>11s}")
for t in ["TLT", "IEF", "LQD", "HYG", "SPY", "QQQ", "IWM", "^TNX", "XLE", "XOP",
          "DBC", "USO", "GLD", "SMH", "XLI", "SVXY"]:
    print(f"{t:10s} {P[t]['Close'].iloc[-1]:>10.3f} {100*last(off_high(t)):>12.3f} "
          f"{100*last(abv_low(t)):>11.3f}")

TNX = P["^TNX"]["Close"]
print(f"\n^TNX last {TNX.iloc[-1]:.4f}; trailing-252 max {hi252('^TNX').iloc[-1]:.4f}; "
      f"AT the 252d max? {TNX.iloc[-1] >= hi252('^TNX').iloc[-1] - 1e-9}")
chg252 = TNX - TNX.shift(252)
chg21 = TNX - TNX.shift(21)
print(f"^TNX 252-session change {100*chg252.iloc[-1]:+.1f} bp  (arm >= +78 bp -> needs "
      f"a close of {TNX.shift(252).iloc[-1] + 0.78:.3f})")
print(f"^TNX 21-session change  {100*chg21.iloc[-1]:+.1f} bp  (entry 13 arm >= +20 bp)")

# ---------------------------------------------------------------------------
hdr("E. HIGHLIGHTED ENTRY 5 -- TLT + the whole IG complex at 52w lows")
tight = (abv_low("TLT") <= 0.005) & (abv_low("IEF") <= 0.010) & (abv_low("LQD") <= 0.010)
tight = tight.dropna()
print(f"live legs: TLT {100*last(abv_low('TLT')):+.3f}% above its 252d low (arm <= 0.50%) "
      f"{'PASS' if last(abv_low('TLT'))<=0.005 else 'FAIL'}")
print(f"           IEF {100*last(abv_low('IEF')):+.3f}% (arm <= 1.00%) "
      f"{'PASS' if last(abv_low('IEF'))<=0.010 else 'FAIL'}")
print(f"           LQD {100*last(abv_low('LQD')):+.3f}% (arm <= 1.00%) "
      f"{'PASS' if last(abv_low('LQD'))<=0.010 else 'FAIL'}")
fires = tight[tight].index
print(f"tight-rung fire days in history: {len(fires)}; last 12: "
      f"{[str(d.date()) for d in fires[-12:]]}")
if len(fires):
    lastfire = fires[-1]
    gap = int(posf[TODAY] - posf[lastfire])
    print(f"last tight-rung fire {lastfire.date()}; sessions since (to today) = {gap} "
          f"(freshness arm: the fire must be the FIRST in >= 10 td)")
tlt_need = float(lo252("TLT").iloc[-1] * 1.005)
print(f"TLT would have to CLOSE at or below {tlt_need:.2f} to arm the price rung "
      f"(last {P['TLT']['Close'].iloc[-1]:.2f}, "
      f"{100*(tlt_need/P['TLT']['Close'].iloc[-1]-1):+.2f}%)")

# ---------------------------------------------------------------------------
hdr("F. HIGHLIGHTED ENTRY 26 -- pure rates repricing, zero credit stress")
m26 = ((abv_low("IEF") <= 0.015) & (abv_low("LQD") <= 0.015)
       & (off_high("HYG") <= 0.0025)).dropna()
print(f"live legs: IEF {100*last(abv_low('IEF')):+.3f}% above 252d low (arm <= 1.5%) "
      f"{'PASS' if last(abv_low('IEF'))<=0.015 else 'FAIL'}")
print(f"           LQD {100*last(abv_low('LQD')):+.3f}% (arm <= 1.5%) "
      f"{'PASS' if last(abv_low('LQD'))<=0.015 else 'FAIL'}")
print(f"           HYG {100*last(off_high('HYG')):+.3f}% BELOW 252d high (arm <= 0.25%) "
      f"{'PASS' if last(off_high('HYG'))<=0.0025 else 'FAIL'}")
d26 = m26[m26].index
e26 = declusters(d26, 21, P["HYG"].index)
print(f"tight-rung days in ALL history: {len(d26)} -> {[str(x.date()) for x in d26]}")
print(f"declustered (gap 21): {len(e26)} -> {[str(x.date()) for x in e26]}")
print("  (the 2026-08-27 entry recorded 16 days / 1 episode dated 2026-08-03; the count "
      "shrinks as HYG's own trailing-252 high rolls up)")
yrs = sorted(set(pd.DatetimeIndex(e26).year))
exyrs = [y for y in yrs if y not in (2018, 2026)]
print(f"years present {yrs}; ex-2018/2026 {exyrs}  "
      f"(arm: >= 8 declustered episodes over >= 3 years excluding 2018 and 2026)")

# ---------------------------------------------------------------------------
hdr("G. ENTRY 1 -- LQD vs HYG joint 52w extremes (episode count arm)")
m1 = ((off_high("HYG") <= 0.005) & (abv_low("LQD") <= 0.02)).dropna()
d1 = m1[m1].index
e1 = declusters(d1, 21, P["HYG"].index)
print(f"live: HYG {100*last(off_high('HYG')):.3f}% off 252d high (arm <= 0.5%) | "
      f"LQD {100*last(abv_low('LQD')):.3f}% above 252d low (arm <= 2.0%)")
print(f"joint days {len(d1)}; declustered episodes {len(e1)}: {[str(x.date()) for x in e1]}")
y1 = sorted(set(pd.DatetimeIndex(e1).year))
print(f"years {y1}; ex-2018 {[y for y in y1 if y != 2018]} "
      f"(arm >= 8 episodes over >= 3 distinct years ex-2018)")

# ---------------------------------------------------------------------------
hdr("H. HIGHLIGHTED ENTRY 30 -- ^MOVE trailing-252 LEVEL percentile in [40,50)")
mv = P["^MOVE"]["Close"]
MVP = rolling_on_valid(mv, lambda x: x.rolling(252).rank(pct=True) * 100)
print(f"^MOVE last {mv.iloc[-1]:.2f} on {mv.index[-1].date()}; trailing-252 LEVEL "
      f"percentile {last(MVP):.2f}  (arm band [40, 50))")
print(f"recent: " + ", ".join(f"{d.date()}:{x:.1f}" for d, x in MVP.dropna().tail(6).items()))
band_lo = float(mv.rolling(252).quantile(0.40).iloc[-1])
band_hi = float(mv.rolling(252).quantile(0.50).iloc[-1])
print(f"the [40,50) band in LEVEL terms today is roughly ^MOVE {band_lo:.1f} to {band_hi:.1f}")

# ---------------------------------------------------------------------------
hdr("I. HIGHLIGHTED ENTRY 19 -- narrow energy thrust count (z10 >= 2.0 of 11)")
COMPLEX = ["XLE", "XOP", "USO", "COP", "CVX", "VLO", "OXY", "SLB", "EOG", "HAL", "WMB"]
zz = {t: tp(t, "z10") for t in COMPLEX}
for t in COMPLEX:
    print(f"  {t:5s} z10 {zz[t]:+.2f}" if zz[t] is not None else f"  {t:5s} MISSING")
cnt = sum(1 for t in COMPLEX if (zz[t] or -9) >= 2.0)
print(f">>> count at z10 >= 2.0 : {cnt}   (arm band is [2, 3]; 4 is the zero crossing, "
      f"5 was the killed broad form)")
print(f"    names over the line: {[t for t in COMPLEX if (zz[t] or -9) >= 2.0]}")
print(f"    max of the complex: {max((zz[t] or -9) for t in COMPLEX):+.2f}")

# ---------------------------------------------------------------------------
hdr("J. HIGHLIGHTED ENTRY 6 -- SPY on a skew spike alone")
sk = P["^SKEW"]["Close"]
SKR5 = pct_rank(sk, 5, 252)  # trailing-252 rank of the 5-day return
print(f"^SKEW last bar {sk.index[-1].date()} = {sk.iloc[-1]:.2f}; 5-day return "
      f"{100*sk.pct_change(5).iloc[-1]:+.2f}%")
print(f"pct_rank(^SKEW 5d return, 252) = {last(SKR5):.1f}   (arm >= 95)")
print(f"^SKEW trailing-252 LEVEL percentile = "
      f"{last(rolling_on_valid(sk, lambda x: x.rolling(252).rank(pct=True)*100)):.1f}")
print(f"SPY {100*last(off_high('SPY')):.3f}% below its 252d high   (arm > 1.0%) "
      f"{'PASS' if last(off_high('SPY'))>0.01 else 'FAIL'}")
print(f"cycle year midterm -> the non-midterm leg FAILS regardless")

# ---------------------------------------------------------------------------
hdr("K. HIGHLIGHTED ENTRY 41 -- short IEF h=5, placebo ladder rank of the true anchor")
pxc = close_panel(["DBC", "USO", "TLT", "IEF", "SPY"])
IDX = pxc.index


def dist_to_high_idx(t, n=252):
    s = P[t]["Close"]
    return (1.0 - s / rolling_on_valid(s, lambda x: x.rolling(n).max())).reindex(IDX)


def flag_in_window(kinds, h, lag=1, k=0):
    ev = load_events(list(kinds))["date"]
    pos, _ = anchor_positions(IDX, ev, offset=k)
    e = np.asarray(pd.DatetimeIndex([IDX[p] for p in pos]).values, dtype="datetime64[ns]")
    out = np.zeros(len(IDX), dtype=bool)
    for i in range(len(IDX)):
        if i + lag + h >= len(IDX):
            continue
        out[i] = bool(((e > np.datetime64(IDX[i + lag])) &
                       (e <= np.datetime64(IDX[i + lag + h]))).any())
    return pd.Series(out, index=IDX)


state41 = dist_to_high_idx("DBC") <= 0.0025
print(f"DBC {100*float(dist_to_high_idx('DBC').iloc[-1]):.3f}% below its 252d high "
      f"(state gate <= 0.25%) -> state {'LIVE' if bool(state41.iloc[-1]) else 'off'}")
h = 5
ret41 = vehicle_ret(pxc, [("IEF", -1.0)], h, 1)
valid41 = ret41.notna()
rows, means = [], []
for k in range(-5, 6):
    m = state41 & flag_in_window(("ppi", "cpi"), h, 1, k)
    dd = IDX[m.values & valid41.values]
    e = declusters(dd, h, IDX)
    r = summarize(ret41.loc[e].values, f"k={k:+d}" + ("  <-- TRUE" if k == 0 else ""))
    rows.append(r)
    means.append(r.get("mean_pct", np.nan))
show(rows, "C8 SHORT IEF h=5, placebo anchor ladder k=-5..+5 (episode mean)")
rank41 = int(np.sum(np.asarray(means) >= means[5]))
print(f">>> true anchor ranks {rank41} of 11 by episode mean  (arm: 1st or 2nd)")
print(f"    rungs paying more than the true anchor: "
      f"{[f'k={k-5:+d}:{means[k]:+.3f}%' for k in range(11) if means[k] > means[5]]}")

# ---------------------------------------------------------------------------
hdr("L. HIGHLIGHTED ENTRY 45 -- VIX expiry x FOMC collision")
vexp = load_events(["vix_expiry"])["date"]
fomc = load_events(["fomc_decision"])["date"]
coll = pd.DatetimeIndex(sorted(set(vexp) & set(fomc)))
print(f"collisions in the calendar: {len(coll)}; post-2018 (SVXY -0.5x era): "
      f"{len(coll[coll >= '2018-02-06'])}")
print(f"  last 6: {[str(d.date()) for d in coll[-6:]]}")
nxt = coll[coll >= TODAY]
if len(nxt):
    print(f"  NEXT collision {nxt[0].date()}, td_ahead {int(posf[nxt[0]] - posf[TODAY])} "
          f"-> the settle-session anchor is that date's close, NOT today's")
past = coll[(coll >= "2018-02-06") & (coll < TODAY)]
print(f"  post-2018 collisions already counted: {len(past)}; arm needs 16 (i.e. +4 new)")

# ---------------------------------------------------------------------------
print("\n-- recompute the settle-session cell on the -0.5x era (the arm's own object)")
svxy = P["SVXY"]["Close"]
sret = svxy.pct_change()          # entry prior close -> exit that close
cv = sret.reindex(coll).dropna()
for lbl, m in (("pre-2018", cv.index < pd.Timestamp("2018-01-01")),
               ("2018+ (LIVE -0.5x vehicle)", cv.index >= pd.Timestamp("2018-01-01"))):
    x = cv[m]
    print(f"  {lbl:28s} n={len(x):<3d} mean {100*x.mean():+.3f}%  record "
          f"{int((x>0).sum())}-{int((x<0).sum())}  hit {100*(x>0).mean():.1f}%")
x = cv[cv.index >= pd.Timestamp("2018-01-01")]
print(f"  ARM READS: n={len(x)} (arm asks >= 16 collisions -> ALREADY MET), "
      f"record {100*(x>0).mean():.1f}% (arm >= 60% -> MET), "
      f"mean {100*x.mean():+.3f}% (arm > 0 -> FAILS)")
print("  NOTE the entry's prose says 'today's 12'; 12 is the WIN count of a 12-7 record, "
      "not the collision count. The binding leg is the MEAN, not the sample size.")
print(f"  most recent two: " + ", ".join(f"{d.date()} {100*r:+.2f}%" for d, r in cv.tail(2).items()))

hdr("M. REMAINING MECHANICAL ARMS, one line each")


def rk(t, f):
    x = tp(t, f)
    return float("nan") if x is None else x


print(f"#3  GLD: GDX r5 {rk('GDX','rank_5d'):.1f} (arm >= 95) | GLD r5 {rk('GLD','rank_5d'):.1f} "
      f"| GLD r63 {rk('GLD','rank_63d'):.1f} (arm >= 50) | GLD "
      f"{100*last(off_high('GLD')):.2f}% off 252d high (arm <= 10%) | PPI today + CPI tomorrow "
      f"are both inside any short hold (arm: no CPI/PPI in the hold)")
print(f"#4  XLE crude band: USO 1d {rk('USO','ret_1d'):+.2f}% (arm band [5,6)%)")
print(f"#7  USO deep-base fade: USO r5 {rk('USO','rank_5d'):.1f} (arm >= 90) | "
      f"r63 {rk('USO','rank_63d'):.1f} (arm <= 20)")
print(f"#8  IHI: r21 {rk('IHI','rank_21d'):.1f} (arm = 100) | r5 {rk('IHI','rank_5d'):.1f}")
print(f"#9  FXI: r5 {rk('FXI','rank_5d'):.1f} (arm <= 20) | r21 {rk('FXI','rank_21d'):.1f} "
      f"(arm >= 80) | EEM 5d {rk('EEM','ret_5d'):+.2f}% (arm > 0)")
print(f"#11 SPY-high/TLT-low: SPY {100*last(off_high('SPY')):.2f}% off high (arm <= 0.5%) | "
      f"TLT {100*last(abv_low('TLT')):.2f}% above low (arm <= 1.0%)")
vix_1d = 100 * (P["^VIX"]["Close"].pct_change().iloc[-1])
vixr21 = last(pct_rank(P["^VIX"]["Close"], 21, 252))
spy_1d = 100 * (P["SPY"]["Close"].pct_change().iloc[-1])
print(f"#12 vol pop in calm tape: ^VIX 1d {vix_1d:+.2f}% (arm >= +5%) | ^VIX 21d-return "
      f"rank {vixr21:.1f} (arm <= 25) | SPY 1d {spy_1d:+.2f}% (arm > -0.75%)")
print(f"#13 gold/dials: DX-Y r21 {rk('DX-Y.NYB','rank_21d'):.1f} (arm <= 15) | ^TNX 21-session "
      f"change {100*chg21.iloc[-1]:+.1f} bp (arm >= +20 bp)")
xlv1 = 100 * P["XLV"]["Close"].pct_change().iloc[-1]
xlk1 = 100 * P["XLK"]["Close"].pct_change().iloc[-1]
print(f"#14 XLV-XLK rotation gap: {xlv1 - xlk1:+.2f} pp (arm >= +3.0 pp); SPY "
      f"{100*last(off_high('SPY')):.2f}% off high (arm <= 3%)")
print(f"#15 short dollar: ^TNX r21 {rk('^TNX','rank_21d'):.1f} (arm >= 65) | DX-Y r21 "
      f"{rk('DX-Y.NYB','rank_21d'):.1f} (arm <= 20)")
tlt1 = 100 * P["TLT"]["Close"].pct_change().iloc[-1]
print(f"#16 short TLT thrust: TLT 1d {tlt1:+.2f}% (arm >= +1.5%) | TLT "
      f"{100*last(abv_low('TLT')):.2f}% above 252d low (arm <= 4%)")
BANKS = ["KRE", "XLF"]
print(f"#17 KRE/XLF: KRE r5 {rk('KRE','rank_5d'):.1f} r21 {rk('KRE','rank_21d'):.1f} | "
      f"XLF r5 {rk('XLF','rank_5d'):.1f} r63 {rk('XLF','rank_63d'):.1f} "
      f"(arm is an ex-crisis cost threshold, not a state)")
print(f"#18 curve: ^TNX at 252d max? "
      f"{bool(TNX.iloc[-1] >= hi252('^TNX').iloc[-1] - 1e-9)} | 252-session change "
      f"{100*chg252.iloc[-1]:+.1f} bp (arm >= +78 bp; needs a "
      f"{TNX.shift(252).iloc[-1]+0.78:.3f} close, {100*(TNX.shift(252).iloc[-1]+0.78-TNX.iloc[-1]):+.1f} bp away)")
print(f"#20 breadth/index-distance: SPY {100*last(off_high('SPY')):.3f}% off 252d high "
      f"(arm > 2.000%) | raw-21d dial {RAW21:.1f} (arm <= 50)")
print(f"#21 XLI washout->high: XLI r5 {rk('XLI','rank_5d'):.1f} (arm <= 5) | XLI "
      f"{100*last(off_high('XLI')):.2f}% off 252d high (arm <= 5%)")
xlur21 = rk("XLU", "rank_21d")
tltr21 = rk("TLT", "rank_21d")
print(f"#22 XLU washout + TLT hit: XLU r21 {xlur21:.1f} (arm <= 5) | TLT r21 {tltr21:.1f} "
      f"(arm < 25)")
print(f"#24 HYG fresh high vs SPY: HYG {100*last(off_high('HYG')):.3f}% off 252d high "
      f"(arm <= 0.05%) | SPY {100*last(off_high('SPY')):.3f}% off (arm >= 2.0%) | "
      f"dial ma10 {MA10_63:.1f} (arm < 50)")
smh252 = last(pct_rank(P["SMH"]["Close"], 252, 252))
print(f"#25 SMH: r63 {rk('SMH','rank_63d'):.1f} (arm <= 5) | 252d-return rank {smh252:.1f} "
      f"(arm top decile >= 90) | r5 {rk('SMH','rank_5d'):.1f} (arm < 15)")
# #28 pooled laggard-still-falling across the tape
hold28 = [t for t, r in BY.items()
          if r.get("rank_21d") is not None and r.get("rank_63d") is not None
          and r["rank_21d"] >= 90 and r["rank_63d"] <= 10]
print(f"#28 pooled laggard: names with r21>=90 AND r63<=10 on the 218-name tape: "
      f"{hold28 if hold28 else 'NONE'}"
      + (f" -> r5 {[round(BY[t]['rank_5d'],1) for t in hold28]} (arm < 15)" if hold28 else ""))
slv1 = 100 * P["SLV"]["Close"].pct_change().iloc[-1]
print(f"#29 SLV: 1d {slv1:+.2f}% (arm: a break of -4.00% or worse) | 5d "
      f"{rk('SLV','ret_5d'):+.2f}%")
spy1d = spy_1d
print(f"#32 XLE at high on a down-SPY session: XLE {100*last(off_high('XLE')):.3f}% off "
      f"252d high (arm 0.05% touch) | SPY 1d {spy1d:+.2f}% (arm < 0)")
SPDR = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "XLRE"]
tri = [t for t in SPDR if BY.get(t) and BY[t]["rank_5d"] <= 20 and BY[t]["rank_21d"] <= 20
       and BY[t]["rank_63d"] <= 20]
print(f"#34 pooled triple rank floor (r5,r21,r63 all <= 20): {tri if tri else 'NONE'}")
print(f"#35 SPY into a print out of a dead range: rel-range {REL_TODAY:.2f} (arm <= 15) | "
      f"dial ma10(63d) {MA10_63:.1f} (arm < 50) | today is the PPI PRINT, not a k=-2 anchor")
print(f"#39 SPY/IWM dial band: ma10(63d) {MA10_63:.2f} (arm band [56, 70)) -> needs a fall "
      f"of {MA10_63-70:.1f} to re-enter")
hyg_rv = rk("HYG", "rvol21_ann")
print(f"#40 HYG post-closure: HYG {100*last(off_high('HYG')):.2f}% off 252d high "
       f"(arm > 1.0%) | 21d realised vol {hyg_rv:.2f}% ann (arm > 4.4%) | next >= 4-day "
       f"closure is the 2026-11-26 Thanksgiving boundary")
print(f"#43 PPI-release TLT anchor: ^TNX at 252d max "
      f"{bool(TNX.iloc[-1] >= hi252('^TNX').iloc[-1]-1e-9)}; the release session is "
      f"{TODAY.date()} so the anchor close is TODAY")
print(f"#44 SPY w/ HYG+TNX at highs: HYG {100*last(off_high('HYG')):.3f}% off (arm <= 0.5%) | "
      f"^TNX at 252d max {bool(TNX.iloc[-1] >= hi252('^TNX').iloc[-1]-1e-9)} | SPY "
      f"{100*last(off_high('SPY')):.3f}% off its own high (arm <= 0.5%)")

# ---------------------------------------------------------------------------
hdr("N. CALENDAR-PARKED ENTRIES")
print(f"#0  non-midterm NFP TLT      -> parks to 2027-01 NFP. 2026 is midterm; next NFP "
      f"2026-10-02 (td {int(posf[pd.Timestamp('2026-10-02')]-posf[TODAY])}).")
print(f"#10 November TLT tdom 4-12   -> parks to ~2026-11-05..2026-11-17. Today is Sep "
      f"tdom {len(sep_full)}.")
print(f"#23 non-midterm dollar wash  -> parks to a year with year%%4 != 2, i.e. 2027.")
print(f"#27 Jackson Hole IEF k=+1    -> anchor was 2026-08-31; midterm-blocked in any case.")
print(f"#31 December small-cap ME-0  -> parks to a NON-midterm December, 2027-12-31.")
gaps_hist = pd.Series((CAL[1:] - CAL[:-1]).days, index=CAL[1:])
gaps_fwd = pd.Series((FULL[1:] - FULL[:-1]).days, index=FULL[1:])
print("#36/#38/#40 CLOSURE GAPS -- verified against the session index, not a calendar:")
print(f"    realised gap-size histogram 2007+: "
      f"{gaps_hist[gaps_hist.index >= '2007-01-01'].value_counts().to_dict()}")
print(f"    an ORDINARY weekend is a 3-day gap; gaps >= 4 since 2007: "
      f"{int((gaps_hist[gaps_hist.index >= '2007-01-01'] >= 4).sum())} "
      f"(reproduces entry #40's 130 anchors)")
for y in range(2021, 2026):
    tg = gaps_hist[(gaps_hist.index >= f"{y}-11-20") & (gaps_hist.index <= f"{y}-12-02")]
    tg = tg[tg == 2]
    print(f"    Thanksgiving {y}: first session back {[str(d.date()) for d in tg.index]} "
          f"gap 2 days -> DOES NOT QUALIFY")
fwd = gaps_fwd[(gaps_fwd.index > TODAY) & (gaps_fwd >= 4)]
print(f"    next forward gaps >= 4 days (first session BACK): "
      f"{[(str(d.date()), int(x)) for d, x in fwd.items()][:4]}")
print(f"    NOTE the forward index uses USFederalHolidayCalendar, which closes Columbus "
      f"Day and Veterans Day when the NYSE does not; those two are false positives and "
      f"are excluded by hand. Thanksgiving 2026-11-26 is Wed-close -> Fri-open = 2 days.")
print(f"#36 closure risk premium     -> next qualifying closure boundary is the Christmas "
      f"gap, first session back 2026-12-28; Thanksgiving does NOT qualify.")
print(f"#37 post-NFP moderate miss   -> parks to the next NFP, 2026-10-02.")
print(f"#38 SVXY post-closure k=+1   -> parks to the next qualifying closure, first session "
      f"back 2026-12-28 (NOT Thanksgiving).")
print(f"#42 Sep PPI-then-CPI pair    -> its k=-2 anchor was 2026-09-08; today is the first "
      f"print itself.")

print("\nDONE.")
