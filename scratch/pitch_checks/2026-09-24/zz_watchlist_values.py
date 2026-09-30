"""Watchlist verdict values for the 2026-09-24 run (tape through the 09-23 close).

Computes every live arm number the tape does not carry, under each entry's own
script definition. Output: zz_watchlist_values_out.txt (same folder).
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
from pitch_lab import *  # noqa: F401,F403,E402
from pitch_lab import (close_panel, load_prices, pct_rank, zscore,  # noqa: E402
                       rolling_on_valid, vehicle_ret, wilder_atr)

warnings.filterwarnings("ignore")
TAPE = json.load(open(ROOT / "data" / "pitch_tape.json"))["tickers"]
D = pd.Timestamp("2026-09-23")


def tp(t: str, *keys: str) -> str:
    r = TAPE.get(t)
    if r is None:
        return f"{t}: NOT IN TAPE"
    return f"{t} " + " ".join(f"{k}={r.get(k)}" for k in keys)


def sec(title: str) -> None:
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def last(s: pd.Series) -> float:
    s = s.dropna()
    return float(s.iloc[-1]) if len(s) else float("nan")


def lastdate(s: pd.Series) -> str:
    s = s.dropna()
    return str(s.index[-1].date()) if len(s) else "none"


SPDR9 = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
FAM29 = ["SPY", "QQQ", "IWM", "DIA", "EFA", "EEM", "EWJ", "FXI", "EWZ",
         "XLK", "XLV", "XLF", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC",
         "SMH", "XBI", "IBB", "KRE", "IHI", "ITB", "XME", "XLE", "XOP", "OIH"]
FAM20 = ["XLK", "XLF", "XLV", "XLY", "XLP", "XLI", "XLE", "XLU", "XLB",
         "SMH", "IBB", "XBI", "IHI", "KRE", "ITA", "XME", "XRT", "XHB",
         "IYR", "OIH"]
ENERGY = ["XLE", "XOP", "USO", "COP", "CVX", "VLO", "OXY", "SLB", "EOG", "HAL", "WMB"]
BANKS11 = ["JPM", "BAC", "C", "WFC", "GS", "MS", "USB", "KEY", "RF", "STT", "SCHW"]
BANKS8 = ["JPM", "BAC", "C", "WFC", "GS", "MS", "BNY", "STT"]
OTHER = ["^VIX", "^VIX3M", "^MOVE", "^TNX", "^SKEW", "TLT", "IEF", "LQD", "HYG",
         "DX-Y.NYB", "UUP", "GLD", "GDX", "SLV", "DBC", "CL=F", "SVXY", "XLRE"]
ALL = sorted(set(SPDR9 + FAM29 + FAM20 + ENERGY + BANKS11 + BANKS8 + OTHER))
PX = load_prices(ALL)
C = {t: PX[t]["Close"].dropna() for t in PX}

sec("0. last bar per key series")
for t in ["SPY", "^VIX", "^MOVE", "^TNX", "^SKEW", "TLT", "HYG", "DX-Y.NYB", "CL=F", "USO", "XLE"]:
    if t in C:
        print(f"  {t:9s} last {C[t].index[-1].date()} close {C[t].iloc[-1]:.4f} "
              f"1d {100 * (C[t].iloc[-1] / C[t].iloc[-2] - 1):+.2f}%")

sec("W12 / W49 / W50 / W32 / W34 volatility")
vix = C["^VIX"]
print(f"  ^VIX 1d {100 * (vix.iloc[-1] / vix.iloc[-2] - 1):+.2f}%  level {vix.iloc[-1]:.2f}  "
      f"pct_rank(vix,21) {last(pct_rank(vix, 21)):.1f}  SPY 1d "
      f"{100 * (C['SPY'].iloc[-1] / C['SPY'].iloc[-2] - 1):+.2f}%")
v3 = C["^VIX3M"]
print(f"  VIX/VIX3M {vix.iloc[-1] / v3.reindex(vix.index).iloc[-1]:.3f} (v3m last {v3.index[-1].date()})")
vx = close_panel(["SVXY", "^VIX", "^VIX3M", "SPY"])["^VIX"]
rng = (rolling_on_valid(vx, lambda x: x.rolling(21).max())
       - rolling_on_valid(vx, lambda x: x.rolling(21).min()))
REL = rolling_on_valid(rng / rolling_on_valid(vx, lambda x: x.rolling(21).mean()),
                       lambda x: x.rolling(252).rank(pct=True) * 100)
print("  VIX 21d relative-range pctile (entry 32/34 relpct), last 8 sessions:")
print("   ", ", ".join(f"{d.date()} {v:.2f}" for d, v in REL.dropna().iloc[-8:].items()))
REL_PROD = rolling_on_valid(rng, lambda x: x.rolling(504).apply(
    lambda w: 100.0 * (w[:-1] < w[-1]).mean(), raw=True))
print("  production abs-range/504d pctile (the site flag, NOT the entry definition), last 4:")
print("   ", ", ".join(f"{d.date()} {v:.2f}" for d, v in REL_PROD.dropna().iloc[-4:].items()))

sec("W29 ^MOVE trailing-252 LEVEL percentile")
mv = C["^MOVE"]
mp = rolling_on_valid(mv, lambda x: x.rolling(252).apply(
    lambda w: 100.0 * (w <= w[-1]).mean(), raw=True))
print(f"  ^MOVE {mv.iloc[-1]:.2f} on {mv.index[-1].date()}  level pctile {last(mp):.1f}  "
      f"(prev {float(mp.dropna().iloc[-2]):.1f})  252 lo {mv.iloc[-252:].min():.2f} hi {mv.iloc[-252:].max():.2f}")

sec("W13 / W15 / W18 / W22 / W52 rates and dollar")
tnx = C["^TNX"]
hi252 = rolling_on_valid(tnx, lambda x: x.rolling(252).max())
print(f"  ^TNX {tnx.iloc[-1]:.3f}  vs 252 max {hi252.iloc[-1]:.3f}  dist "
      f"{100 * (tnx.iloc[-1] / hi252.iloc[-1] - 1):+.3f}%  21-session chg "
      f"{tnx.iloc[-1] - tnx.iloc[-22]:+.3f}pt  252-session chg {100 * (tnx.iloc[-1] - tnx.iloc[-253]):+.1f}bp")
print(f"  pct_rank ^TNX 21 {last(pct_rank(tnx, 21)):.1f}  DX r21 {last(pct_rank(C['DX-Y.NYB'], 21)):.1f}  "
      f"UUP r21 {last(pct_rank(C['UUP'], 21)):.1f}")
# W18 OOS episodes: filter-then-decluster at gap 10 on the curve panel
cp = close_panel(["^TNX", "TLT", "IEF"]).dropna(how="any")
t2 = cp["^TNX"]
h2 = rolling_on_valid(t2, lambda x: x.rolling(252).max())
chg = (t2 - t2.shift(252)) * 100.0
st = ((t2 / h2 - 1.0) >= -0.0025) & (chg >= 78)
dd = cp[["TLT", "IEF"]].pct_change().dropna()
beta = float(np.polyfit(dd["IEF"].values, dd["TLT"].values, 1)[0])
R8 = vehicle_ret(cp, [("IEF", 1.0), ("TLT", -1.0 / beta)], 8, 1)
pos = pd.Series(range(len(cp)), index=cp.index)
kept, lastp = [], -10 ** 9
for d in cp.index[st.values]:
    if pos[d] - lastp >= 10:
        kept.append(d)
        lastp = pos[d]
print(f"  W18 TLT hedge 1/beta {1 / beta:.3f}; state days since 09-01: "
      + ", ".join(f"{d.date()}({chg[d]:+.1f}bp)" for d in cp.index[st.values] if d >= pd.Timestamp('2026-09-01')))
print("  W18 fdc(gap10) episodes since 2026-09-01: "
      + ", ".join(f"{d.date()} R8={100 * R8.get(d, np.nan):+.3f}%" for d in kept if d >= pd.Timestamp('2026-09-01')))
ep1 = pd.Timestamp("2026-09-09")
if ep1 in R8.index:
    print(f"  W18 OOS episode 1 (sig 09-09) realized h=8 curve return {1e4 * R8[ep1]:+.1f} bps")
    a, b = pd.Timestamp("2026-09-10"), pd.Timestamp("2026-09-22")
    ri, rt = cp.at[b, "IEF"] / cp.at[a, "IEF"] - 1, cp.at[b, "TLT"] / cp.at[a, "TLT"] - 1
    print(f"  W18 hand check 09-10 -> 09-22: IEF {100 * ri:+.3f}%  TLT {100 * rt:+.3f}%  "
          f"curve {1e4 * (ri - rt / beta):+.1f} bps")
print("  CL=F vs USO last 5 closes (roll-seam check):")
for d in C["USO"].index[-5:]:
    print(f"    {d.date()} USO {C['USO'].get(d, np.nan):.2f}  CL=F {C['CL=F'].get(d, np.nan):.2f}")

sec("W5 / W11 / W16 / W1 / W23 / W25 / W39 / W43 / W53 duration and credit")
for t in ["TLT", "IEF", "LQD"]:
    s = C[t]
    lo = rolling_on_valid(s, lambda x: x.rolling(252).min())
    print(f"  {t} above 252 low {100 * (s.iloc[-1] / lo.iloc[-1] - 1):.3f}%  1d "
          f"{100 * (s.iloc[-1] / s.iloc[-2] - 1):+.2f}%  r5 {last(pct_rank(s, 5)):.1f}  r21 {last(pct_rank(s, 21)):.1f}")
IDX = C["TLT"].index


def above_low(t):
    s = C[t]
    return ((s / s.rolling(252).min() - 1.0) * 100).reindex(IDX)


TL, IE, LQ = above_low("TLT"), above_low("IEF"), above_low("LQD")
P5 = pd.Series(range(len(IDX)), index=IDX)


def first_in(mask, gap=10):
    days = IDX[mask.reindex(IDX, fill_value=False).values]
    keep, lp = [], -10 ** 9
    for d in days:
        p = int(P5[d])
        if p - lp >= gap:
            keep.append(d)
        lp = p
    return pd.DatetimeIndex(keep)


par = (TL <= 0.5).fillna(False)
cell = ((TL <= 0.5) & (IE <= 1.0) & (LQ <= 1.0)).fillna(False)
print("  W5 parent-state days since 09-01: " + ", ".join(str(d.date()) for d in IDX[par.values] if d >= pd.Timestamp('2026-09-01')))
print("  W5 cell-state days since 09-01:   " + ", ".join(str(d.date()) for d in IDX[cell.values] if d >= pd.Timestamp('2026-09-01')))
print("  W5 first_in(gap10) anchors since 08-01 parent: "
      + ", ".join(str(d.date()) for d in first_in(par) if d >= pd.Timestamp('2026-08-01'))
      + " | cell: " + ", ".join(str(d.date()) for d in first_in(cell) if d >= pd.Timestamp('2026-08-01')))
hy = C["HYG"]
hyhi = rolling_on_valid(hy, lambda x: x.rolling(252).max())
rv = hy.pct_change().rolling(21).std() * np.sqrt(252) * 100
print(f"  HYG off 252 high {100 * (hy.iloc[-1] / hyhi.iloc[-1] - 1):+.3f}%  21d rvol {rv.iloc[-1]:.2f}%  "
      f"pitch_lab.zscore {last(zscore(hy)):+.2f}  | {tp('HYG', 'z10', 'ret_1d', 'rank_5d')}")
print(f"  IEF r5 (pct_rank) {last(pct_rank(C['IEF'], 5)):.1f}  | {tp('IEF', 'rank_5d', 'z10')}")

sec("W3 / W4 / W7 / W19 / W31 / W40 / W63 commodities and energy")
for t in ["GDX", "GLD", "SLV", "USO", "DBC", "XLE", "CL=F"]:
    s = C[t]
    hi = rolling_on_valid(s, lambda x: x.rolling(252).max())
    print(f"  {t:5s} 1d {100 * (s.iloc[-1] / s.iloc[-2] - 1):+.2f}%  r5 {last(pct_rank(s, 5)):.1f}  "
          f"r21 {last(pct_rank(s, 21)):.1f}  r63 {last(pct_rank(s, 63)):.1f}  off 252 high "
          f"{100 * (s.iloc[-1] / hi.iloc[-1] - 1):+.2f}%  last {s.index[-1].date()}")
u = PX["USO"]
atr = wilder_atr(u["High"].values, u["Low"].values, u["Close"].values)
print(f"  USO 1d move in prior-day Wilder ATR: {(u['Close'].iloc[-1] - u['Close'].iloc[-2]) / atr[-2]:.2f}")
print("  W19 energy z10 (pitch_lab.zscore | tape z10):")
cnt_pl = cnt_tp = 0
for t in ENERGY:
    zl = last(zscore(C[t]))
    zt = TAPE.get(t, {}).get("z10")
    cnt_pl += zl >= 2.0
    cnt_tp += (zt is not None and zt >= 2.0)
    print(f"    {t:4s} {zl:+.2f} | {zt}")
print(f"  W19 count at z10 >= 2: pitch_lab {cnt_pl}  tape {cnt_tp}")
for t in ["USO", "CL=F"]:
    s = C[t]
    r5, r21 = pct_rank(s, 5), pct_rank(s, 21)
    prior = r21.shift(1).rolling(10).max()
    stt = (r5 <= 3) & (prior >= 90)
    tail = pd.DataFrame({"r5": r5, "r21": r21, "prior10max": prior, "state": stt}).dropna().iloc[-8:]
    print(f"  W63 {t} round-trip state (r5<=3 & prior-10 r21 max >= 90), last 8 sessions:")
    for d, r in tail.iterrows():
        print(f"    {d.date()} r5 {r.r5:5.1f} r21 {r.r21:5.1f} prior10max {r.prior10max:5.1f} state {bool(r.state)}")

sec("W21 nine SPDRs: r5 and distance from 252 high")
for t in SPDR9:
    s = C[t]
    hi = rolling_on_valid(s, lambda x: x.rolling(252).max())
    print(f"  {t} r5 {last(pct_rank(s, 5)):5.1f}  r21 {last(pct_rank(s, 21)):5.1f}  r63 {last(pct_rank(s, 63)):5.1f}  "
          f"off high {100 * (s.iloc[-1] / hi.iloc[-1] - 1):+.2f}%  63d ret {100 * (s.iloc[-1] / s.iloc[-64] - 1):+.2f}%  "
          f"1d {100 * (s.iloc[-1] / s.iloc[-2] - 1):+.2f}%")
print(f"  XLV-XLK 1d gap {100 * ((C['XLV'].iloc[-1] / C['XLV'].iloc[-2]) - (C['XLK'].iloc[-1] / C['XLK'].iloc[-2])):+.2f}pp")

sec("W27 29-ETF family and full tape: r21 >= 90 AND r63 <= 10")
for t in FAM29:
    if t not in C:
        print(f"  {t} missing")
        continue
    s = C[t]
    a, b, c = last(pct_rank(s, 21)), last(pct_rank(s, 63)), last(pct_rank(s, 5))
    if a >= 90 or b <= 10:
        print(f"  {t} r21 {a:.1f} r63 {b:.1f} r5 {c:.1f}{'  <-- JOINT' if a >= 90 and b <= 10 else ''}")
hits = [(k, v['rank_21d'], v['rank_63d'], v['rank_5d']) for k, v in TAPE.items()
        if v.get('rank_21d') is not None and v.get('rank_63d') is not None
        and v['rank_21d'] >= 90 and v['rank_63d'] <= 10]
print(f"  tape holders of r21>=90 & r63<=10: {hits}")

sec("W47 20-ETF family and full tape: r5 <= 2 AND r63 >= 90")
for t in FAM20:
    if t not in C:
        print(f"  {t} missing")
        continue
    s = C[t]
    a, b = last(pct_rank(s, 5)), last(pct_rank(s, 63))
    if a <= 10 or b >= 85:
        print(f"  {t} r5 {a:.1f} r63 {b:.1f}{'  <-- CORNER' if a <= 2 and b >= 90 else ''}")
hits = [(k, v['rank_5d'], v['rank_63d']) for k, v in TAPE.items()
        if v.get('rank_5d') is not None and v.get('rank_63d') is not None
        and v['rank_5d'] <= 2 and v['rank_63d'] >= 90]
print(f"  tape holders of r5<=2 & r63>=90: {hits}")

sec("W8 / W9 / W24 / W48 single-family ranks")
for t in ["IHI", "FXI", "EEM", "SMH", "XLV", "IBB", "XBI"]:
    s = C[t]
    print(f"  {t} r5 {last(pct_rank(s, 5)):.1f} r21 {last(pct_rank(s, 21)):.1f} r63 {last(pct_rank(s, 63)):.1f} "
          f"5d ret {100 * (s.iloc[-1] / s.iloc[-6] - 1):+.2f}%")

sec("W17 bank breadth (11-name complex)")
r5s = {t: last(pct_rank(C[t], 5)) for t in BANKS11}
r63s = {t: last(pct_rank(C[t], 63)) for t in BANKS11}
n20 = sum(v <= 20 for v in r5s.values())
print("  " + ", ".join(f"{t} r5 {r5s[t]:.1f}/r63 {r63s[t]:.1f}" for t in BANKS11))
print(f"  at r5 <= 20: {n20} of 11 ({100 * n20 / 11:.0f}%)  median r63 {np.median(list(r63s.values())):.1f}")
print(f"  KRE r5 {last(pct_rank(C['KRE'], 5)):.1f}  XLF r5 {last(pct_rank(C['XLF'], 5)):.1f}")

sec("W51 eight-name bank universe: 1d move in prior-day Wilder ATR, gap share")
spy1 = 100 * (C['SPY'].iloc[-1] / C['SPY'].iloc[-2] - 1)
print(f"  SPY 1d {spy1:+.2f}% (gate > -1%)")
for t in BANKS8:
    f = PX[t]
    a = wilder_atr(f["High"].values, f["Low"].values, f["Close"].values)
    mv1 = f["Close"].iloc[-1] - f["Close"].iloc[-2]
    gap = f["Open"].iloc[-1] - f["Close"].iloc[-2]
    gs = gap / mv1 if mv1 != 0 else float("nan")
    print(f"  {t:4s} 1d {100 * mv1 / f['Close'].iloc[-2]:+.2f}%  {mv1 / a[-2]:+.2f} ATR  gap share {gs:.2f}")

sec("W61 XLU washout x TLT (XLU r21 <= 5 AND TLT r21 < 25), GAP 21 clusters")
xr, trr = pct_rank(C["XLU"], 21), pct_rank(C["TLT"], 21)
j = pd.DataFrame({"xlu_r21": xr, "tlt_r21": trr}).dropna()
j["state"] = (j.xlu_r21 <= 5) & (j.tlt_r21 < 25)
print(j[j.index >= pd.Timestamp("2026-09-14")].round(2).to_string())
xs = C["XLU"]
xhi = rolling_on_valid(xs, lambda x: x.rolling(252).min())
print(f"  XLU above 252 low {100 * (xs.iloc[-1] / xhi.iloc[-1] - 1):.2f}%")

sec("W6 ^SKEW / W33 / W11 / W20 / W23 SPY")
sk = C["^SKEW"]
print(f"  ^SKEW last {sk.index[-1].date()} {sk.iloc[-1]:.2f}  r5 {last(pct_rank(sk, 5)):.1f}")
spy = C["SPY"]
shi = rolling_on_valid(spy, lambda x: x.rolling(252).max())
sma = spy.rolling(200).mean()
print(f"  SPY off 252 high {100 * (spy.iloc[-1] / shi.iloc[-1] - 1):+.2f}%  vs 200d {100 * (spy.iloc[-1] / sma.iloc[-1] - 1):+.2f}%")

sec("W55 SPDR9 63d ranking today (preview only; anchor is the 09-30 close)")
r63d = {t: 100 * (C[t].iloc[-1] / C[t].iloc[-64] - 1) for t in SPDR9}
srt = sorted(r63d.items(), key=lambda kv: -kv[1])
print("  " + ", ".join(f"{t} {v:+.2f}%" for t, v in srt))
