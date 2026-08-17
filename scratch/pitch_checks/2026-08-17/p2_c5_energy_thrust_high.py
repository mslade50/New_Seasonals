"""C5 round 1+2 - the energy complex thrusting to a 52w high, LONG XLE.

Cell: XLE 5d rank 100.0 (+7.67%), -0.33% off its 52w high, 63d rank only 59.1
(not extended on a quarter). Pitched as an INVERSION, i.e. the opposite side of
the book's own reflex, which the script has to verify rather than assume.

Trigger is defined TWO ways because a rank can jump on a denominator roll (the
2026-08-13 bite):
  A. rank form:      rank5 >= 98  &  within 1% of the 52w high  &  63d rank mid
  B. magnitude form: 5d move >= k * Wilder-14 ATR  &  same two gates
Both are reported; the write-up must say which is the real finding.

Registry debts:
  * 2026-08-13 "Long XLE on a crude one-day thrust" - XLE's crude beta 0.479,
    residual +0.291% at sign p 0.596; also 47 OVS SHORT signals on USO >= +5%
    days at avgR +0.29. Book overlap is section 6 here.
  * 2026-08-10 "Energy's 5d washout into a CPI print" - the long-continuation
    form must be shown NOT to be the book's LT Trend ST OS setup.
  * 2026-08-14 "producers against the barrel" - a 63d relative spread is a
    BEAR-TAPE SELECTOR by construction (41.7% below SPY's 200d vs 20.3% base).
    Bear-tape fraction is reported for every definition here.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

H_MAIN = 5
TICK = "XLE"

raw = load_prices(["XLE", "XOP", "OIH", "USO", "SPY"])
pan = close_panel(["XLE", "XOP", "OIH", "USO", "SPY"])
pan = pan.dropna(subset=["XLE", "SPY"])
IDX = pan.index
xle = pan["XLE"]

ohlc = raw["XLE"].reindex(IDX)
atr = pd.Series(np.asarray(wilder_atr(ohlc["High"], ohlc["Low"], ohlc["Close"], 14),
                           dtype=float), index=IDX)
r5 = pct_rank(xle, 5)
r63 = pct_rank(xle, 63)
hi52 = xle.rolling(252).max()
off_hi = xle / hi52 - 1.0
mv5 = xle - xle.shift(5)
mv5_atr = mv5 / atr
spy200 = pan["SPY"].rolling(200).mean()
below_spy200 = pan["SPY"] < spy200
base_bear = float(below_spy200.mean())

print(f"XLE panel {IDX[0].date()} .. {IDX[-1].date()}  N={len(IDX)}")
print(f"TODAY: rank5={r5.iloc[-1]:.1f} rank63={r63.iloc[-1]:.1f} off52wh={100*off_hi.iloc[-1]:+.2f}% "
      f"ATR14={atr.iloc[-1]:.2f} 5d move={mv5.iloc[-1]:+.2f} = {mv5_atr.iloc[-1]:+.2f} ATR "
      f"({100*(xle.iloc[-1]/xle.iloc[-6]-1):+.2f}%)")
print(f"base rate SPY below 200d = {100*base_bear:.1f}%")

NEAR_HI = off_hi >= -0.01
MID63 = (r63 >= 30) & (r63 <= 75)
RANK = ((r5 >= 98) & NEAR_HI & MID63).fillna(False)
K = 2.5  # today's 5d move in ATRs, floored down to a round number (printed above)
MAG = ((mv5_atr >= K) & NEAR_HI & MID63).fillna(False)

for nm, m in (("RANK form", RANK), ("MAG form", MAG)):
    s = IDX[m.values]
    print(f"{nm}: N={len(s)} days, episodes(10td)={len(declusters(s, 10, IDX))}, "
          f"bear-tape {100*float(below_spy200.loc[s].mean()) if len(s) else float('nan'):.1f}%")

variants = {
    "rank5>=98 ALONE": (r5 >= 98).fillna(False),
    "rank5>=98 + near-high": ((r5 >= 98) & NEAR_HI).fillna(False),
    "rank5>=98 + 63d mid": ((r5 >= 98) & MID63).fillna(False),
    "RANK BASE (all three)": RANK,
    "rank5>=95 all three": ((r5 >= 95) & NEAR_HI & MID63).fillna(False),
    "rank5>=99 all three": ((r5 >= 99) & NEAR_HI & MID63).fillna(False),
    "MAG >=2.0 ATR all three": ((mv5_atr >= 2.0) & NEAR_HI & MID63).fillna(False),
    "MAG >=2.5 ATR all three": MAG,
    "MAG >=3.0 ATR all three": ((mv5_atr >= 3.0) & NEAR_HI & MID63).fillna(False),
    "MAG >=2.5 ATR ALONE": (mv5_atr >= 2.5).fillna(False),
    "near-high + 63d mid ONLY (no thrust)": (NEAR_HI & MID63).fillna(False),
}

battery(pan, RANK, [("XLE", 1.0)], H_MAIN, "C5 LONG XLE, rank form",
        cost_bps=3.0, variants=variants, min_gap=10,
        event_kinds=("opex", "vix_expiry", "jackson_hole"))

battery(pan, MAG, [("XLE", 1.0)], H_MAIN, f"C5 LONG XLE, magnitude form (>={K} ATR)",
        cost_bps=3.0, min_gap=10, event_kinds=("opex", "vix_expiry"))

# ---------------------------------------------------------------- horizons
print("\n" + "=" * 92)
print("1b. HORIZON SCAN (both definitions), episode level")
print("=" * 92)
for nm, m in (("RANK", RANK), ("MAG", MAG)):
    s = IDX[m.values]
    show(horizon_scan(pan, s, [("XLE", 1.0)], hs=(1, 2, 3, 5, 8, 10), min_gap=10),
         f"{nm} form, long XLE")

# ---------------------------------------------------------------- gate attribution
print("\n" + "=" * 92)
print("2. GATE ATTRIBUTION - what does each gate actually add? (h=5, episodes)")
print("=" * 92)
f5 = fwd_lag(xle, H_MAIN, 1)
rows = []
for lbl, m in variants.items():
    s = IDX[m.fillna(False).values]
    if len(s) == 0:
        rows.append({"gate": lbl, "n_days": 0})
        continue
    e = declusters(s, 10, IDX)
    v = f5.reindex(e).dropna()
    rows.append({"gate": lbl, "n_days": len(s), "n_epi": len(v),
                 "mean_pct": round(100 * v.mean(), 3),
                 "median_pct": round(100 * v.median(), 3),
                 "hit": round(100 * (v > 0).mean(), 1),
                 "bear_frac": round(100 * float(below_spy200.loc[s].mean()), 1),
                 "sign_p": round(sign_test(int((v > 0).sum()), len(v)), 4)})
rows.append({"gate": "UNCONDITIONAL XLE", "n_days": len(IDX), "n_epi": int(f5.notna().sum()),
             "mean_pct": round(100 * f5.mean(), 3), "median_pct": round(100 * f5.median(), 3),
             "hit": round(100 * (f5.dropna() > 0).mean(), 1),
             "bear_frac": round(100 * base_bear, 1), "sign_p": np.nan})
show(rows, "gate ladder")

# ---------------------------------------------------------------- complex breadth
print("\n" + "=" * 92)
print("3. DOES THE COMPLEX CONFIRM? same cell on XOP / OIH / USO, and the XLE cell")
print("   measured in each vehicle")
print("=" * 92)
epi = declusters(IDX[RANK.values], 10, IDX)
rows = []
for t in ("XLE", "XOP", "OIH", "USO"):
    s = pan[t].dropna()
    v = fwd_lag(s, H_MAIN, 1).reindex(epi).dropna()
    unc = fwd_lag(s, H_MAIN, 1).dropna()
    rows.append({"vehicle": t, "n_epi": len(v),
                 "mean_pct": round(100 * v.mean(), 3),
                 "uncond_pct": round(100 * unc.mean(), 3),
                 "edge_pp": round(100 * (v.mean() - unc.mean()), 3),
                 "hit": round(100 * (v > 0).mean(), 1)})
show(rows, "XLE-triggered cell traded in each energy vehicle, h=5")

rows = []
for t in ("XOP", "OIH"):
    s = pan[t].dropna()
    rr5 = pct_rank(s, 5)
    rr63 = pct_rank(s, 63)
    oh = s / s.rolling(252).max() - 1.0
    m = ((rr5 >= 98) & (oh >= -0.01) & (rr63 >= 30) & (rr63 <= 75)).fillna(False)
    sd = s.index[m.reindex(s.index, fill_value=False).values]
    if len(sd) < 3:
        rows.append({"ticker": t, "n_epi": 0})
        continue
    e = declusters(sd, 10, s.index)
    v = fwd_lag(s, H_MAIN, 1).reindex(e).dropna()
    unc = fwd_lag(s, H_MAIN, 1).dropna()
    rows.append({"ticker": t, "n_epi": len(v), "mean_pct": round(100 * v.mean(), 3),
                 "uncond_pct": round(100 * unc.mean(), 3),
                 "edge_pp": round(100 * (v.mean() - unc.mean()), 3),
                 "hit": round(100 * (v > 0).mean(), 1)})
show(rows, "same DEFINITION applied to each name's own state (reference class)")

# ---------------------------------------------------------------- SPY-relative
print("\n" + "=" * 92)
print("4. IS IT JUST TAPE? XLE minus SPY and beta-neutral residual on the cell")
print("=" * 92)
rx = fwd_lag(xle, H_MAIN, 1)
rp = fwd_lag(pan["SPY"], H_MAIN, 1)
ok = rx.notna() & rp.notna()
beta = float(np.polyfit(rp[ok], rx[ok], 1)[0])
show([summarize(rx.reindex(epi).dropna().values, "XLE leg"),
      summarize(rp.reindex(epi).dropna().values, "SPY leg"),
      summarize((rx - rp).reindex(epi).dropna().values, "equal-$ XLE-SPY"),
      summarize((rx - beta * rp).reindex(epi).dropna().values,
                f"beta-neutral resid (beta={beta:.2f})"),
      summarize((rx - beta * rp)[ok].values, "resid all days (control)")],
     f"h={H_MAIN}, episodes N={len(epi)}")

# ---------------------------------------------------------------- era + oil regime
print("\n" + "=" * 92)
print("5. ROUND 2 - era, oil regime, concentration")
print("=" * 92)
v = f5.reindex(epi).dropna()
show(era_split(v.index, v.values), "era split (episodes)")
uso = pan["USO"].dropna()
uso63 = pct_rank(uso, 63).reindex(epi)
hi_oil = uso63 >= 50
sub = v.reindex(uso63.dropna().index).dropna()
if len(sub):
    hm = hi_oil.reindex(sub.index).fillna(False).values
    show([summarize(sub[hm].values, "USO 63d rank >= 50 at trigger"),
          summarize(sub[~hm].values, "USO 63d rank < 50 (today = 8.3)")],
         "oil-regime split (USO history starts 2006)")
print("\nconcentration:", cluster_note(v.index, v.values))
print("episode dates:", ", ".join(str(d.date()) for d in v.index))

# ---------------------------------------------------------------- BOOK OVERLAP
print("\n" + "=" * 92)
print("6. BOOK OVERLAP - what did the systematic book ACTUALLY do on these days?")
print("=" * 92)
led = pd.read_parquet(Path(__file__).resolve().parents[3] / "data" / "backtest_trades_full.parquet")
led["Signal Date"] = pd.to_datetime(led["Signal Date"])
ENERGY = {"XLE", "XOP", "OIH", "USO", "XOM", "CVX", "COP", "SLB", "HAL", "VLO",
          "OXY", "PSX", "MPC", "EOG", "PXD", "DVN", "FANG", "WMB", "KMI", "OKE",
          "ERX", "ERY", "GUSH", "DRIP", "UCO", "SCO", "BNO", "DBO", "XES", "IEZ",
          "AMLP", "FCG", "PBW", "TAN", "NRGU"}
trig_days = set(IDX[RANK.values])
# a trade whose signal date sits within [-1, +5] td of any trigger day
pos = pd.Series(range(len(IDX)), index=IDX)
win = set()
for d in trig_days:
    p = pos.get(d)
    if p is None:
        continue
    for q in range(max(0, p - 1), min(len(IDX), p + 6)):
        win.add(IDX[q])
win10 = set()
for d in trig_days:
    p = pos.get(d)
    if p is None:
        continue
    for q in range(max(0, p - 5), min(len(IDX), p + 11)):
        win10.add(IDX[q])
sub10 = led[led["Ticker"].isin(ENERGY) & led["Signal Date"].isin(win10)]
print(f"ledger energy trades with a signal inside [-5,+10] td of a RANK trigger: {len(sub10)}")
if len(sub10):
    print(sub10.groupby(["Strategy", "Direction"]).agg(
        n=("R_Multiple", "size"), avgR=("R_Multiple", "mean")).round(3).to_string())
# the wider MAG trigger has more history; use it for the real book read
winM = set()
for d in IDX[MAG.values]:
    p = pos.get(d)
    if p is None:
        continue
    for q in range(max(0, p - 5), min(len(IDX), p + 11)):
        winM.add(IDX[q])
subM = led[led["Ticker"].isin(ENERGY) & led["Signal Date"].isin(winM)]
print(f"\nledger energy trades inside [-5,+10] td of a MAGNITUDE trigger: {len(subM)}")
if len(subM):
    print(subM.groupby(["Strategy", "Direction"]).agg(
        n=("R_Multiple", "size"), avgR=("R_Multiple", "mean"),
        totR=("R_Multiple", "sum")).round(3).to_string())
sub = led[led["Ticker"].isin(ENERGY) & led["Signal Date"].isin(win)]
print(f"\nledger energy trades with a signal inside [-1,+5] td of a RANK trigger: {len(sub)}")
if len(sub):
    g = sub.groupby(["Direction"]).agg(n=("R_Multiple", "size"),
                                       avgR=("R_Multiple", "mean"),
                                       totR=("R_Multiple", "sum"))
    print(g.round(3).to_string())
    print("\nby strategy x direction:")
    print(sub.groupby(["Strategy", "Direction"]).agg(
        n=("R_Multiple", "size"), avgR=("R_Multiple", "mean")).round(3).to_string())
allen = led[led["Ticker"].isin(ENERGY)]
print(f"\nALL ledger energy trades: {len(allen)}  "
      f"long {int((allen['Direction']=='Long').sum())} / short {int((allen['Direction']=='Short').sum())}")
print(allen.groupby("Direction").agg(n=("R_Multiple", "size"), avgR=("R_Multiple", "mean")).round(3).to_string())

# what does the book's own book say about a 52w-high thrust?
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from strategy_config import STRATEGY_BOOK  # noqa: E402
print("\nSTRATEGY_BOOK entries by direction (is the book structurally short a thrust?):")
for cfg in STRATEGY_BOOK:
    st = cfg.get("settings", {})
    print(f"  {cfg.get('name',''):28s} dir={st.get('trade_direction')}  "
          f"entry={st.get('entry_type')}  hold={cfg.get('execution',{}).get('hold_days')}  "
          f"XLE_in_universe={'XLE' in cfg.get('universe_tickers', [])}")

# ---------------------------------------------------------------- 7. THE DIVERGENCE COLLISION
print("\n" + "=" * 92)
print("7. IS THE THRUST CELL SEPARABLE FROM THE 2026-08-14 XLE-minus-USO 63d")
print("   DIVERGENCE KILL? (that kill's trigger was the divergence itself)")
print("=" * 92)
div = xle.pct_change(63) - pan["USO"].pct_change(63)
print(f"TODAY's XLE-USO 63d divergence = {100*div.iloc[-1]:+.2f}pp  "
      f"(the 08-14 kill's live number was +18.69pp; its sign-flip threshold was 18pp)")
for thr in (0.15, 0.18, 0.19, 0.25):
    dm = (div >= thr).fillna(False)
    s_div = IDX[dm.values]
    s_thr = IDX[RANK.values]
    both = sorted(set(s_div) & set(s_thr))
    print(f"  divergence >= {100*thr:.0f}pp: {len(s_div)} days; overlap with the thrust "
          f"trigger = {len(both)} of {len(s_thr)} thrust days ({100*len(both)/max(1,len(s_thr)):.1f}%)")
# and the thrust cell measured WITH and WITHOUT a live divergence
dm18 = (div >= 0.18).reindex(epi).fillna(False)
if len(epi):
    ve = f5.reindex(epi).dropna()
    m18 = dm18.reindex(ve.index).fillna(False).values
    show([summarize(ve[m18].values, "thrust episodes WITH divergence>=18pp (today's state)"),
          summarize(ve[~m18].values, "thrust episodes WITHOUT")],
         "thrust cell split by the divergence the 08-14 kill was about")
