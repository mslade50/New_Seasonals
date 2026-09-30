"""C1 round 1: short SPY when HYG's 21d return rank <= 10 with SPY within 2% of
its 252 closing high.  Pre-specified sign SHORT, h=5 and h=10, lag=1.

Claimed mechanism: credit leads equity (Merton).  Gate attribution:
  near-high days with vs without the HYG weakness, and HYG weakness split by
  DURATION (IEF also weak) vs SPREAD (HYG/IEF ratio weak, PIT beta-IEF
  residual weak).  The registry already says duration-driven HYG flushes are a
  different object (2026-09-11) and that the credit-specific residual has
  failed six times.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
BAR = pd.Timestamp("2026-09-23")
px = close_panel(["SPY", "HYG", "IEF"]).dropna().loc[:BAR]
spy, hyg, ief = px["SPY"], px["HYG"], px["IEF"]
LEGS = [("SPY", -1.0)]

hr21 = pct_rank(hyg, 21)
ir21 = pct_rank(ief, 21)
ratio = hyg / ief
rr21 = pct_rank(ratio, 21)
dh, di = hyg.pct_change(), ief.pct_change()
beta = (dh.rolling(252).cov(di) / di.rolling(252).var()).shift(1)
resid21 = (dh - beta * di).rolling(21).sum()
res_rank = resid21.rolling(252).rank(pct=True) * 100
hi252 = spy.rolling(252).max()
off = spy / hi252 - 1.0
near2 = off >= -0.02
hyg_vol = load_prices(["HYG"])["HYG"]["Volume"].loc[:BAR]
vr = hyg_vol / hyg_vol.rolling(63).mean().shift(1)

print(f"panel {px.index[0].date()} .. {px.index[-1].date()} n={len(px)}")
st = pd.DataFrame({"SPY_off_hi%": 100 * off, "HYG_r21": hr21, "IEF_r21": ir21,
                   "HYG/IEF_r21": rr21, "resid_rank": res_rank,
                   "HYG_1d%": 100 * dh, "HYG_vol_x63": vr.reindex(px.index),
                   "HYG_off_lo%": 100 * (hyg / hyg.rolling(252).min() - 1)}).tail(6)
print(st.round(2).to_string())

trig = (hr21 <= 10) & near2


def cell(mask, h, label, gap=None):
    r = vehicle_ret(px, LEGS, h)
    v = r.notna()
    days = px.index[mask.reindex(px.index, fill_value=False).values & v.values]
    if len(days) == 0:
        return {"label": label, "n": 0}
    epi = declusters(days, gap or h, px.index)
    vals = r.loc[epi].values
    base = r[v]
    w = int((vals > 0).sum())
    o = summarize(vals, label)
    o["n_days"] = len(days)
    o["edge_pp"] = round(o["mean_pct"] - 100 * base.mean(), 3)
    o["rec"] = f"{w}-{len(vals)-w}"
    o["p_vs_downrate"] = round(sign_test(w, len(vals), float((base > 0).mean())), 4)
    return o


for h in (5, 10):
    rows = [cell(near2, h, "PARENT SPY within 2% of 252 high"),
            cell(trig, h, "CHILD + HYG r21 <= 10"),
            cell(near2 & (hr21 > 10), h, "COMPLEMENT near-high, HYG r21 > 10"),
            cell(near2 & (hr21 > 10) & (hr21 <= 30), h, "near-high, HYG r21 (10,30]"),
            cell(trig & (ir21 <= 20), h, "CHILD & IEF r21 <= 20 (DURATION-driven)"),
            cell(trig & (ir21 > 20) & (ir21 < 50), h, "CHILD & IEF r21 (20,50)"),
            cell(trig & (ir21 >= 50), h, "CHILD & IEF r21 >= 50 (SPREAD-driven)"),
            cell(trig & (rr21 <= 10), h, "CHILD & HYG/IEF ratio r21 <= 10"),
            cell(trig & (rr21 > 10), h, "CHILD & HYG/IEF ratio r21 > 10"),
            cell(trig & (res_rank <= 10), h, "CHILD & PIT beta-IEF resid rank <= 10"),
            cell(trig & (res_rank > 10), h, "CHILD & PIT resid rank > 10"),
            cell(near2 & (res_rank <= 10), h, "near-high & resid rank <= 10 (credit-only)"),
            summarize(vehicle_ret(px, LEGS, h).dropna().values, "CTRL all days (short)")]
    show(rows, f"C1 SHORT SPY h={h}, lag=1, episodes gap=h")

print("\n  live-state bucket on 09-23: IEF r21 and ratio r21 above tell which bucket")
vari = {}
for thr in (5, 10, 15):
    for w in (0.01, 0.02, 0.03):
        vari[f"HYG r21<={thr} SPY within {int(100*w)}%"] = (hr21 <= thr) & (off >= -w)
for h in (5, 10):
    battery(px, trig, LEGS, h, "C1 short SPY, HYG r21<=10 & SPY within 2% of high",
            cost_bps=1.0, variants=vari, event_kinds=("nfp",))

# midterm split, h=5 / h=10
for h in (5, 10):
    r = vehicle_ret(px, LEGS, h)
    days = px.index[trig.values & r.notna().values]
    epi = declusters(days, h, px.index)
    yrs = pd.DatetimeIndex(epi).year
    mid = (yrs % 4) == 2
    show([summarize(r.loc[epi].values[mid], f"h={h} midterm"),
          summarize(r.loc[epi].values[~mid], f"h={h} non-midterm")], "midterm split")

# book overlap: SPY/QQQ longs signalled inside the would-be hold window
led = pd.read_parquet(ROOT / "data" / "backtest_trades_full.parquet")
for col in ("Ticker", "Strategy", "Direction", "Signal Date", "R_Multiple"):
    assert col in led.columns, col
led["Signal Date"] = pd.to_datetime(led["Signal Date"])
idx = px.index
pos = pd.Series(range(len(idx)), index=idx)
r10 = vehicle_ret(px, LEGS, 10)
epi = declusters(idx[trig.values & r10.notna().values], 10, idx)
win = set()
for d in epi:
    p = pos[d]
    win.update(idx[p: min(len(idx), p + 12)])
sq = led[led["Ticker"].isin(["SPY", "QQQ"])]
ov = sq[sq["Signal Date"].isin(win)]
print(f"\nBOOK: SPY/QQQ ledger rows {len(sq)}; signalled inside a C1 hold window: {len(ov)}")
if len(ov):
    print(ov.groupby(["Strategy", "Direction"])["R_Multiple"].agg(["count", "mean"]).round(3).to_string())
print(sq.groupby(["Ticker", "Strategy", "Direction"]).size().to_string())
s = load_prices(["SPY"])["SPY"].loc[:BAR]
a = float(wilder_atr(s["High"], s["Low"], s["Close"])[-1])
print(f"SPY close {s['Close'].iloc[-1]:.2f} Wilder-14 ATR {a:.2f} ({100*a/s['Close'].iloc[-1]:.2f}%)")
