"""Risk-adjusted impact of a Friday-only EOD-DD(0.25 ATR) exit, per-trade R.
Sharpe = mean(R)/std(R);  Sortino = mean(R)/downside_dev (MAR=0).
Paired bootstrap (resample trades) gives a CI on the Sharpe/Sortino change.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EOD_DD = 0.25
np.random.seed(12345)

t = pd.read_parquet(ROOT / "data" / "backtest_trades_full.parquet")

# entry-day close lookup across both price caches
close_map = {}
for fn in ["master_prices.parquet", "overflow_prices.parquet"]:
    p = ROOT / "data" / fn
    if not p.exists():
        continue
    px = pd.read_parquet(p)
    px["date"] = pd.to_datetime(px["date"])
    for tk, g in px.groupby("ticker"):
        s = g.set_index("date")["Close"]
        close_map[tk] = s if tk not in close_map else close_map[tk].combine_first(s)


def build(strat):
    d = t[t["Strategy"] == strat].copy()
    d["Entry Date"] = pd.to_datetime(d["Entry Date"])
    d["ec"] = d.apply(lambda r: close_map.get(r["Ticker"], pd.Series(dtype=float)).get(r["Entry Date"], np.nan), axis=1)
    d = d.dropna(subset=["ec"]).copy()
    is_long = d["Direction"].eq("Long")
    risk_ps = d["stop_atr"] * d["ATR"]
    gain_ps = np.where(is_long, d["ec"] - d["Entry Price"], d["Entry Price"] - d["ec"])
    dd = -gain_ps / d["ATR"]
    trig = (d["Entry Date"].dt.weekday == 4) & (dd > EOD_DD)
    r_old = d["R_Multiple"].to_numpy(float)
    r_new = r_old.copy()
    r_new[trig.to_numpy()] = (gain_ps / risk_ps)[trig.to_numpy()]
    return r_old, r_new, int(trig.sum())


def metrics(r):
    mean = r.mean()
    std = r.std(ddof=1)
    dd_dev = np.sqrt(np.mean(np.minimum(r, 0.0) ** 2))   # downside dev, MAR=0
    sharpe = mean / std if std else np.nan
    sortino = mean / dd_dev if dd_dev else np.nan
    return dict(N=len(r), mean=mean, std=std, sharpe=sharpe,
                downside=dd_dev, sortino=sortino, totR=r.sum(),
                worst=r.min(), p5=np.percentile(r, 5), pct_neg=(r < 0).mean())


def show(strat):
    r_old, r_new, ntrig = build(strat)
    mo, mn = metrics(r_old), metrics(r_new)
    print(f"\n{'='*72}\n{strat}   (N={mo['N']}, EOD-DD triggers={ntrig})")
    hdr = ["metric", "original", "EOD-DD", "delta"]
    print(f"  {hdr[0]:<12}{hdr[1]:>11}{hdr[2]:>11}{hdr[3]:>11}")
    for k, lbl in [("totR", "total R"), ("mean", "mean R"), ("std", "std R"),
                   ("downside", "downside d"), ("sharpe", "Sharpe/tr"),
                   ("sortino", "Sortino/tr"), ("pct_neg", "% losers"),
                   ("worst", "worst R"), ("p5", "5th pctile")]:
        print(f"  {lbl:<12}{mo[k]:>11.3f}{mn[k]:>11.3f}{mn[k]-mo[k]:>+11.3f}")

    # paired bootstrap on Sharpe / Sortino change
    n = len(r_old)
    dS, dSo = [], []
    for _ in range(5000):
        idx = np.random.randint(0, n, n)
        a, b = metrics(r_old[idx]), metrics(r_new[idx])
        dS.append(b["sharpe"] - a["sharpe"])
        dSo.append(b["sortino"] - a["sortino"])
    dS, dSo = np.array(dS), np.array(dSo)
    print(f"  bootstrap dSharpe : median={np.median(dS):+.4f}  "
          f"90% CI=[{np.percentile(dS,5):+.4f},{np.percentile(dS,95):+.4f}]  "
          f"P(improve)={ (dS>0).mean():.0%}")
    print(f"  bootstrap dSortino: median={np.median(dSo):+.4f}  "
          f"90% CI=[{np.percentile(dSo,5):+.4f},{np.percentile(dSo,95):+.4f}]  "
          f"P(improve)={ (dSo>0).mean():.0%}")


for s in ["Oversold Low Volume", "LT Trend ST OS"]:
    show(s)
