"""Portfolio-level rundown: book (post frag-bands + OLV gate ledger) + trend
sleeve at 0.6x NAV, all on the flat $750k basis.

Part 1: combined monthly series + headline stats (full / 2016+ / 2020+)
Part 2: worst months/drawdowns, annual table
Part 3: stationary block bootstrap -> 1y PnL and 1y/3y maxDD distributions
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NAV = 750_000.0
TREND_FRACTION = 0.6

# --- book monthly returns (flat basis, current ledger incl. all new rules) ---
daily = pd.read_parquet(ROOT / "data" / "backtest_daily_pnl.parquet")
daily["date"] = pd.to_datetime(daily["date"])
book_m = (daily.set_index("date")["pnl_flat"]
          .groupby(pd.Grouper(freq="ME")).sum() / NAV)
book_m = book_m[book_m.index >= "2003-01-31"]

# --- trend sleeve monthly net returns, FINAL 12-name universe (same-close;
# next-open shaves ~0.05-0.06 Sharpe — noted in the writeup, not modeled) ---
src = open(ROOT / "scratch" / "tf_universe_study.py").read()
src = src.replace("Path(__file__).resolve().parents[1]", "Path('.')")
ns = {}
exec(src.split("px, irx = load_prices()")[0], ns)
px, irx = ns["load_prices"]()
sig = ns["build_combo"](px)
FINAL = ["SPY", "QQQ", "IWM", "EFA", "EEM", "FXI", "VNQ", "GLD", "SLV", "DBC", "TLT", "LQD"]
trend = ns["run"](px, irx, sig, FINAL)["net"]
trend.index = pd.to_datetime(trend.index)

common = book_m.index.intersection(trend.index)
book = book_m.reindex(common).fillna(0.0)
tr = trend.reindex(common).fillna(0.0)
combo = book + TREND_FRACTION * tr


def stats(r, label, since=None):
    x = r.copy()
    if since:
        x = x[x.index >= since]
    curve = (1 + x).cumprod()
    yrs = len(x) / 12
    cagr = curve.iloc[-1] ** (1 / yrs) - 1
    vol = x.std() * np.sqrt(12)
    sharpe = x.mean() / x.std() * np.sqrt(12)
    dd = (curve / curve.cummax() - 1).min()
    t = x.mean() / (x.std() / np.sqrt(len(x)))
    return (f"{label:<22} N={len(x):>3}  CAGR {cagr*100:6.2f}%  vol {vol*100:5.2f}%  "
            f"Sharpe {sharpe:5.2f}  maxDD {dd*100:6.1f}%  t={t:5.2f}  "
            f"worst-mo {x.min()*100:+5.1f}%  %pos {100*(x>0).mean():4.1f}")


print("=== monthly return series, flat $750k basis ===")
for since, tag in [(None, "full"), ("2016-01-01", "2016+"), ("2020-01-01", "2020+")]:
    print(stats(book, f"book only {tag}", since))
    print(stats(combo, f"book + 0.6x trend {tag}", since))
    print()

print("=== worst 10 combined months ===")
w = combo.sort_values().head(10)
for d, v in w.items():
    print(f"  {d.date()}  {v*100:+6.2f}%  (${v*NAV:+,.0f})  book {book.loc[d]*100:+6.2f}%  "
          f"trend {tr.loc[d]*100:+5.2f}%")

print("\n=== annual, combined (flat $) ===")
ann = combo.groupby(combo.index.year).apply(lambda x: (1 + x).prod() - 1)
annb = book.groupby(book.index.year).apply(lambda x: (1 + x).prod() - 1)
for y in ann.index:
    print(f"  {y}: combined {ann[y]*100:+6.1f}% (${ann[y]*NAV:+,.0f})  book {annb[y]*100:+6.1f}%")

# --- Part 3: stationary block bootstrap (expected block 6 months) ---
rng = np.random.default_rng(7)


def boot_paths(r, horizon, n=20_000, p=1 / 6):
    v = r.values
    n_obs = len(v)
    out = np.empty((n, horizon))
    for k in range(n):
        i = rng.integers(n_obs)
        for j in range(horizon):
            out[k, j] = v[i]
            i = rng.integers(n_obs) if rng.random() < p else (i + 1) % n_obs
    return out


def dd_of(paths):
    curves = np.cumprod(1 + paths, axis=1)
    peaks = np.maximum.accumulate(curves, axis=1)
    return (curves / peaks - 1).min(axis=1)


for tag, series in [("full-history", combo), ("2016+ regime", combo[combo.index >= "2016-01-01"])]:
    p12 = boot_paths(series, 12)
    p36 = boot_paths(series, 36)
    pnl12 = (np.prod(1 + p12, axis=1) - 1)
    dd12, dd36 = dd_of(p12), dd_of(p36)
    q = lambda a, x: np.percentile(a, x)
    print(f"\n=== bootstrap ({tag}, stationary blocks ~6mo, 20k paths) ===")
    print(f"1y PnL:   p5 {q(pnl12,5)*100:+6.1f}% (${q(pnl12,5)*NAV:+,.0f})   "
          f"median {q(pnl12,50)*100:+6.1f}% (${q(pnl12,50)*NAV:+,.0f})   "
          f"p95 {q(pnl12,95)*100:+6.1f}%")
    print(f"P(down year): {(pnl12<0).mean()*100:.1f}%")
    print(f"1y maxDD: median {q(dd12,50)*100:6.1f}%   p95 {q(dd12,5)*100:6.1f}% "
          f"(5% of years worse than this)")
    print(f"3y maxDD: median {q(dd36,50)*100:6.1f}%   p95 {q(dd36,5)*100:6.1f}%")
