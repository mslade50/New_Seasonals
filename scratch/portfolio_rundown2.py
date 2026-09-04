"""Rundown part 2: signal-alpha robustness and concentration."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
led = pd.read_parquet(ROOT / "data" / "backtest_trades_full.parquet")
led["Signal Date"] = pd.to_datetime(led["Signal Date"])
led["yr"] = led["Signal Date"].dt.year

print("=== per-strategy split-half (avgR first half vs second half of ITS OWN life) ===")
rows = []
for s, g in led.groupby("Strategy"):
    g = g.sort_values("Signal Date")
    mid = len(g) // 2
    a, b = g.iloc[:mid], g.iloc[mid:]
    rows.append({
        "Strategy": s[:26], "N": len(g),
        "spanA": f"{a['Signal Date'].min().year}-{a['Signal Date'].max().year}",
        "avgR_A": round(a.R_Multiple.mean(), 2),
        "spanB": f"{b['Signal Date'].min().year}-{b['Signal Date'].max().year}",
        "avgR_B": round(b.R_Multiple.mean(), 2),
        "decay": round(b.R_Multiple.mean() - a.R_Multiple.mean(), 2),
        "PnL_share%": round(100 * g.PnL_flat_750k.sum() / led.PnL_flat_750k.sum(), 1),
    })
print(pd.DataFrame(rows).sort_values("PnL_share%", ascending=False).to_string(index=False))

print("\n=== concentration ===")
tot = led.PnL_flat_750k.sum()
by_tier = led.groupby("Tier").PnL_flat_750k.sum() / tot * 100
print("PnL by tier %:", by_tier.round(1).to_dict())
tick = led.groupby(led.Ticker.str.upper()).PnL_flat_750k.sum().sort_values(ascending=False)
print(f"top-10 tickers = {tick.head(10).sum()/tot*100:.1f}% of PnL: "
      f"{[(t, round(v/1000)) for t, v in tick.head(10).items()]}")
neg = tick[tick < 0]
print(f"worst-5 tickers: {[(t, round(v/1000)) for t, v in neg.head(5).items()] if False else [(t, round(v/1000)) for t, v in tick.tail(5).items()]}")

# single-name vs ETF/index split (rough: carets + known ETFs vs rest)
etfish = led.Ticker.str.upper().str.match(r"^(SPY|QQQ|IWM|DIA|SMH|XL[A-Z]|EW[A-Z]|EEM|EFA|FXI|GLD|SLV|USO|DBC|UUP|TLT|IEF|LQD|HYG|VNQ|\^)")
print(f"single-stock share of PnL: {led[~etfish].PnL_flat_750k.sum()/tot*100:.1f}% "
      f"(subject to universe survivorship)")

print("\n=== rolling 12m windows, book monthly (flat) ===")
daily = pd.read_parquet(ROOT / "data" / "backtest_daily_pnl.parquet")
daily["date"] = pd.to_datetime(daily["date"])
m = daily.set_index("date")["pnl_flat"].groupby(pd.Grouper(freq="ME")).sum() / 750000
m = m[m.index >= "2003-01-31"]
roll_ret = (1 + m).rolling(12).apply(np.prod) - 1
roll_sharpe = m.rolling(12).mean() / m.rolling(12).std() * np.sqrt(12)
print(f"rolling-12m return: min {roll_ret.min()*100:+.1f}% ({roll_ret.idxmin().date()}), "
      f"p5 {np.nanpercentile(roll_ret, 5)*100:+.1f}%, median {np.nanpercentile(roll_ret, 50)*100:+.1f}%")
print(f"rolling-12m Sharpe: min {roll_sharpe.min():.2f} ({roll_sharpe.idxmin().date()}), "
      f"p5 {np.nanpercentile(roll_sharpe.dropna(), 5):.2f}")
neg12 = (roll_ret.dropna() < 0)
print(f"negative 12m windows: {neg12.sum()}/{len(neg12)} ({neg12.mean()*100:.1f}%)")

print("\n=== trade-quality trend (book avgR by era) ===")
for lo, hi in [(2003, 2008), (2009, 2014), (2015, 2020), (2021, 2026)]:
    g = led[(led.yr >= lo) & (led.yr <= hi)]
    print(f"  {lo}-{hi}: N={len(g):>4}  avgR {g.R_Multiple.mean():+.3f}  "
          f"win {100*(g.R_Multiple>0).mean():.1f}%  PnL ${g.PnL_flat_750k.sum()/1000:+,.0f}k")
