"""Time-in-market (daily MTM) Sharpe / Sortino per book variant.

The headline daily Sharpe books each trade's whole R on its exit day over the full
business-day calendar, so idle days dilute it. This reconstructs the real daily
mark-to-market: each open position contributes its daily close-to-close R every
day it is held (entry day anchored to the fill, exit day to the realized exit
price), summed across concurrent positions. The ratios are then computed over
only the days the book is actually in the market (>=1 open position), annualized
by sqrt(252). Same constant-per-trade-risk basis as seasonal_sharpe.py.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = r"C:\Users\McKinley Slade\dev\New_Seasonals"
sys.path.insert(0, ROOT)
import scripts.seasonal_edge as se
from scripts.seasonal_sharpe import dedup, ratios

PARQUET = os.path.join(ROOT, "data", "seasonal_ideas_backtest.parquet")


def _daily_mtm(book, closes):
    """Return (daily_R Series over business days, in_market_day count). Each trade
    spreads its R across held days via close-to-close changes / risk_per_unit."""
    contrib = {}        # date -> summed daily R
    in_market = {}      # date -> count of open positions
    for r in book.itertuples():
        c = closes.get(se._norm_ticker(r.ticker))
        if c is None:
            continue
        s = 1.0 if r.direction == "long" else -1.0
        risk = float(r.risk_per_unit) if r.risk_per_unit else np.nan
        if not np.isfinite(risk) or risk <= 0:
            continue
        days = c.index[(c.index >= r.entry_date) & (c.index <= r.exit_date)]
        if len(days) == 0:
            continue
        prev = float(r.entry_price)        # fill on entry day
        for i, d in enumerate(days):
            px = float(r.exit_price) if d == days[-1] else float(c.loc[d])
            contrib[d] = contrib.get(d, 0.0) + s * (px - prev) / risk
            in_market[d] = in_market.get(d, 0) + 1
            prev = float(c.loc[d])
    if not contrib:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    idx = pd.date_range(min(contrib), max(contrib), freq="B")
    daily = pd.Series(contrib).reindex(idx, fill_value=0.0).sort_index()
    occ = pd.Series(in_market).reindex(idx, fill_value=0).sort_index()
    return daily, occ


def stats(name, book, closes):
    daily, occ = _daily_mtm(book, closes)
    if daily.empty:
        print(f"{name}: no data"); return
    inmkt = daily[occ > 0]
    sh_tim, so_tim = ratios(inmkt, 252)            # time-in-market
    sh_cal, so_cal = ratios(daily, 252)            # full-calendar MTM
    pct_in = 100 * (occ > 0).mean()
    avg_conc = occ[occ > 0].mean()
    R = book["R"].astype(float)
    print(f"{name:38s} | TIM Sharpe {sh_tim:5.2f} Sortino {so_tim:5.2f} | "
          f"calMTM Sharpe {sh_cal:5.2f} | %inMkt {pct_in:4.0f} | "
          f"avgConc {avg_conc:4.1f} | N {len(book):5d} TotR {R.sum():6.0f}")


def main(path=PARQUET):
    df = pd.read_parquet(path)
    df["entry_date"] = pd.to_datetime(df.entry_date); df["exit_date"] = pd.to_datetime(df.exit_date)
    df["asset"] = np.where(df.channel == "detect_seasonal", "stock", "macro")
    df = dedup(df)
    print(f"Loading prices for MTM reconstruction ...")
    closes = {k: v["Close"] for k, v in se.load_prices(list(se.IDEA_UNIVERSE), include_overflow=True).items()
              if v is not None and not v.empty}
    print(f"  {len(closes)} price series\n")

    V1 = df[~((df.asset == "stock") & (df.direction == "short"))]
    V2 = df[((df.asset == "stock") & (df.direction == "long")) | ((df.asset == "macro") & (df.direction == "short"))]
    books = {
        "raw (all, deduped)": df,
        "V1: exclude stock shorts": V1,
        "V2: stock-long + macro-short": V2,
        "long-only (both assets)": df[df.direction == "long"],
        "V1 minus midterm (cycle != 2)": V1[V1.cycle != 2],
        "V2 minus midterm (cycle != 2)": V2[V2.cycle != 2],
    }
    print("(TIM = time-in-market: ratios over days with >=1 open position, ann. sqrt(252))\n")
    for name, b in books.items():
        stats(name, b, closes)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else PARQUET)
