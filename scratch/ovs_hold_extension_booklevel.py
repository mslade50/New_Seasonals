"""Book-level impact of the OVS hold extension (any-loss, no filters).

Daily MTM for the ENTIRE ledger on the flat $750k basis, then the same with
OVS losers extended to T+5. Only OVS rows change; every other strategy's
series is identical, so the delta is exactly the OVS extension deltas placed
on the book curve. Universe cleanup (crypto/caret) applied book-wide for the
baseline too, so both sides use the tradeable book."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASIS = 750_000.0
ASOF = pd.Timestamp("2026-07-10")

ledger = pd.read_parquet(ROOT / "data" / "backtest_trades_full.parquet")
drop = ledger["Ticker"].str.endswith("-USD") | (
    ledger["Ticker"].str.startswith("^") & ~ledger["Ticker"].isin(["^GSPC", "^NDX"]))
book = ledger[~drop].copy()
for c in ("Signal Date", "Entry Date", "Exit Date"):
    book[c] = pd.to_datetime(book[c])
is_ovs = book["Strategy"] == "Overbot Vol Spike"
print(f"book trades: {len(book)} ({is_ovs.sum()} OVS)")

prices = pd.read_parquet(ROOT / "data" / "master_prices.parquet")
prices["date"] = pd.to_datetime(prices["date"])
# Ledger tickers use ^GSPC/^NDX but MTM marks should be on the traded ETF?
# The ledger books Entry/Exit prices on the SIGNAL ticker's series for IOB
# (engine models ^GSPC bars) so mark on the same series it was booked on.
need = set(book["Ticker"].unique()) | {"SPY"}
prices = prices[prices["ticker"].isin(need)]
px = {t: g.set_index("date").sort_index() for t, g in prices.groupby("ticker")}
missing = [t for t in book["Ticker"].unique() if t not in px]
if missing:
    print(f"WARNING no prices for {missing} — dropping their trades from both sides")
    book = book[~book["Ticker"].isin(missing)]
    is_ovs = book["Strategy"] == "Overbot Vol Spike"

# ---- OVS extension replay (any loss at T+2 time exit) ----
affected = is_ovs & (book["Exit Type"] == "Time") & (book["R_Multiple"] < 0)
new_exit = {}
for idx, row in book[affected].iterrows():
    df = px.get(row["Ticker"])
    if df is None or row["Exit Date"] not in df.index:
        continue
    pos = df.index.get_loc(row["Exit Date"])
    if abs(df["Close"].iloc[pos] / row["Exit Price"] - 1) > 0.01:
        continue
    ext = df.iloc[pos + 1 : pos + 4]
    if len(ext) < 3:
        continue
    tgt = row["Entry Price"] - row["tgt_atr"] * row["ATR"]
    ex_d, ex_p = ext.index[-1], ext["Close"].iloc[-1]
    for d, day in ext.iterrows():
        if day["Low"] <= tgt:
            ex_d, ex_p = d, tgt
            break
    new_exit[idx] = (ex_d, ex_p)
print(f"extensions: {len(new_exit)}")

sign = {"Long": 1.0, "Short": -1.0}


def build_book_curve(extend: bool) -> pd.Series:
    daily = defaultdict(float)
    for idx, row in book.iterrows():
        if extend and idx in new_exit:
            ex_d, ex_p = new_exit[idx]
        else:
            ex_d, ex_p = row["Exit Date"], row["Exit Price"]
        df = px[row["Ticker"]]
        sgn = sign[row["Direction"]]
        sh, prev = row["Shares_flat"], row["Entry Price"]
        for d, day in df.loc[row["Entry Date"]:ex_d].iterrows():
            mark = ex_p if d == ex_d else day["Close"]
            daily[d] += sgn * sh * (mark - prev)
            prev = mark
    return pd.Series(daily).sort_index()


cal = px["SPY"].index
base = build_book_curve(False)
ext = build_book_curve(True)
cal = cal[(cal >= base.index.min()) & (cal <= ASOF)]
base = base.reindex(cal, fill_value=0.0)
ext = ext.reindex(cal, fill_value=0.0)

for wname, start in [("full", None), ("last 5y", ASOF - pd.DateOffset(years=5)),
                     ("last 3y", ASOF - pd.DateOffset(years=3))]:
    print(f"\n--- BOOK, {wname} ---")
    print(f"{'variant':10s} {'PnL$':>11} {'Sharpe':>7} {'vol%ann':>8} {'maxDD$':>9} {'worstday$':>10}")
    for n, pnl in [("baseline", base), ("OVS-ext", ext)]:
        p = pnl if start is None else pnl[pnl.index >= start]
        r = p / BASIS
        curve = p.cumsum()
        dd = (curve - curve.cummax()).min()
        print(f"{n:10s} {p.sum():>11,.0f} {r.mean()/r.std()*np.sqrt(252):>7.2f} "
              f"{r.std()*np.sqrt(252)*100:>8.2f} {dd:>9,.0f} {p.min():>10,.0f}")
