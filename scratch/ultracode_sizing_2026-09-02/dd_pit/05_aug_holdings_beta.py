"""dd_pit step 5: why was the 126d return-beta 0.19 while the book's realised
beta in the Aug-2026 episode was ~1.1? Rebuild the book's HOLDINGS-BASED beta
day by day from the ledger's open positions (flat-basis shares x entry price,
signed by direction, x each ticker's own 126d OLS beta to SPY from
master_prices), and price a hedge at that beta over the same armed days.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
NAV = 750_000.0
pd.set_option("display.width", 250, "display.max_rows", 200, "display.float_format", "{:,.2f}".format)

led = pd.read_parquet(ROOT / "data/backtest_trades_full.parquet")
for c in ["Entry Date", "Exit Date"]:
    led[c] = pd.to_datetime(led[c])
led = led[led["PnL_flat_750k"].notna() & led["Entry Date"].notna()]
if "Shares_flat" not in led.columns:
    led["Shares_flat"] = led["Shares"] * led["Risk_flat_750k"] / led["Risk_compounded"].replace(0, np.nan)
sgn = np.where(led["Direction"].astype(str) == "Short", -1.0, 1.0)
led["notional"] = led["Shares_flat"].fillna(0) * led["Entry Price"] * sgn

days = pd.bdate_range("2026-07-15", "2026-09-01")
tick = sorted(set(led.loc[(led["Entry Date"] <= days[-1]) & (led["Exit Date"] >= days[0]), "Ticker"].astype(str).str.replace(".", "-", regex=False)) | {"SPY"})
px = pq.read_table(ROOT / "data/master_prices.parquet", columns=["ticker", "date", "Close"], filters=[("ticker", "in", tick)]).to_pandas().pivot(index="date", columns="ticker", values="Close")
px.index = pd.to_datetime(px.index); px = px[px.index <= "2026-09-01"]
ret = px.pct_change(fill_method=None)
spy = ret["SPY"]
beta126 = pd.DataFrame({t: ret[t].rolling(126).cov(spy) / spy.rolling(126).var() for t in ret.columns}).shift(1).clip(-1, 3)

rows = []
for d in days:
    opn = led[(led["Entry Date"] <= d) & (led["Exit Date"] >= d)]
    if opn.empty:
        rows.append(dict(date=d, n_open=0, gross_pct=0, net_pct=0, beta_hold=0)); continue
    t = opn["Ticker"].astype(str).str.replace(".", "-", regex=False)
    b = beta126.reindex([d]).iloc[0].reindex(t.values).fillna(1.0).values
    # mark at the day's close
    mk = px.reindex([d]).iloc[0].reindex(t.values).values
    notional = opn["Shares_flat"].fillna(0).values * np.where(np.isnan(mk), opn["Entry Price"].values, mk) * np.where(opn["Direction"].astype(str) == "Short", -1.0, 1.0)
    rows.append(dict(date=d, n_open=len(opn), gross_pct=float(np.abs(notional).sum() / NAV * 100), net_pct=float(notional.sum() / NAV * 100),
                     beta_hold=float((notional * b).sum() / NAV), olv_net_pct=float(notional[(opn["Strategy"] == "Oversold Low Volume").values].sum() / NAV * 100)))
H = pd.DataFrame(rows).set_index("date")
seg = pd.read_csv(HERE / "hedge_aug2026_daily.csv", index_col=0, parse_dates=True)
H = H.join(seg[["dial_pit", "armed_pit", "beta_hat", "spy_pct", "book_usd", "hedge_usd"]])
H["hedge_at_hold_beta_usd"] = -H.armed_pit * H.beta_hold.shift(1).fillna(0) * H.spy_pct / 100 * NAV
print(H.round(2).to_string())
a = H.armed_pit > 0
print(f"\narmed days (PIT) {int(a.sum())}: mean holdings beta {H.beta_hold[a].mean():.2f} (range {H.beta_hold[a].min():.2f}..{H.beta_hold[a].max():.2f}), mean gross {H.gross_pct[a].mean():.0f}% NAV, net {H.net_pct[a].mean():.0f}% NAV, OLV net {H.olv_net_pct[a].mean():.0f}% NAV")
print(f"ex-ante 126d return-beta on armed days: {H.beta_hat[a].mean():.2f}")
print(f"hedge at return-beta: ${H.hedge_usd[a].sum():,.0f}; hedge at holdings-beta (lag-1): ${H.hedge_at_hold_beta_usd[a].sum():,.0f}; book on armed days ${H.book_usd[a].sum():,.0f}")
# realised beta of the book on armed days vs each ex-ante measure
ga = H[a]
print(f"realised OLS beta of book on armed days: {np.polyfit(ga.spy_pct / 100, ga.book_usd / NAV, 1)[0]:.2f}")
H.round(3).to_csv(HERE / "hedge_aug2026_holdings.csv")
