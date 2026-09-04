"""Refined tradeable hybrid under the intraday-vs-gap entry rule:
  - LIMIT (open - 0.25 ATR): US single stocks + US index ETFs (trade during the move)
  - MARKET-ON-OPEN: international index ETFs (gap at the US open, no pullback)
Stock longs come from the main candidates; index ETFs from the proxy candidates.
(Non-index macro: commodities/bonds/FX not included — separate sleeve.)"""
import sys
import numpy as np
import pandas as pd

ROOT = r"C:\Users\McKinley Slade\dev\New_Seasonals"
sys.path.insert(0, ROOT)
import scripts.seasonal_edge as se
from scripts.seasonal_ticket_sim import simulate_ticket
from scripts.seasonal_sharpe import ratios

US_ETF = {"SPY", "QQQ", "DIA", "IWM", "IJH", "ONEQ", "SOXX", "IYT", "VXX"}


def sim_from(cand_path, want_tickers, want_dir, mode, prices):
    cand = pd.read_parquet(cand_path)
    cand["asof"] = pd.to_datetime(cand["asof"])
    rows = []
    for r in cand.itertuples():
        if want_tickers is not None and r.ticker not in want_tickers:
            continue
        if want_dir is not None and r.direction != want_dir:
            continue
        px = prices.get(r.ticker)
        if px is None or px.empty:
            continue
        tk = {"ticker": r.ticker, "direction": r.direction, "entry": float(r.t_entry),
              "stop": float(r.t_stop), "target": float(r.t_target), "time_stop_days": int(r.time_stop_days)}
        o = simulate_ticket(tk, px, r.asof, entry_mode=mode, entry_atr_mult=0.25)
        if o is None or not o.get("filled", True):
            continue
        rows.append({"ticker": r.ticker, "direction": r.direction, "asof": r.asof,
                     "entry_date": o["entry_date"], "exit_date": pd.Timestamp(o["exit_date"]),
                     "R": o["R"], "cycle": int(pd.Timestamp(r.asof).year % 4)})
    return pd.DataFrame(rows)


MAIN = ROOT + r"\data\seasonal_ideas_candidates.parquet"
PRX = ROOT + r"\data\seasonal_proxy_candidates.parquet"

# preload all needed prices
maincand = pd.read_parquet(MAIN); prxcand = pd.read_parquet(PRX)
all_t = set(maincand["ticker"]) | set(prxcand["ticker"])
prices = {t: se.load_prices([t]).get(se._norm_ticker(t)) for t in all_t}

# stock longs: from main candidates, detect_seasonal == stock channel, limit entry
sc = maincand.copy()
stock_tk = set(sc[sc["channel"] == "detect_seasonal"]["ticker"])
stock = sim_from(MAIN, stock_tk, "long", "limit", prices)
stock["sleeve"] = "stock-long (limit)"

intl_etf = set(prxcand["ticker"]) - US_ETF
us = sim_from(PRX, US_ETF, None, "limit", prices); us["sleeve"] = "US idx ETF (limit)"
intl = sim_from(PRX, intl_etf, None, "t1_open", prices); intl["sleeve"] = "Intl idx ETF (open)"

book = pd.concat([stock, us, intl], ignore_index=True)
# dedup one open per ticker+direction
book = book.sort_values(["ticker", "direction", "entry_date"])
keep, last = [], {}
for x in book.itertuples():
    k = (x.ticker, x.direction)
    if last.get(k) is None or x.entry_date > last[k]:
        keep.append(x.Index); last[k] = x.exit_date
book = book.loc[keep]
full = pd.date_range(book["exit_date"].min().normalize(), book["exit_date"].max().normalize(), freq="B")


def prof(name, b):
    if len(b) == 0:
        print(f"{name}: none"); return
    R = b["R"].astype(float); pf = R[R > 0].sum() / abs(R[R < 0].sum())
    m = b.groupby(b["exit_date"].dt.normalize())["R"].sum().reindex(full, fill_value=0).resample("ME").sum()
    sh, so = ratios(m, 12)
    print(f"{name:30s} N{len(b):5d} avgR{R.mean():+.3f} PF{pf:.2f} TotR{R.sum():+6.0f} Sharpe{sh:.2f} Sortino{so:.2f}")


print("=== REFINED TRADEABLE HYBRID (geography entry rule) ===")
for s in ["stock-long (limit)", "US idx ETF (limit)", "Intl idx ETF (open)"]:
    prof("  " + s, book[book.sleeve == s])
prof("REFINED HYBRID (all)", book)
prof("  refined + ex-midterm", book[book.cycle != 2])
print("\nfor reference, prior 'all-open-macro' hybrid was Sharpe 1.23 / ex-mid 1.55")
