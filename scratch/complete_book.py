"""Complete tradeable book with the CORRECTED entry rule:
  LIMIT (open - 0.25 ATR): US single stocks + US equity index ETFs
      (the only instruments whose underlying trades during the US cash session)
  MARKET-ON-OPEN: everything else — international equity ETFs, commodity/bond ETFs,
      futures, FX (underlying moves overnight/24h -> gaps at the US open)
"""
import sys
import numpy as np
import pandas as pd

ROOT = r"C:\Users\McKinley Slade\dev\New_Seasonals"
sys.path.insert(0, ROOT)
import scripts.seasonal_edge as se
from scripts.seasonal_ticket_sim import simulate_ticket
from scripts.seasonal_sharpe import ratios

US_IDX_ETF = {"SPY", "QQQ", "DIA", "IWM", "IJH", "ONEQ", "SOXX", "IYT", "VXX"}
MAIN = ROOT + r"\data\seasonal_ideas_candidates.parquet"
PRX = ROOT + r"\data\seasonal_proxy_candidates.parquet"
mc = pd.read_parquet(MAIN); mc["asof"] = pd.to_datetime(mc["asof"])
pc = pd.read_parquet(PRX); pc["asof"] = pd.to_datetime(pc["asof"])
prices = {t: se.load_prices([t]).get(se._norm_ticker(t)) for t in set(mc["ticker"]) | set(pc["ticker"])}


def sim(cand, mode, sleeve):
    rows = []
    for r in cand.itertuples():
        px = prices.get(r.ticker)
        if px is None or px.empty:
            continue
        tk = {"ticker": r.ticker, "direction": r.direction, "entry": float(r.t_entry),
              "stop": float(r.t_stop), "target": float(r.t_target), "time_stop_days": int(r.time_stop_days)}
        o = simulate_ticket(tk, px, r.asof, entry_mode=mode, entry_atr_mult=0.25)
        if o is None or not o.get("filled", True):
            continue
        rows.append({"ticker": r.ticker, "direction": r.direction, "entry_date": o["entry_date"],
                     "exit_date": pd.Timestamp(o["exit_date"]), "R": o["R"],
                     "cycle": int(pd.Timestamp(r.asof).year % 4), "sleeve": sleeve})
    return pd.DataFrame(rows)


stock_tk = set(mc[mc["channel"] == "detect_seasonal"]["ticker"])
stock = sim(mc[(mc["channel"] == "detect_seasonal") & (mc["direction"] == "long")], "limit", "US stocks (limit)")
us_etf = sim(pc[pc["ticker"].isin(US_IDX_ETF)], "limit", "US idx ETF (limit)")
intl_etf = sim(pc[~pc["ticker"].isin(US_IDX_ETF)], "t1_open", "Intl idx ETF (open)")
nonidx = sim(mc[(mc["channel"] == "detect_cross_asset") & (~mc["ticker"].str.startswith("^"))],
             "t1_open", "commod/bond/FX (open)")

book = pd.concat([stock, us_etf, intl_etf, nonidx], ignore_index=True)
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


print("=== COMPLETE TRADEABLE BOOK (corrected entry rule) ===")
for s in ["US stocks (limit)", "US idx ETF (limit)", "Intl idx ETF (open)", "commod/bond/FX (open)"]:
    prof("  " + s, book[book.sleeve == s])
prof("COMPLETE BOOK (all)", book)
prof("  complete + ex-midterm", book[book.cycle != 2])
