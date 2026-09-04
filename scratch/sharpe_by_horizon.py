"""Sharpe and Sortino of the complete tradeable book, broken down by time frame
(time-stop horizon: 5d / 10d / 21d). Same corrected entry rule as complete_book."""
import sys
import numpy as np
import pandas as pd

ROOT = r"C:\Users\McKinley Slade\dev\New_Seasonals"
sys.path.insert(0, ROOT)
import scripts.seasonal_edge as se
from scripts.seasonal_ticket_sim import simulate_ticket
from scripts.seasonal_sharpe import ratios

US_IDX_ETF = {"SPY", "QQQ", "DIA", "IWM", "IJH", "ONEQ", "SOXX", "IYT", "VXX"}
mc = pd.read_parquet(ROOT + r"\data\seasonal_ideas_candidates.parquet"); mc["asof"] = pd.to_datetime(mc["asof"])
pc = pd.read_parquet(ROOT + r"\data\seasonal_proxy_candidates.parquet"); pc["asof"] = pd.to_datetime(pc["asof"])
prices = {t: se.load_prices([t]).get(se._norm_ticker(t)) for t in set(mc["ticker"]) | set(pc["ticker"])}


def sim(cand, mode):
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
        rows.append({"ticker": r.ticker, "direction": r.direction, "tsd": int(r.time_stop_days),
                     "entry_date": o["entry_date"], "exit_date": pd.Timestamp(o["exit_date"]),
                     "R": o["R"], "cycle": int(pd.Timestamp(r.asof).year % 4), "bars": o["bars_held"]})
    return pd.DataFrame(rows)


book = pd.concat([
    sim(mc[(mc["channel"] == "detect_seasonal") & (mc["direction"] == "long")], "limit"),
    sim(pc[pc["ticker"].isin(US_IDX_ETF)], "limit"),
    sim(pc[~pc["ticker"].isin(US_IDX_ETF)], "t1_open"),
    sim(mc[(mc["channel"] == "detect_cross_asset") & (~mc["ticker"].str.startswith("^"))], "t1_open"),
], ignore_index=True)
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
    print(f"{name:18s} N{len(b):5d} Win{100*(R>0).mean():5.1f}% AvgR{R.mean():+.3f} MedHold{int(b['bars'].median())}b "
          f"PF{pf:.2f} TotR{R.sum():+6.0f} Sharpe{sh:.2f} Sortino{so:.2f}")


print("=== COMPLETE BOOK by time-stop horizon ===")
for h in [5, 10, 21]:
    prof(f"{h}d horizon", book[book.tsd == h])
prof("ALL", book)
print("\n=== by horizon, ex-midterm ===")
for h in [5, 10, 21]:
    prof(f"{h}d ex-midterm", book[(book.tsd == h) & (book.cycle != 2)])
