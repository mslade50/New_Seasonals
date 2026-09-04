"""Does the 0.25-ATR limit only hurt INTERNATIONAL ETFs (gap-at-open) and not US
ones (intraday-traded)? Split the macro proxy book US vs intl, open vs limit."""
import sys
import numpy as np
import pandas as pd

ROOT = r"C:\Users\McKinley Slade\dev\New_Seasonals"
sys.path.insert(0, ROOT)
import scripts.seasonal_edge as se
from scripts.seasonal_ticket_sim import simulate_ticket
from scripts.seasonal_sharpe import ratios

US = {"SPY", "QQQ", "DIA", "IWM", "IJH", "ONEQ", "SOXX", "IYT", "VXX"}  # US-market, intraday
# everything else in the proxy set is an international country/region ETF (gaps at US open)

cand = pd.read_parquet(ROOT + r"\data\seasonal_proxy_candidates.parquet")
cand["asof"] = pd.to_datetime(cand["asof"])
prices = {t: se.load_prices([t]).get(se._norm_ticker(t)) for t in cand["ticker"].unique()}


def sim_book(mode):
    out = []
    for r in cand.itertuples():
        px = prices.get(r.ticker)
        if px is None or px.empty:
            continue
        tk = {"ticker": r.ticker, "direction": r.direction, "entry": float(r.t_entry),
              "stop": float(r.t_stop), "target": float(r.t_target), "time_stop_days": int(r.time_stop_days)}
        o = simulate_ticket(tk, px, r.asof, entry_mode=mode, entry_atr_mult=0.25)
        if o is None or not o.get("filled", True):
            continue
        out.append({"ticker": r.ticker, "direction": r.direction, "asof": r.asof,
                    "entry_date": o["entry_date"], "exit_date": pd.Timestamp(o["exit_date"]),
                    "R": o["R"], "grp": "US" if r.ticker in US else "Intl"})
    d = pd.DataFrame(out)
    # dedup per ticker+direction (one open position at a time)
    d = d.sort_values(["ticker", "direction", "entry_date"])
    keep, last = [], {}
    for x in d.itertuples():
        k = (x.ticker, x.direction)
        if last.get(k) is None or x.entry_date > last[k]:
            keep.append(x.Index); last[k] = x.exit_date
    return d.loc[keep]


def stat(b):
    if len(b) == 0:
        return "  (none)"
    R = b["R"].astype(float); pf = R[R > 0].sum() / abs(R[R < 0].sum())
    full = pd.date_range(b["exit_date"].min().normalize(), b["exit_date"].max().normalize(), freq="B")
    m = b.groupby(b["exit_date"].dt.normalize())["R"].sum().reindex(full, fill_value=0).resample("ME").sum()
    sh, _ = ratios(m, 12)
    return f"N{len(b):4d}  Win {100*(R>0).mean():4.1f}%  AvgR {R.mean():+.3f}  PF {pf:.2f}  TotR {R.sum():+5.0f}  Sharpe {sh:.2f}"


opn = sim_book("t1_open"); lim = sim_book("limit")
print("=== MACRO PROXY: market-on-open vs limit 0.25 ATR, by geography (deduped) ===\n")
for grp in ["US", "Intl"]:
    print(f"-- {grp} --")
    print(f"  open  : {stat(opn[opn.grp == grp])}")
    print(f"  limit : {stat(lim[lim.grp == grp])}")
    print()
print("-- ALL macro proxies --")
print(f"  open  : {stat(opn)}")
print(f"  limit : {stat(lim)}")
