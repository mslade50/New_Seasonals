"""Equity curves + lumpiness by horizon (5/10/21d) for the complete tradeable book.
Tests the '21d is lumpy from holding more correlated beta' hypothesis."""
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
                     "entry_date": pd.Timestamp(o["entry_date"]), "exit_date": pd.Timestamp(o["exit_date"]), "R": o["R"]})
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
bdays = pd.bdate_range(book["entry_date"].min(), book["exit_date"].max())

print("=== lumpiness by horizon ===")
print(f"{'hz':4s} {'TotR':>6} {'mo.std':>7} {'maxMo':>6} {'minMo':>6} {'maxDD':>6} {'top5mo%':>8} {'avgConc':>8} {'Sharpe':>7}")
for h in [5, 10, 21]:
    b = book[book.tsd == h]
    m = b.groupby(b["exit_date"].dt.normalize())["R"].sum().reindex(bdays, fill_value=0).resample("ME").sum()
    eq = b.sort_values("exit_date")["R"].cumsum(); dd = (eq - eq.cummax()).min()
    top5 = m.nlargest(5).sum() / m.sum() * 100
    conc = np.mean([((b["entry_date"] <= d) & (b["exit_date"] >= d)).sum() for d in bdays[::5]])
    sh, _ = ratios(m, 12)
    print(f"{h:>2}d  {b['R'].sum():+6.0f} {m.std():7.1f} {m.max():6.0f} {m.min():6.0f} {dd:6.0f} "
          f"{top5:7.0f}% {conc:8.1f} {sh:7.2f}")

# --- plot ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 9))
colors = {5: "#1f77b4", 10: "#2ca02c", 21: "#d62728"}
for h in [5, 10, 21]:
    d = book[book.tsd == h].sort_values("exit_date")
    eq = d["R"].cumsum()
    a1.plot(d["exit_date"].values, eq.values, label=f"{h}d (Tot {d['R'].sum():.0f}R)", color=colors[h], lw=1.6)
    a2.plot(d["exit_date"].values, (eq - eq.cummax()).values, color=colors[h], lw=1.1, label=f"{h}d")
a1.set_title("Cumulative R by exit date, per horizon"); a1.legend(); a1.grid(alpha=.3); a1.set_ylabel("cum R")
a2.set_title("Underwater (R below peak) per horizon"); a2.legend(); a2.grid(alpha=.3); a2.set_ylabel("R")
fig.tight_layout()
out = ROOT + r"\scratch\horizon_curves.png"
fig.savefig(out, dpi=110)
print(f"\nSaved -> {out}")
