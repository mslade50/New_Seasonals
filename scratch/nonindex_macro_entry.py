"""Non-index macro (commodities/bonds/FX/crypto from the MAIN candidates): open vs
limit 0.25 ATR, split by instrument type. These were never run through the entry
analysis even though their tickets were already on disk."""
import sys
import numpy as np
import pandas as pd

ROOT = r"C:\Users\McKinley Slade\dev\New_Seasonals"
sys.path.insert(0, ROOT)
import scripts.seasonal_edge as se
from scripts.seasonal_ticket_sim import simulate_ticket
from scripts.seasonal_sharpe import ratios

cand = pd.read_parquet(ROOT + r"\data\seasonal_ideas_candidates.parquet")
cand["asof"] = pd.to_datetime(cand["asof"])
# non-index macro = cross-asset channel, not a cash index (^...)
macro = cand[(cand["channel"] == "detect_cross_asset") & (~cand["ticker"].str.startswith("^"))].copy()


def kind(t):
    if t.endswith("=F"):
        return "futures (24h)"
    if t.endswith("=X") or t == "DX-Y.NYB":
        return "FX (24h)"
    if t.endswith("-USD"):
        return "crypto (24h)"
    return "US-session ETF"


macro["kind"] = macro["ticker"].map(kind)
print("non-index macro tickers by type:")
for k, g in macro.groupby("kind"):
    print(f"  {k:16s}: {sorted(g['ticker'].unique())}")
prices = {t: se.load_prices([t]).get(se._norm_ticker(t)) for t in macro["ticker"].unique()}


def sim_book(mode):
    rows = []
    for r in macro.itertuples():
        px = prices.get(r.ticker)
        if px is None or px.empty:
            continue
        tk = {"ticker": r.ticker, "direction": r.direction, "entry": float(r.t_entry),
              "stop": float(r.t_stop), "target": float(r.t_target), "time_stop_days": int(r.time_stop_days)}
        o = simulate_ticket(tk, px, r.asof, entry_mode=mode, entry_atr_mult=0.25)
        if o is None or not o.get("filled", True):
            continue
        rows.append({"ticker": r.ticker, "kind": r.kind, "direction": r.direction,
                     "entry_date": o["entry_date"], "exit_date": pd.Timestamp(o["exit_date"]), "R": o["R"]})
    d = pd.DataFrame(rows).sort_values(["ticker", "direction", "entry_date"])
    keep, last = [], {}
    for x in d.itertuples():
        k = (x.ticker, x.direction)
        if last.get(k) is None or x.entry_date > last[k]:
            keep.append(x.Index); last[k] = x.exit_date
    return d.loc[keep]


opn = sim_book("t1_open"); lim = sim_book("limit")
full = pd.date_range(min(opn["exit_date"].min(), lim["exit_date"].min()).normalize(),
                     max(opn["exit_date"].max(), lim["exit_date"].max()).normalize(), freq="B")


def stat(b):
    if len(b) == 0:
        return "(none)"
    R = b["R"].astype(float); pf = R[R > 0].sum() / abs(R[R < 0].sum()) if (R < 0).any() else np.inf
    m = b.groupby(b["exit_date"].dt.normalize())["R"].sum().reindex(full, fill_value=0).resample("ME").sum()
    sh, _ = ratios(m, 12)
    return f"N{len(b):4d} Win{100*(R>0).mean():4.1f}% AvgR{R.mean():+.3f} PF{pf:.2f} TotR{R.sum():+5.0f} Sharpe{sh:.2f}"


print("\n=== NON-INDEX MACRO: open vs limit, by type (deduped) ===")
for k in ["US-session ETF", "futures (24h)", "FX (24h)", "crypto (24h)"]:
    print(f"-- {k} --")
    print(f"   open : {stat(opn[opn.kind == k])}")
    print(f"   limit: {stat(lim[lim.kind == k])}")
print("-- ALL non-index macro --")
print(f"   open : {stat(opn)}")
print(f"   limit: {stat(lim)}")
