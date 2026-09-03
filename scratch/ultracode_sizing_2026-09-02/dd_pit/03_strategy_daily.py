"""dd_pit step 3: rebuild the per-strategy daily MTM payload through the
ledger's last exit (2026-09-01) from data/backtest_trades_full.parquet, using
the SAME engine the site uses (pages.strat_backtester.get_daily_mtm_series per
Strategy||Tier on build_site.page_shaped frames), flat $750k basis. Prices are
read straight from data/master_prices.parquet (no data_provider R2 refresh).

Writes strategy_daily_extended.parquet (wide, one column per Strategy||Tier
plus book = row sum) and validates against dist/data/strategy_daily.json over
the overlap.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


class _NoOp:
    def __getattr__(self, name): return self
    def __call__(self, *a, **k): return self
    def __bool__(self): return False
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def cache_data(self, *a, **k):
        def deco(fn): return fn
        return deco
    cache_resource = cache_data


sys.modules["streamlit"] = _NoOp()
from pages.strat_backtester import get_daily_mtm_series  # noqa: E402
from build_site import load_ledger, page_shaped  # noqa: E402

df = load_ledger()
df = df[df["PnL_flat_750k"].notna()].copy()
print(f"ledger {len(df)} trades, signal {df['Signal Date'].min().date()} .. {df['Signal Date'].max().date()}, exit max {df['Exit Date'].max().date()}")
flat = page_shaped(df)

tickers = sorted(set(flat["Ticker"].astype(str).str.replace(".", "-", regex=False)) | {"SPY"})
print(f"loading {len(tickers)} tickers from master_prices ...")
tab = pq.read_table(ROOT / "data/master_prices.parquet", columns=["ticker", "date", "Close"],
                    filters=[("ticker", "in", tickers)]).to_pandas()
tab["date"] = pd.to_datetime(tab["date"])
tab = tab[tab["date"] <= "2026-09-01"]          # today's in-progress bar is excluded
md = {t: g.set_index("date")[["Close"]].sort_index() for t, g in tab.groupby("ticker")}
missing = [t for t in tickers if t not in md]
print(f"  {len(md)} tickers priced; missing {len(missing)}: {missing[:20]}")

start = flat["Date"].min()
groups = {}
for (s, t), g in flat.groupby(["Strategy", "Tier"]):
    key = f"{s}||{t}"
    groups[key] = get_daily_mtm_series(g, md, start_date=start)
    print(f"  MTM {key}: {len(g)} trades, sum {groups[key].sum():,.0f} vs booked {g['PnL'].sum():,.0f}")
idx = None
for s in groups.values():
    idx = s.index if idx is None else idx.union(s.index)
idx = idx[idx <= "2026-09-01"]
W = pd.DataFrame({k: v.reindex(idx).fillna(0.0) for k, v in groups.items()}, index=idx)
W.index.name = "date"
W["book"] = W.sum(axis=1)
W.to_parquet(HERE / "strategy_daily_extended.parquet")
print(f"\nwrote strategy_daily_extended.parquet: {W.shape}, {idx.min().date()} .. {idx.max().date()}")
print(f"book total ${W['book'].sum():,.0f} vs ledger PnL_flat sum ${df['PnL_flat_750k'].sum():,.0f}")

# --- validation vs the site payload (ends 2026-08-07) ---
sd = json.load(open(ROOT / "dist/data/strategy_daily.json"))
dates = pd.to_datetime(sd["dates"])
S = pd.DataFrame(sd["series"], index=dates).fillna(0.0)
tot = pd.Series(sd["total_flat"], index=dates, dtype=float)
common = S.index.intersection(W.index)
print(f"\nvalidation vs dist/data/strategy_daily.json over {common.min().date()} .. {common.max().date()} ({len(common)} days)")
mine_book = W.loc[common, "book"]; site_book = tot.reindex(common)
site_sum = S.reindex(common).sum(axis=1)
print(f"  corr(book, site total_flat) {mine_book.corr(site_book):.5f}; corr(book, site series-sum) {mine_book.corr(site_sum):.5f}")
print(f"  sum mine ${mine_book.sum():,.0f}  site total_flat ${site_book.sum():,.0f}  site series-sum ${site_sum.sum():,.0f}")
d = (mine_book - site_sum)
big = d[d.abs() > 500]
print(f"  days with |diff| > $500 vs site series-sum: {len(big)}; largest {d.abs().max():,.0f} on {d.abs().idxmax().date()}")
print(f"  by year (mine - site series-sum):")
print((d.groupby(d.index.year).sum()).round(0).to_string())
missing_cols = [c for c in S.columns if c not in W.columns]; new_cols = [c for c in W.columns if c not in S.columns and c != "book"]
print(f"  columns only in site: {missing_cols}; only in mine: {new_cols}")
