"""Marginal effect of a factor sleeve on the combined book. Book monthly return
= PnL_flat_750k by exit month / 750k. Sleeve overlays at 0.5x NAV (funded from
idle cash / light margin). 2016-09+ window.
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUN_DATE = pd.Timestamp("2026-07-02")

fac = pd.read_parquet(HERE / "factor_etf_prices.parquet")
fac.index = pd.to_datetime(fac.index).normalize()
fac = fac[fac.index < RUN_DATE]
mret = fac.resample("ME").last().pct_change()
mret.index = mret.index.to_period("M")

tr = pd.read_parquet(ROOT / "data" / "backtest_trades_full.parquet")
tr["Exit Date"] = pd.to_datetime(tr["Exit Date"])
book = (tr.groupby(tr["Exit Date"].dt.to_period("M"))["PnL_flat_750k"].sum() / 750_000.0)

frag = pd.read_parquet(ROOT / "data" / "rd2_fragility.parquet")
frag.index = pd.to_datetime(frag.index).normalize()
dial = frag["63d"].dropna().rolling(10, min_periods=1).mean()
dial_d = dial.reindex(pd.date_range(dial.index.min(), dial.index.max()), method="ffill", limit=5)
dial_mmean = dial_d.resample("ME").mean()
dial_mmean.index = dial_mmean.index.to_period("M")

W = pd.Period("2016-09", "M")
idx = mret.index[(mret.index >= W)]
book = book.reindex(idx).fillna(0)

MQUV = mret[["MTUM", "QUAL", "USMV", "VLUE"]].mean(axis=1)


def stats_line(r, name):
    sh = r.mean() * 12 / (r.std() * np.sqrt(12))
    eq = (1 + r).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    return f"{name:26s} avg={100*r.mean():+.2f}%/mo vol={100*r.std()*np.sqrt(12):.1f}% Sharpe={sh:.2f} maxDD={100*dd:.1f}%"


print("=== 2016-09+ monthly, book flat-750k basis ===")
print(stats_line(book, "book alone"))
hi = dial_mmean.reindex(idx) >= 50
print(f"book in high-frag months (N={int(hi.sum())}): avg={100*book[hi].mean():+.2f}%/mo | other: {100*book[~hi].mean():+.2f}%/mo")

for nm, sleeve in [("SPY", mret["SPY"]), ("USMV", mret["USMV"]), ("MQUV", MQUV)]:
    s = sleeve.reindex(idx)
    comb = book + 0.5 * s
    print(stats_line(comb, f"book + 0.5x {nm}"))
    print(f"    sleeve-in-high-frag drag: {100*(0.5*s[hi]).mean():+.2f}%/mo over {int(hi.sum())} months")
