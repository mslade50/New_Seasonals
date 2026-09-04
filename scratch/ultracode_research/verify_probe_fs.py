from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

root = Path(r"C:\Users\McKinley Slade\dev\New_Seasonals")

led = pd.read_parquet(root / "data" / "backtest_trades_full.parquet")
pnl = led.groupby(pd.to_datetime(led["Exit Date"]).dt.to_period("M"))["PnL_flat_750k"].sum() / 750_000

def sh(r):
    return float(r.mean() / r.std() * np.sqrt(12))

for lab, sl in [("2016-09..2026-06", pnl.loc["2016-09":"2026-06"]),
                ("2016-09..2026-07 (incl partial Jul)", pnl.loc["2016-09":"2026-07"]),
                ("2016-08..2026-06", pnl.loc["2016-08":"2026-06"]),
                ("2016-07..2026-06", pnl.loc["2016-07":"2026-06"])]:
    print(f"book {lab}: N={len(sl)} avg {sl.mean()*100:+.2f}%/mo Sharpe {sh(sl):.2f}")

# BIL episode t variants using researcher's own episode sums
theirs = np.array([-2.4, 6.5, 21.8, -3.9, 0.3, -3.4, 5.8, 5.1])
for lab, arr in [("their 8 sums", theirs),
                 ("their sums ddof=0", None)]:
    pass
t = theirs.mean() / (theirs.std(ddof=1) / np.sqrt(8))
print(f"their sums ddof=1: t={t:.2f} p={2*stats.t.sf(abs(t),7):.3f}")
t0 = theirs.mean() / (theirs.std(ddof=0) / np.sqrt(8))
print(f"their sums ddof=0: t={t0:.2f} p={2*stats.t.sf(abs(t0),7):.3f}")
# weighted by episode length? episode mean monthly active, weights n
n = np.array([1, 2, 3, 5, 2, 4, 1, 1])
epmean = theirs / n
t2 = epmean.mean() / (epmean.std(ddof=1) / np.sqrt(8))
print(f"episode MEAN-monthly ddof=1: t={t2:.2f} p={2*stats.t.sf(abs(t2),7):.3f}")
# log-active or normal z? try normal p for t=1.79 df large
print(f"norm p for 1.79: {2*stats.norm.sf(1.79):.3f}")
