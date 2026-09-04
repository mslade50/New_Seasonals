"""Check: long AAPL / short QQQ 5d snapback after extreme divergence.

Candidate: AAPL 5d rank 1.2 while QQQ 5d rank 100 as of 2026-08-05.
Cells:
  A. AAPL 5d rank < 10 AND QQQ 5d rank > 85 -> fwd 5d spread (AAPL-QQQ)
  B. spread form: (AAPL 5d ret - QQQ 5d ret) at its 252d 2nd pctile ->
     fwd 5d spread
  C. control: unconditional fwd 5d spread drift
Also prints AAPL's last earnings date (post-print drift risk).
Declustered 5 td. Era split. No bar after 2026-08-05 is used.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[5]
CUTOFF = pd.Timestamp("2026-08-05")

mp = pd.read_parquet(ROOT / "data" / "master_prices.parquet",
                     columns=["ticker", "date", "Close"])


def load(tkr: str) -> pd.Series:
    df = mp[mp["ticker"] == tkr].set_index("date").sort_index()[["Close"]]
    df.index = pd.to_datetime(df.index).normalize()
    df = df[~df.index.duplicated(keep="last")]
    return df.loc[:CUTOFF, "Close"]


df = pd.DataFrame({"aapl": load("AAPL"), "qqq": load("QQQ")}).dropna()
r5 = df.pct_change(5)


def rank252(s: pd.Series) -> pd.Series:
    return s.rolling(252).apply(lambda w: (w <= w.iloc[-1]).mean() * 100, raw=False)


rk_a, rk_q = rank252(r5["aapl"]), rank252(r5["qqq"])
sprd5 = r5["aapl"] - r5["qqq"]
rk_sprd = rank252(sprd5)

fwd5 = df.shift(-5) / df - 1
fwd_spread = fwd5["aapl"] - fwd5["qqq"]


def decluster(mask: pd.Series, gap: int = 5) -> pd.DatetimeIndex:
    out, last = [], None
    idx = mask.index
    for i, v in enumerate(mask.values):
        if v and (last is None or i - last > gap):
            out.append(idx[i])
            last = i
    return pd.DatetimeIndex(out)


def stats(x: pd.Series, label: str) -> None:
    x = x.dropna()
    if len(x) < 3:
        print(f"{label:52s} N={len(x)}")
        return
    t = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
    print(f"{label:52s} {x.mean()*1e4:+7.1f} bps  t {t:+5.2f}  N {len(x):3d}"
          f"  hit {(x > 0).mean():.2f}  worst {x.min()*1e4:+.0f}")


mA = (rk_a < 10) & (rk_q > 85)
mB = rk_sprd < 2
for name, m in [("A rank divergence", mA), ("B spread 2nd pctile", mB)]:
    days = decluster(m.fillna(False))
    print(f"=== cell {name}: {len(days)} declustered ===")
    stats(fwd_spread.reindex(days), "  fwd 5d spread (AAPL-QQQ)")
    stats(fwd5["aapl"].reindex(days), "  fwd 5d AAPL outright")
    d18 = days[days >= "2018-01-01"]
    stats(fwd_spread.reindex(d18), "  spread 2018+")
    stats(fwd_spread.reindex(days[days < "2018-01-01"]), "  spread pre-2018")
    print("  last 8 episodes:", " ".join(d.strftime("%Y-%m-%d") for d in days[-8:]))

stats(fwd_spread.dropna(), "unconditional fwd 5d spread")
print("current: rk_aapl=%.1f rk_qqq=%.1f rk_spread=%.1f"
      % (rk_a.iloc[-1], rk_q.iloc[-1], rk_sprd.iloc[-1]))

try:
    earn = pd.read_parquet(ROOT / "data" / "earnings_calendar.parquet")
    ea = earn[earn["ticker"] == "AAPL"]
    dcol = [c for c in ea.columns if "date" in c.lower()][0]
    past = pd.to_datetime(ea[dcol])
    past = past[past <= CUTOFF]
    print("AAPL most recent earnings:", past.max())
except Exception as e:  # noqa: BLE001
    print("earnings lookup failed:", e)
