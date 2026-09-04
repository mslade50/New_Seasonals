"""Three quick kill-checks in one pass (no bar after 2026-08-05):

1. USO crash bounce: 5d <= -10% -> fwd 5d, split by 252d return sign
   (current: 5d -11.2%, 252d +50.9%).
2. GDX one-day spike: 1d >= +6% -> fwd 5d, split by above/below 200d SMA
   (current: +7.4% 1d, 3.9% below 200d, 252d +53%).
3. FXI momentum: 21d rank > 95 -> fwd 10d (current rank 97.2).
Declustered. Era splits where N allows.
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


def decluster(mask: pd.Series, gap: int) -> pd.DatetimeIndex:
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
        print(f"{label:56s} N={len(x)}")
        return
    t = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
    print(f"{label:56s} {x.mean()*1e4:+7.1f} bps  t {t:+5.2f}  N {len(x):3d}"
          f"  hit {(x > 0).mean():.2f}  worst {x.min()*1e4:+.0f}")


# --- 1. USO ---
uso = load("USO")
r5, r252 = uso.pct_change(5), uso.pct_change(252)
fwd5 = uso.shift(-5) / uso - 1
m = r5 <= -0.10
days = decluster(m.fillna(False), 5)
print(f"=== USO 5d <= -10%: {len(days)} declustered ===")
stats(fwd5.reindex(days), "  fwd 5d, all")
up = days[r252.reindex(days) > 0]
dn = days[r252.reindex(days) <= 0]
stats(fwd5.reindex(up), "  fwd 5d | 252d ret > 0 (today's shape)")
stats(fwd5.reindex(dn), "  fwd 5d | 252d ret <= 0")
stats(fwd5.reindex(up[up >= "2018-01-01"]), "  up-regime 2018+")
print("  up-regime episodes:", " ".join(d.strftime("%Y-%m-%d") for d in up))

# --- 2. GDX ---
gdx = load("GDX")
r1 = gdx.pct_change()
sma200 = gdx.rolling(200).mean()
fwd5g = gdx.shift(-5) / gdx - 1
m = r1 >= 0.06
days = decluster(m.fillna(False), 5)
print(f"=== GDX 1d >= +6%: {len(days)} declustered ===")
stats(fwd5g.reindex(days), "  fwd 5d, all")
above = days[(gdx > sma200).reindex(days).fillna(False)]
below = days[(gdx <= sma200).reindex(days).fillna(False)]
stats(fwd5g.reindex(above), "  fwd 5d | above 200d")
stats(fwd5g.reindex(below), "  fwd 5d | below 200d (today)")
r252g = gdx.pct_change(252)
belowup = below[r252g.reindex(below) > 0]
stats(fwd5g.reindex(belowup), "  fwd 5d | below 200d AND 252d>0 (exact)")
stats(fwd5g.reindex(days[days >= "2018-01-01"]), "  all spikes 2018+")

# --- 3. FXI ---
fxi = load("FXI")
r21 = fxi.pct_change(21)
rank = r21.rolling(252).apply(lambda w: (w <= w.iloc[-1]).mean() * 100, raw=False)
fwd10 = fxi.shift(-10) / fxi - 1
m = rank > 95
days = decluster(m.fillna(False), 10)
print(f"=== FXI 21d rank > 95: {len(days)} declustered ===")
stats(fwd10.reindex(days), "  fwd 10d, all")
stats(fwd10.reindex(days[days >= "2018-01-01"]), "  2018+")
stats(fwd10.dropna(), "  FXI unconditional fwd 10d")
