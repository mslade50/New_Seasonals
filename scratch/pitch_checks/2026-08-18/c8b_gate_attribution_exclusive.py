"""C8b -- clean gate attribution + C7 outlier audit.

C8: the "deep ONLY" cell in c8 still CONTAINED the 52w-low days. Redo the
attribution EXCLUSIVELY (deep & NOT at a new 52w low) so the fresh-low gate
is priced against its true complement.

C7: identify the extreme forward returns so a kill is not resting on a data
artifact, and re-price the short on the MEDIAN and on a winsorized mean.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import strategy_config as sc  # noqa: E402
from pitch_lab import PRICES_PATH, declusters, show, summarize  # noqa: E402

BAD, GAP = {"SOXS"}, 21
mp = pd.read_parquet(PRICES_PATH, columns=["date", "ticker", "Close"])
uni = sorted(set(sc.LIQUID_PLUS_COMMODITIES) - BAD)
mp = mp[mp["ticker"].isin(uni)]
mp["date"] = pd.to_datetime(mp["date"])
px = mp.pivot_table(index="date", columns="ticker", values="Close",
                    aggfunc="last").sort_index()
px = px.loc[:, px.notna().sum() >= 1000]

sma200 = px.rolling(200).mean()
d200 = px / sma200 - 1.0
at_low = px <= px.rolling(252).min() * 1.0000001
FWD = {h: px.shift(-(1 + h)) / px.shift(-1) - 1.0 for h in range(1, 11)}


def cells(mask: pd.DataFrame, h: int) -> pd.DataFrame:
    recs, f = [], FWD[h]
    for t in px.columns:
        mt = mask[t].fillna(False) & f[t].notna()
        d = px.index[mt.values]
        if len(d) == 0:
            continue
        for dt in declusters(d, GAP, px.index):
            recs.append((t, dt, f.at[dt, t]))
    return pd.DataFrame(recs, columns=["tkr", "date", "ret"])


print("### C8b: EXCLUSIVE gate attribution -- does the fresh 52w low add "
      "anything to 'deep below the 200d'? ###")
deep = d200 <= -0.20
for h in (3, 5, 10):
    both = cells(deep & at_low, h)
    only = cells(deep & ~at_low, h)
    show([summarize(both["ret"].values, f"h={h} deep AND new 52w low"),
          summarize(only["ret"].values, f"h={h} deep, NOT at a new low")],
         f"h={h}")
    diff = 100 * (both["ret"].mean() - only["ret"].mean())
    se = np.sqrt(both["ret"].var(ddof=1) / len(both)
                 + only["ret"].var(ddof=1) / len(only))
    print(f"  fresh-low GATE contribution = {diff:+.3f}pp  welch t = "
          f"{(both['ret'].mean()-only['ret'].mean())/se:+.2f}")

print("\n  same, 2018+ only:")
for h in (3, 5, 10):
    both = cells(deep & at_low, h)
    only = cells(deep & ~at_low, h)
    b = both[both["date"] >= "2018-01-01"]["ret"].values
    o = only[only["date"] >= "2018-01-01"]["ret"].values
    show([summarize(b, f"h={h} deep AND new low, 2018+"),
          summarize(o, f"h={h} deep NOT at low, 2018+")])

print("\n\n### C7 outlier audit: extension >= 95th own-pctile, worst short "
      "outcomes ###")
extp = d200.expanding(756).rank(pct=True) * 100.0
m = (extp >= 95.0) & (d200 > 0)
for h in (5, 10):
    df = cells(m, h)
    df["pnl"] = -df["ret"]
    w = df.sort_values("pnl").head(6)
    print(f"\nh={h}: worst 6 SHORT outcomes")
    print(w.assign(pnl_pct=(100 * w["pnl"]).round(1))[
        ["tkr", "date", "pnl_pct"]].to_string(index=False))
    lo, hi = np.percentile(df["pnl"], [1, 99])
    wins = np.clip(df["pnl"].values, lo, hi)
    show([summarize(df["pnl"].values, f"h={h} SHORT raw (N={len(df)})"),
          summarize(wins, f"h={h} SHORT winsorized 1/99"),
          summarize(df.loc[df["date"] >= "2018-01-01", "pnl"].values,
                    f"h={h} SHORT 2018+")])
    print(f"  median SHORT pnl = {100*np.median(df['pnl']):+.3f}%")
