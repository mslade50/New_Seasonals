"""Per-strategy and portfolio R stats across deciles of the risk-dial score.

Deciles are computed on the DAILY score distribution (10d MA of each dial,
2016-07..today), so each band covers ~10% of trading days; trades then land in
whichever band their signal-date score falls in. The daily series is exactly 0
on >10% of days, so the bottom deciles merge into one 0-band (include_lowest
guards it — v1 of this script silently dropped the 136 score-zero trades).
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
trades = pd.read_parquet(ROOT / "data" / "backtest_trades_full.parquet")
frag = pd.read_parquet(ROOT / "data" / "rd2_fragility.parquet")
frag.index = pd.to_datetime(frag.index).normalize()

trades["Signal Date"] = pd.to_datetime(trades["Signal Date"]).dt.normalize()
trades = trades.dropna(subset=["R_Multiple"]).copy()


def joined(col: str) -> tuple[pd.DataFrame, pd.Series]:
    s = frag[col].dropna().rolling(10, min_periods=1).mean()
    grid = pd.date_range(s.index.min(), s.index.max(), freq="D")
    sf = s.reindex(grid).ffill(limit=5)
    d = trades.copy()
    d["score"] = sf.reindex(d["Signal Date"]).values
    return d.dropna(subset=["score"]), s


def decile_bands(daily: pd.Series):
    qs = daily.quantile(np.arange(0, 1.01, 0.10)).values
    qs[-1] = max(qs[-1] + 1, 200)
    # dedupe with tolerance (bottom deciles are all ~0 -> one merged band)
    edges = [-0.001]
    for q in qs[1:]:
        if q > edges[-1] + 0.5:
            edges.append(float(q))
    labels = [f"{max(edges[i], 0):.0f}-{edges[i+1]:.0f}" for i in range(len(edges) - 2)]
    labels.append(f"{edges[-2]:.0f}+")
    return edges, labels


def cut(d, edges, labels):
    return pd.cut(d.score, bins=edges, labels=labels, include_lowest=True)


def table(d: pd.DataFrame, edges, labels) -> pd.DataFrame:
    d = d.copy()
    d["band"] = cut(d, edges, labels)
    assert d["band"].notna().all(), "trades fell outside decile bins"
    g = d.groupby("band", observed=False)["R_Multiple"]
    return pd.DataFrame({
        "N": g.size(),
        "avgR": g.mean().round(3),
        "medR": g.median().round(3),
        "win%": g.apply(lambda s: (s > 0).mean() * 100).round(1),
        "totR": g.sum().round(1),
    })


for col in ["63d", "21d", "5d"]:
    d, daily = joined(col)
    edges, labels = decile_bands(daily)
    pct_zero = (daily == 0).mean() * 100
    is_ovs = d["Strategy"].str.contains("Overbot Vol|OVS", case=False, na=False)
    print("=" * 78)
    print(f"DIAL {col} (10d MA) — {pct_zero:.0f}% of days at exactly 0; "
          f"decile edges: " + ", ".join(f"{e:.1f}" for e in edges[1:-1]))
    print(f"\n--- PORTFOLIO ex-OVS ({(~is_ovs).sum()} trades) ---")
    print(table(d[~is_ovs], edges, labels).to_string())
    if col == "63d":
        print(f"\n--- OVS only, exempt live ({is_ovs.sum()} trades) ---")
        print(table(d[is_ovs], edges, labels).to_string())

print("=" * 78)
print("PER-STRATEGY avgR (N) by 63d-MA10 decile band")
d, daily = joined("63d")
edges, labels = decile_bands(daily)
d["band"] = cut(d, edges, labels)
rows = []
order = d.groupby("Strategy").size().sort_values(ascending=False).index
for strat in order:
    g = d[d.Strategy == strat]
    row = {"Strategy": strat[:26], "Ntot": len(g), "allR": round(g.R_Multiple.mean(), 2)}
    for b in labels:
        gb = g[g.band == b]
        row[b] = f"{gb.R_Multiple.mean():+.2f} ({len(gb)})" if len(gb) >= 3 else (
            f". ({len(gb)})" if len(gb) else "")
    rows.append(row)
print(pd.DataFrame(rows).to_string(index=False))
print("\n(cells with N<3 shown as '.'; OVS is exempt from the live multiplier)")
