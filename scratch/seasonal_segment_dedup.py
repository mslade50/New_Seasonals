"""Re-cut the seasonal backtest: dedup overlapping re-emissions (one open position
per ticker+direction — 'if it's already on, it's done'), then segment by the
macro-short / stock-long rule. Also report the target-horizon mix and realized
holding period. Operates on the existing weekly backtest parquet (no re-run)."""
import os
import sys

import numpy as np
import pandas as pd

ROOT = r"C:\Users\McKinley Slade\dev\New_Seasonals"
sys.path.insert(0, ROOT)
df = pd.read_parquet(os.path.join(ROOT, "data", "seasonal_ideas_backtest.parquet"))
df["entry_date"] = pd.to_datetime(df["entry_date"])
df["exit_date"] = pd.to_datetime(df["exit_date"])
# stocks = equity seasonal channel; macro = cross-asset channel
df["asset"] = np.where(df["channel"] == "detect_seasonal", "stock", "macro")

pd.set_option("display.width", 240)


def dedup(d):
    """Greedy non-overlapping per (ticker, direction): skip any flag whose entry
    falls on/before the last kept trade's exit for that name+direction."""
    d = d.sort_values(["ticker", "direction", "entry_date"])
    keep, last_exit = [], {}
    for r in d.itertuples():
        k = (r.ticker, r.direction)
        le = last_exit.get(k)
        if le is None or r.entry_date > le:
            keep.append(r.Index)
            last_exit[k] = r.exit_date
    return d.loc[keep]


def agg(g):
    R = g["R"].astype(float); w, l = R[R > 0], R[R < 0]
    pf = w.sum() / abs(l.sum()) if l.sum() else np.inf
    d = g.sort_values("exit_date"); eq = d["R"].cumsum()
    dd = float((eq - eq.cummax()).min())
    return {"N": len(g), "Win%": round(100 * (R > 0).mean(), 1), "AvgR": round(R.mean(), 3),
            "MedR": round(R.median(), 3), "TotR": round(R.sum(), 1),
            "AvgR/Std": round(R.mean() / R.std(), 3) if R.std() else np.nan,
            "PF": round(pf, 2) if np.isfinite(pf) else np.inf, "maxDD_R": round(dd, 1)}


print(f"raw weekly trades: {len(df)}")
dd_all = dedup(df)
print(f"after dedup (one open per ticker+direction): {len(dd_all)}  "
      f"({100*len(dd_all)/len(df):.0f}% kept) -> this removes the 'same trade 3x' stacking\n")

# --- 2x2 asset x direction, DEDUPED ---
print("=== 2x2 asset x direction (deduped) ===")
rows = {f"{a}/{dirn}": agg(g) for (a, dirn), g in dd_all.groupby(["asset", "direction"])}
print(pd.DataFrame(rows).T.to_string())

# --- book variants (all deduped) ---
V1 = dd_all[~((dd_all.asset == "stock") & (dd_all.direction == "short"))]          # drop stock shorts
V2 = dd_all[((dd_all.asset == "stock") & (dd_all.direction == "long")) |           # stock-long + macro-short
            ((dd_all.asset == "macro") & (dd_all.direction == "short"))]
books = {"raw (all, deduped)": dd_all,
         "V1: drop stock shorts (keep macro long+short + stock long)": V1,
         "V2: stock LONG-only + macro SHORT-only": V2}
print("\n=== book variants (deduped) ===")
print(pd.DataFrame({k: agg(g) for k, g in books.items()}).T.to_string())

# --- chosen book (V1) by cycle ---
print("\n=== V1 by cycle year (deduped) ===")
print(pd.DataFrame({int(c): agg(g) for c, g in V1.groupby("cycle")}).T.to_string())

# --- target horizon mix + realized hold ---
print("\n=== target horizon (time-stop) mix, deduped all ===")
hz = dd_all.groupby("time_stop_days").agg(
    N=("R", "size"), share_pct=("R", lambda x: round(100 * len(x) / len(dd_all), 1)),
    realized_hold_bars=("bars_held", "mean"), avgR=("R", "mean"),
    pct_target=("exit_type", lambda x: round(100 * (x == "Target").mean(), 1)),
    pct_stop=("exit_type", lambda x: round(100 * (x == "Stop").mean(), 1)),
    pct_time=("exit_type", lambda x: round(100 * (x == "Time").mean(), 1)))
print(hz.round(2).to_string())
print(f"\noverall realized hold: mean {dd_all.bars_held.mean():.1f} bars, "
      f"median {dd_all.bars_held.median():.0f}; exit mix "
      f"Tgt {100*(dd_all.exit_type=='Target').mean():.0f}% / "
      f"Stop {100*(dd_all.exit_type=='Stop').mean():.0f}% / "
      f"Time {100*(dd_all.exit_type=='Time').mean():.0f}%")

# --- equity curve for V1 ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 8))
for label, g, col in [("V1 segmented (deduped)", V1, "#1a7f37"),
                      ("raw all (deduped)", dd_all, "#888"),
                      ("V2 strict", V2, "#c0392b")]:
    d = g.sort_values("exit_date"); eq = d["R"].cumsum()
    a1.plot(d["exit_date"].values, eq.values, label=label, lw=1.6, color=col)
a1.set_title("Seasonal ideas — segmented + deduped, cumulative R"); a1.legend(); a1.grid(alpha=.3); a1.set_ylabel("cum R")
d = V1.sort_values("exit_date"); eq = d["R"].cumsum()
a2.plot(d["exit_date"].values, (eq - eq.cummax()).values, color="#c0392b", lw=1)
a2.set_title("V1 underwater (R below peak)"); a2.grid(alpha=.3); a2.set_ylabel("R")
fig.tight_layout()
out = os.path.join(ROOT, "scratch", "seasonal_segment_dedup.png")
fig.savefig(out, dpi=110)
print(f"\nSaved -> {out}")
