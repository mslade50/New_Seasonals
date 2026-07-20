"""Cost of a 50%-of-NAV notional cap per position, measured on the ledger
(flat $750k basis, GRM-scaled sizes as booked).

Two readings of "per position":
  A. per LEG: each trade's own notional clipped to 50% NAV
  B. per TICKER: concurrent open notional in one ticker (all strategies)
     capped at 50% NAV — legs admitted in entry order, later legs scaled
     down/skipped so the running total stays under the cap.
Lost PnL is proportional to the clipped fraction (linear sizing).
"""
from __future__ import annotations

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAV = 750_000.0
CAP = 0.50 * NAV

df = pd.read_parquet(ROOT / "data" / "backtest_trades_full.parquet")
for c in ["Entry Date", "Exit Date", "Signal Date"]:
    df[c] = pd.to_datetime(df[c])
df["notional"] = df["Shares_flat"].abs() * df["Entry Price"]
df["frac"] = df["notional"] / NAV

print("--- per-leg notional as % of NAV, by strategy ---")
g = df.groupby("Strategy")["frac"]
tab = pd.DataFrame({"n": g.size(), "p50": g.median(), "p95": g.quantile(.95),
                    "max": g.max(), ">50%": g.apply(lambda s: (s > .5).sum())})
print((tab.sort_values("max", ascending=False)
       .style.format if False else tab.sort_values("max", ascending=False)
       .round(3).to_string()))

# A: per-leg cap
df["clipA"] = (CAP / df["notional"]).clip(upper=1.0)
df["lostA"] = df["PnL_flat_750k"] * (1 - df["clipA"])
a_bind = df[df["clipA"] < 1]
print(f"\nA. per-LEG 50% cap: binds on {len(a_bind)} of {len(df)} trades")
print(f"   PnL foregone: ${df['lostA'].sum():,.0f} "
      f"(book total flat PnL ${df['PnL_flat_750k'].sum():,.0f})")
print(a_bind.groupby("Strategy")["lostA"].agg(["count", "sum"]).round(0).to_string())

# B: per-ticker concurrent cap, all strategies pooled
df2 = df.sort_values(["Entry Date", "Signal Date"]).reset_index(drop=True)
lostB = 0.0
binds = []
for tkr, g in df2.groupby("Ticker"):
    g = g.sort_values(["Entry Date", "Signal Date"])
    open_legs = []  # (exit_date, notional_admitted)
    for i, r in g.iterrows():
        open_legs = [(x, n) for x, n in open_legs if x > r["Entry Date"]]
        used = sum(n for _, n in open_legs)
        room = max(0.0, CAP - used)
        clip = min(1.0, room / r["notional"]) if r["notional"] > 0 else 1.0
        if clip < 1.0:
            binds.append({"Ticker": tkr, "Strategy": r["Strategy"],
                          "Entry Date": r["Entry Date"], "clip": clip,
                          "lost": r["PnL_flat_750k"] * (1 - clip),
                          "PnL": r["PnL_flat_750k"]})
            lostB += r["PnL_flat_750k"] * (1 - clip)
        open_legs.append((r["Exit Date"], r["notional"] * clip))

b = pd.DataFrame(binds)
print(f"\nB. per-TICKER concurrent 50% cap: binds on {len(b)} legs")
print(f"   PnL foregone: ${lostB:,.0f}")
if len(b):
    print(b.groupby("Strategy").agg(n=("lost", "count"), lost=("lost", "sum")).round(0).to_string())
    print("\n   ten largest single-leg impacts:")
    print(b.reindex(b["lost"].abs().sort_values(ascending=False).index)
           .head(10)[["Ticker", "Strategy", "Entry Date", "clip", "PnL", "lost"]]
           .round(2).to_string(index=False))
    yr = b.groupby(b["Entry Date"].dt.year)["lost"].sum().round(0)
    print("\n   foregone PnL by year:")
    print(yr.to_string())

# where do the big per-ticker stacks live? max concurrent notional per ticker (uncapped)
rows = []
for tkr, g in df2.groupby("Ticker"):
    ev = []
    for _, r in g.iterrows():
        ev.append((r["Entry Date"], r["notional"]))
        ev.append((r["Exit Date"], -r["notional"]))
    ev.sort()
    run = peak = 0.0
    for _, n in ev:
        run += n
        peak = max(peak, run)
    rows.append((tkr, peak / NAV))
pk = pd.DataFrame(rows, columns=["Ticker", "peak_frac"]).sort_values("peak_frac", ascending=False)
print(f"\ntickers whose peak concurrent notional ever exceeded 50% NAV: {(pk['peak_frac']>.5).sum()}")
print(pk.head(12).round(2).to_string(index=False))
