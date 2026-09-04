"""3x ETF Overbot Fade: performance when the ENTRY day is a Monday that
gaps DOWN (entry-day open < prior close by > 0.1 ATR). Ledger + price cache;
ATR = the ledger's frozen entry ATR."""
from __future__ import annotations

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
led = pd.read_parquet(ROOT / "data" / "backtest_trades_full.parquet")

for strat in ["3x ETF Overbot Fade", "3x Bear ETF Overbot Fade"]:
    d = led[led["Strategy"] == strat].copy()
    if d.empty:
        continue
    d["Entry Date"] = pd.to_datetime(d["Entry Date"])
    tickers = sorted(d["Ticker"].unique())
    px = pd.read_parquet(ROOT / "data" / "master_prices.parquet",
                         filters=[("ticker", "in", tickers)])
    px["date"] = pd.to_datetime(px["date"]).dt.normalize()
    frames = {t: g.sort_values("date").drop_duplicates("date").set_index("date")
              for t, g in px.groupby("ticker")}

    rows = []
    for _, r in d.iterrows():
        f = frames.get(r["Ticker"])
        if f is None or r["Entry Date"] not in f.index:
            continue
        i = f.index.get_loc(r["Entry Date"])
        if i == 0 or pd.isna(r["ATR"]) or r["ATR"] <= 0:
            continue
        gap_atr = (f.iloc[i]["Open"] - f.iloc[i - 1]["Close"]) / r["ATR"]
        rows.append({"R": r["R_Multiple"], "wd": r["Entry Date"].weekday(),
                     "gap_atr": gap_atr, "ticker": r["Ticker"],
                     "date": r["Entry Date"].date()})
    t = pd.DataFrame(rows)

    def line(lbl, seg):
        if not len(seg):
            print(f"  {lbl:<34} n=  0")
            return
        w = seg[seg.R > 0].R.sum()
        l = -seg[seg.R < 0].R.sum()
        pf = w / l if l > 0 else float("inf")
        print(f"  {lbl:<34} n={len(seg):>3}  avgR {seg.R.mean():>6.2f}  "
              f"totR {seg.R.sum():>7.1f}  win {(seg.R>0).mean():>4.0%}  PF {pf:>5.2f}")

    print(f"\n=== {strat} ({len(t)} trades matched) ===")
    line("ALL", t)
    line("Monday entries", t[t.wd == 0])
    line("Monday + gap down > 0.1 ATR", t[(t.wd == 0) & (t.gap_atr < -0.1)])
    line("Monday, no gap down", t[(t.wd == 0) & (t.gap_atr >= -0.1)])
    line("Any day + gap down > 0.1 ATR", t[t.gap_atr < -0.1])
    line("Non-Monday", t[t.wd != 0])
    sub = t[(t.wd == 0) & (t.gap_atr < -0.1)]
    if len(sub):
        print("  trades:", ", ".join(f"{r.ticker} {r.date} {r.R:+.2f}R"
                                     for r in sub.itertuples()))
