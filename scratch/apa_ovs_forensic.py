"""Why didn't APA fire OVS on 2026-03-26/27? Replays every OVS filter
condition per day for mid-late March 2026 (indicators.py conventions)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

px = pd.read_parquet(ROOT / "data" / "master_prices.parquet")
px["date"] = pd.to_datetime(px["date"])
df = px[px["ticker"] == "APA"].set_index("date").sort_index()

for w in [2, 5, 10, 21, 252]:
    ret = df["Close"].pct_change(w, fill_method=None)
    df[f"rank_{w}"] = ret.expanding(min_periods=252).rank(pct=True) * 100.0

hl = df["High"] - df["Low"]
hc = (df["High"] - df["Close"].shift()).abs()
lc = (df["Low"] - df["Close"].shift()).abs()
df["ATR"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
df["ret_atr"] = (df["Close"] - df["Close"].shift(1)) / df["ATR"]
df["atr_pct"] = df["ATR"] / df["Close"] * 100
df["vol_ok"] = df["Volume"] >= 100000
df["consec21"] = (df["rank_21"] > 85).rolling(3).sum() == 3

sz = pd.read_parquet(ROOT / "data" / "atr_seasonal_ranks.parquet")
sz = sz[sz["ticker"] == "APA"].set_index(pd.to_datetime(sz.loc[sz["ticker"] == "APA", "Date"]))

earn = pd.read_parquet(ROOT / "data" / "earnings_calendar.parquet")
apa_earn = pd.to_datetime(earn[earn["symbol"] == "APA"]["date"]) if "symbol" in earn.columns else pd.Series(dtype="datetime64[ns]")
if apa_earn.empty and "ticker" in earn.columns:
    apa_earn = pd.to_datetime(earn[earn["ticker"] == "APA"]["date"])

win = df.loc["2026-03-16":"2026-03-31"]
print(f"{'date':10s} {'close':>7} {'r2':>5} {'r5':>5} {'r10':>5} {'r21':>5} {'c21x3':>5} "
      f"{'r252':>5} {'barbell':>7} {'sznl5':>5} {'ret/ATR':>7} {'verdict'}")
for d, row in win.iterrows():
    s5 = sz["atr_sznl_5d"].get(d, np.nan)
    nearest_earn = (apa_earn - d).abs().min() if len(apa_earn) else pd.NaT
    conds = {
        "r2>85": row["rank_2"] > 85, "r5>85": row["rank_5"] > 85,
        "r10>85": row["rank_10"] > 85, "r21x3": bool(row["consec21"]),
        "barbell": not (65 <= row["rank_252"] <= 95),
        "sznl5<85": (np.isnan(s5) or s5 < 85),
        "up>.25atr": row["ret_atr"] > 0.25,
    }
    fails = [k for k, v in conds.items() if not v]
    print(f"{d.date()} {row['Close']:>7.2f} {row['rank_2']:>5.1f} {row['rank_5']:>5.1f} "
          f"{row['rank_10']:>5.1f} {row['rank_21']:>5.1f} {str(bool(row['consec21'])):>5s} "
          f"{row['rank_252']:>5.1f} {str(conds['barbell']):>7s} {s5:>5.1f} {row['ret_atr']:>7.2f} "
          f"{'SIGNAL' if not fails else 'no: ' + ','.join(fails)}")

print(f"\nAPA earnings dates near window: "
      f"{sorted(x.date() for x in apa_earn if abs((x - pd.Timestamp('2026-03-26')).days) < 45)}")
