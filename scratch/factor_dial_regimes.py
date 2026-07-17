"""Daily Fama-French factor returns conditioned on the dial regime.

Regime known at close t-1 (dial 10d-MA + SPY proximity to 252d high);
factor return measured day t. Factors are academic LONG-SHORT daily series
(percent/day): Mom (winners-losers; negative = laggards outperform),
HML (value-growth), SMB (small-big), RMW (profitable-weak), CMA, ST_Rev
(short-term reversal), MktRF. Window = the dial series (2016-07+), one
decade — treat as conditional description, not proven edges (the factor-
seasonality memory's post-2013 discipline applies).
"""
import io
import os
import re
import zipfile

import numpy as np
import pandas as pd
import requests

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
DAILY_SOURCES = [
    ("F-F_Research_Data_Factors_daily_CSV.zip", ["Mkt-RF", "SMB", "HML"]),
    ("F-F_Momentum_Factor_daily_CSV.zip", ["Mom"]),
    ("F-F_ST_Reversal_Factor_daily_CSV.zip", ["ST_Rev"]),
    ("F-F_Research_Data_5_Factors_2x3_daily_CSV.zip", ["RMW", "CMA"]),
]
CACHE = os.path.join(_ROOT, "scratch", "ff_daily_factors.parquet")


def fetch_daily() -> pd.DataFrame:
    if os.path.exists(CACHE):
        return pd.read_parquet(CACHE)
    merged = None
    for zip_name, keep in DAILY_SOURCES:
        r = requests.get(BASE + zip_name, timeout=60)
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            text = z.read(z.namelist()[0]).decode("latin-1")
        rows = []
        for line in text.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if parts and re.fullmatch(r"\d{8}", parts[0]):
                rows.append(parts)
        header = None
        for line in text.splitlines():
            if "," in line and not re.match(r"\s*\d{8},", line):
                cand = [c.strip() for c in line.split(",")]
                if any(c in ("Mkt-RF", "Mom", "ST_Rev", "RMW") for c in cand):
                    header = [c for c in cand if c]     # factor names only
                    break
        assert header, f"no header found in {zip_name}"
        width = len(rows[0])
        cols = ["Date"] + header
        cols = cols[:width] + [f"_x{i}" for i in range(width - len(cols))]
        df = pd.DataFrame(rows, columns=cols)
        df.index = pd.to_datetime(df.pop("Date"), format="%Y%m%d")
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.replace([-99.99, -999], np.nan)
        cols = [c for c in keep if c in df.columns]
        assert cols, f"{zip_name}: wanted {keep}, got {list(df.columns)}"
        part = df[cols]
        merged = part if merged is None else merged.join(part, how="outer")
    merged = merged.rename(columns={"Mkt-RF": "MktRF"})
    merged.to_parquet(CACHE)
    return merged


ff = fetch_daily()
print(f"factors: {list(ff.columns)}, {ff.index.min().date()} -> {ff.index.max().date()}")

frag = pd.read_parquet(os.path.join(_ROOT, "data", "rd2_fragility.parquet"))
s63 = frag["63d"].dropna().sort_index()
ma10 = s63.rolling(10, min_periods=1).mean()
mp = pd.read_parquet(os.path.join(_ROOT, "data", "master_prices.parquet"),
                     filters=[("ticker", "==", "SPY")])
spy = (mp.assign(date=pd.to_datetime(mp["date"]))
       .set_index("date")["Close"].sort_index().reindex(s63.index).ffill())
near5 = (spy / spy.rolling(252, min_periods=60).max() - 1) >= -0.05

state = pd.DataFrame({"dial": ma10, "near5": near5}).dropna()
regime = pd.Series("off_highs", index=state.index)
regime[state["near5"] & (state["dial"] < 20)] = "clean_air(<20)"
regime[state["near5"] & (state["dial"] >= 20) & (state["dial"] < 50)] = "mid(20-50)"
regime[state["near5"] & (state["dial"] >= 50)] = "elevated(>=50)"

# regime at t-1 conditions factor return at t
joined = ff.join(regime.rename("regime").shift(1), how="inner").dropna(subset=["regime"])
print(f"joined: {joined.index.min().date()} -> {joined.index.max().date()}, "
      f"{len(joined)} days\n")

FACTORS = ["MktRF", "Mom", "HML", "SMB", "RMW", "CMA", "ST_Rev"]
ORDER = ["clean_air(<20)", "mid(20-50)", "elevated(>=50)", "off_highs"]

rows = []
for reg in ORDER:
    g = joined[joined["regime"] == reg]
    row = {"regime": reg, "days": len(g)}
    for f in FACTORS:
        v = g[f].dropna()
        ann = v.mean() * 252
        t = v.mean() / (v.std() / np.sqrt(len(v))) if len(v) > 10 else np.nan
        row[f] = f"{ann:+.1f} ({t:+.1f})"
    rows.append(row)
pd.set_option("display.width", 200)
print("annualized % (daily t-stat), factor return conditioned on prior-close regime:")
print(pd.DataFrame(rows).set_index("regime").to_string())

uncond = {f: f"{joined[f].mean() * 252:+.1f}" for f in FACTORS}
print(f"\nunconditional (2016+): {uncond}")
