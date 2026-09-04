"""How an added trailing-252d-return-in-ATR gate reshapes the OLV return stream.

Filter: ret_atr_252d = (Close_t - Close_{t-252}) / ATR_14_t  (indicators.py:197)
Thresholds tested: > 0, > 3, > 5 ATR  (baseline = no filter).

The gate is applied at SIGNAL time and only removes signals, so subsetting the
real OLV fills in the rebuilt ledger by ret_atr_252d-at-signal is exact for
R-based shape (R_Multiple is sizing/cap-independent).
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\McKinley Slade\dev\New_Seasonals")
LEDGER = ROOT / "data" / "backtest_trades_full.parquet"
PRICES = ROOT / "data" / "master_prices.parquet"
W = 252

def atr14(g: pd.DataFrame) -> pd.Series:
    h, l, c = g["High"], g["Low"], g["Close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(14).mean()

def ret_atr_252_series(g: pd.DataFrame) -> pd.Series:
    g = g.sort_values("date")
    atr = atr14(g)
    return ((g["Close"] - g["Close"].shift(W)) / atr).set_axis(g["date"])

# --- OLV trades from the rebuilt ledger ---
led = pd.read_parquet(LEDGER)
olv = led[led["Strategy"] == "Oversold Low Volume"].copy()
olv["Signal Date"] = pd.to_datetime(olv["Signal Date"])

# --- ret_atr_252d at signal date, per ticker ---
mp = pd.read_parquet(PRICES)
mp = mp[mp["ticker"].isin(olv["Ticker"].unique())]
metric_by_ticker = {t: ret_atr_252_series(g) for t, g in mp.groupby("ticker")}

def lookup(row):
    s = metric_by_ticker.get(row["Ticker"])
    if s is None:
        return np.nan
    s = s[~s.index.duplicated()]
    hit = s.reindex([row["Signal Date"]]).iloc[0]
    if pd.isna(hit):  # signal date not an exact bar; take last bar <= signal
        prior = s[s.index <= row["Signal Date"]]
        hit = prior.iloc[-1] if len(prior) else np.nan
    return hit

olv["ret_atr_252"] = olv.apply(lookup, axis=1)
n_nan = olv["ret_atr_252"].isna().sum()

def stats(df: pd.DataFrame, base_n: int, base_totR: float) -> dict:
    R = df["R_Multiple"].astype(float)
    pnl = df["PnL_flat_750k"].astype(float)
    wins = R[R > 0]; losses = R[R < 0]
    pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else np.inf
    eq = R.sort_index()  # already chronological-ish; resort by signal date below
    return {
        "N": len(df),
        "%kept": 100 * len(df) / base_n,
        "Win%": 100 * (R > 0).mean(),
        "AvgR": R.mean(),
        "MedR": R.median(),
        "TotR": R.sum(),
        "%TotR_kept": 100 * R.sum() / base_totR if base_totR else np.nan,
        "StdR": R.std(),
        "AvgR/Std": R.mean() / R.std() if R.std() else np.nan,
        "PF": pf,
        "PnL_flat": pnl.sum(),
        "WorstR": R.min(),
        "P05": R.quantile(0.05),
        "P95": R.quantile(0.95),
        "Skew": R.skew(),
    }

def max_dd_in_R(df: pd.DataFrame) -> float:
    d = df.sort_values("Exit Date")
    eq = d["R_Multiple"].astype(float).cumsum()
    return float((eq - eq.cummax()).min())

def trades_per_year(df):
    return df.assign(yr=pd.to_datetime(df["Exit Date"]).dt.year).groupby("yr")["R_Multiple"].agg(["count", "sum"])

base = olv.dropna(subset=["ret_atr_252"])
base_n, base_totR = len(base), base["R_Multiple"].sum()

print(f"OLV trades in ledger: {len(olv)}  (NaN metric dropped: {n_nan}) -> usable {base_n}")
print(f"ret_atr_252 distribution over OLV signals: "
      f"min {base.ret_atr_252.min():.1f} | p25 {base.ret_atr_252.quantile(.25):.1f} | "
      f"med {base.ret_atr_252.median():.1f} | p75 {base.ret_atr_252.quantile(.75):.1f} | "
      f"max {base.ret_atr_252.max():.1f}")
print(f"  share with ret_atr_252 <= 0: {100*(base.ret_atr_252<=0).mean():.1f}%")
print()

rows = []
cuts = {"baseline (all)": base,
        ">0 ATR": base[base.ret_atr_252 > 0],
        ">3 ATR": base[base.ret_atr_252 > 3],
        ">5 ATR": base[base.ret_atr_252 > 5]}
for name, df in cuts.items():
    s = stats(df, base_n, base_totR)
    s["cut"] = name
    s["MaxDD_R"] = max_dd_in_R(df)
    rows.append(s)

res = pd.DataFrame(rows).set_index("cut")
cols = ["N", "%kept", "Win%", "AvgR", "MedR", "TotR", "%TotR_kept", "StdR",
        "AvgR/Std", "PF", "MaxDD_R", "WorstR", "P05", "P95", "Skew", "PnL_flat"]
pd.set_option("display.width", 220, "display.float_format", lambda x: f"{x:.2f}")
print("=== WHOLE OLV BOOK (liquid + overflow) ===")
print(res[cols].to_string())

# By tier
for tier in ["Liquid", "Overflow"]:
    bt = base[base["Tier"] == tier]
    btot = bt["R_Multiple"].sum(); bn = len(bt)
    rr = []
    for name, df0 in {"baseline": bt, ">0": bt[bt.ret_atr_252>0],
                      ">3": bt[bt.ret_atr_252>3], ">5": bt[bt.ret_atr_252>5]}.items():
        s = stats(df0, bn, btot); s["cut"] = name; s["MaxDD_R"] = max_dd_in_R(df0); rr.append(s)
    print(f"\n=== {tier} tier ===")
    print(pd.DataFrame(rr).set_index("cut")[["N","%kept","Win%","AvgR","TotR","%TotR_kept","AvgR/Std","PF","MaxDD_R"]].to_string())

# Year-by-year total R for baseline vs >3 (the likely sweet spot) to show curve shape
print("\n=== Total R by exit year: baseline vs >0 vs >3 vs >5 ===")
yb = trades_per_year(base)["sum"].rename("baseline")
y0 = trades_per_year(base[base.ret_atr_252>0])["sum"].rename(">0")
y3 = trades_per_year(base[base.ret_atr_252>3])["sum"].rename(">3")
y5 = trades_per_year(base[base.ret_atr_252>5])["sum"].rename(">5")
yr = pd.concat([yb, y0, y3, y5], axis=1).fillna(0)
print(yr.to_string())

# --- Equity-curve shape (cumulative R, chronological by exit date) ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9))
for name, df0, color in [("baseline (all 344)", base, "#444"),
                         (">0 ATR", base[base.ret_atr_252 > 0], "#1f77b4"),
                         (">3 ATR", base[base.ret_atr_252 > 3], "#2ca02c"),
                         (">5 ATR", base[base.ret_atr_252 > 5], "#d62728")]:
    d = df0.sort_values("Exit Date")
    eq = d["R_Multiple"].astype(float).cumsum()
    ax1.plot(pd.to_datetime(d["Exit Date"]).values, eq.values, label=name, color=color, lw=1.6)
    dd = eq - eq.cummax()
    ax2.plot(pd.to_datetime(d["Exit Date"]).values, dd.values, label=name, color=color, lw=1.2)
ax1.set_title("OLV cumulative R by exit date — trailing-252d-return-in-ATR gate")
ax1.legend(); ax1.grid(alpha=.3); ax1.set_ylabel("cumulative R")
ax2.set_title("Underwater (drawdown in R)")
ax2.legend(); ax2.grid(alpha=.3); ax2.set_ylabel("R below peak")
fig.tight_layout()
out = ROOT / "scratch" / "olv_perf_atr_curves.png"
fig.savefig(out, dpi=110)
print(f"\nSaved curve -> {out}")
