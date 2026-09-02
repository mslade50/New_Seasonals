"""Within-strategy adds, step 1: per-leg cluster features + per-leg daily MTM.

For each target strategy, every ledger leg gets (as of its ENTRY date, using only
information knowable then):
  n_open          open legs of the same strategy (entered earlier, still open)
  n_open_names    distinct open tickers
  n_same_sector   open legs in the same sector (data/sector_map.parquet + a manual
                  map for the 3x ETFs)
  n_same_ticker   open legs in the same ticker
  rho63_mean/max  trailing-63d daily-return correlation of the new name with the
                  open names (window ends the session BEFORE entry)
  stack_age_td    business days since the earliest still-open leg entered (0 = solo)
  rung_ladder     OLV only: the recency-ladder rung the engine applied (0.5/0.7/1.0)
  unit_pnl/risk   PnL and risk at Size_Mult = 1 (the nominal full size) so candidate
                  multipliers can be replayed on a common basis
Also writes a per-leg daily MTM vector (from master_prices closes, reconciled to the
booked PnL) so sleeve variance / Euler contributions can be measured.

OVS tranche rows (near/far) are collapsed to one trade per (Ticker, Signal Date,
Entry Date). Outputs: within_strategy_adds_features.parquet, _mtm.parquet.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
NAV = 750_000.0
TARGETS = ["Oversold Low Volume", "52wh Breakout", "Overbot Vol Spike", "Weak Close Decent Sznls",
           "LT Trend ST OS", "3x ETF Overbot Fade", "3x Bear ETF Overbot Fade", "3x Leader Gap Fade"]

LEV3X_SECTOR = {
    **{t: "Index" for t in ["SPXL", "TQQQ", "UDOW", "TNA", "MIDU", "SPXS", "SQQQ", "SDOW", "TZA"]},
    **{t: "Intl Index" for t in ["YINN", "BRZU", "EDC", "MEXX", "YANG", "EDZ"]},
    **{t: "Technology" for t in ["SOXL", "SOXS", "TECL", "TECS", "WEBL", "WEBS"]},
    **{t: "Financial Services" for t in ["FAS", "FAZ", "DPST"]},
    **{t: "Healthcare" for t in ["LABU", "LABD", "CURE"]},
    **{t: "Energy" for t in ["ERX", "ERY", "GUSH", "DRIP"]},
    **{t: "Real Estate" for t in ["DRN", "DRV"]},
    **{t: "Treasury" for t in ["TMF", "TMV"]},
    **{t: "Basic Materials" for t in ["NUGT", "JNUG", "DUST", "JDST"]},
    "NAIL": "Consumer Cyclical", "RETL": "Consumer Cyclical", "DFEN": "Industrials",
}

led = pd.read_parquet(ROOT / "data/backtest_trades_full.parquet")
led = led[led["PnL_flat_750k"].notna() & led["Strategy"].isin(TARGETS)].copy()

# ---- collapse OVS tranches to trades
ovs = led[led.Strategy == "Overbot Vol Spike"]
keys = ["Strategy", "Tier", "Ticker", "Direction", "Signal Date", "Entry Date"]
agg = ovs.groupby(keys, as_index=False).agg(
    trade_id=("trade_id", "min"), **{"Exit Date": ("Exit Date", "max")}, **{"Exit Type": ("Exit Type", "first")},
    **{"Entry Price": ("Entry Price", "first")}, **{"Exit Price": ("Exit Price", "mean")},
    PnL_flat_750k=("PnL_flat_750k", "sum"), Risk_flat_750k=("Risk_flat_750k", "sum"),
    Shares_flat=("Shares_flat", "sum"), ATR=("ATR", "first"), **{"Range %": ("Range %", "first")},
    **{"Risk bps": ("Risk bps", "first")}, Size_Mult=("Size_Mult", "first"), stop_atr=("stop_atr", "first"))
agg["R_Multiple"] = agg["PnL_flat_750k"] / agg["Risk_flat_750k"]
agg["n_tranche"] = ovs.groupby(keys).size().values
cols = list(agg.columns)
rest = led[led.Strategy != "Overbot Vol Spike"].copy()
rest["n_tranche"] = 1
rest["Exit Price"] = rest["Exit Price"].astype(float)
df = pd.concat([rest[[c for c in cols if c in rest.columns]], agg], ignore_index=True)
df = df.sort_values(["Strategy", "Entry Date", "trade_id"]).reset_index(drop=True)

# ---- sector
sec = pd.read_parquet(ROOT / "data/sector_map.parquet").set_index("ticker")["sector"].to_dict()
sec.update(LEV3X_SECTOR)
df["sector"] = df["Ticker"].map(sec).fillna("UNKNOWN")
# UNKNOWN never matches anything (treated as its own singleton sector per ticker)
df.loc[df.sector == "UNKNOWN", "sector"] = "UNK_" + df.loc[df.sector == "UNKNOWN", "Ticker"]

# ---- unit size (Size_Mult = 1) basis + OLV ladder rung inference
df["unit_pnl"] = df["PnL_flat_750k"] / df["Size_Mult"]
df["unit_risk"] = df["Risk_flat_750k"] / df["Size_Mult"]
def olv_rung(m):
    # engine Size_Mult for OLV = ladder rung x residual (earnings override 15/52.5, ticker-cap clip)
    for r in (1.0, 0.7, 0.5):
        for res in (1.0, 15 / 52.5):
            if abs(m - r * res) < 0.02:
                return r, res
    # cap-clipped legs: pick the rung whose residual is <= 1
    for r in (0.5, 0.7, 1.0):
        if m <= r + 1e-9:
            return r, m / r
    return 1.0, m
df["rung_ladder"] = 1.0
df["residual_mult"] = df["Size_Mult"]
isolv = df.Strategy == "Oversold Low Volume"
rr = df.loc[isolv, "Size_Mult"].map(olv_rung)
df.loc[isolv, "rung_ladder"] = [x[0] for x in rr]
df.loc[isolv, "residual_mult"] = [x[1] for x in rr]

# ---- prices for corr + MTM
ticks = sorted(df.Ticker.unique())
tbl = pq.read_table(ROOT / "data/master_prices.parquet", columns=["ticker", "date", "Close"],
                    filters=[("ticker", "in", ticks)])
px = tbl.to_pandas().pivot(index="date", columns="ticker", values="Close").sort_index()
px.index = pd.to_datetime(px.index).tz_localize(None) if getattr(px.index, "tz", None) is not None else pd.to_datetime(px.index)
ret = px.pct_change()
pos = pd.Series(np.arange(len(px)), index=px.index)

def bdays(a, b):
    return int(np.busday_count(np.datetime64(a, "D"), np.datetime64(b, "D")))

feat_rows = []
mtm_rows = []
for s, g in df.groupby("Strategy", sort=False):
    g = g.sort_values(["Entry Date", "trade_id"])
    open_ = []  # dicts of open legs
    for i, t in g.iterrows():
        ed = t["Entry Date"]
        open_ = [o for o in open_ if o["exit"] >= ed]
        names = sorted({o["tk"] for o in open_})
        n_same_sec = sum(1 for o in open_ if o["sec"] == t["sector"] and o["tk"] != t["Ticker"])
        n_same_tk = sum(1 for o in open_ if o["tk"] == t["Ticker"])
        n_same_sec_names = len({o["tk"] for o in open_ if o["sec"] == t["sector"] and o["tk"] != t["Ticker"]})
        rho_m = rho_x = np.nan
        others = [n for n in names if n != t["Ticker"]]
        if others and t["Ticker"] in ret.columns:
            w = ret.loc[:ed - pd.Timedelta(days=1)].tail(63)
            cs = [w[t["Ticker"]].corr(w[n]) for n in others if n in w.columns]
            cs = [c for c in cs if pd.notna(c)]
            if cs:
                rho_m, rho_x = float(np.mean(cs)), float(np.max(cs))
        age = bdays(min(o["entry"] for o in open_), ed) if open_ else 0
        open_risk = sum(o["risk"] for o in open_)
        same_day = sum(1 for o in open_ if o["entry"] == ed)
        feat_rows.append(dict(idx=i, n_open=len(open_), n_open_names=len(names), n_same_sector=n_same_sec,
                              n_same_sector_names=n_same_sec_names, n_same_ticker=n_same_tk,
                              rho63_mean=rho_m, rho63_max=rho_x, stack_age_td=age, open_risk_bps=open_risk / NAV * 1e4,
                              same_day_prior=same_day, rho_sum=float(np.nansum([c for c in ([] if not others else [
                                  ret.loc[:ed - pd.Timedelta(days=1)].tail(63)[t["Ticker"]].corr(
                                      ret.loc[:ed - pd.Timedelta(days=1)].tail(63)[n]) for n in others if n in ret.columns])]))))
        open_.append(dict(tk=t["Ticker"], sec=t["sector"], entry=ed, exit=t["Exit Date"], risk=t["Risk_flat_750k"]))
        # per-leg daily MTM from closes, reconciled to booked PnL
        if t["Ticker"] in px.columns and ed in pos.index and t["Exit Date"] in pos.index:
            a, b = pos[ed], pos[t["Exit Date"]]
            closes = px[t["Ticker"]].iloc[a:b + 1].values
            sign = 1.0 if t["Direction"] == "Long" else -1.0
            sh = t["Shares_flat"] * sign
            if len(closes) >= 1 and np.isfinite(closes).all():
                path = np.concatenate([[t["Entry Price"]], closes[:-1], [t["Exit Price"]]]) if len(closes) > 1 else np.array([t["Entry Price"], t["Exit Price"]])
                dp = np.diff(path) * sh
                dp[-1] += t["PnL_flat_750k"] - dp.sum()   # residual (slippage, fills) lands on the exit day
                for d_, v in zip(px.index[a:b + 1], dp):
                    mtm_rows.append((s, i, d_, float(v)))
            else:
                mtm_rows.append((s, i, t["Exit Date"], float(t["PnL_flat_750k"])))
        else:
            mtm_rows.append((s, i, t["Exit Date"], float(t["PnL_flat_750k"])))

F = pd.DataFrame(feat_rows).set_index("idx")
df = df.join(F)
df["yr"] = df["Entry Date"].dt.year
df.to_parquet(OUT / "within_strategy_adds_features.parquet")
M = pd.DataFrame(mtm_rows, columns=["Strategy", "idx", "date", "pnl"])
M.to_parquet(OUT / "within_strategy_adds_mtm.parquet")
print("features", df.shape, "mtm rows", len(M))
print(df.groupby("Strategy").agg(N=("trade_id", "size"), n_open_mean=("n_open", "mean"), same_sec=("n_same_sector", "mean"),
                                 same_tk=("n_same_ticker", "mean"), rho=("rho63_mean", "mean"), age=("stack_age_td", "mean")).round(2).to_string())
# reconciliation check
chk = M.groupby("idx").pnl.sum().reindex(df.index)
print("MTM reconciles to booked PnL:", np.allclose(chk.fillna(0).values, df["PnL_flat_750k"].values, atol=1.0))
