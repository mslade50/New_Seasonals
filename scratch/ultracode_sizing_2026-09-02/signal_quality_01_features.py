"""Signal-quality study (2026-09-02), step 1: build the per-trade feature table.

One row per TRADE (OVS scale-out tranches collapsed: R = sum PnL / sum Risk),
with every feature knowable at the signal close (or, for the T+1 open gap, at
the next open) joined from master_prices, the dial, VIX, P/C, breadth, the
earnings calendar and the ledger itself.

Inputs : data/backtest_trades_full.parquet, data/master_prices.parquet (column
         + ticker filtered), data/rd2_fragility.parquet (10d MA of 63d; rows
         before 2026-07-02 are the recompute vintage), data/cboe_putcall.parquet,
         data/sector_map.parquet, data/earnings_calendar.parquet,
         strategy_config.STRATEGY_BOOK (filter thresholds -> extremity feature).
Output : scratch/ultracode_sizing_2026-09-02/signal_quality_features.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import strategy_config as sc  # noqa: E402

pd.set_option("display.width", 250, "display.max_columns", 60)

# ------------------------------------------------------------------ ledger
led = pd.read_parquet(ROOT / "data/backtest_trades_full.parquet")
led = led[led["PnL_flat_750k"].notna()].copy()
key = ["Strategy", "Tier", "Ticker", "Signal Date", "Entry Date"]
agg = {
    "PnL_flat_750k": "sum", "Risk_flat_750k": "sum", "Shares_flat": "sum",
    "Direction": "first", "Exit Date": "max", "Exit Type": "first", "Entry Price": "first",
    "Signal Close": "first", "T+1 Open": "first", "ATR": "first", "stop_atr": "first",
    "tgt_atr": "first", "Range %": "first", "Size_Mult": "first", "Risk bps": "first",
    "hold_days_target": "first", "Entry Criteria": "first", "trade_id": "min",
}
T = led.groupby(key, as_index=False).agg(agg)
T["n_tranche"] = led.groupby(key).size().values
T["R"] = T["PnL_flat_750k"] / T["Risk_flat_750k"]
T["win"] = (T["R"] > 0).astype(int)
T["year"] = T["Signal Date"].dt.year
T["overflow"] = (T["Tier"] == "Overflow").astype(int)
T["hold_td"] = np.busday_count(T["Entry Date"].values.astype("datetime64[D]"),
                               T["Exit Date"].values.astype("datetime64[D]"))
T["gap_atr"] = (T["T+1 Open"] - T["Signal Close"]) / T["ATR"]
T["wait_td"] = np.busday_count(T["Signal Date"].values.astype("datetime64[D]"),
                               T["Entry Date"].values.astype("datetime64[D]"))
print(f"ledger rows {len(led)} -> trades {len(T)}")

# ------------------------------------------------------------------ prices
tickers = sorted(set(T["Ticker"]) | {"SPY", "^VIX", "QQQ", "IWM"})
px = pq.read_table(ROOT / "data/master_prices.parquet",
                   columns=["ticker", "date", "Open", "High", "Low", "Close", "Volume"],
                   filters=[("ticker", "in", tickers)]).to_pandas()
px = px.sort_values(["ticker", "date"])
print(f"price rows {len(px):,} tickers {px.ticker.nunique()} (ledger tickers {T.Ticker.nunique()})")


def indicators(g: pd.DataFrame) -> pd.DataFrame:
    g = g.set_index("date")
    c = g["Close"]
    out = pd.DataFrame(index=g.index)
    for w in [2, 5, 10, 21, 63, 252]:
        r = c.pct_change(w, fill_method=None)
        out[f"ret_{w}d"] = r
        if w != 63:
            out[f"rank_{w}d"] = r.expanding(min_periods=252).rank(pct=True) * 100.0
    hl = g["High"] - g["Low"]
    hc = (g["High"] - c.shift()).abs()
    lc = (g["Low"] - c.shift()).abs()
    atr = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
    out["atr_pct"] = atr / c * 100.0
    out["atr_pct_rank"] = out["atr_pct"].expanding(min_periods=252).rank(pct=True) * 100.0
    sma200 = c.rolling(200).mean()
    sma50 = c.rolling(50).mean()
    sma10 = c.rolling(10).mean()
    out["dist200"] = (c / sma200 - 1) * 100.0
    out["dist50"] = (c / sma50 - 1) * 100.0
    out["dist10_atr"] = (c - sma10) / atr
    out["above200"] = (c > sma200).astype(float)
    out["sma200_slope"] = (sma200 / sma200.shift(21) - 1) * 100.0
    out["hi252_dist"] = (c / g["High"].rolling(252).max() - 1) * 100.0
    out["lo252_dist"] = (c / g["Low"].rolling(252).min() - 1) * 100.0
    vma63 = g["Volume"].rolling(63).mean()
    out["vol_ratio"] = g["Volume"] / vma63
    out["vol_ratio_10d_rank"] = (g["Volume"].rolling(10).mean() / vma63).expanding(min_periods=252).rank(pct=True) * 100.0
    out["dollar_vol_m"] = (c * vma63) / 1e6
    denom = hl.replace(0, np.nan)
    out["range_pct"] = ((c - g["Low"]) / denom).fillna(0.5)
    out["move1_atr"] = (c - c.shift(1)) / atr
    dn = (c < c.shift(1)).astype(int)
    out["consec_down"] = dn.groupby((dn != dn.shift()).cumsum()).cumsum() * dn
    up = (c > c.shift(1)).astype(int)
    out["consec_up"] = up.groupby((up != up.shift()).cumsum()).cumsum() * up
    out["rv21"] = c.pct_change().rolling(21).std() * np.sqrt(252) * 100
    out["rv252"] = c.pct_change().rolling(252).std() * np.sqrt(252) * 100
    out["vol_of_vol"] = out["rv21"] / out["rv252"]
    out["age_years"] = (np.arange(len(c)) / 252.0)
    return out


ind = {}
for t, g in px.groupby("ticker"):
    ind[t] = indicators(g)
print("indicators done")

# breadth over the loaded panel (ledger universe) and by sector
sector = pd.read_parquet(ROOT / "data/sector_map.parquet").set_index("ticker")["sector"]
above = pd.DataFrame({t: d["above200"] for t, d in ind.items()})
breadth_all = above.mean(axis=1)
sec_breadth = {}
for s, grp in sector.groupby(sector):
    cols = [t for t in grp.index if t in above.columns]
    if len(cols) >= 5:
        sec_breadth[s] = above[cols].mean(axis=1)
sec_breadth = pd.DataFrame(sec_breadth)
# 21d breadth change
breadth_chg21 = breadth_all - breadth_all.shift(21)

# market
spy = ind["SPY"]
vix = px[px.ticker == "^VIX"].set_index("date")["Close"]
vix_pct252 = vix.rolling(252, min_periods=252).apply(lambda w: (w <= w[-1]).mean() * 100, raw=True)
spy_rv21 = spy["rv21"]
vrp = vix - spy_rv21  # implied minus realized (vol risk premium proxy)

frag = pd.read_parquet(ROOT / "data/rd2_fragility.parquet")
dial = frag["63d"].rolling(10).mean()
dial_raw21 = frag["21d"]
try:
    pitd = pd.read_parquet(HERE / "cross_strategy_regime_pit_dial.parquet")
    dial_pit = pitd["pit"].rolling(10).mean()
    print(f"PIT dial rows available: {dial_pit.notna().sum()}")
except Exception as e:  # noqa: BLE001
    dial_pit = pd.Series(dtype=float)
    print("no PIT dial", e)

pc = pd.read_parquet(ROOT / "data/cboe_putcall.parquet")["equity"].dropna().sort_index()
pc.index = pd.to_datetime(pc.index)
pc_ma = pc.rolling(10, min_periods=10).mean()
pc_pct = pc_ma.rolling(252, min_periods=252).apply(lambda w: (w <= w[-1]).mean() * 100, raw=True)

earn = pd.read_parquet(ROOT / "data/earnings_calendar.parquet", columns=["ticker", "date"])
earn["date"] = pd.to_datetime(earn["date"]).dt.normalize()
earn_map = {t: np.sort(g["date"].unique()) for t, g in earn.groupby("ticker")}

# ------------------------------------------------------------------ ledger-derived features
T = T.sort_values(["Strategy", "Ticker", "Signal Date"]).reset_index(drop=True)
T["days_since_last_sig"] = np.nan
T["prior_sig_21td"] = 0
for (s, tk), g in T.groupby(["Strategy", "Ticker"]):
    d = g["Signal Date"].values.astype("datetime64[D]")
    prev = np.r_[np.datetime64("NaT"), d[:-1]]
    ok = ~pd.isna(prev)
    dsl = np.full(len(d), np.nan)
    dsl[ok] = np.busday_count(prev[ok], d[ok])
    T.loc[g.index, "days_since_last_sig"] = dsl
    # count of prior signals within 21 td (excluding today)
    cnt = np.zeros(len(d), dtype=int)
    for i in range(len(d)):
        cnt[i] = int(np.sum((d[:i] < d[i]) & (np.busday_count(d[:i], d[i]) <= 21)))
    T.loc[g.index, "prior_sig_21td"] = cnt
T["is_first_sig_ticker"] = T["days_since_last_sig"].isna().astype(int)
T["days_since_last_sig_c"] = T["days_since_last_sig"].fillna(999).clip(upper=999)

sig_day = T.groupby(["Strategy", "Signal Date"]).size().rename("n_sig_strat_day")
T = T.merge(sig_day, left_on=["Strategy", "Signal Date"], right_index=True, how="left")
sig_day_book = T.groupby("Signal Date").size().rename("n_sig_book_day")
T = T.merge(sig_day_book, left_on="Signal Date", right_index=True, how="left")
# strategy signals in trailing 5 td (book-wide signal density), excluding today
bd = pd.bdate_range("2003-01-01", "2026-09-30")
sig_ts = T.groupby("Signal Date").size().reindex(bd).fillna(0)
T["book_sig_5td"] = sig_ts.rolling(5).sum().shift(1).reindex(T["Signal Date"]).values
strat_ts = {s: g.groupby("Signal Date").size().reindex(bd).fillna(0) for s, g in T.groupby("Strategy")}
T["strat_sig_21td"] = [strat_ts[s].rolling(21).sum().shift(1).get(d, np.nan) for s, d in zip(T["Strategy"], T["Signal Date"])]

# open legs of the same strategy at signal date (entered before, exit >= signal date)
T["open_legs_strat"] = 0
for s, g in T.groupby("Strategy"):
    ent = g["Entry Date"].values
    ex = g["Exit Date"].values
    sd = g["Signal Date"].values
    n = np.array([int(np.sum((ent < x) & (ex >= x))) for x in sd])
    T.loc[g.index, "open_legs_strat"] = n

# ------------------------------------------------------------------ join price/market features
feat_cols = ["ret_2d", "ret_5d", "ret_10d", "ret_21d", "ret_63d", "ret_252d", "rank_2d", "rank_5d", "rank_10d",
             "rank_21d", "rank_252d", "atr_pct", "atr_pct_rank", "dist200", "dist50", "dist10_atr", "above200",
             "sma200_slope", "hi252_dist", "lo252_dist", "vol_ratio", "vol_ratio_10d_rank", "dollar_vol_m",
             "range_pct", "move1_atr", "consec_down", "consec_up", "rv21", "rv252", "vol_of_vol", "age_years"]
rows = []
for i, r in T.iterrows():
    d = r["Signal Date"]
    tk = r["Ticker"]
    f = {}
    if tk in ind and d in ind[tk].index:
        f.update(ind[tk].loc[d, feat_cols].to_dict())
    rows.append(f)
F = pd.DataFrame(rows, index=T.index)
T = pd.concat([T, F], axis=1)
missing = T["atr_pct"].isna().mean()
print(f"price-feature missing share: {missing:.3%}")

d = T["Signal Date"]
T["spy_dist200"] = spy["dist200"].reindex(d).values
T["spy_ret10"] = spy["ret_10d"].reindex(d).values * 100
T["spy_ret21"] = spy["ret_21d"].reindex(d).values * 100
T["spy_ret63"] = spy["ret_63d"].reindex(d).values * 100
T["spy_rv21"] = spy_rv21.reindex(d).values
T["spy_hi252_dist"] = spy["hi252_dist"].reindex(d).values
T["vix"] = vix.reindex(d).values
T["vix_pct252"] = vix_pct252.reindex(d).values
T["vrp"] = vrp.reindex(d).values
T["dial"] = dial.reindex(d).values
T["dial_raw21"] = dial_raw21.reindex(d).values
T["dial_pit"] = dial_pit.reindex(d).values if len(dial_pit) else np.nan
T["pc_pct_lag1"] = pc_pct.reindex(d - pd.tseries.offsets.BDay(1), method="ffill").values
T["breadth200"] = breadth_all.reindex(d).values * 100
T["breadth_chg21"] = breadth_chg21.reindex(d).values * 100
T["sector"] = T["Ticker"].map(sector).fillna("Unknown")
T["sector_breadth200"] = [sec_breadth[s].get(dd, np.nan) * 100 if s in sec_breadth else np.nan
                          for s, dd in zip(T["sector"], d)]
T["rel_ret_21d"] = T["ret_21d"] * 100 - T["spy_ret21"]
T["rel_ret_63d"] = T["ret_63d"] * 100 - T["spy_ret63"]

# earnings proximity (business days to next announcement on/after signal, and since last before signal)
nxt, prv = [], []
for tk, dd in zip(T["Ticker"], T["Signal Date"]):
    arr = earn_map.get(tk)
    if arr is None or len(arr) == 0:
        nxt.append(np.nan); prv.append(np.nan); continue
    dd64 = np.datetime64(dd, "D")
    a = arr.astype("datetime64[D]")
    j = np.searchsorted(a, dd64)
    nxt.append(np.busday_count(dd64, a[j]) if j < len(a) else np.nan)
    prv.append(np.busday_count(a[j - 1], dd64) if j > 0 else np.nan)
T["td_to_next_earn"] = nxt
T["td_since_last_earn"] = prv
T["has_earn"] = T["td_to_next_earn"].notna().astype(int)

# ------------------------------------------------------------------ filter extremity per strategy
book = {s["name"]: s for s in sc.STRATEGY_BOOK}


def extremity(row) -> float:
    st = book.get(row["Strategy"])
    if st is None:
        return np.nan
    params = st.get("settings", st)
    vals = []
    for pf in params.get("perf_filters", []):
        col = f"rank_{pf['window']}d"
        if col not in row or pd.isna(row[col]):
            continue
        if pf["logic"] == "<":
            vals.append(pf["thresh"] - row[col])
        elif pf["logic"] == ">":
            vals.append(row[col] - pf["thresh"])
    return float(np.mean(vals)) if vals else np.nan


T["filt_extremity"] = T.apply(extremity, axis=1)
# strategy-neutral "how oversold/overbought" for dip-buys and fades: signed 5d rank distance from 50
T["rank5_dist50"] = (T["rank_5d"] - 50).abs()

T.to_parquet(HERE / "signal_quality_features.parquet", index=False)
print(T.groupby("Strategy").size().to_string())
print(T[["R", "win", "atr_pct", "dist200", "dial", "vix", "td_to_next_earn", "filt_extremity"]].describe().T.to_string())
