"""
ml/features.py — Per-trade point-in-time feature assembly.

One code path for both training (ml/dataset.py) and inference
(ml/score_daily.py): ticker-level indicators come from the repo's own
indicators.calculate_indicators (read-only import), guaranteeing parity with
what the scanner/backtester compute. Look-ahead columns that function produces
(NextOpen, pivot columns) are never selected as features — see
config.TICKER_FEATURES and tests/test_ml_no_lookahead.py.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indicators import calculate_indicators  # noqa: E402  (repo root import)
from ml import config  # noqa: E402

OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]


# -----------------------------------------------------------------------------
# Raw data loaders
# -----------------------------------------------------------------------------

def load_price_map(tickers, prices_path: str = None) -> dict:
    """{ticker: OHLCV DataFrame indexed by normalized date} from master_prices."""
    path = prices_path or config.MASTER_PRICES_PATH
    tickers = [str(t) for t in tickers]
    try:
        df = pd.read_parquet(path, filters=[("ticker", "in", tickers)])
    except Exception:
        df = pd.read_parquet(path)
    df = df[df["ticker"].isin(set(tickers))]
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    out = {}
    for tkr, grp in df.groupby("ticker"):
        g = grp.drop_duplicates(subset="date", keep="last").set_index("date").sort_index()
        out[tkr] = g[OHLCV_COLS]
    return out


def load_sznl_map(primary: str = None, backup: str = None) -> dict:
    """{ticker: Series(date -> seasonal_rank)} — mirrors the loader in
    pages/strat_backtester.py without the Streamlit dependency."""

    def load_raw(path):
        try:
            df = pd.read_csv(path)
            if "ticker" not in df.columns or "Date" not in df.columns:
                return pd.DataFrame()
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.dropna(subset=["Date", "ticker"])
            df["Date"] = df["Date"].dt.normalize()
            df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
            return df
        except Exception:
            return pd.DataFrame()

    df_p = load_raw(primary or config.SZNL_CSV_PRIMARY)
    df_b = load_raw(backup or config.SZNL_CSV_BACKUP)
    if df_p.empty and df_b.empty:
        return {}
    if df_p.empty:
        final = df_b
    elif df_b.empty:
        final = df_p
    else:
        final = pd.concat([df_p, df_b], axis=0).drop_duplicates(
            subset=["ticker", "Date"], keep="first")
    final = final.sort_values("Date")
    return {t: g.set_index("Date")["seasonal_rank"] for t, g in final.groupby("ticker")}


def load_atr_sznl_map(path: str = None, tickers=None) -> dict:
    """{ticker: DataFrame[atr_sznl_*] indexed by date}."""
    df = pd.read_parquet(path or config.ATR_SZNL_PATH)
    if tickers is not None:
        df = df[df["ticker"].isin(set(tickers))]
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    cols = [c for c in df.columns if c.startswith("atr_sznl_")]
    return {t: g.set_index("Date")[cols].sort_index() for t, g in df.groupby("ticker")}


# -----------------------------------------------------------------------------
# Per-ticker feature frame
# -----------------------------------------------------------------------------

def compute_ticker_feature_frame(ohlcv: pd.DataFrame, sznl_map: dict, ticker: str,
                                 atr_sznl: pd.DataFrame = None) -> pd.DataFrame:
    """Full-history per-date feature frame for one ticker.

    Reuses indicators.calculate_indicators for scan parity, then adds the few
    cheap derived columns the plan specifies. Returns only feature columns.
    """
    ind = calculate_indicators(ohlcv, sznl_map, ticker)

    ind["close_vs_sma50_atr"] = (ind["Close"] - ind["SMA50"]) / ind["ATR"]
    ind["close_vs_sma200_atr"] = (ind["Close"] - ind["SMA200"]) / ind["ATR"]
    ind["sma50_gt_sma200"] = (ind["SMA50"] > ind["SMA200"]).astype(float)
    ind["pct_off_52w_high"] = (ind["Close"] / ind["High_52w"] - 1.0) * 100.0
    # RangePct comes back 0..1 — keep native scale

    feats = ind[[c for c in config.TICKER_FEATURES if c in ind.columns]].copy()

    if atr_sznl is not None and not atr_sznl.empty:
        joined = atr_sznl.reindex(feats.index)
        for c in config.ATR_SZNL_FEATURES:
            feats[c] = joined[c] if c in joined.columns else np.nan
    else:
        for c in config.ATR_SZNL_FEATURES:
            feats[c] = np.nan
    return feats


# -----------------------------------------------------------------------------
# Trade-level assembly (shared by training and inference)
# -----------------------------------------------------------------------------

def assemble_features(trades: pd.DataFrame, price_map: dict, sznl_map: dict,
                      atr_sznl_map: dict, market_frame: pd.DataFrame,
                      date_col: str = "Signal Date") -> pd.DataFrame:
    """Build the model feature matrix for a frame of trades/signals.

    `trades` must carry: Ticker, <date_col>, Strategy, Direction, Tier,
    hold_days_target, stop_atr, tgt_atr. Returns a DataFrame aligned to
    trades.index with config.ALL_FEATURES columns (NaN where unavailable —
    the model is NaN-tolerant).
    """
    trades = trades.copy()
    trades[date_col] = pd.to_datetime(trades[date_col]).dt.normalize()

    # Per-ticker frames computed once
    frames = {}
    for tkr in trades["Ticker"].unique():
        if tkr in price_map and len(price_map[tkr]) > 0:
            frames[tkr] = compute_ticker_feature_frame(
                price_map[tkr], sznl_map, tkr, atr_sznl_map.get(tkr))

    ticker_cols = config.TICKER_FEATURES + config.ATR_SZNL_FEATURES
    rows = []
    for idx, row in trades.iterrows():
        frame = frames.get(row["Ticker"])
        dt = row[date_col]
        if frame is not None and dt in frame.index:
            rows.append(frame.loc[dt, ticker_cols])
        else:
            rows.append(pd.Series(np.nan, index=ticker_cols))
    feat = pd.DataFrame(rows, index=trades.index)

    # Market context joined on the signal date
    mkt = market_frame.reindex(trades[date_col].values)
    mkt.index = trades.index
    for c in config.MARKET_FEATURES:
        feat[c] = mkt[c] if c in mkt.columns else np.nan

    # Trade/meta
    feat["hold_days_target"] = pd.to_numeric(trades.get("hold_days_target"), errors="coerce")
    feat["stop_atr"] = pd.to_numeric(trades.get("stop_atr"), errors="coerce").fillna(0.0)
    feat["tgt_atr"] = pd.to_numeric(trades.get("tgt_atr"), errors="coerce").fillna(0.0)
    feat["dow"] = trades[date_col].dt.dayofweek.astype(float)
    feat["month"] = trades[date_col].dt.month.astype(float)
    for c in config.CATEGORICAL_FEATURES:
        feat[c] = trades.get(c, pd.Series("Unknown", index=trades.index)).astype(str)

    return feat[config.ALL_FEATURES]
