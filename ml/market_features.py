"""
ml/market_features.py — Point-in-time market-context features.

Built entirely from data/master_prices.parquet (verified to contain ^VIX,
^VIX3M, ^GSPC, the 11 SPDR sector ETFs, HYG and IEF). Every computation uses
trailing windows only; the no-lookahead unit test exercises this module.
"""

import numpy as np
import pandas as pd

from ml import config


def load_market_closes(prices_path: str = None) -> pd.DataFrame:
    """Wide frame of Close prices (index=date, columns=ticker) for the
    market-context tickers."""
    path = prices_path or config.MASTER_PRICES_PATH
    try:
        df = pd.read_parquet(path, columns=["ticker", "date", "Close"],
                             filters=[("ticker", "in", list(config.MARKET_TICKERS))])
    except Exception:
        df = pd.read_parquet(path, columns=["ticker", "date", "Close"])
    df = df[df["ticker"].isin(config.MARKET_TICKERS)]
    wide = df.pivot_table(index="date", columns="ticker", values="Close", aggfunc="last")
    wide.index = pd.to_datetime(wide.index).normalize()
    return wide.sort_index()


def build_market_frame(closes: pd.DataFrame) -> pd.DataFrame:
    """Compute the market-context feature frame from a wide close-price frame.

    All features are trailing-only:
      vix_close, vix_5d_chg, vix_term, spx_vs_sma200_pct, spx_ret_21d_rank,
      spx_rv_21d, breadth_pct_above_200d, hyg_ief_z63
    """
    out = pd.DataFrame(index=closes.index)

    # The union date index leaves NaN holes wherever one ticker has a date
    # others lack; a single hole poisons rolling windows (SMA200 etc.) for a
    # full window length. Trailing ffill (past values only — no lookahead),
    # capped at 5 sessions so genuinely dead series stay NaN.
    closes = closes.ffill(limit=5)

    vix = closes.get("^VIX")
    vix3m = closes.get("^VIX3M")
    spx = closes.get("^GSPC")

    if vix is not None:
        out["vix_close"] = vix
        out["vix_5d_chg"] = vix - vix.shift(5)
        if vix3m is not None:
            out["vix_term"] = vix / vix3m
        else:
            out["vix_term"] = np.nan
    else:
        out["vix_close"] = np.nan
        out["vix_5d_chg"] = np.nan
        out["vix_term"] = np.nan

    if spx is not None:
        sma200 = spx.rolling(200).mean()
        out["spx_vs_sma200_pct"] = (spx / sma200 - 1.0) * 100.0
        ret21 = spx.pct_change(21, fill_method=None)
        out["spx_ret_21d_rank"] = ret21.expanding(min_periods=252).rank(pct=True) * 100.0
        logret = np.log(spx / spx.shift(1))
        out["spx_rv_21d"] = logret.rolling(21).std() * np.sqrt(252) * 100.0
    else:
        out["spx_vs_sma200_pct"] = np.nan
        out["spx_ret_21d_rank"] = np.nan
        out["spx_rv_21d"] = np.nan

    sectors = [t for t in config.SECTOR_ETFS if t in closes.columns]
    if sectors:
        sec = closes[sectors]
        above = sec.gt(sec.rolling(200).mean())
        # only count sectors that actually have data on the date
        have_data = sec.notna() & sec.rolling(200).mean().notna()
        denom = have_data.sum(axis=1).replace(0, np.nan)
        out["breadth_pct_above_200d"] = (above & have_data).sum(axis=1) / denom * 100.0
    else:
        out["breadth_pct_above_200d"] = np.nan

    hyg, ief = closes.get("HYG"), closes.get("IEF")
    if hyg is not None and ief is not None:
        ratio = hyg / ief
        mu = ratio.rolling(63).mean()
        sd = ratio.rolling(63).std()
        out["hyg_ief_z63"] = (ratio - mu) / sd
    else:
        out["hyg_ief_z63"] = np.nan

    return out[config.MARKET_FEATURES]


def get_market_frame(prices_path: str = None) -> pd.DataFrame:
    return build_market_frame(load_market_closes(prices_path))
