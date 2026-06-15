"""
signal_chart_common.py - shared helpers for per-trade signal charts.

The chart renderer (scripts/build_signal_charts.py) and the site builder
(scripts/build_site.py) must agree on (a) the stable chart key and (b) the
MAE/MFE math, so both live here. This module has NO heavy deps (no mplfinance)
so build_site can import it cheaply.
"""
import re

import pandas as pd

PRE_TD = 126   # trading days of context before the signal bar
POST_TD = 63   # trading days after the exit bar

# Relative layout shared by the R2 key (under charts/) and the local dir.
REL_ROOT = "signals"


def slug(s):
    return re.sub(r"[^A-Za-z0-9]+", "_", str(s)).strip("_")


def chart_relpath(strategy, ticker, signal_date):
    """Stable relative path for a trade's chart:
    signals/<strategy>/<TICKER>_<YYYYMMDD>.png

    Keyed on Strategy + Ticker + Signal Date only - NOT trade_id (which is
    np.arange and reshuffles when new signals fire) and NOT exit type (which
    can flip on a data revision). Disjoint liquid/overflow universes guarantee
    a (strategy, ticker, signal-date) triple is unique across tiers.
    """
    sd = pd.Timestamp(signal_date)
    return f"{REL_ROOT}/{slug(strategy)}/{slug(ticker)}_{sd:%Y%m%d}.png"


def _nearest_pos(idx, d):
    return int(idx.get_indexer([pd.Timestamp(d)], method="nearest")[0])


def trade_geometry(trade, prices):
    """Window slice bounds + MAE/MFE + stop/target levels for one trade.

    `trade` is a ledger row (mapping with Direction, Entry Price, ATR,
    stop_atr, tgt_atr, Signal/Entry/Exit Date); `prices` its OHLCV DataFrame
    indexed by Date. Returns a dict, or None if prices are empty.

    MAE/MFE are measured over the holding period (entry..exit) from the bars'
    highs/lows; R uses per-share risk = stop_atr * ATR (reproduces ledger R).
    """
    if prices is None or prices.empty:
        return None
    idx = prices.index
    is_long = str(trade["Direction"]) == "Long"
    entry_px = float(trade["Entry Price"])
    atr = float(trade["ATR"])
    stop_atr = float(trade["stop_atr"])
    tgt_atr = float(trade["tgt_atr"])

    sig_pos = _nearest_pos(idx, trade["Signal Date"])
    ent_pos = _nearest_pos(idx, trade["Entry Date"])
    exit_pos = _nearest_pos(idx, trade["Exit Date"])
    lo = max(0, sig_pos - PRE_TD)
    hi = min(len(idx) - 1, exit_pos + POST_TD)
    post_short = (exit_pos + POST_TD) > (len(idx) - 1)

    hold = prices.iloc[ent_pos:exit_pos + 1]
    risk_ps = stop_atr * atr  # per-share risk
    if is_long:
        mfe_move = hold["High"].max() - entry_px
        mae_move = hold["Low"].min() - entry_px
        stop_px = entry_px - stop_atr * atr
        tgt_px = entry_px + tgt_atr * atr
    else:
        mfe_move = entry_px - hold["Low"].min()
        mae_move = entry_px - hold["High"].max()
        stop_px = entry_px + stop_atr * atr
        tgt_px = entry_px - tgt_atr * atr

    return {
        "sig_pos": sig_pos, "ent_pos": ent_pos, "exit_pos": exit_pos,
        "lo": lo, "hi": hi, "post_short": post_short,
        "entry_px": entry_px, "stop_px": stop_px, "tgt_px": tgt_px,
        "mfe_pct": mfe_move / entry_px * 100.0,
        "mae_pct": mae_move / entry_px * 100.0,
        "mfe_r": (mfe_move / risk_ps) if risk_ps else float("nan"),
        "mae_r": (mae_move / risk_ps) if risk_ps else float("nan"),
    }


def lookup_prices(px_map, ticker):
    """Fetch a ticker's OHLCV from a {ticker: df} map, tolerating the
    yfinance dot->dash convention (BRK.B vs BRK-B)."""
    p = px_map.get(ticker)
    if p is None or getattr(p, "empty", True):
        p = px_map.get(str(ticker).replace(".", "-"))
    return p
