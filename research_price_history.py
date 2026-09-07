"""Raw bars for replaying already-frozen research order levels.

The adjusted strategy cache must never be a fallback for these dollar levels.
Missing/failed raw downloads stop grading and leave existing evidence intact.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def replay_cutoff(as_of) -> pd.Timestamp:
    """Exclude today's potentially incomplete bar, including manual intraday runs."""
    today = pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize()
    return min(pd.Timestamp(as_of).normalize() + pd.Timedelta(days=1), today)


def load_raw_prices(tickers, *, start, end, download=None) -> pd.DataFrame:
    """Return long-form raw OHLC through the exclusive ``end`` date.

    Bounds/tickers come from pending journal orders, not the full strategy book.
    An injected downloader makes the price-basis contract testable offline.
    """
    wanted = sorted({str(t).upper() for t in tickers})
    columns = ["ticker", "date", "Open", "High", "Low", "Close"]
    if not wanted or pd.Timestamp(start) >= pd.Timestamp(end):
        return pd.DataFrame(columns=columns)
    if download is None:
        import yfinance as yf
        download = yf.download
    frames = []
    for offset in range(0, len(wanted), 50):
        batch = wanted[offset:offset + 50]
        raw = download(batch, start=str(pd.Timestamp(start).date()), end=str(pd.Timestamp(end).date()),
                       auto_adjust=False, back_adjust=False, progress=False)
        if raw is None or raw.empty:
            raise ValueError("raw research price history unavailable; grading stopped")
        if len(batch) > 1 and not isinstance(raw.columns, pd.MultiIndex):
            raise ValueError("raw multi-ticker history lacks ticker columns; grading stopped")
        for ticker in batch:
            try:
                frame = raw.xs(ticker, level="Ticker", axis=1).copy() if isinstance(raw.columns, pd.MultiIndex) else raw.copy()
            except KeyError as exc:
                raise ValueError(f"raw research history missing for {ticker}; grading stopped") from exc
            if isinstance(frame.columns, pd.MultiIndex):
                frame.columns = frame.columns.get_level_values(0)
            frame.columns = [str(c).capitalize() for c in frame.columns]
            if not {"Open", "High", "Low", "Close"} <= set(frame.columns):
                raise ValueError(f"raw OHLC incomplete for {ticker}; grading stopped")
            frame = frame[["Open", "High", "Low", "Close"]].apply(pd.to_numeric, errors="coerce")
            frame = frame.dropna()
            if frame.empty or not np.isfinite(frame.to_numpy()).all() or (frame <= 0).any().any():
                raise ValueError(f"raw OHLC invalid for {ticker}; grading stopped")
            frame.index = pd.to_datetime(frame.index).tz_localize(None).normalize()
            frame = frame[(frame.index >= pd.Timestamp(start).normalize()) & (frame.index < pd.Timestamp(end).normalize())]
            frame["ticker"] = ticker
            frame["date"] = frame.index
            frames.append(frame.reset_index(drop=True))
    result = pd.concat(frames, ignore_index=True).drop_duplicates(["ticker", "date"], keep="last")
    result.attrs["price_basis"] = "RAW"
    result.attrs["source"] = "yfinance:auto_adjust=False,back_adjust=False"
    return result[columns]
