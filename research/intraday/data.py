"""Local-only data contracts for the intraday research lab.

The production loader can refresh missing data from R2.  This module
deliberately cannot: research runs consume only frames or parquet files that
the caller explicitly supplies.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd

BAR_MINUTES = 15
BAR_DELTA = pd.Timedelta(minutes=BAR_MINUTES)
SESSION_START = pd.Timestamp("09:30").time()
SESSION_LAST_BAR = pd.Timestamp("15:45").time()
REQUIRED_COLUMNS = ("ts", "open", "high", "low", "close", "volume")


class IntradayDataError(ValueError):
    """Raised when bars cannot support a defensible event-clock study."""


class LookaheadError(ValueError):
    """Raised when a signal claims to use information not yet available."""


def normalize_bars(frame: pd.DataFrame, *, ticker: str = "") -> pd.DataFrame:
    """Validate and normalize one ticker's regular-session 15-minute bars.

    Timestamps are interpreted as bar *start* times in New York.  Naive input
    is therefore assumed to already be ET, matching the existing parquet
    contract.  Aware input is converted to New York time and made naive.
    Missing sessions/bars are allowed; completeness is measured separately.
    """

    label = f" for {ticker.upper()}" if ticker else ""
    if not isinstance(frame, pd.DataFrame):
        raise IntradayDataError(f"bars{label} must be a pandas DataFrame")

    work = frame.copy()
    work.columns = [str(column).strip().lower() for column in work.columns]
    if "ts" not in work.columns and isinstance(work.index, pd.DatetimeIndex):
        work = work.reset_index()
        work = work.rename(columns={work.columns[0]: "ts"})

    missing = [column for column in REQUIRED_COLUMNS if column not in work.columns]
    if missing:
        raise IntradayDataError(f"bars{label} missing columns: {missing}")

    work = work.loc[:, REQUIRED_COLUMNS].copy()
    work["ts"] = pd.to_datetime(work["ts"], errors="coerce")
    if work["ts"].isna().any():
        raise IntradayDataError(f"bars{label} contain invalid timestamps")
    if isinstance(work["ts"].dtype, pd.DatetimeTZDtype):
        work["ts"] = work["ts"].dt.tz_convert("America/New_York").dt.tz_localize(None)

    for column in ("open", "high", "low", "close", "volume"):
        work[column] = pd.to_numeric(work[column], errors="coerce")
    if work[list(REQUIRED_COLUMNS[1:])].isna().any().any():
        raise IntradayDataError(f"bars{label} contain missing or non-numeric OHLCV")
    if not np.isfinite(work[list(REQUIRED_COLUMNS[1:])].to_numpy(dtype=float)).all():
        raise IntradayDataError(f"bars{label} contain non-finite OHLCV")

    work = work.sort_values("ts").reset_index(drop=True)
    if work["ts"].duplicated().any():
        duplicate = work.loc[work["ts"].duplicated(), "ts"].iloc[0]
        raise IntradayDataError(f"bars{label} contain duplicate timestamp {duplicate}")

    times = work["ts"].dt.time
    outside_session = (times < SESSION_START) | (times > SESSION_LAST_BAR)
    aligned = (
        work["ts"].dt.minute.mod(BAR_MINUTES).eq(0)
        & work["ts"].dt.second.eq(0)
        & work["ts"].dt.microsecond.eq(0)
    )
    if outside_session.any() or (~aligned).any():
        bad = work.loc[outside_session | ~aligned, "ts"].iloc[0]
        raise IntradayDataError(
            f"bars{label} contain a non-regular-session or non-15m timestamp: {bad}"
        )

    if (work[["open", "high", "low", "close"]] <= 0).any().any():
        raise IntradayDataError(f"bars{label} contain non-positive prices")
    if (work["volume"] < 0).any():
        raise IntradayDataError(f"bars{label} contain negative volume")

    max_body = work[["open", "close"]].max(axis=1)
    min_body = work[["open", "close"]].min(axis=1)
    invalid_ohlc = (
        work["high"].lt(max_body)
        | work["low"].gt(min_body)
        | work["high"].lt(work["low"])
    )
    if invalid_ohlc.any():
        bad = work.loc[invalid_ohlc, "ts"].iloc[0]
        raise IntradayDataError(f"bars{label} have impossible OHLC at {bad}")

    counts = work.groupby(work["ts"].dt.normalize()).size()
    if (counts > 26).any():
        bad_day = counts[counts > 26].index[0]
        raise IntradayDataError(
            f"bars{label} have more than 26 bars on {bad_day.date()}"
        )
    return work


def normalize_frame_map(frames: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Upper-case ticker keys and validate every supplied frame."""

    normalized: dict[str, pd.DataFrame] = {}
    for raw_ticker, frame in frames.items():
        ticker = str(raw_ticker).strip().upper()
        if not ticker:
            raise IntradayDataError("ticker keys cannot be blank")
        if ticker in normalized:
            raise IntradayDataError(f"duplicate ticker after normalization: {ticker}")
        normalized[ticker] = normalize_bars(frame, ticker=ticker)
    if not normalized:
        raise IntradayDataError("no intraday frames supplied")
    return normalized


def load_parquet_frames(
    data_dir: str | Path,
    *,
    tickers: Iterable[str] | None = None,
    interval: str = "15min",
) -> dict[str, pd.DataFrame]:
    """Read existing ``{TICKER}_{interval}.parquet`` files from local disk.

    No cache helper, credential, network, or R2 path is imported here.  When
    ``tickers`` is provided, missing files fail loudly instead of shrinking a
    research universe silently.
    """

    root = Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"intraday data directory does not exist: {root}")
    suffix = f"_{interval}.parquet"
    if tickers is None:
        paths = sorted(
            path for path in root.glob(f"*{suffix}") if path.name != "_meta.parquet"
        )
    else:
        paths = [root / f"{str(ticker).upper()}{suffix}" for ticker in tickers]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing intraday parquet(s): {missing}")

    frames: dict[str, pd.DataFrame] = {}
    for path in paths:
        ticker = path.name[: -len(suffix)].upper()
        if ticker == "_META":
            continue
        frames[ticker] = pd.read_parquet(path)
    return normalize_frame_map(frames)
