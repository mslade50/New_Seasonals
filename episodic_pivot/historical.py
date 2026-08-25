"""Observed-panel EP candidate census and fixed-horizon diagnostic study.

This is deliberately not described as a backtest.  The repo lacks a historical
listed/delisted universe, point-in-time news archive, reliable release times,
and broad premarket bars.  Every eligibility feature below is either known at
the prior close or explicitly marked ex-post.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import HistoricalPolicy


_PRICE_COLUMNS = ["ticker", "date", "Open", "High", "Low", "Close", "Volume"]


@dataclass
class TickerStudyResult:
    events: pd.DataFrame
    anomalies: pd.DataFrame
    counts: dict[str, int]


def _era(year: int) -> str:
    if year <= 2009:
        return "1999-2009"
    if year <= 2014:
        return "2010-2014"
    if year <= 2019:
        return "2015-2019"
    if year <= 2022:
        return "2020-2022"
    return "2023+"


def _gap_band(value: float) -> str:
    if value < 15:
        return "10-15"
    if value < 20:
        return "15-20"
    if value < 30:
        return "20-30"
    return ">30"


def _rvol_band(value: float) -> str:
    if value < 4:
        return "2-4x"
    if value < 8:
        return "4-8x"
    return ">=8x"


def _sample_period(year: int) -> str:
    if year <= 2019:
        return "DEVELOPMENT_1999_2019"
    if year <= 2023:
        return "VALIDATION_2020_2023"
    return "HOLDOUT_2024_2026"


def _empty_events() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "ticker",
            "date",
            "previous_session",
            "previous_close",
            "event_open",
            "event_high",
            "event_low",
            "event_close",
            "event_volume",
            "gap_pct",
            "prior_addv_63",
            "event_rvol_20",
            "prior_63d_return_pct",
            "earnings_date_match",
            "data_quality_clean",
            "gap_band",
            "rvol_band",
            "era",
            "sample_period",
        ]
    )


def study_ticker(
    ticker: str,
    prices: pd.DataFrame,
    *,
    policy: HistoricalPolicy,
    earnings_dates: Iterable[pd.Timestamp] = (),
    include_outcomes: bool = False,
) -> TickerStudyResult:
    """Build strict v0 events for one ticker using prior-session features only."""

    frame = prices.copy()
    frame = frame.rename(columns={c.lower(): c for c in ["Open", "High", "Low", "Close", "Volume"]})
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame = frame.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    for column in ("Open", "High", "Low", "Close", "Volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    previous_close = frame["Close"].shift(1)
    close_ratio = frame["Close"] / previous_close
    gap_pct = 100.0 * (frame["Open"] / previous_close - 1.0)
    close_change_pct = 100.0 * (close_ratio - 1.0)
    invalid = (
        frame[["Open", "High", "Low", "Close"]].le(0).any(axis=1)
        | frame[["Open", "High", "Low", "Close", "Volume"]].isna().any(axis=1)
        | (frame["Volume"] <= 0)
        | (frame["High"] < frame["Low"])
    )
    half_double = close_ratio.between(0.45, 0.55) | close_ratio.between(1.80, 2.20)
    extreme = gap_pct.abs().ge(50) | close_change_pct.abs().ge(50)
    suspicious_nearby = (
        (half_double | invalid | extreme)
        .astype("int8")
        .rolling(5, center=True, min_periods=1)
        .max()
        .astype(bool)
    )
    anomaly_mask = invalid | half_double | extreme
    anomalies = frame.loc[anomaly_mask, ["date", "Open", "High", "Low", "Close", "Volume"]].copy()
    if not anomalies.empty:
        anomalies.insert(0, "ticker", ticker)
        anomalies["gap_pct"] = gap_pct[anomaly_mask].to_numpy()
        anomalies["close_change_pct"] = close_change_pct[anomaly_mask].to_numpy()
        anomalies["invalid_bar"] = invalid[anomaly_mask].to_numpy()
        anomalies["half_or_double_cliff"] = half_double[anomaly_mask].to_numpy()
        anomalies["extreme_move"] = extreme[anomaly_mask].to_numpy()

    prior_dollar_volume = (frame["Close"] * frame["Volume"]).shift(1)
    prior_addv_63 = prior_dollar_volume.rolling(63, min_periods=63).mean()
    prior_volume_20 = frame["Volume"].shift(1).rolling(20, min_periods=20).mean()
    event_rvol_20 = frame["Volume"] / prior_volume_20
    prior_63d_return_pct = 100.0 * (previous_close / frame["Close"].shift(64) - 1.0)
    prior_bar_count = pd.Series(np.arange(len(frame)), index=frame.index)

    open_observable = (
        (prior_bar_count >= policy.min_prior_bars)
        & (previous_close >= policy.min_prior_close)
        & (prior_addv_63 >= policy.min_prior_addv_63)
        & (gap_pct >= policy.min_open_gap_pct)
        & ~invalid
    )
    volume_confirmed = open_observable & (event_rvol_20 >= policy.min_event_volume_rvol_20)
    prior_confirmed_count = (
        volume_confirmed.astype("int8")
        .shift(1, fill_value=0)
        .rolling(policy.first_event_lookback_sessions, min_periods=1)
        .sum()
    )
    first_event = volume_confirmed & prior_confirmed_count.eq(0)
    neglected = prior_63d_return_pct <= policy.max_prior_63d_return_pct
    strict = first_event & neglected

    counts = {
        "bars": int(len(frame)),
        "open_observable": int(open_observable.sum()),
        "volume_confirmed_ex_post": int(volume_confirmed.sum()),
        "first_confirmed_in_126": int(first_event.sum()),
        "strict_with_neglect": int(strict.sum()),
        "anomalies": int(anomaly_mask.sum()),
    }
    if not strict.any():
        return TickerStudyResult(_empty_events(), anomalies, counts)

    event_rows = frame.index[strict]
    events = pd.DataFrame(
        {
            "ticker": ticker,
            "date": frame.loc[event_rows, "date"].to_numpy(),
            "previous_session": frame["date"].shift(1).loc[event_rows].to_numpy(),
            "previous_close": previous_close.loc[event_rows].to_numpy(),
            "event_open": frame.loc[event_rows, "Open"].to_numpy(),
            "event_high": frame.loc[event_rows, "High"].to_numpy(),
            "event_low": frame.loc[event_rows, "Low"].to_numpy(),
            "event_close": frame.loc[event_rows, "Close"].to_numpy(),
            "event_volume": frame.loc[event_rows, "Volume"].to_numpy(),
            "gap_pct": gap_pct.loc[event_rows].to_numpy(),
            "prior_addv_63": prior_addv_63.loc[event_rows].to_numpy(),
            "event_rvol_20": event_rvol_20.loc[event_rows].to_numpy(),
            "prior_63d_return_pct": prior_63d_return_pct.loc[event_rows].to_numpy(),
            "data_quality_clean": (~suspicious_nearby.loc[event_rows]).to_numpy(),
        }
    )
    earnings_set = {pd.Timestamp(value).normalize() for value in earnings_dates}
    events["earnings_date_match"] = [
        pd.Timestamp(day).normalize() in earnings_set
        or pd.Timestamp(previous).normalize() in earnings_set
        for day, previous in zip(events["date"], events["previous_session"])
    ]
    events["gap_band"] = [_gap_band(value) for value in events["gap_pct"]]
    events["rvol_band"] = [_rvol_band(value) for value in events["event_rvol_20"]]
    events["era"] = [_era(pd.Timestamp(value).year) for value in events["date"]]
    events["sample_period"] = [
        _sample_period(pd.Timestamp(value).year) for value in events["date"]
    ]

    if include_outcomes:
        events["event_day_open_to_close_pct"] = 100.0 * (
            events["event_close"] / events["event_open"] - 1.0
        )
        for horizon in policy.horizons:
            close_forward = frame["Close"].shift(-horizon)
            next_open = frame["Open"].shift(-1)
            confirmed_close = frame["Close"].shift(-horizon)
            events[f"open_to_close_{horizon}d_pct"] = (
                100.0
                * (close_forward.loc[event_rows].to_numpy() / events["event_open"].to_numpy() - 1.0)
            )
            events[f"next_open_to_close_{horizon}d_pct"] = (
                100.0
                * (
                    confirmed_close.loc[event_rows].to_numpy()
                    / next_open.loc[event_rows].to_numpy()
                    - 1.0
                )
            )

        # MFE/MAE use the longest fixed horizon and are left NaN when censored.
        longest = max(policy.horizons)
        mfe = []
        mae = []
        for row_index, entry in zip(event_rows, events["event_open"]):
            window = frame.iloc[row_index : row_index + longest + 1]
            if len(window) < longest + 1:
                mfe.append(np.nan)
                mae.append(np.nan)
            else:
                mfe.append(100.0 * (float(window["High"].max()) / entry - 1.0))
                mae.append(100.0 * (float(window["Low"].min()) / entry - 1.0))
        events[f"mfe_{longest}d_pct"] = mfe
        events[f"mae_{longest}d_pct"] = mae

    return TickerStudyResult(events, anomalies, counts)


def load_observed_panel(
    master_path: str | Path,
    overflow_path: str | Path | None = None,
) -> pd.DataFrame:
    master = pd.read_parquet(master_path, columns=_PRICE_COLUMNS)
    master["_source_priority"] = 0
    frames = [master]
    if overflow_path and Path(overflow_path).exists():
        overflow = pd.read_parquet(overflow_path, columns=_PRICE_COLUMNS)
        overflow["_source_priority"] = 1
        frames.append(overflow)
    panel = pd.concat(frames, ignore_index=True)
    panel["ticker"] = panel["ticker"].astype(str).str.upper().str.strip()
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    panel = panel.sort_values(["ticker", "date", "_source_priority"])
    panel = panel.drop_duplicates(["ticker", "date"], keep="first")
    return panel.drop(columns="_source_priority")


def load_earnings_map(path: str | Path | None) -> dict[str, set[pd.Timestamp]]:
    if not path or not Path(path).exists():
        return {}
    frame = pd.read_parquet(path, columns=["ticker", "date"])
    frame["ticker"] = frame["ticker"].astype(str).str.upper().str.strip()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame = frame.dropna().drop_duplicates(["ticker", "date"])
    return {
        ticker: set(group["date"].tolist())
        for ticker, group in frame.groupby("ticker", sort=False)
    }


def load_current_company_symbols(path: str | Path) -> set[str]:
    """Return the current FMP operating-company slice.

    This removes ETFs, FX, indices, and futures from the diagnostic panel but is
    still explicitly survivor-biased; it must not be presented as historical
    point-in-time membership.
    """

    frame = pd.read_parquet(path)
    mask = pd.Series(True, index=frame.index)
    for column in ("isEtf", "isFund"):
        if column in frame:
            mask &= ~frame[column].fillna(False).astype(bool)
    if "isActivelyTrading" in frame:
        mask &= frame["isActivelyTrading"].fillna(False).astype(bool)
    symbols = frame.loc[mask, "ticker"].astype(str).str.upper().str.strip()
    return set(symbols)


def run_observed_panel_study(
    panel: pd.DataFrame,
    *,
    policy: HistoricalPolicy,
    earnings_map: dict[str, set[pd.Timestamp]] | None = None,
    include_outcomes: bool = False,
    benchmark_prices: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    events = []
    anomalies = []
    totals = {
        "tickers": 0,
        "bars": 0,
        "open_observable": 0,
        "volume_confirmed_ex_post": 0,
        "first_confirmed_in_126": 0,
        "strict_with_neglect": 0,
        "anomalies": 0,
    }
    earnings_map = earnings_map or {}
    for ticker, frame in panel.groupby("ticker", sort=True):
        result = study_ticker(
            ticker,
            frame,
            policy=policy,
            earnings_dates=earnings_map.get(ticker, ()),
            include_outcomes=include_outcomes,
        )
        totals["tickers"] += 1
        for key, value in result.counts.items():
            totals[key] += value
        if not result.events.empty:
            events.append(result.events)
        if not result.anomalies.empty:
            anomalies.append(result.anomalies)

    event_frame = pd.concat(events, ignore_index=True) if events else _empty_events()
    anomaly_frame = pd.concat(anomalies, ignore_index=True) if anomalies else pd.DataFrame()
    if not event_frame.empty:
        cluster = event_frame.groupby("date")["ticker"].transform("size")
        event_frame["event_date_cluster_size"] = cluster
        event_frame["market_wide_cluster"] = cluster >= 10
        if include_outcomes and benchmark_prices is not None and not benchmark_prices.empty:
            event_frame = attach_benchmark_outcomes(
                event_frame, benchmark_prices, policy=policy
            )
    return event_frame, anomaly_frame, totals


def attach_benchmark_outcomes(
    events: pd.DataFrame,
    benchmark_prices: pd.DataFrame,
    *,
    policy: HistoricalPolicy,
) -> pd.DataFrame:
    """Attach SPY-like fixed-horizon returns and simple arithmetic excess."""

    benchmark = benchmark_prices.copy()
    benchmark["date"] = pd.to_datetime(benchmark["date"]).dt.normalize()
    benchmark = benchmark.sort_values("date").drop_duplicates("date", keep="last")
    benchmark = benchmark.set_index("date")
    lookup = pd.DataFrame(index=benchmark.index)
    lookup["benchmark_event_day_open_to_close_pct"] = 100.0 * (
        benchmark["Close"] / benchmark["Open"] - 1.0
    )
    for horizon in policy.horizons:
        lookup[f"benchmark_open_to_close_{horizon}d_pct"] = 100.0 * (
            benchmark["Close"].shift(-horizon) / benchmark["Open"] - 1.0
        )
        lookup[f"benchmark_next_open_to_close_{horizon}d_pct"] = 100.0 * (
            benchmark["Close"].shift(-horizon) / benchmark["Open"].shift(-1) - 1.0
        )
    out = events.merge(lookup.reset_index(), on="date", how="left")
    if "event_day_open_to_close_pct" in out:
        out["excess_event_day_open_to_close_pct"] = (
            out["event_day_open_to_close_pct"]
            - out["benchmark_event_day_open_to_close_pct"]
        )
    for horizon in policy.horizons:
        event_column = f"open_to_close_{horizon}d_pct"
        confirmed_column = f"next_open_to_close_{horizon}d_pct"
        if event_column in out:
            out[f"excess_open_to_close_{horizon}d_pct"] = (
                out[event_column] - out[f"benchmark_open_to_close_{horizon}d_pct"]
            )
        if confirmed_column in out:
            out[f"excess_next_open_to_close_{horizon}d_pct"] = (
                out[confirmed_column]
                - out[f"benchmark_next_open_to_close_{horizon}d_pct"]
            )
    return out


def diagnostic_summary(events: pd.DataFrame) -> pd.DataFrame:
    outcome_columns = [column for column in events if column.endswith("d_pct")]
    if events.empty or not outcome_columns:
        return pd.DataFrame()
    clean = events[events["data_quality_clean"]].copy()
    groups = [
        "sample_period",
        "era",
        "gap_band",
        "rvol_band",
        "earnings_date_match",
    ]
    rows = []
    for keys, frame in clean.groupby(groups, dropna=False):
        row = dict(zip(groups, keys))
        row["n"] = len(frame)
        for column in outcome_columns:
            row[f"{column}_mean"] = frame[column].mean()
            row[f"{column}_median"] = frame[column].median()
        rows.append(row)
    return pd.DataFrame(rows)


def clustered_outcome_summary(
    events: pd.DataFrame,
    *,
    cluster_column: str,
    bootstrap_samples: int = 2_000,
    seed: int = 17,
) -> pd.DataFrame:
    """Equal-weight clusters and bootstrap their mean; never treat rows as IID."""

    if events.empty or cluster_column not in events:
        return pd.DataFrame()
    clean = events[events["data_quality_clean"]].copy()
    outcome_columns = [
        column
        for column in clean
        if column.startswith("excess_") and column.endswith("_pct")
    ]
    if not outcome_columns:
        return pd.DataFrame()
    cluster_means = clean.groupby(cluster_column, dropna=False)[outcome_columns].mean()
    rng = np.random.default_rng(seed)
    rows = []
    for column in outcome_columns:
        values = cluster_means[column].dropna().to_numpy(dtype=float)
        if not len(values):
            continue
        if len(values) == 1:
            low = high = float(values[0])
        else:
            draws = rng.choice(values, size=(bootstrap_samples, len(values)), replace=True)
            boot_means = draws.mean(axis=1)
            low, high = np.quantile(boot_means, [0.025, 0.975])
        rows.append(
            {
                "cluster_basis": cluster_column,
                "outcome": column,
                "n_events": int(clean[column].notna().sum()),
                "n_clusters": int(len(values)),
                "equal_weight_cluster_mean": float(values.mean()),
                "equal_weight_cluster_median": float(np.median(values)),
                "bootstrap_mean_ci_2_5": float(low),
                "bootstrap_mean_ci_97_5": float(high),
            }
        )
    return pd.DataFrame(rows)
