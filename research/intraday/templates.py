"""Pre-registered universal signal templates for the intraday research lab."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import time
from math import isfinite

import numpy as np
import pandas as pd

from .data import BAR_DELTA, normalize_frame_map
from .eligibility import EligibilityConfig, calculate_eligibility

GAP_FIRST_HOUR_TEMPLATE_ID = "gap_first_hour_residual_continuation_v0"
INTRADAY_SHOCK_TEMPLATE_ID = "intraday_residual_shock_reversal_v0"

DEFAULT_SECTOR_PROXIES = {
    "BASIC MATERIALS": "XLB",
    "COMMUNICATION SERVICES": "XLC",
    "CONSUMER CYCLICAL": "XLY",
    "CONSUMER DEFENSIVE": "XLP",
    "ENERGY": "XLE",
    "FINANCIAL SERVICES": "XLF",
    "HEALTHCARE": "XLV",
    "INDUSTRIALS": "XLI",
    "REAL ESTATE": "XLRE",
    "TECHNOLOGY": "XLK",
    "UTILITIES": "XLU",
    "COMMODITY": "DBC",
}


@dataclass(frozen=True)
class GapFirstHourConfig:
    decision_time: str = "10:30"
    entry_time: str = "10:45"
    exit_bar_time: str = "15:45"
    min_abs_residual_gap: float = 0.01
    min_abs_residual_first_hour: float = 0.005
    market_weight: float = 0.5

    def __post_init__(self) -> None:
        decision = _parse_clock(self.decision_time)
        entry = _parse_clock(self.entry_time)
        exit_bar = _parse_clock(self.exit_bar_time)
        if not _minutes(decision) < _minutes(entry) < _minutes(exit_bar):
            raise ValueError("gap clocks must satisfy decision < entry < exit bar")
        if (
            not isfinite(self.min_abs_residual_gap)
            or not isfinite(self.min_abs_residual_first_hour)
            or self.min_abs_residual_gap < 0
            or self.min_abs_residual_first_hour < 0
        ):
            raise ValueError("gap thresholds cannot be negative")
        if not isfinite(self.market_weight) or not 0 <= self.market_weight <= 1:
            raise ValueError("market_weight must be in [0, 1]")


@dataclass(frozen=True)
class IntradayShockConfig:
    evaluation_bar_time: str = "13:00"
    entry_time: str = "13:30"
    exit_bar_time: str = "15:45"
    min_abs_residual_shock: float = 0.015
    market_weight: float = 0.5

    def __post_init__(self) -> None:
        evaluation_bar = _parse_clock(self.evaluation_bar_time)
        entry = _parse_clock(self.entry_time)
        exit_bar = _parse_clock(self.exit_bar_time)
        feature_available_minutes = _minutes(evaluation_bar) + 15
        if not feature_available_minutes < _minutes(entry) < _minutes(exit_bar):
            raise ValueError(
                "shock clocks must satisfy evaluation-bar close < entry < exit bar"
            )
        if not isfinite(self.min_abs_residual_shock) or self.min_abs_residual_shock < 0:
            raise ValueError("shock threshold cannot be negative")
        if not isfinite(self.market_weight) or not 0 <= self.market_weight <= 1:
            raise ValueError("market_weight must be in [0, 1]")


def _parse_clock(clock_text: str) -> time:
    try:
        parsed = time.fromisoformat(clock_text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid event-clock time: {clock_text!r}") from exc
    if parsed.second or parsed.microsecond:
        raise ValueError("event-clock times must align to whole minutes")
    if parsed.minute % 15:
        raise ValueError("event-clock times must align to the 15-minute grid")
    return parsed


def _minutes(clock_value: time) -> int:
    return clock_value.hour * 60 + clock_value.minute


def prepare_metadata(
    tickers: Iterable[str],
    metadata: pd.DataFrame | None,
    *,
    market_ticker: str,
) -> pd.DataFrame:
    """Normalize ticker/sector/sector-proxy metadata.

    ``sector_proxy`` is optional.  Known sectors use a fixed SPDR mapping;
    unknown sectors fall back to the market proxy and remain visibly labeled
    ``UNKNOWN`` in every signal row.
    """

    ticker_set = {str(ticker).upper() for ticker in tickers}
    if metadata is None or metadata.empty:
        meta = pd.DataFrame({"ticker": sorted(ticker_set), "sector": "UNKNOWN"})
    else:
        meta = metadata.copy()
        meta.columns = [str(column).strip().lower() for column in meta.columns]
        if "ticker" not in meta.columns:
            raise ValueError("metadata must contain a ticker column")
        if "sector" not in meta.columns:
            meta["sector"] = "UNKNOWN"
        meta["ticker"] = meta["ticker"].astype(str).str.upper().str.strip()
        meta = meta.drop_duplicates("ticker", keep="last")
        meta = meta.loc[meta["ticker"].isin(ticker_set)].copy()
        missing = ticker_set.difference(meta["ticker"])
        if missing:
            meta = pd.concat(
                [meta, pd.DataFrame({"ticker": sorted(missing), "sector": "UNKNOWN"})],
                ignore_index=True,
            )
    meta["sector"] = meta["sector"].fillna("UNKNOWN").astype(str).str.strip()
    if "sector_proxy" not in meta.columns:
        meta["sector_proxy"] = pd.NA
    mapped_proxy = meta["sector"].str.upper().map(DEFAULT_SECTOR_PROXIES)
    meta["sector_proxy"] = (
        meta["sector_proxy"]
        .fillna(mapped_proxy)
        .fillna(market_ticker)
        .astype(str)
        .str.upper()
    )
    return meta[["ticker", "sector", "sector_proxy"]].reset_index(drop=True)


def _clock(day: pd.Timestamp, clock_text: str) -> pd.Timestamp:
    parsed = _parse_clock(clock_text)
    return day.normalize() + pd.Timedelta(hours=parsed.hour, minutes=parsed.minute)


def _sessions(frame: pd.DataFrame) -> dict[pd.Timestamp, pd.DataFrame]:
    return {
        pd.Timestamp(day): group.reset_index(drop=True)
        for day, group in frame.groupby(frame["ts"].dt.normalize(), sort=True)
    }


def _bar_at(session: pd.DataFrame, ts: pd.Timestamp) -> pd.Series | None:
    match = session.loc[session["ts"].eq(ts)]
    return None if match.empty else match.iloc[0]


def _open_and_return_through(
    session: pd.DataFrame,
    evaluation_bar_ts: pd.Timestamp,
) -> tuple[float, float] | None:
    open_bar = _bar_at(
        session, evaluation_bar_ts.normalize() + pd.Timedelta(hours=9, minutes=30)
    )
    evaluation_bar = _bar_at(session, evaluation_bar_ts)
    if open_bar is None or evaluation_bar is None:
        return None
    session_open = float(open_bar["open"])
    cumulative_return = float(evaluation_bar["close"] / session_open - 1.0)
    return session_open, cumulative_return


def _scheduled_close(session: pd.DataFrame, day: pd.Timestamp) -> float | None:
    close_bar = _bar_at(session, _clock(day, "15:45"))
    return None if close_bar is None else float(close_bar["close"])


def _residual(
    asset_return: float,
    market_return: float,
    sector_return: float,
    *,
    market_weight: float,
    same_proxy: bool,
) -> float:
    if same_proxy:
        reference = market_return
    else:
        reference = (
            market_weight * market_return + (1.0 - market_weight) * sector_return
        )
    return asset_return - reference


def _eligibility_lookup(
    eligibility: pd.DataFrame,
) -> dict[tuple[str, pd.Timestamp], dict]:
    lookup: dict[tuple[str, pd.Timestamp], dict] = {}
    for row in eligibility.to_dict("records"):
        lookup[
            (str(row["ticker"]).upper(), pd.Timestamp(row["trade_date"]).normalize())
        ] = row
    return lookup


def _candidate_tickers(
    frames: Mapping[str, pd.DataFrame],
    meta: pd.DataFrame,
    market_ticker: str,
    candidates: Iterable[str] | None,
) -> list[str]:
    if candidates is not None:
        requested = {str(ticker).upper() for ticker in candidates}
        missing = requested.difference(frames)
        if missing:
            raise ValueError(f"explicit candidate frame(s) missing: {sorted(missing)}")
        return sorted(requested)
    proxies = set(meta["sector_proxy"].astype(str).str.upper()) | {market_ticker}
    return sorted(set(frames).difference(proxies))


def generate_gap_first_hour_signals(
    frames: Mapping[str, pd.DataFrame],
    metadata: pd.DataFrame | None = None,
    *,
    market_ticker: str = "SPY",
    candidates: Iterable[str] | None = None,
    eligibility: pd.DataFrame | None = None,
    eligibility_config: EligibilityConfig | None = None,
    config: GapFirstHourConfig | None = None,
    frames_are_normalized: bool = False,
) -> pd.DataFrame:
    """Generate aligned residual-gap/first-hour continuation candidates.

    The first-hour snapshot uses the four completed bars through 10:30 ET,
    decides at 10:30, and leaves a full bar before entering at 10:45. The rule enters in
    the direction of the residual overnight gap only when the response confirms
    that direction.
    """

    eligibility_config = eligibility_config or EligibilityConfig()
    config = config or GapFirstHourConfig()
    market_ticker = market_ticker.upper()
    bars = dict(frames) if frames_are_normalized else normalize_frame_map(frames)
    if market_ticker not in bars:
        raise ValueError(f"market proxy {market_ticker} is missing")
    meta = prepare_metadata(bars, metadata, market_ticker=market_ticker)
    meta_by_ticker = meta.set_index("ticker").to_dict("index")
    eligibility = (
        eligibility
        if eligibility is not None
        else calculate_eligibility(
            bars,
            eligibility_config,
            calendar_ticker=market_ticker,
            frames_are_normalized=True,
        )
    )
    eligible = _eligibility_lookup(eligibility)
    sessions = {ticker: _sessions(frame) for ticker, frame in bars.items()}
    canonical_days = sorted(
        {day for ticker_sessions in sessions.values() for day in ticker_sessions}
    )
    previous_market_day = {
        day: canonical_days[index - 1] if index else None
        for index, day in enumerate(canonical_days)
    }
    output: list[dict] = []

    candidate_list = _candidate_tickers(bars, meta, market_ticker, candidates)
    missing_proxies = sorted(
        {
            str(meta_by_ticker[ticker]["sector_proxy"]).upper()
            for ticker in candidate_list
        }.difference(bars)
    )
    if missing_proxies:
        raise ValueError(f"required sector proxy frame(s) missing: {missing_proxies}")
    for ticker in candidate_list:
        details = meta_by_ticker[ticker]
        sector_proxy = str(details["sector_proxy"]).upper()
        for day, asset_day in sessions[ticker].items():
            gate = eligible.get((ticker, day))
            if gate is None or not bool(gate["eligible"]):
                continue
            market_day = sessions[market_ticker].get(day)
            sector_day = sessions[sector_proxy].get(day)
            if market_day is None or sector_day is None:
                continue

            decision_ts = _clock(day, config.decision_time)
            entry_bar_ts = _clock(day, config.entry_time)
            feature_bar_ts = decision_ts - BAR_DELTA
            asset_move = _open_and_return_through(asset_day, feature_bar_ts)
            market_move = _open_and_return_through(market_day, feature_bar_ts)
            sector_move = _open_and_return_through(sector_day, feature_bar_ts)
            if asset_move is None or market_move is None or sector_move is None:
                continue

            prior_day = previous_market_day.get(day)
            if prior_day is None:
                continue
            asset_prior_day = sessions[ticker].get(prior_day)
            market_prior_day = sessions[market_ticker].get(prior_day)
            sector_prior_day = sessions[sector_proxy].get(prior_day)
            if (
                asset_prior_day is None
                or market_prior_day is None
                or sector_prior_day is None
            ):
                continue
            asset_prior = _scheduled_close(asset_prior_day, prior_day)
            market_prior = _scheduled_close(market_prior_day, prior_day)
            sector_prior = _scheduled_close(sector_prior_day, prior_day)
            if asset_prior is None or market_prior is None or sector_prior is None:
                continue
            asset_open, asset_first_hour = asset_move
            market_open, market_first_hour = market_move
            sector_open, sector_first_hour = sector_move
            asset_gap = asset_open / asset_prior - 1.0
            market_gap = market_open / market_prior - 1.0
            sector_gap = sector_open / sector_prior - 1.0
            same_proxy = sector_proxy in {market_ticker, ticker}
            residual_gap = _residual(
                asset_gap,
                market_gap,
                sector_gap,
                market_weight=config.market_weight,
                same_proxy=same_proxy,
            )
            residual_first_hour = _residual(
                asset_first_hour,
                market_first_hour,
                sector_first_hour,
                market_weight=config.market_weight,
                same_proxy=same_proxy,
            )
            aligned = residual_gap * residual_first_hour > 0
            if not (
                aligned
                and abs(residual_gap) >= config.min_abs_residual_gap
                and abs(residual_first_hour) >= config.min_abs_residual_first_hour
            ):
                continue

            side = int(np.sign(residual_gap))
            exit_bar_ts = _clock(day, config.exit_bar_time)
            output.append(
                {
                    "template_id": GAP_FIRST_HOUR_TEMPLATE_ID,
                    "ticker": ticker,
                    "sector": details["sector"],
                    "sector_proxy": sector_proxy,
                    "trade_date": day,
                    "side": side,
                    "decision_ts": decision_ts,
                    "feature_bar_ts": feature_bar_ts,
                    "feature_available_ts": feature_bar_ts + BAR_DELTA,
                    "entry_bar_ts": entry_bar_ts,
                    "entry_ts": entry_bar_ts,
                    "exit_bar_ts": exit_bar_ts,
                    "exit_ts": exit_bar_ts + BAR_DELTA,
                    "asset_gap": asset_gap,
                    "market_gap": market_gap,
                    "sector_gap": sector_gap,
                    "residual_gap": residual_gap,
                    "asset_first_hour": asset_first_hour,
                    "market_first_hour": market_first_hour,
                    "sector_first_hour": sector_first_hour,
                    "residual_first_hour": residual_first_hour,
                    "signal_strength": abs(residual_gap) + abs(residual_first_hour),
                    "price_proxy": gate["price_proxy"],
                    "median_dollar_volume": gate["median_dollar_volume"],
                    "data_completeness": gate["data_completeness"],
                }
            )
    return pd.DataFrame(output)


def generate_intraday_shock_signals(
    frames: Mapping[str, pd.DataFrame],
    metadata: pd.DataFrame | None = None,
    *,
    market_ticker: str = "SPY",
    candidates: Iterable[str] | None = None,
    eligibility: pd.DataFrame | None = None,
    eligibility_config: EligibilityConfig | None = None,
    config: IntradayShockConfig | None = None,
    frames_are_normalized: bool = False,
) -> pd.DataFrame:
    """Generate residual-shock reversal candidates at a fixed event clock.

    The bar labelled 13:00 closes at 13:15; the template then leaves a full bar
    before entering at 13:30. It measures the cumulative session-open move and
    The pre-registered direction fades the market/sector-adjusted shock.
    """

    eligibility_config = eligibility_config or EligibilityConfig()
    config = config or IntradayShockConfig()
    market_ticker = market_ticker.upper()
    bars = dict(frames) if frames_are_normalized else normalize_frame_map(frames)
    if market_ticker not in bars:
        raise ValueError(f"market proxy {market_ticker} is missing")
    meta = prepare_metadata(bars, metadata, market_ticker=market_ticker)
    meta_by_ticker = meta.set_index("ticker").to_dict("index")
    eligibility = (
        eligibility
        if eligibility is not None
        else calculate_eligibility(
            bars,
            eligibility_config,
            calendar_ticker=market_ticker,
            frames_are_normalized=True,
        )
    )
    eligible = _eligibility_lookup(eligibility)
    sessions = {ticker: _sessions(frame) for ticker, frame in bars.items()}
    output: list[dict] = []

    candidate_list = _candidate_tickers(bars, meta, market_ticker, candidates)
    missing_proxies = sorted(
        {
            str(meta_by_ticker[ticker]["sector_proxy"]).upper()
            for ticker in candidate_list
        }.difference(bars)
    )
    if missing_proxies:
        raise ValueError(f"required sector proxy frame(s) missing: {missing_proxies}")
    for ticker in candidate_list:
        details = meta_by_ticker[ticker]
        sector_proxy = str(details["sector_proxy"]).upper()
        for day, asset_day in sessions[ticker].items():
            gate = eligible.get((ticker, day))
            if gate is None or not bool(gate["eligible"]):
                continue
            market_day = sessions[market_ticker].get(day)
            sector_day = sessions[sector_proxy].get(day)
            if market_day is None or sector_day is None:
                continue

            evaluation_bar_ts = _clock(day, config.evaluation_bar_time)
            feature_available_ts = evaluation_bar_ts + BAR_DELTA
            entry_bar_ts = _clock(day, config.entry_time)
            asset_move = _open_and_return_through(asset_day, evaluation_bar_ts)
            market_move = _open_and_return_through(market_day, evaluation_bar_ts)
            sector_move = _open_and_return_through(sector_day, evaluation_bar_ts)
            if asset_move is None or market_move is None or sector_move is None:
                continue
            _, asset_shock = asset_move
            _, market_shock = market_move
            _, sector_shock = sector_move
            residual_shock = _residual(
                asset_shock,
                market_shock,
                sector_shock,
                market_weight=config.market_weight,
                same_proxy=sector_proxy in {market_ticker, ticker},
            )
            if abs(residual_shock) < config.min_abs_residual_shock:
                continue

            exit_bar_ts = _clock(day, config.exit_bar_time)
            output.append(
                {
                    "template_id": INTRADAY_SHOCK_TEMPLATE_ID,
                    "ticker": ticker,
                    "sector": details["sector"],
                    "sector_proxy": sector_proxy,
                    "trade_date": day,
                    "side": -int(np.sign(residual_shock)),
                    "decision_ts": feature_available_ts,
                    "feature_bar_ts": evaluation_bar_ts,
                    "feature_available_ts": feature_available_ts,
                    "entry_bar_ts": entry_bar_ts,
                    "entry_ts": entry_bar_ts,
                    "exit_bar_ts": exit_bar_ts,
                    "exit_ts": exit_bar_ts + BAR_DELTA,
                    "asset_shock": asset_shock,
                    "market_shock": market_shock,
                    "sector_shock": sector_shock,
                    "residual_shock": residual_shock,
                    "signal_strength": abs(residual_shock),
                    "price_proxy": gate["price_proxy"],
                    "median_dollar_volume": gate["median_dollar_volume"],
                    "data_completeness": gate["data_completeness"],
                }
            )
    return pd.DataFrame(output)
