"""Build and analyze a leveraged-ETF flow pressure monitor.

The daily official fund files are end-of-day observations. They are therefore
used only to forecast the next session and later. The intraday feature uses
only prior-day AUM plus the proxy return available at 15:30 ET.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_DIR / "data"
OUTPUT_DIR = PACKAGE_DIR / "outputs"
MASTER_PRICES = PROJECT_ROOT / "data" / "master_prices.parquet"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cache_io import download_to_local  # noqa: E402
from research.letf_flow_monitor.config import (  # noqa: E402
    FUNDS,
    PROSHARES_URL,
    PROXIES,
    FundSpec,
)


def _numeric(series: pd.Series) -> pd.Series:
    """Parse dollar/comma/percent formatted numeric columns."""
    return pd.to_numeric(
        series.astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip()
        .replace({"": np.nan, "-": np.nan, "nan": np.nan}),
        errors="coerce",
    )


def split_adjusted_share_flow(
    shares_k: pd.Series,
    nav: pd.Series,
    prior_nav: pd.Series,
    split_tolerance: float = 0.20,
) -> pd.DataFrame:
    """Return split-adjusted share change and its supporting diagnostics.

    On a split day, the file's current ``Prior NAV`` is on the new share
    basis while the preceding row's ``NAV`` is on the old basis. The ratio
    converts preceding shares to the current basis. Small differences are
    ignored because ordinary prior NAV should simply equal the previous NAV.
    """
    prev_shares = shares_k.shift(1)
    prev_nav = nav.shift(1)
    raw_ratio = prev_nav / prior_nav.replace(0, np.nan)
    is_split = (raw_ratio < 1 - split_tolerance) | (
        raw_ratio > 1 + split_tolerance
    )
    split_factor = raw_ratio.where(is_split, 1.0).replace(
        [np.inf, -np.inf], np.nan
    )
    adjusted_prev_shares = prev_shares * split_factor
    delta_shares_k = shares_k - adjusted_prev_shares
    return pd.DataFrame(
        {
            "previous_shares_k_adjusted": adjusted_prev_shares,
            "split_factor": split_factor,
            "split_detected": is_split.fillna(False),
            "delta_shares_k": delta_shares_k,
        },
        index=shares_k.index,
    )


def parse_fund_history(raw: bytes, spec: FundSpec) -> pd.DataFrame:
    """Normalize one official ProShares historical NAV CSV."""
    df = pd.read_csv(io.BytesIO(raw))
    required = {
        "Date",
        "NAV",
        "Prior NAV",
        "Shares Outstanding (000)",
        "Assets Under Management",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{spec.ticker}: missing columns {sorted(missing)}")

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df["Date"], errors="coerce").dt.normalize(),
            "nav": _numeric(df["NAV"]),
            "prior_nav": _numeric(df["Prior NAV"]),
            "shares_outstanding_k": _numeric(df["Shares Outstanding (000)"]),
            "aum_usd": _numeric(df["Assets Under Management"]),
        }
    ).dropna(subset=["date", "nav", "shares_outstanding_k"])
    out = out.sort_values("date").drop_duplicates("date", keep="last")
    out = out.reset_index(drop=True)

    split = split_adjusted_share_flow(
        out["shares_outstanding_k"], out["nav"], out["prior_nav"]
    )
    out = pd.concat([out, split], axis=1)
    out["primary_flow_usd"] = out["delta_shares_k"] * 1_000 * out["nav"]
    out["ticker"] = spec.ticker
    out["benchmark"] = spec.benchmark
    out["proxy"] = spec.proxy
    out["leverage"] = spec.leverage
    out["source_url"] = PROSHARES_URL.format(ticker=spec.ticker)
    return out


def fetch_proshares_history(
    refresh: bool = False,
    timeout: int = 30,
) -> pd.DataFrame:
    """Download official fund histories, with a local raw-file cache."""
    raw_dir = DATA_DIR / "raw_proshares"
    raw_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    failures: list[str] = []
    session = requests.Session()
    session.headers.update({"User-Agent": "NewSeasonals-LETF-flow-research/1.0"})

    for spec in FUNDS:
        path = raw_dir / f"{spec.ticker}-historical_nav.csv"
        if refresh or not path.exists():
            try:
                response = session.get(
                    PROSHARES_URL.format(ticker=spec.ticker), timeout=timeout
                )
                response.raise_for_status()
                path.write_bytes(response.content)
            except Exception as exc:
                if not path.exists():
                    failures.append(f"{spec.ticker}: {type(exc).__name__}: {exc}")
                    continue
                print(
                    f"[warning] refresh failed for {spec.ticker}; using cache: {exc}",
                    file=sys.stderr,
                )
        try:
            frames.append(parse_fund_history(path.read_bytes(), spec))
        except Exception as exc:
            failures.append(f"{spec.ticker}: {type(exc).__name__}: {exc}")

    if failures:
        raise RuntimeError("Official flow history failures:\n" + "\n".join(failures))
    history = pd.concat(frames, ignore_index=True)
    history = history.sort_values(["date", "benchmark", "ticker"])
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    history.to_parquet(DATA_DIR / "fund_history.parquet", index=False)
    return history


def load_proxy_daily(
    intraday_cache: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    if not MASTER_PRICES.exists():
        raise FileNotFoundError(f"Missing {MASTER_PRICES}")
    prices = pd.read_parquet(
        MASTER_PRICES,
        filters=[("ticker", "in", list(PROXIES))],
        columns=["ticker", "date", "Open", "High", "Low", "Close", "Volume"],
    )
    prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()
    prices = prices.rename(columns={"ticker": "proxy"})
    # The official fund file can arrive one session ahead of the nightly daily
    # cache. Fill only those missing tail sessions from the regular-hours
    # intraday cache so the latest report is not a row of NaNs.
    if intraday_cache:
        additions: list[pd.DataFrame] = []
        for proxy, bars in intraday_cache.items():
            if bars is None or bars.empty:
                continue
            tail = bars.copy()
            tail["date"] = tail["ts"].dt.normalize()
            tail = tail.groupby("date", as_index=False).agg(
                Open=("open", "first"),
                High=("high", "max"),
                Low=("low", "min"),
                Close=("close", "last"),
                Volume=("volume", "sum"),
            )
            latest_daily = prices.loc[prices["proxy"] == proxy, "date"].max()
            tail = tail[tail["date"] > latest_daily]
            if not tail.empty:
                tail["proxy"] = proxy
                additions.append(tail)
        if additions:
            prices = pd.concat([prices, *additions], ignore_index=True)
    prices = prices.sort_values(["proxy", "date"])
    group = prices.groupby("proxy", group_keys=False)
    prices["proxy_return"] = group["Close"].pct_change()
    prices["daily_dollar_volume"] = prices["Close"] * prices["Volume"]
    prices["trailing_vol_20"] = group["proxy_return"].transform(
        lambda x: x.shift(1).rolling(20, min_periods=15).std()
    )
    for horizon in (1, 3, 5):
        prices[f"forward_return_{horizon}d"] = group["Close"].transform(
            lambda x, h=horizon: x.shift(-h) / x - 1
        )
        prices[f"forward_rv_{horizon}d"] = group["proxy_return"].transform(
            lambda x, h=horizon: np.sqrt(
                sum(x.shift(-i).pow(2) for i in range(1, h + 1)) / h
            )
        )
    prices["forward_abs_return_1d"] = prices["forward_return_1d"].abs()
    prices["forward_range_1d"] = (
        group["High"].shift(-1) - group["Low"].shift(-1)
    ) / prices["Close"]
    return prices


def trailing_zscore(series: pd.Series, window: int = 252) -> pd.Series:
    """Point-in-time z-score using only observations before the current row."""
    prior = series.shift(1)
    mean = prior.rolling(window, min_periods=max(60, window // 2)).mean()
    std = prior.rolling(window, min_periods=max(60, window // 2)).std()
    return (series - mean) / std.replace(0, np.nan)


def prior_quantile(
    series: pd.Series,
    q: float,
    window: int = 252,
) -> pd.Series:
    """Point-in-time rolling threshold that excludes the current row."""
    return series.shift(1).rolling(
        window, min_periods=max(60, window // 2)
    ).quantile(q)


def build_daily_features(
    history: pd.DataFrame,
    intraday_cache: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    prices = load_proxy_daily(intraday_cache=intraday_cache)
    fund = history.copy()
    fund["prior_aum_usd"] = fund.groupby("ticker")["aum_usd"].shift(1)
    fund = fund.merge(
        prices[["proxy", "date", "proxy_return"]],
        on=["proxy", "date"],
        how="left",
        validate="many_to_one",
    )
    fund["mechanical_demand_usd"] = (
        fund["prior_aum_usd"]
        * fund["leverage"]
        * (fund["leverage"] - 1)
        * fund["proxy_return"]
    )
    fund["flow_exposure_usd"] = fund["leverage"] * fund["primary_flow_usd"]
    fund["estimated_demand_usd"] = (
        fund["mechanical_demand_usd"] + fund["flow_exposure_usd"]
    )
    fund["bull_flow_exposure_usd"] = fund["flow_exposure_usd"].where(
        fund["leverage"] > 0, 0.0
    )
    fund["inverse_flow_exposure_usd"] = fund["flow_exposure_usd"].where(
        fund["leverage"] < 0, 0.0
    )
    fund["gross_primary_flow_usd"] = fund["primary_flow_usd"].abs()
    fund["rebalance_coefficient_usd"] = (
        fund["prior_aum_usd"] * fund["leverage"] * (fund["leverage"] - 1)
    )
    fund.to_parquet(DATA_DIR / "fund_pressure.parquet", index=False)

    cols = [
        "aum_usd",
        "prior_aum_usd",
        "primary_flow_usd",
        "gross_primary_flow_usd",
        "flow_exposure_usd",
        "bull_flow_exposure_usd",
        "inverse_flow_exposure_usd",
        "mechanical_demand_usd",
        "estimated_demand_usd",
        "rebalance_coefficient_usd",
    ]
    keys = ["benchmark", "proxy", "date"]
    daily = fund.groupby(keys, as_index=False)[cols].sum(min_count=1)
    counts = fund.groupby(keys, as_index=False)["ticker"].nunique().rename(
        columns={"ticker": "fund_count"}
    )
    daily = daily.merge(counts, on=keys, how="left", validate="one_to_one")
    expected_counts = pd.Series(
        {benchmark: sum(f.benchmark == benchmark for f in FUNDS)
         for benchmark in {f.benchmark for f in FUNDS}}
    )
    daily["expected_fund_count"] = daily["benchmark"].map(expected_counts)
    # Do not compare early 2x-only history with the later complete 2x+3x
    # complex. All published research begins once all four configured funds
    # are present for that benchmark.
    daily = daily[daily["fund_count"] == daily["expected_fund_count"]].copy()

    daily = daily.merge(prices, on=["proxy", "date"], how="left")
    daily["gross_flow_pct_aum"] = (
        daily["gross_primary_flow_usd"] / daily["prior_aum_usd"]
    )
    daily["flow_exposure_pct_dollar_volume"] = (
        daily["flow_exposure_usd"] / daily["daily_dollar_volume"]
    )
    daily["mechanical_pct_dollar_volume"] = (
        daily["mechanical_demand_usd"] / daily["daily_dollar_volume"]
    )
    daily["estimated_demand_pct_dollar_volume"] = (
        daily["estimated_demand_usd"] / daily["daily_dollar_volume"]
    )
    daily = daily.sort_values(["benchmark", "date"]).reset_index(drop=True)

    for column in (
        "gross_flow_pct_aum",
        "flow_exposure_pct_dollar_volume",
        "mechanical_pct_dollar_volume",
        "estimated_demand_pct_dollar_volume",
    ):
        daily[f"{column}_z"] = daily.groupby("benchmark")[column].transform(
            trailing_zscore
        )
        daily[f"{column}_p10"] = daily.groupby("benchmark")[column].transform(
            lambda x: prior_quantile(x, 0.10)
        )
        daily[f"{column}_p90"] = daily.groupby("benchmark")[column].transform(
            lambda x: prior_quantile(x, 0.90)
        )

    daily["abs_estimated_demand_pct_dollar_volume"] = daily[
        "estimated_demand_pct_dollar_volume"
    ].abs()
    daily["abs_estimated_demand_pct_dollar_volume_p90"] = daily.groupby(
        "benchmark"
    )["abs_estimated_demand_pct_dollar_volume"].transform(
        lambda x: prior_quantile(x, 0.90)
    )
    daily["forward_vol_ratio_5d"] = (
        daily["forward_rv_5d"] / daily["trailing_vol_20"]
    )
    daily.to_parquet(DATA_DIR / "benchmark_daily_features.parquet", index=False)
    return daily


def _load_intraday_proxy(proxy: str, refresh: bool) -> pd.DataFrame | None:
    path = DATA_DIR / "intraday" / f"{proxy}.parquet"
    key = f"intraday/15min/{proxy}.parquet"
    if refresh or not path.exists():
        ok = download_to_local(key, str(path))
        if not ok and not path.exists():
            print(f"[warning] no intraday cache for {proxy}", file=sys.stderr)
            return None
    frame = pd.read_parquet(path)
    frame["ts"] = pd.to_datetime(frame["ts"])
    return frame.sort_values("ts")


def build_intraday_features(
    daily: pd.DataFrame,
    refresh: bool = False,
    intraday_cache: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Build 15:30 ET pressure estimates using only information then known."""
    frames: list[pd.DataFrame] = []
    for proxy in PROXIES:
        bars = (
            intraday_cache.get(proxy)
            if intraday_cache is not None
            else _load_intraday_proxy(proxy, refresh=refresh)
        )
        if bars is None or bars.empty:
            continue
        bars = bars.copy()
        bars["date"] = bars["ts"].dt.normalize()
        bars["time"] = bars["ts"].dt.strftime("%H:%M")
        session = bars.groupby("date").agg(
            session_close=("close", "last"),
            session_volume=("volume", "sum"),
        )
        # 15:15-labelled bar spans 15:15-15:30; its close is the 15:30 price.
        cutoff = bars[bars["time"] == "15:15"].set_index("date")["close"].rename(
            "close_1530"
        )
        final_half_hour = (
            bars[bars["time"].isin(["15:30", "15:45"])]
            .assign(dollar_volume=lambda x: x["close"] * x["volume"])
            .groupby("date")["dollar_volume"]
            .sum()
            .rename("final_30m_dollar_volume")
        )
        x = pd.concat([session, cutoff, final_half_hour], axis=1).reset_index()
        x["proxy"] = proxy
        x = x.merge(
            daily[
                [
                    "date",
                    "proxy",
                    "benchmark",
                    "Close",
                    "rebalance_coefficient_usd",
                    "forward_return_1d",
                ]
            ],
            on=["date", "proxy"],
            how="inner",
        ).sort_values("date")
        x["previous_close"] = x["Close"].shift(1)
        x["return_to_1530"] = x["close_1530"] / x["previous_close"] - 1
        x["return_1530_to_close"] = x["session_close"] / x["close_1530"] - 1
        x["mechanical_demand_1530_usd"] = (
            x["rebalance_coefficient_usd"] * x["return_to_1530"]
        )
        x["mechanical_1530_pct_final_30m_dv"] = (
            x["mechanical_demand_1530_usd"] / x["final_30m_dollar_volume"]
        )
        x["signed_remaining_return"] = (
            np.sign(x["mechanical_demand_1530_usd"])
            * x["return_1530_to_close"]
        )
        x["abs_mechanical_1530_pct_final_30m_dv"] = x[
            "mechanical_1530_pct_final_30m_dv"
        ].abs()
        x["abs_pressure_p90"] = prior_quantile(
            x["abs_mechanical_1530_pct_final_30m_dv"], 0.90
        )
        frames.append(x)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True).sort_values(["benchmark", "date"])
    out.to_parquet(DATA_DIR / "benchmark_intraday_features.parquet", index=False)
    return out


def _normal_pvalue(t_value: float) -> float:
    if not np.isfinite(t_value):
        return np.nan
    return math.erfc(abs(t_value) / math.sqrt(2))


def first_episode_event(
    frame: pd.DataFrame,
    raw_event: pd.Series,
    cooldown: int = 5,
) -> pd.Series:
    """Keep the first event after ``cooldown`` quiet benchmark sessions."""
    raw = raw_event.fillna(False).astype(bool)
    prior_recent = raw.groupby(frame["benchmark"]).transform(
        lambda x: x.shift(1).rolling(cooldown, min_periods=1).max()
    ).fillna(False)
    return raw & ~prior_recent.astype(bool)


def _newey_west_t(values: pd.Series, max_lag: int) -> float:
    """HAC t-statistic for a sample mean."""
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    n = len(x)
    if n < 2:
        return np.nan
    mean = x.mean()
    residual = x - mean
    gamma0 = float(np.dot(residual, residual) / n)
    long_run_variance = gamma0
    for lag in range(1, min(max_lag, n - 1) + 1):
        weight = 1 - lag / (max_lag + 1)
        gamma = float(np.dot(residual[lag:], residual[:-lag]) / n)
        long_run_variance += 2 * weight * gamma
    if long_run_variance <= 0:
        return np.nan
    return mean / math.sqrt(long_run_variance / n)


def _event_result(
    name: str,
    event_definition: str,
    rows: pd.DataFrame,
    target: str,
    expected_direction: int,
    target_label: str,
    hac_lags: int = 5,
) -> dict:
    use = rows.dropna(subset=[target]).copy()
    if use.empty:
        return {
            "hypothesis": name,
            "event_definition": event_definition,
            "target": target_label,
            "event_dates": 0,
            "event_mean": np.nan,
            "baseline_mean": np.nan,
            "difference": np.nan,
            "t_stat": np.nan,
            "p_value_two_sided": np.nan,
            "assessment": "insufficient history",
        }
    baseline = use.groupby("benchmark")[target].transform("mean")
    event = use.loc[use["event"], ["date", "benchmark", target]].copy()
    event["baseline"] = baseline.loc[event.index]
    event["excess"] = event[target] - event["baseline"]
    # One observation per calendar date prevents the four correlated index
    # complexes from being counted as four independent market events.
    clustered = event.groupby("date", as_index=False).agg(
        event_value=(target, "mean"),
        baseline=("baseline", "mean"),
        excess=("excess", "mean"),
    )
    n = len(clustered)
    mean_excess = clustered["excess"].mean()
    t_value = _newey_west_t(clustered["excess"], max_lag=hac_lags)
    p_value = _normal_pvalue(t_value)
    aligned = expected_direction * mean_excess
    if n < 20:
        assessment = "insufficient event count"
    elif aligned > 0 and p_value < 0.10:
        assessment = "supportive"
    elif aligned < 0 and p_value < 0.10:
        assessment = "contradictory"
    else:
        assessment = "inconclusive"
    return {
        "hypothesis": name,
        "event_definition": event_definition,
        "target": target_label,
        "event_dates": n,
        "first_event": clustered["date"].min() if n else pd.NaT,
        "last_event": clustered["date"].max() if n else pd.NaT,
        "event_mean": clustered["event_value"].mean(),
        "baseline_mean": clustered["baseline"].mean(),
        "difference": mean_excess,
        "t_stat": t_value,
        "p_value_two_sided": p_value,
        "assessment": assessment,
    }


def run_event_studies(
    daily: pd.DataFrame,
    intraday: pd.DataFrame,
) -> pd.DataFrame:
    results: list[dict] = []

    x = daily.copy()
    x["event"] = first_episode_event(
        x, x["gross_flow_pct_aum"] > x["gross_flow_pct_aum_p90"]
    )
    results.append(
        _event_result(
            "Large creations/redemptions precede volatility expansion",
            "Gross primary flow/AUM above its prior 252-session 90th percentile",
            x,
            "forward_vol_ratio_5d",
            1,
            "next-5d RMS volatility / trailing-20d volatility",
        )
    )

    x = daily.copy()
    x["event"] = first_episode_event(
        x,
        x["abs_estimated_demand_pct_dollar_volume"]
        > x["abs_estimated_demand_pct_dollar_volume_p90"],
    )
    results.append(
        _event_result(
            "Large total LETF pressure precedes volatility expansion",
            "Absolute estimated demand/dollar volume above prior 252-session 90th percentile",
            x,
            "forward_vol_ratio_5d",
            1,
            "next-5d RMS volatility / trailing-20d volatility",
        )
    )

    x = daily.copy()
    x["event"] = first_episode_event(
        x,
        x["flow_exposure_pct_dollar_volume"]
        < x["flow_exposure_pct_dollar_volume_p10"],
    )
    results.append(
        _event_result(
            "Bearish primary-market flow is a contrarian turn signal",
            "Net flow exposure/dollar volume below prior 252-session 10th percentile",
            x,
            "forward_return_5d",
            1,
            "next-5d proxy return",
        )
    )

    x = daily.copy()
    x["event"] = first_episode_event(
        x,
        x["flow_exposure_pct_dollar_volume"]
        > x["flow_exposure_pct_dollar_volume_p90"],
    )
    results.append(
        _event_result(
            "Bullish primary-market flow is a contrarian turn signal",
            "Net flow exposure/dollar volume above prior 252-session 90th percentile",
            x,
            "forward_return_5d",
            -1,
            "next-5d proxy return",
        )
    )

    if not intraday.empty:
        x = intraday.copy()
        x["event"] = first_episode_event(
            x,
            x["abs_mechanical_1530_pct_final_30m_dv"] > x["abs_pressure_p90"],
        )
        results.append(
            _event_result(
                "Mechanical rebalance pressure continues into the close",
                "Absolute 15:30 modeled demand/final-30m dollar volume above prior 252-session 90th percentile",
                x,
                "signed_remaining_return",
                1,
                "direction-adjusted 15:30-to-close return",
                hac_lags=1,
            )
        )
        x["signed_next_day_reversal"] = -np.sign(
            x["mechanical_demand_1530_usd"]
        ) * x["forward_return_1d"]
        results.append(
            _event_result(
                "Extreme mechanical pressure reverses next session",
                "Absolute 15:30 modeled demand/final-30m dollar volume above prior 252-session 90th percentile",
                x,
                "signed_next_day_reversal",
                1,
                "opposite-direction next-session return",
            )
        )

    result = pd.DataFrame(results)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_DIR / "event_studies.csv", index=False)
    return result


def run_pressure_robustness(daily: pd.DataFrame) -> pd.DataFrame:
    """Describe stability of the predefined total-pressure volatility test."""
    x = daily.copy()
    x["event"] = first_episode_event(
        x,
        x["abs_estimated_demand_pct_dollar_volume"]
        > x["abs_estimated_demand_pct_dollar_volume_p90"],
    )

    def summarize(sample: pd.DataFrame, segment_type: str, segment: str) -> dict:
        use = sample.dropna(subset=["forward_vol_ratio_5d"]).copy()
        use["baseline"] = use.groupby("benchmark")[
            "forward_vol_ratio_5d"
        ].transform("mean")
        events = use[use["event"]].copy()
        events["excess"] = events["forward_vol_ratio_5d"] - events["baseline"]
        clustered = events.groupby("date", as_index=False).agg(
            event_mean=("forward_vol_ratio_5d", "mean"),
            baseline_mean=("baseline", "mean"),
            excess=("excess", "mean"),
        )
        t_value = _newey_west_t(clustered["excess"], 5)
        p_value = _normal_pvalue(t_value)
        effect = clustered["excess"].mean()
        if len(clustered) < 20:
            assessment = "insufficient event count"
        elif effect > 0 and p_value < 0.10:
            assessment = "supportive"
        elif effect < 0 and p_value < 0.10:
            assessment = "contradictory"
        else:
            assessment = "inconclusive"
        return {
            "segment_type": segment_type,
            "segment": segment,
            "event_dates": len(clustered),
            "event_mean": clustered["event_mean"].mean(),
            "baseline_mean": clustered["baseline_mean"].mean(),
            "difference": effect,
            "t_stat": t_value,
            "p_value_two_sided": p_value,
            "assessment": assessment,
        }

    rows = [
        summarize(group, "benchmark", benchmark)
        for benchmark, group in x.groupby("benchmark")
    ]
    rows.extend(
        [
            summarize(x[x["date"].dt.year <= 2017], "period", "2010-2017"),
            summarize(x[x["date"].dt.year >= 2018], "period", "2018-present"),
            summarize(x[x["date"].dt.year != 2020], "stress exclusion", "excluding 2020"),
        ]
    )
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "pressure_robustness.csv", index=False)
    return out


def _money(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    sign = "-" if value < 0 else ""
    value = abs(value)
    if value >= 1e9:
        return f"{sign}${value / 1e9:.2f}bn"
    return f"{sign}${value / 1e6:.1f}mm"


def write_reports(
    daily: pd.DataFrame,
    studies: pd.DataFrame,
    robustness: pd.DataFrame,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    latest_dates = daily.groupby("benchmark")["date"].max().rename("latest_date")
    latest = daily.merge(latest_dates, left_on="benchmark", right_index=True)
    latest = latest[latest["date"] == latest["latest_date"]].copy()
    latest["pressure_extreme"] = (
        latest["abs_estimated_demand_pct_dollar_volume"]
        > latest["abs_estimated_demand_pct_dollar_volume_p90"]
    )
    benchmark_support = robustness[
        robustness["segment_type"] == "benchmark"
    ].set_index("segment")["assessment"]
    latest["benchmark_vol_evidence"] = latest["benchmark"].map(benchmark_support)
    latest["flow_turn_state"] = np.select(
        [
            latest["flow_exposure_pct_dollar_volume"]
            < latest["flow_exposure_pct_dollar_volume_p10"],
            latest["flow_exposure_pct_dollar_volume"]
            > latest["flow_exposure_pct_dollar_volume_p90"],
        ],
        ["extreme bearish flow", "extreme bullish flow"],
        default="ordinary",
    )
    snapshot_cols = [
        "date",
        "benchmark",
        "proxy",
        "Close",
        "proxy_return",
        "aum_usd",
        "primary_flow_usd",
        "gross_primary_flow_usd",
        "flow_exposure_usd",
        "mechanical_demand_usd",
        "estimated_demand_usd",
        "estimated_demand_pct_dollar_volume",
        "gross_flow_pct_aum_z",
        "pressure_extreme",
        "benchmark_vol_evidence",
        "flow_turn_state",
    ]
    latest[snapshot_cols].to_csv(OUTPUT_DIR / "latest_snapshot.csv", index=False)
    fund_pressure = pd.read_parquet(DATA_DIR / "fund_pressure.parquet")
    latest_fund_date = fund_pressure["date"].max()
    latest_funds = fund_pressure[fund_pressure["date"] == latest_fund_date].copy()
    latest_funds["flow_state"] = np.select(
        [latest_funds["primary_flow_usd"] > 0, latest_funds["primary_flow_usd"] < 0],
        ["creation", "redemption"],
        default="unchanged",
    )
    latest_fund_cols = [
        "date",
        "ticker",
        "benchmark",
        "leverage",
        "aum_usd",
        "primary_flow_usd",
        "flow_state",
        "flow_exposure_usd",
        "mechanical_demand_usd",
        "estimated_demand_usd",
    ]
    latest_funds[latest_fund_cols].sort_values(
        "primary_flow_usd"
    ).to_csv(OUTPUT_DIR / "latest_fund_flows.csv", index=False)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "timing": {
            "primary_flows": "official end-of-day; actionable no earlier than next session",
            "mechanical_1530": "prior-day AUM plus proxy return through 15:30 ET",
        },
        "funds": [asdict(fund) for fund in FUNDS],
        "latest": json.loads(latest[snapshot_cols].to_json(orient="records", date_format="iso")),
        "latest_funds": json.loads(
            latest_funds[latest_fund_cols].to_json(orient="records", date_format="iso")
        ),
        "studies": json.loads(studies.to_json(orient="records", date_format="iso")),
        "pressure_robustness": json.loads(
            robustness.to_json(orient="records", date_format="iso")
        ),
    }
    (OUTPUT_DIR / "monitor.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    lines = [
        "# Leveraged ETF Flow Monitor",
        "",
        f"Generated {payload['generated_at_utc']}. Official primary-market flows are post-close data; the current reading is for the next-session watchlist, not a same-day trade signal.",
        "",
        "## Latest post-close snapshot",
        "",
        "| Date | Complex | Return | Gross flow | Flow exposure | Mechanical | Combined | % proxy daily $vol | Pressure extreme | Complex-level vol evidence | Flow state |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for _, row in latest.sort_values("benchmark").iterrows():
        lines.append(
            f"| {row['date'].date()} | {row['benchmark']} | {row['proxy_return']:.2%} | "
            f"{_money(row['gross_primary_flow_usd'])} | {_money(row['flow_exposure_usd'])} | "
            f"{_money(row['mechanical_demand_usd'])} | "
            f"{_money(row['estimated_demand_usd'])} | "
            f"{row['estimated_demand_pct_dollar_volume']:.2%} | "
            f"{'YES' if row['pressure_extreme'] else 'no'} | "
            f"{row['benchmark_vol_evidence']} | {row['flow_turn_state']} |"
        )
    lines.extend(
        [
            "",
            "## Largest current creations and redemptions",
            "",
            "Positive flow is a creation; negative flow is a redemption. Flow exposure applies the fund's signed leverage, so it is the directional benchmark implication.",
            "",
            "| Fund | Complex | L | Flow | State | Flow exposure |",
            "|---|---|---:|---:|---|---:|",
        ]
    )
    for _, row in latest_funds.reindex(
        latest_funds["primary_flow_usd"].abs().sort_values(ascending=False).index
    ).head(10).iterrows():
        lines.append(
            f"| {row['ticker']} | {row['benchmark']} | {row['leverage']:+d}x | "
            f"{_money(row['primary_flow_usd'])} | {row['flow_state']} | "
            f"{_money(row['flow_exposure_usd'])} |"
        )
    lines.extend(
        [
            "",
            "## Predefined hypothesis tests",
            "",
            "| Hypothesis | Event dates | Event mean | Baseline | Difference | p-value | Assessment |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for _, row in studies.iterrows():
        lines.append(
            f"| {row['hypothesis']} | {int(row['event_dates'])} | "
            f"{row['event_mean']:.4f} | {row['baseline_mean']:.4f} | "
            f"{row['difference']:.4f} | {row['p_value_two_sided']:.3f} | "
            f"{row['assessment']} |"
        )
    lines.extend(
        [
            "",
            "## Total-pressure volatility robustness",
            "",
            "| Slice | Events | Event mean | Baseline | Difference | p-value | Assessment |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for _, row in robustness.iterrows():
        lines.append(
            f"| {row['segment_type']}: {row['segment']} | {int(row['event_dates'])} | "
            f"{row['event_mean']:.4f} | {row['baseline_mean']:.4f} | "
            f"{row['difference']:.4f} | {row['p_value_two_sided']:.3f} | "
            f"{row['assessment']} |"
        )
    lines.extend(
        [
            "",
            "`Pressure extreme` is an objective threshold crossing. `Complex-level vol evidence` says whether that particular benchmark's historical effect clears the exploratory gate; it prevents a pooled result from being presented as validated everywhere.",
            "",
            "## Interpretation guardrails",
            "",
            "- `Primary flow` is inferred from split-adjusted changes in official shares outstanding.",
            "- `Mechanical` is a model estimate, not an observed closing-auction order.",
            "- `Combined` adds flow-associated exposure (`leverage x flow`) to modeled rebalancing. AP and swap-counterparty pre-hedging can shift the actual execution time.",
            "- Statistical results are research diagnostics. A supportive result is not production authorization; costs, stability, and a frozen out-of-sample gate still matter.",
            "- Events are reduced to the first signal after five quiet benchmark sessions, same-date complexes are clustered, and p-values use Newey-West standard errors.",
        ]
    )
    (OUTPUT_DIR / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    history = fetch_proshares_history(refresh=refresh)
    intraday_cache = {
        proxy: _load_intraday_proxy(proxy, refresh=refresh) for proxy in PROXIES
    }
    intraday_cache = {
        proxy: frame for proxy, frame in intraday_cache.items() if frame is not None
    }
    daily = build_daily_features(history, intraday_cache=intraday_cache)
    intraday = build_intraday_features(
        daily, refresh=False, intraday_cache=intraday_cache
    )
    studies = run_event_studies(daily, intraday)
    robustness = run_pressure_robustness(daily)
    write_reports(daily, studies, robustness)
    print(f"fund rows: {len(history):,}")
    print(f"daily complex rows: {len(daily):,}")
    print(f"intraday rows: {len(intraday):,}")
    print(f"report: {OUTPUT_DIR / 'REPORT.md'}")
    return daily, intraday, studies


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="refresh official ProShares and R2 intraday caches",
    )
    args = parser.parse_args(argv)
    run(refresh=args.refresh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
