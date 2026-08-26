"""Leakage-aware UVXY research harness.

The original hypothesis and its shorter-horizon overlays are deliberately
frozen and simple:

* VIX Range Compression activates using the production dashboard definition.
* The prior session's ex-VRC fragility dial is in its trailing upper tercile.
* Equal-weight 5d and 21d horizon-native overlays are compared with the original
  63d sizing-basis specification without selecting a winner after seeing returns.
* Buy UVXY at the next open and sell at the fifth post-signal session's close.

The module is research-only. It does not import or mutate the strategy book, order
staging, portfolio state, or private-site artifacts.
"""

from __future__ import annotations

import hashlib
import json
import os
import warnings
from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

UVXY_1_5X_START_DATE = pd.Timestamp("2018-02-28")
SIMPLE_DIAL_PIT_START = pd.Timestamp("2026-07-16")
OFFICIAL_UVXY_URL = "https://www.proshares.com/our-etfs/strategic/uvxy"
REGISTERED_SIMPLE_SIGNALS = (
    "Distribution Dominance",
    "VIX Range Compression",
    "Defensive Leadership",
    "Pre-FOMC Rally",
    "Low Absorption Ratio",
    "Seasonal Rank Divergence",
    "Dispersion",
)
EX_VRC_SIMPLE_SIGNALS = tuple(
    signal for signal in REGISTERED_SIMPLE_SIGNALS if signal != "VIX Range Compression"
)
NATIVE_FRAGILITY_HORIZONS = (5, 21, 63)


@dataclass(frozen=True)
class StrategyConfig:
    """Frozen primary specification and conservative implementation assumptions."""

    start_date: str = "2018-03-01"
    vix_range_window: int = 21
    percentile_lookback: int = 504
    percentile_min_fraction: float = 0.80
    vrc_percentile_threshold: float = 15.0
    vix_floor: float = 13.0
    vix_sma_window: int = 20
    vix_sma_min_fraction: float = 0.80
    fragility_decay_sessions: int = 63
    simple_signal_count: int = 7
    fragility_ma_window: int = 10
    native_fragility_smooth_window: int = 5
    fragility_rank_threshold: float = 67.0
    rearm_off_sessions: int = 5
    hold_sessions: int = 5
    round_trip_cost_bps: float = 12.0
    illustrative_sleeve_weight: float = 0.02
    bootstrap_samples: int = 20_000
    random_seed: int = 8_251_626


def _normalise_index(frame: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    out = frame.copy()
    out.index = pd.to_datetime(out.index).tz_localize(None).normalize()
    return out[~out.index.duplicated(keep="last")].sort_index()


def rolling_percentile(
    series: pd.Series,
    lookback: int,
    min_fraction: float = 0.80,
) -> pd.Series:
    """Rank today's value against only the preceding ``lookback`` observations.

    The current value is never part of its reference distribution. This matches
    the dashboard's effective convention while making the no-lookahead contract
    explicit and testable.
    """

    values = series.to_numpy(dtype=float)
    result = np.full(len(values), np.nan, dtype=float)
    # Production checks the prior window plus today's valid observation against
    # int(lookback * min_fraction). Because today's value is already known to
    # be finite here, the required number of finite *prior* observations is one
    # smaller (402 for the 504-session, 80% production setting).
    min_valid = max(int(lookback * min_fraction) - 1, 0)
    for index in range(lookback, len(values)):
        current = values[index]
        if not np.isfinite(current):
            continue
        history = values[index - lookback : index]
        history = history[np.isfinite(history)]
        if len(history) < min_valid:
            continue
        result[index] = float((history < current).sum() / len(history) * 100.0)
    return pd.Series(result, index=series.index, name=series.name)


def decayed_weight(history: pd.Series, decay_sessions: int = 63) -> pd.Series:
    """Return 1 while a signal is on, then linearly decay to zero."""

    fired = history.fillna(False).astype(bool)
    position = pd.Series(np.arange(len(fired), dtype=float), index=fired.index)
    last_fire = position.where(fired).ffill()
    days_since = position - last_fire
    weight = (1.0 - days_since / decay_sessions).clip(lower=0.0)
    weight.loc[fired] = 1.0
    return weight.fillna(0.0)


def build_native_fragility_dials(
    signal_history: pd.DataFrame,
    *,
    horizons: Iterable[int] = NATIVE_FRAGILITY_HORIZONS,
    smooth_window: int = 5,
) -> pd.DataFrame:
    """Build horizon-matched equal-weight dials from the six non-VRC signals.

    These overlays intentionally avoid the incumbent dial's outcome-fitted
    weights, regime multipliers, and VRC contribution. A component is 1 while
    active and then decays linearly over its named horizon. The mean component
    weight is smoothed once over five sessions, matching the persisted dial
    basis without adding the slower 10-session sizing average.
    """

    missing = set(EX_VRC_SIMPLE_SIGNALS).difference(signal_history.columns)
    if missing:
        raise ValueError(f"signal history missing registered inputs: {sorted(missing)}")
    if smooth_window < 1:
        raise ValueError("smooth_window must be positive")

    history = _normalise_index(signal_history.loc[:, EX_VRC_SIMPLE_SIGNALS])
    history = history.astype("boolean").fillna(False).astype(bool)
    result: dict[str, pd.Series] = {}
    for horizon in horizons:
        horizon = int(horizon)
        if horizon < 1:
            raise ValueError("fragility horizons must be positive")
        components = [
            decayed_weight(history[signal], horizon)
            for signal in EX_VRC_SIMPLE_SIGNALS
        ]
        raw = pd.concat(components, axis=1).mean(axis=1) * 100.0
        result[f"{horizon}d"] = raw.rolling(
            smooth_window, min_periods=1
        ).mean()
    return pd.DataFrame(result, index=history.index)


def remove_vrc_from_simple_dial(
    simple_smoothed: pd.Series,
    vrc_weight_smoothed: pd.Series,
    signal_count: int = 7,
) -> pd.Series:
    """Algebraically remove VRC from the registered equal-weight simple dial."""

    if signal_count < 2:
        raise ValueError("signal_count must be at least 2")
    ex_vrc = (
        simple_smoothed * float(signal_count)
        - vrc_weight_smoothed * 100.0
    ) / float(signal_count - 1)
    # Do not clip: an out-of-bounds result is a useful parity failure showing
    # that the supplied VRC component did not come from the same source vintage
    # as the persisted simple dial.
    return ex_vrc.rename("simple_ex_vrc")


def vrc_activation(vrc_on: pd.Series, off_sessions: int = 5) -> pd.Series:
    """Activate only after at least ``off_sessions`` consecutive off sessions."""

    if off_sessions < 1:
        raise ValueError("off_sessions must be positive")
    on = vrc_on.fillna(False).astype(bool)
    previous_off = (~on.shift(1, fill_value=False)).astype(int)
    rearmed = previous_off.rolling(off_sessions, min_periods=off_sessions).sum().eq(
        off_sessions
    )
    return (on & rearmed).rename("vrc_activation")


def _load_price_panel(master_path: Path, tickers: Iterable[str]) -> dict[str, pd.DataFrame]:
    tickers = list(tickers)
    prices = pd.read_parquet(master_path, filters=[("ticker", "in", tickers)])
    required = {"ticker", "date", "Open", "High", "Low", "Close"}
    missing_columns = required.difference(prices.columns)
    if missing_columns:
        raise ValueError(f"master price cache missing columns: {sorted(missing_columns)}")
    prices["date"] = pd.to_datetime(prices["date"]).dt.tz_localize(None).dt.normalize()
    result: dict[str, pd.DataFrame] = {}
    for ticker, group in prices.groupby("ticker", sort=False):
        frame = group.set_index("date")[["Open", "High", "Low", "Close"]]
        result[str(ticker)] = _normalise_index(frame)
    missing_tickers = set(tickers).difference(result)
    if missing_tickers:
        raise ValueError(f"master price cache missing tickers: {sorted(missing_tickers)}")
    return result


def _file_fingerprint(
    path: Path,
    *,
    loaded_rows: int | None = None,
) -> dict[str, Any]:
    """Fingerprint an input so a later cache refresh cannot mimic this vintage."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = path.stat()
    fingerprint = {
        "path": str(path.resolve()),
        "sha256": digest.hexdigest(),
        "bytes": int(stat.st_size),
        "modified_ns": int(stat.st_mtime_ns),
    }
    if loaded_rows is not None:
        fingerprint["loaded_rows"] = int(loaded_rows)
    return fingerprint


def _recompute_registered_signal_history(
    spy_path: Path,
    closes_path: Path,
    sp500_path: Path,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Recompute one frozen current-vintage history from the risk-cache trio.

    The dashboard module is Streamlit-backed, so the import is kept lazy and a
    compatibility shim is applied for this machine's older Streamlit runtime.
    No files are written and the returned histories are explicitly research-
    vintage, not point-in-time observations.
    """

    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    import streamlit as st

    if not hasattr(st, "fragment"):
        def _fragment(function):
            return function

        st.fragment = _fragment

    from daily_risk_report import compute_all_signals

    spy = _normalise_index(pd.read_parquet(spy_path))
    closes = _normalise_index(pd.read_parquet(closes_path))
    sp500 = _normalise_index(pd.read_parquet(sp500_path))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        computed = compute_all_signals(spy, closes, sp500)

    histories: dict[str, pd.Series] = {}
    for signal in REGISTERED_SIMPLE_SIGNALS:
        payload = computed["signals_ordered"].get(signal, {})
        history = payload.get("signal_history")
        if history is None or not hasattr(history, "empty") or history.empty:
            raise ValueError(f"recomputed signal history missing {signal!r}")
        histories[signal] = (
            _normalise_index(history)
            .reindex(spy.index)
            .astype("boolean")
            .fillna(False)
            .astype(bool)
        )
    return pd.DataFrame(histories, index=spy.index), {
        "risk_spy_rows": len(spy),
        "risk_closes_rows": len(closes),
        "risk_sp500_rows": len(sp500),
    }


def load_inputs(data_dir: str | Path) -> dict[str, Any]:
    """Load only the local files needed for this research run."""

    data_path = Path(data_dir)
    master_path = data_path / "master_prices.parquet"
    simple_path = data_path / "rd2_fragility_simple.parquet"
    incumbent_path = data_path / "rd2_fragility.parquet"
    signal_history_path = data_path / "signal_fire_history.parquet"
    risk_spy_path = data_path / "rd2_spy_ohlc.parquet"
    risk_closes_path = data_path / "rd2_closes.parquet"
    risk_sp500_path = data_path / "rd2_sp500_closes.parquet"
    repository_root = Path(__file__).resolve().parents[2]
    classification_reference = (
        repository_root / "reference" / "sp500_risk_classification.csv"
    )
    classification_legacy = (
        repository_root / "data" / "sp500_risk_classification.csv"
    )
    classification_path = (
        classification_reference
        if classification_reference.exists()
        else classification_legacy
    )
    context_paths = {
        "risk_classification": classification_path,
        "seasonal_ranks": repository_root / "seasonal_ranks.csv",
        "seasonal_ranks_extension": repository_root / "sznl_ranks.csv",
        "signal_horizon_stats": repository_root
        / "data"
        / "signal_horizon_stats.json",
        "daily_risk_report_code": repository_root / "daily_risk_report.py",
        "risk_dashboard_code": repository_root
        / "pages"
        / "risk_dashboard_v2.py",
        "fragility_core_code": repository_root / "fragility_core.py",
        "research_backtest_code": Path(__file__).resolve(),
    }
    required_paths = (
        master_path,
        simple_path,
        incumbent_path,
        signal_history_path,
        risk_spy_path,
        risk_closes_path,
        risk_sp500_path,
    )
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)
    for label, path in context_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"missing research dependency {label}: {path}")

    prices = _load_price_panel(master_path, ["UVXY", "^VIX", "^VIX3M", "SPY"])
    simple = _normalise_index(pd.read_parquet(simple_path))
    incumbent = _normalise_index(pd.read_parquet(incumbent_path))
    signal_history = _normalise_index(pd.read_parquet(signal_history_path))
    research_signal_history, risk_rows = _recompute_registered_signal_history(
        risk_spy_path,
        risk_closes_path,
        risk_sp500_path,
    )
    if "simple" not in simple.columns:
        raise ValueError("rd2_fragility_simple.parquet is missing 'simple'")
    missing_incumbent = {"5d", "21d", "63d"}.difference(incumbent.columns)
    if missing_incumbent:
        raise ValueError(
            f"rd2_fragility.parquet missing columns: {sorted(missing_incumbent)}"
        )
    if "VIX Range Compression" not in signal_history.columns:
        raise ValueError(
            "signal_fire_history.parquet is missing 'VIX Range Compression'"
        )
    loaded_price_rows = sum(len(frame) for frame in prices.values())
    return {
        "prices": prices,
        "simple": simple["simple"].astype(float),
        "incumbent": incumbent[["5d", "21d", "63d"]].astype(float),
        "vrc_component_history": signal_history["VIX Range Compression"].astype(bool),
        "research_signal_history": research_signal_history,
        "paths": {
            "master_prices": str(master_path.resolve()),
            "simple_fragility": str(simple_path.resolve()),
            "incumbent_fragility": str(incumbent_path.resolve()),
            "signal_history": str(signal_history_path.resolve()),
            "risk_spy_ohlc": str(risk_spy_path.resolve()),
            "risk_closes": str(risk_closes_path.resolve()),
            "risk_sp500_closes": str(risk_sp500_path.resolve()),
            **{
                label: str(path.resolve())
                for label, path in context_paths.items()
            },
        },
        "fingerprints": {
            "master_prices": _file_fingerprint(
                master_path, loaded_rows=loaded_price_rows
            ),
            "simple_fragility": _file_fingerprint(simple_path, loaded_rows=len(simple)),
            "incumbent_fragility": _file_fingerprint(
                incumbent_path, loaded_rows=len(incumbent)
            ),
            "signal_history": _file_fingerprint(
                signal_history_path, loaded_rows=len(signal_history)
            ),
            "risk_spy_ohlc": _file_fingerprint(
                risk_spy_path, loaded_rows=risk_rows["risk_spy_rows"]
            ),
            "risk_closes": _file_fingerprint(
                risk_closes_path, loaded_rows=risk_rows["risk_closes_rows"]
            ),
            "risk_sp500_closes": _file_fingerprint(
                risk_sp500_path, loaded_rows=risk_rows["risk_sp500_rows"]
            ),
            **{
                label: _file_fingerprint(path)
                for label, path in context_paths.items()
            },
        },
    }


def build_features(inputs: dict[str, Any], config: StrategyConfig) -> pd.DataFrame:
    """Build point-in-time transforms on the UVXY trading-session calendar."""

    prices = inputs["prices"]
    index = prices["UVXY"].index
    vix = prices["^VIX"]["Close"].reindex(index).ffill(limit=3)
    vix3m = prices["^VIX3M"]["Close"].reindex(index).ffill(limit=3)
    spy = prices["SPY"]["Close"].reindex(index).ffill(limit=3)

    range_metric = (
        vix.rolling(config.vix_range_window).max()
        - vix.rolling(config.vix_range_window).min()
    )
    compression_pctile = rolling_percentile(
        range_metric,
        config.percentile_lookback,
        config.percentile_min_fraction,
    ).rename("compression_pctile")
    vix_sma = vix.rolling(
        config.vix_sma_window,
        min_periods=int(config.vix_sma_window * config.vix_sma_min_fraction),
    ).mean()
    vrc_on = (
        (compression_pctile < config.vrc_percentile_threshold)
        & (vix > config.vix_floor)
        & (vix > vix_sma)
    ).fillna(False)
    # The persisted simple dial was built from the shorter risk-cache history,
    # whose exact VRC states are stored in signal_fire_history.parquet. Use that
    # source where available, then continue with today's recomputation after its
    # final row. This keeps the algebraic subtraction on the same warm-up basis.
    stored_vrc = inputs["vrc_component_history"].reindex(index).astype("boolean")
    vrc_component_on = stored_vrc.fillna(vrc_on.astype("boolean")).astype(bool)
    vrc_decay_5d = (
        decayed_weight(vrc_component_on, config.fragility_decay_sessions)
        .rolling(5, min_periods=1)
        .mean()
        .rename("vrc_decay_5d")
    )

    simple = inputs["simple"].reindex(index)
    simple_ex_vrc = remove_vrc_from_simple_dial(
        simple,
        vrc_decay_5d,
        config.simple_signal_count,
    )
    fragility_ma10 = simple_ex_vrc.rolling(
        config.fragility_ma_window,
        min_periods=config.fragility_ma_window,
    ).mean()
    fragility_rank = rolling_percentile(
        fragility_ma10,
        config.percentile_lookback,
        config.percentile_min_fraction,
    ).rename("fragility_rank")

    native_dials = build_native_fragility_dials(
        inputs["research_signal_history"],
        smooth_window=config.native_fragility_smooth_window,
    )
    native_columns: dict[str, pd.Series] = {}
    for horizon in NATIVE_FRAGILITY_HORIZONS:
        label = f"{horizon}d"
        dial = native_dials[label].reindex(index).rename(
            f"native_fragility_{label}"
        )
        rank = rolling_percentile(
            dial,
            config.percentile_lookback,
            config.percentile_min_fraction,
        ).rename(f"native_fragility_{label}_rank")
        native_columns[dial.name] = dial
        native_columns[rank.name] = rank

    incumbent = inputs["incumbent"].reindex(index)
    incumbent_ma10 = incumbent["63d"].rolling(
        config.fragility_ma_window,
        min_periods=config.fragility_ma_window,
    ).mean()
    incumbent_rank = rolling_percentile(
        incumbent_ma10,
        config.percentile_lookback,
        config.percentile_min_fraction,
    ).rename("incumbent_rank")
    incumbent_native_ranks = {
        f"incumbent_{horizon}d_rank": rolling_percentile(
            incumbent[f"{horizon}d"],
            config.percentile_lookback,
            config.percentile_min_fraction,
        ).rename(f"incumbent_{horizon}d_rank")
        for horizon in NATIVE_FRAGILITY_HORIZONS
    }

    term_ratio = (vix / vix3m).replace([np.inf, -np.inf], np.nan)
    term_rank = rolling_percentile(
        term_ratio,
        config.percentile_lookback,
        config.percentile_min_fraction,
    ).rename("term_rank")

    features = pd.DataFrame(
        {
            "uvxy_open": prices["UVXY"]["Open"],
            "uvxy_close": prices["UVXY"]["Close"],
            "spy_close": spy,
            "vix": vix,
            "vix3m": vix3m,
            "vix_sma": vix_sma,
            "vix_range": range_metric,
            "compression_pctile": compression_pctile,
            "vrc_on": vrc_on,
            "vrc_component_on": vrc_component_on,
            "vrc_decay_5d": vrc_decay_5d,
            "simple": simple,
            "simple_ex_vrc": simple_ex_vrc,
            "fragility_ma10": fragility_ma10,
            "fragility_rank": fragility_rank,
            "incumbent_ma10": incumbent_ma10,
            "incumbent_rank": incumbent_rank,
            "term_ratio": term_ratio,
            "term_rank": term_rank,
            **native_columns,
            **incumbent_native_ranks,
        },
        index=index,
    )
    features.index.name = "date"
    return features


def _vrc_mask(features: pd.DataFrame, config: StrategyConfig, threshold: float) -> pd.Series:
    return (
        (features["compression_pctile"] < threshold)
        & (features["vix"] > config.vix_floor)
        & (features["vix"] > features["vix_sma"])
    ).fillna(False)


def build_signals(
    features: pd.DataFrame,
    config: StrategyConfig,
    *,
    vrc_threshold: float | None = None,
    fragility_threshold: float | None = None,
) -> pd.DataFrame:
    """Build the primary interaction and standalone comparators."""

    vrc_threshold = (
        config.vrc_percentile_threshold if vrc_threshold is None else vrc_threshold
    )
    fragility_threshold = (
        config.fragility_rank_threshold
        if fragility_threshold is None
        else fragility_threshold
    )
    vrc_on = _vrc_mask(features, config, float(vrc_threshold))
    activation = vrc_activation(vrc_on, config.rearm_off_sessions)

    lagged_frag_high = features["fragility_rank"].shift(1) >= float(
        fragility_threshold
    )
    lagged_full_high = features["incumbent_rank"].shift(1) >= float(
        fragility_threshold
    )
    lagged_full_abs = features["incumbent_ma10"].shift(1) >= 50.0
    unavailable = pd.Series(np.nan, index=features.index, dtype=float)
    native_rank_columns = {
        horizon: features.get(
            f"native_fragility_{horizon}d_rank", unavailable
        ).shift(1)
        for horizon in NATIVE_FRAGILITY_HORIZONS
    }
    lagged_native_high = {
        horizon: rank >= float(fragility_threshold)
        for horizon, rank in native_rank_columns.items()
    }
    lagged_short_mean_high = (
        (native_rank_columns[5] + native_rank_columns[21]) / 2.0
    ) >= float(fragility_threshold)
    lagged_incumbent_native_high = {
        horizon: features.get(f"incumbent_{horizon}d_rank", unavailable).shift(1)
        >= float(fragility_threshold)
        for horizon in (5, 21)
    }

    frag_cross = lagged_frag_high & ~lagged_frag_high.shift(1, fill_value=False)
    signals = pd.DataFrame(
        {
            "primary": activation & lagged_frag_high,
            "vrc_only": activation,
            "fragility_only": frag_cross,
            "vrc_x_ex_vrc_5d": activation & lagged_native_high[5],
            "vrc_x_ex_vrc_21d": activation & lagged_native_high[21],
            "vrc_x_ex_vrc_5d_21d_both": (
                activation & lagged_native_high[5] & lagged_native_high[21]
            ),
            "vrc_x_ex_vrc_5d_21d_mean": activation & lagged_short_mean_high,
            "vrc_x_ex_vrc_63d_native": activation & lagged_native_high[63],
            "vrc_x_incumbent_rank": activation & lagged_full_high,
            "vrc_x_incumbent_50": activation & lagged_full_abs,
            "vrc_x_incumbent_5d_rank": (
                activation & lagged_incumbent_native_high[5]
            ),
            "vrc_x_incumbent_21d_rank": (
                activation & lagged_incumbent_native_high[21]
            ),
        },
        index=features.index,
    ).fillna(False)
    return signals.astype(bool)


def forward_trade_return(
    features: pd.DataFrame,
    signal_position: int,
    hold_sessions: int,
    round_trip_cost_bps: float,
) -> dict[str, Any] | None:
    """Next-open to final-close return for one close-known signal."""

    entry_position = signal_position + 1
    exit_position = signal_position + hold_sessions
    if entry_position >= len(features) or exit_position >= len(features):
        return None
    entry_open = float(features["uvxy_open"].iloc[entry_position])
    exit_close = float(features["uvxy_close"].iloc[exit_position])
    if not np.isfinite(entry_open) or not np.isfinite(exit_close) or entry_open <= 0:
        return None

    side_cost = float(round_trip_cost_bps) / 2.0 / 10_000.0
    gross_return = exit_close / entry_open - 1.0
    net_return = (exit_close * (1.0 - side_cost)) / (
        entry_open * (1.0 + side_cost)
    ) - 1.0
    return {
        "signal_position": signal_position,
        "entry_position": entry_position,
        "exit_position": exit_position,
        "signal_date": features.index[signal_position],
        "entry_date": features.index[entry_position],
        "exit_date": features.index[exit_position],
        "entry_open": entry_open,
        "exit_close": exit_close,
        "gross_return": gross_return,
        "net_return": net_return,
    }


def run_trades(
    signal: pd.Series,
    features: pd.DataFrame,
    config: StrategyConfig,
    *,
    label: str,
    pit_start: pd.Timestamp | None = SIMPLE_DIAL_PIT_START,
) -> pd.DataFrame:
    """Run non-overlapping trades for a signal series."""

    start = pd.Timestamp(config.start_date)
    records: list[dict[str, Any]] = []
    last_exit_position = -1
    for signal_position in np.flatnonzero(signal.reindex(features.index).fillna(False)):
        signal_position = int(signal_position)
        if features.index[signal_position] < start:
            continue
        record = forward_trade_return(
            features,
            signal_position,
            config.hold_sessions,
            config.round_trip_cost_bps,
        )
        if record is None or record["entry_position"] <= last_exit_position:
            continue
        record["variant"] = label
        signal_date = record["signal_date"]
        for column in (
            "vix",
            "vix3m",
            "compression_pctile",
            "fragility_ma10",
            "fragility_rank",
            "incumbent_ma10",
            "incumbent_rank",
            "native_fragility_5d",
            "native_fragility_5d_rank",
            "native_fragility_21d",
            "native_fragility_21d_rank",
            "native_fragility_63d",
            "native_fragility_63d_rank",
            "incumbent_5d_rank",
            "incumbent_21d_rank",
            "term_ratio",
            "term_rank",
        ):
            record[column] = (
                float(features.at[signal_date, column])
                if column in features.columns
                else np.nan
            )
        gate_position = signal_position - 1
        record["gate_date"] = (
            features.index[gate_position]
            if gate_position >= 0
            else pd.NaT
        )
        for column in (
            "fragility_ma10",
            "fragility_rank",
            "incumbent_ma10",
            "incumbent_rank",
            "native_fragility_5d",
            "native_fragility_5d_rank",
            "native_fragility_21d",
            "native_fragility_21d_rank",
            "native_fragility_63d",
            "native_fragility_63d_rank",
            "incumbent_5d_rank",
            "incumbent_21d_rank",
        ):
            record[f"gate_{column}"] = (
                float(features.iloc[gate_position][column])
                if gate_position >= 0 and column in features.columns
                else np.nan
            )
        record["calendar_year"] = int(signal_date.year)
        record["genuine_pit"] = bool(
            pit_start is not None and signal_date >= pd.Timestamp(pit_start)
        )
        records.append(record)
        last_exit_position = int(record["exit_position"])

    columns = [
        "variant",
        "signal_date",
        "gate_date",
        "entry_date",
        "exit_date",
        "entry_open",
        "exit_close",
        "gross_return",
        "net_return",
        "vix",
        "vix3m",
        "compression_pctile",
        "fragility_ma10",
        "fragility_rank",
        "incumbent_ma10",
        "incumbent_rank",
        "native_fragility_5d",
        "native_fragility_5d_rank",
        "native_fragility_21d",
        "native_fragility_21d_rank",
        "native_fragility_63d",
        "native_fragility_63d_rank",
        "incumbent_5d_rank",
        "incumbent_21d_rank",
        "gate_fragility_ma10",
        "gate_fragility_rank",
        "gate_incumbent_ma10",
        "gate_incumbent_rank",
        "gate_native_fragility_5d",
        "gate_native_fragility_5d_rank",
        "gate_native_fragility_21d",
        "gate_native_fragility_21d_rank",
        "gate_native_fragility_63d",
        "gate_native_fragility_63d_rank",
        "gate_incumbent_5d_rank",
        "gate_incumbent_21d_rank",
        "term_ratio",
        "term_rank",
        "calendar_year",
        "genuine_pit",
    ]
    return pd.DataFrame.from_records(records, columns=columns)


def summarize_trades(
    trades: pd.DataFrame,
    config: StrategyConfig,
    sample_end: pd.Timestamp,
) -> dict[str, Any]:
    """Summarize event-level returns without pretending sparse events are daily IID."""

    returns = trades.get("net_return", pd.Series(dtype=float)).dropna().astype(float)
    if returns.empty:
        return {
            "n_trades": 0,
            "years_represented": 0,
            "mean_return": None,
            "median_return": None,
            "win_rate": None,
            "t_stat": None,
            "one_sided_p": None,
            "best_return": None,
            "worst_return": None,
            "unit_notional_compound": None,
            "sleeve_total_return": None,
            "sleeve_annualized_return": None,
            "sleeve_exit_marked_max_drawdown": None,
        }

    t_result = stats.ttest_1samp(returns, 0.0, alternative="greater")
    sleeve_returns = returns * config.illustrative_sleeve_weight
    sleeve_curve = (1.0 + sleeve_returns).cumprod()
    sleeve_drawdown = sleeve_curve / sleeve_curve.cummax() - 1.0
    sample_years = max(
        (pd.Timestamp(sample_end) - pd.Timestamp(config.start_date)).days / 365.25,
        1 / 365.25,
    )
    sleeve_total = float(sleeve_curve.iloc[-1] - 1.0)
    sleeve_annualized = float((1.0 + sleeve_total) ** (1.0 / sample_years) - 1.0)
    return {
        "n_trades": len(returns),
        "years_represented": int(trades["calendar_year"].nunique()),
        "mean_return": float(returns.mean()),
        "median_return": float(returns.median()),
        "win_rate": float((returns > 0).mean()),
        "t_stat": float(t_result.statistic),
        "one_sided_p": float(t_result.pvalue),
        "best_return": float(returns.max()),
        "worst_return": float(returns.min()),
        "unit_notional_compound": float((1.0 + returns).prod() - 1.0),
        "sleeve_total_return": sleeve_total,
        "sleeve_annualized_return": sleeve_annualized,
        "sleeve_exit_marked_max_drawdown": float(sleeve_drawdown.min()),
    }


def year_block_bootstrap(
    trades: pd.DataFrame,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    """Resample whole calendar-year event clusters."""

    if trades.empty:
        return {"ci_low": None, "ci_high": None, "prob_mean_le_zero": None}
    grouped = {
        int(year): group["net_return"].to_numpy(dtype=float)
        for year, group in trades.groupby("calendar_year")
    }
    years = np.array(sorted(grouped), dtype=int)
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=float)
    for sample in range(samples):
        chosen = rng.choice(years, size=len(years), replace=True)
        values = np.concatenate([grouped[int(year)] for year in chosen])
        means[sample] = values.mean()
    return {
        "ci_low": float(np.quantile(means, 0.05)),
        "ci_high": float(np.quantile(means, 0.95)),
        "prob_mean_le_zero": float((means <= 0.0).mean()),
    }


def leave_one_year_out(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year in sorted(trades.get("calendar_year", pd.Series(dtype=int)).unique()):
        remaining = trades[trades["calendar_year"] != year]["net_return"]
        rows.append(
            {
                "excluded_year": int(year),
                "n_trades": len(remaining),
                "mean_return": float(remaining.mean()) if len(remaining) else np.nan,
                "compound_return": (
                    float((1.0 + remaining).prod() - 1.0) if len(remaining) else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _all_forward_returns(features: pd.DataFrame, config: StrategyConfig) -> pd.Series:
    start_position = int(features.index.searchsorted(pd.Timestamp(config.start_date)))
    values: dict[pd.Timestamp, float] = {}
    for position in range(start_position, len(features)):
        record = forward_trade_return(
            features,
            position,
            config.hold_sessions,
            config.round_trip_cost_bps,
        )
        if record is not None:
            values[record["signal_date"]] = float(record["net_return"])
    return pd.Series(values, name="forward_net_return", dtype=float)


def matched_control_analysis(
    primary: pd.DataFrame,
    features: pd.DataFrame,
    vrc_state: pd.Series,
    config: StrategyConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Descriptively match to non-VRC dates by year, VIX, and VIX/VIX3M slope."""

    if primary.empty:
        return pd.DataFrame(), {
            "mean_control": None,
            "mean_lift": None,
            "n_pairs": 0,
            "inference": "descriptive_only",
        }
    forwards = _all_forward_returns(features, config)
    vix_rank = rolling_percentile(features["vix"], config.percentile_lookback)
    rows = []
    for trade in primary.itertuples(index=False):
        date = pd.Timestamp(trade.signal_date)
        current_vix_rank = vix_rank.get(date, np.nan)
        current_term_rank = features.at[date, "term_rank"]
        if not np.isfinite(current_vix_rank) or not np.isfinite(current_term_rank):
            continue
        vix_bucket = min(int(current_vix_rank // 20), 4)
        term_bucket = min(int(current_term_rank // (100 / 3)), 2)
        candidate_index = forwards.index
        candidate_vix = vix_rank.reindex(candidate_index)
        candidate_term = features["term_rank"].reindex(candidate_index)
        candidate_mask = (
            (candidate_index.year == date.year)
            & ((candidate_vix // 20).clip(upper=4) == vix_bucket)
            & ((candidate_term // (100 / 3)).clip(upper=2) == term_bucket)
            & ~vrc_state.reindex(candidate_index).fillna(False).to_numpy()
        )
        candidates = forwards.loc[candidate_mask]
        positions = features.index.get_indexer(candidates.index)
        signal_position = features.index.get_loc(date)
        candidates = candidates[np.abs(positions - signal_position) > 10]
        if candidates.empty:
            continue
        control = float(candidates.mean())
        rows.append(
            {
                "signal_date": date,
                "strategy_return": float(trade.net_return),
                "matched_control_return": control,
                "lift": float(trade.net_return - control),
                "n_controls": len(candidates),
            }
        )
    matched = pd.DataFrame(rows)
    if matched.empty:
        return matched, {
            "mean_control": None,
            "mean_lift": None,
            "n_pairs": 0,
            "inference": "descriptive_only",
        }
    return matched, {
        "mean_control": float(matched["matched_control_return"].mean()),
        "mean_lift": float(matched["lift"].mean()),
        "n_pairs": len(matched),
        "inference": (
            "descriptive_only; controls overlap and are reused, so no independent "
            "paired p-value is reported"
        ),
    }


def run_sensitivity(
    features: pd.DataFrame,
    config: StrategyConfig,
) -> pd.DataFrame:
    rows = []
    for fragility_threshold in (50.0, 67.0, 80.0):
        for vrc_threshold in (10.0, 15.0, 20.0):
            signals = build_signals(
                features,
                config,
                vrc_threshold=vrc_threshold,
                fragility_threshold=fragility_threshold,
            )
            for hold_sessions in (3, 5, 10):
                for cost_bps in (12.0, 25.0, 50.0):
                    variant_config = replace(
                        config,
                        hold_sessions=hold_sessions,
                        round_trip_cost_bps=cost_bps,
                    )
                    trades = run_trades(
                        signals["primary"],
                        features,
                        variant_config,
                        label="sensitivity",
                    )
                    summary = summarize_trades(trades, variant_config, features.index[-1])
                    rows.append(
                        {
                            "fragility_threshold": fragility_threshold,
                            "vrc_threshold": vrc_threshold,
                            "hold_sessions": hold_sessions,
                            "round_trip_cost_bps": cost_bps,
                            **summary,
                        }
                    )
    return pd.DataFrame(rows)


def run_short_horizon_overlay_grid(
    features: pd.DataFrame,
    config: StrategyConfig,
) -> pd.DataFrame:
    """Evaluate the predeclared short overlays without selecting a winner."""

    variants = (
        "vrc_x_ex_vrc_5d",
        "vrc_x_ex_vrc_21d",
        "vrc_x_ex_vrc_5d_21d_both",
        "vrc_x_ex_vrc_5d_21d_mean",
        "vrc_x_ex_vrc_63d_native",
    )
    rows: list[dict[str, Any]] = []
    for threshold in (60.0, 67.0, 75.0):
        signals = build_signals(
            features,
            config,
            fragility_threshold=threshold,
        )
        for cost_bps in (12.0, 25.0, 50.0):
            variant_config = replace(
                config,
                hold_sessions=5,
                round_trip_cost_bps=cost_bps,
            )
            for variant in variants:
                trades = run_trades(
                    signals[variant],
                    features,
                    variant_config,
                    label=variant,
                    pit_start=None,
                )
                summary = summarize_trades(
                    trades,
                    variant_config,
                    features.index[-1],
                )
                rows.append(
                    {
                        "variant": variant,
                        "fragility_threshold": threshold,
                        "round_trip_cost_bps": cost_bps,
                        **summary,
                    }
                )
    return pd.DataFrame(rows)


def _term_structure_summary(trades: pd.DataFrame) -> list[dict[str, Any]]:
    if trades.empty:
        return []
    buckets = pd.cut(
        trades["term_rank"],
        bins=[-np.inf, 100 / 3, 200 / 3, np.inf],
        labels=["contango / low spot", "middle", "backwardation / high spot"],
    )
    rows = []
    for bucket, group in trades.assign(term_bucket=buckets).groupby(
        "term_bucket", observed=True
    ):
        rows.append(
            {
                "bucket": str(bucket),
                "n_trades": len(group),
                "mean_return": float(group["net_return"].mean()),
                "win_rate": float((group["net_return"] > 0).mean()),
            }
        )
    return rows


def evaluate_failure_rules(
    primary: pd.DataFrame,
    summaries: dict[str, dict[str, Any]],
    bootstrap: dict[str, Any],
    sensitivity: pd.DataFrame,
    cost_50_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    primary_summary = summaries["primary"]
    vrc = summaries["vrc_only"]
    primary_mean = primary_summary["mean_return"]
    vrc_mean = vrc["mean_return"]
    years = primary["calendar_year"].nunique() if not primary.empty else 0
    without_best_two = primary.nlargest(2, "net_return").index
    trimmed = primary.drop(index=without_best_two)["net_return"] if len(primary) > 2 else pd.Series(dtype=float)
    year_pnl = primary.groupby("calendar_year")["net_return"].sum() if not primary.empty else pd.Series(dtype=float)
    positive_year_pnl = year_pnl.clip(lower=0.0)
    max_positive_share = (
        float(positive_year_pnl.max() / positive_year_pnl.sum())
        if positive_year_pnl.sum() > 0
        else 1.0
    )
    base_cost_neighbors = sensitivity[
        sensitivity["round_trip_cost_bps"].eq(12.0)
    ]
    positive_neighbor_fraction = float(
        (base_cost_neighbors["mean_return"].fillna(-np.inf) > 0).mean()
    )
    ex_2020 = primary[primary["calendar_year"] != 2020]["net_return"]

    checks = [
        ("At least 30 non-overlapping events", len(primary) >= 30, len(primary)),
        ("At least six represented years", years >= 6, int(years)),
        ("Net mean return is positive", primary_mean is not None and primary_mean > 0, primary_mean),
        ("Mean edge exceeds five times baseline cost", primary_mean is not None and primary_mean >= 0.006, primary_mean),
        (
            "Bootstrap share of resampled means <=0 is at most 10%",
            bootstrap.get("prob_mean_le_zero") is not None
            and bootstrap["prob_mean_le_zero"] <= 0.10,
            bootstrap.get("prob_mean_le_zero"),
        ),
        (
            "Interaction improves on VRC alone",
            primary_mean is not None and vrc_mean is not None and primary_mean > vrc_mean,
            None if primary_mean is None or vrc_mean is None else primary_mean - vrc_mean,
        ),
        (
            "Mean stays positive after removing best two trades",
            len(trimmed) > 0 and float(trimmed.mean()) > 0,
            float(trimmed.mean()) if len(trimmed) else None,
        ),
        (
            "No year supplies more than 50% of positive P&L",
            max_positive_share <= 0.50,
            max_positive_share,
        ),
        (
            "Mean remains positive at 50 bps round trip",
            cost_50_summary.get("mean_return") is not None
            and cost_50_summary["mean_return"] > 0,
            cost_50_summary.get("mean_return"),
        ),
        (
            "Mean remains positive excluding 2020",
            len(ex_2020) > 0 and float(ex_2020.mean()) > 0,
            float(ex_2020.mean()) if len(ex_2020) else None,
        ),
        (
            "A majority of neighboring definitions stay positive",
            positive_neighbor_fraction >= 0.50,
            positive_neighbor_fraction,
        ),
        (
            "At least one qualifying trade is genuinely PIT",
            bool(primary["genuine_pit"].any()) if not primary.empty else False,
            int(primary["genuine_pit"].sum()) if not primary.empty else 0,
        ),
    ]
    return [
        {"rule": rule, "passed": bool(passed), "observed": observed}
        for rule, passed, observed in checks
    ]


def run_research(data_dir: str | Path, config: StrategyConfig | None = None) -> dict[str, Any]:
    """Run the full primary backtest, comparators, and falsification battery."""

    config = config or StrategyConfig()
    inputs = load_inputs(data_dir)
    features = build_features(inputs, config)
    signals = build_signals(features, config)

    recomputed_fragility_variants = {
        "vrc_x_ex_vrc_5d",
        "vrc_x_ex_vrc_21d",
        "vrc_x_ex_vrc_5d_21d_both",
        "vrc_x_ex_vrc_5d_21d_mean",
        "vrc_x_ex_vrc_63d_native",
        "vrc_x_incumbent_5d_rank",
        "vrc_x_incumbent_21d_rank",
    }
    trades_by_variant = {
        label: run_trades(
            signals[label],
            features,
            config,
            label=label,
            pit_start=None if label in recomputed_fragility_variants else SIMPLE_DIAL_PIT_START,
        )
        for label in signals.columns
    }
    short_horizon_trades = pd.concat(
        [
            trades_by_variant[label]
            for label in sorted(recomputed_fragility_variants)
        ],
        ignore_index=True,
    ).sort_values(["signal_date", "variant"], ignore_index=True)
    summaries = {
        label: summarize_trades(trades, config, features.index[-1])
        for label, trades in trades_by_variant.items()
    }
    primary = trades_by_variant["primary"]
    bootstrap = year_block_bootstrap(
        primary,
        config.bootstrap_samples,
        config.random_seed,
    )
    looy = leave_one_year_out(primary)
    sensitivity = run_sensitivity(features, config)
    short_horizon_overlay_grid = run_short_horizon_overlay_grid(features, config)
    cost_50_config = replace(config, round_trip_cost_bps=50.0)
    cost_50_trades = run_trades(
        signals["primary"], features, cost_50_config, label="primary_cost_50"
    )
    cost_50_summary = summarize_trades(cost_50_trades, cost_50_config, features.index[-1])
    matched, matched_summary = matched_control_analysis(
        primary,
        features,
        features["vrc_on"],
        config,
    )
    failures = evaluate_failure_rules(
        primary,
        summaries,
        bootstrap,
        sensitivity,
        cost_50_summary,
    )

    unconditional = _all_forward_returns(features, config)
    unconditional_summary = {
        "n_overlapping_observations": len(unconditional),
        "mean_return": float(unconditional.mean()),
        "median_return": float(unconditional.median()),
        "win_rate": float((unconditional > 0).mean()),
    }
    vrc_selected = primary["net_return"]
    primary_dates = set(pd.to_datetime(primary["signal_date"]))
    vrc_unselected = trades_by_variant["vrc_only"]
    vrc_unselected = vrc_unselected[
        ~pd.to_datetime(vrc_unselected["signal_date"]).isin(primary_dates)
    ]["net_return"]
    interaction_lift = {
        "selected_vrc_mean": float(vrc_selected.mean()) if len(vrc_selected) else None,
        "other_vrc_mean": float(vrc_unselected.mean()) if len(vrc_unselected) else None,
        "lift": (
            float(vrc_selected.mean() - vrc_unselected.mean())
            if len(vrc_selected) and len(vrc_unselected)
            else None
        ),
    }
    short_overlay_robustness: dict[str, dict[str, Any]] = {}
    for label in (
        "vrc_x_ex_vrc_5d",
        "vrc_x_ex_vrc_21d",
        "vrc_x_ex_vrc_5d_21d_both",
        "vrc_x_ex_vrc_5d_21d_mean",
        "vrc_x_ex_vrc_63d_native",
    ):
        variant = trades_by_variant[label]
        trimmed = (
            variant.drop(index=variant.nlargest(2, "net_return").index)["net_return"]
            if len(variant) > 2
            else pd.Series(dtype=float)
        )
        ex_2020 = variant[variant["calendar_year"] != 2020]["net_return"]
        short_overlay_robustness[label] = {
            "mean_without_best_two": (
                float(trimmed.mean()) if len(trimmed) else None
            ),
            "mean_excluding_2020": (
                float(ex_2020.mean()) if len(ex_2020) else None
            ),
            "year_block_bootstrap": year_block_bootstrap(
                variant,
                config.bootstrap_samples,
                config.random_seed + len(label),
            ),
        }

    short_rank_pair = features[
        ["native_fragility_5d_rank", "native_fragility_21d_rank"]
    ].dropna()
    short_rank_correlation = (
        float(short_rank_pair.corr().iloc[0, 1])
        if len(short_rank_pair) > 1
        else None
    )

    last = features.dropna(subset=["vix", "uvxy_close"]).iloc[-1]
    simple_overlap = features["simple"].notna() & features["simple_ex_vrc"].notna()
    ex_vrc_values = features.loc[simple_overlap, "simple_ex_vrc"]
    ex_vrc_out_of_bounds = (ex_vrc_values < 0.0) | (ex_vrc_values > 100.0)
    vrc_component_history = inputs["vrc_component_history"]
    research_signal_history = inputs["research_signal_history"]
    metadata = {
        "data_start": features.index.min().strftime("%Y-%m-%d"),
        "data_end": features.index.max().strftime("%Y-%m-%d"),
        "primary_start": config.start_date,
        "uvxy_1_5x_start_date": UVXY_1_5X_START_DATE.strftime("%Y-%m-%d"),
        "simple_dial_pit_start": SIMPLE_DIAL_PIT_START.strftime("%Y-%m-%d"),
        "latest_uvxy_close": float(last["uvxy_close"]),
        "latest_vix": float(last["vix"]),
        "latest_fragility_rank": (
            float(last["fragility_rank"]) if np.isfinite(last["fragility_rank"]) else None
        ),
        "vrc_component_history_start": vrc_component_history.index.min().strftime(
            "%Y-%m-%d"
        ),
        "vrc_component_history_end": vrc_component_history.index.max().strftime(
            "%Y-%m-%d"
        ),
        "short_overlay_signal_vintage_start": (
            research_signal_history.index.min().strftime("%Y-%m-%d")
        ),
        "short_overlay_signal_vintage_end": (
            research_signal_history.index.max().strftime("%Y-%m-%d")
        ),
        "short_overlay_genuinely_pit_rows": 0,
        "vrc_activations_after_short_overlay_vintage": int(
            signals.loc[
                signals.index > research_signal_history.index.max(),
                "vrc_only",
            ].sum()
        ),
        "ex_vrc_parity": {
            "out_of_bounds_rows": int(ex_vrc_out_of_bounds.sum()),
            "minimum": float(ex_vrc_values.min()) if len(ex_vrc_values) else None,
            "maximum": float(ex_vrc_values.max()) if len(ex_vrc_values) else None,
        },
        "source_paths": inputs["paths"],
        "input_fingerprints": inputs["fingerprints"],
    }
    failed_count = sum(not item["passed"] for item in failures)
    verdict = {
        "status": "reject_as_alpha" if failed_count else "research_gate_passed",
        "actionability": "pass / research-only" if failed_count else "paper-track candidate",
        "failed_rules": failed_count,
        "total_rules": len(failures),
        "reason": (
            "The primary return is non-positive, the sample is sparse, and no "
            "qualifying event exists in the genuinely point-in-time segment."
            if failed_count
            else "All preregistered research gates passed; live deployment still requires approval."
        ),
    }
    return {
        "config": asdict(config),
        "metadata": metadata,
        "verdict": verdict,
        "summaries": summaries,
        "unconditional": unconditional_summary,
        "bootstrap": bootstrap,
        "matched_control": matched_summary,
        "interaction_lift": interaction_lift,
        "short_overlay_rank_correlation_5d_21d": short_rank_correlation,
        "short_overlay_robustness": short_overlay_robustness,
        "term_structure": _term_structure_summary(primary),
        "failure_rules": failures,
        "features": features,
        "signals": signals,
        "trades": primary,
        "trades_by_variant": trades_by_variant,
        "short_horizon_trades": short_horizon_trades,
        "sensitivity": sensitivity,
        "short_horizon_overlay_grid": short_horizon_overlay_grid,
        "leave_one_year_out": looy,
        "matched_control_pairs": matched,
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def serializable_results(results: dict[str, Any]) -> dict[str, Any]:
    excluded = {
        "features",
        "signals",
        "trades",
        "trades_by_variant",
        "short_horizon_trades",
        "sensitivity",
        "short_horizon_overlay_grid",
        "leave_one_year_out",
        "matched_control_pairs",
    }
    return _json_ready({key: value for key, value in results.items() if key not in excluded})


def _pct(value: Any, digits: int = 2) -> str:
    if value is None or not np.isfinite(float(value)):
        return "—"
    return f"{float(value) * 100:.{digits}f}%"


def _num(value: Any, digits: int = 2) -> str:
    if value is None or not np.isfinite(float(value)):
        return "—"
    return f"{float(value):.{digits}f}"


def _return_bars_svg(trades: pd.DataFrame) -> str:
    if trades.empty:
        return "<p>No qualifying trades.</p>"
    width, height = 980, 270
    pad_left, pad_right = 54, 20
    usable_w = width - pad_left - pad_right
    baseline = 125
    values = trades["net_return"].to_numpy(dtype=float)
    max_abs = max(float(np.abs(values).max()), 0.01)
    scale = 94 / max_abs
    step = usable_w / len(values)
    bars = []
    labels = []
    for index, row in enumerate(trades.itertuples(index=False)):
        value = float(row.net_return)
        bar_h = abs(value) * scale
        x = pad_left + index * step + step * 0.18
        y = baseline - bar_h if value >= 0 else baseline
        colour = "#1f8a70" if value >= 0 else "#c84b4b"
        date = pd.Timestamp(row.signal_date).strftime("%Y-%m-%d")
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{step * 0.64:.1f}" '
            f'height="{bar_h:.1f}" fill="{colour}" rx="2"><title>{date}: '
            f'{value * 100:+.2f}%</title></rect>'
        )
        labels.append(
            f'<text x="{x + step * 0.32:.1f}" y="{height - 18}" '
            f'text-anchor="end" transform="rotate(-55 {x + step * 0.32:.1f} '
            f'{height - 18})" font-size="10" fill="#667085">{date[:7]}</text>'
        )
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        'aria-label="Net return by qualifying UVXY trade">'
        f'<line x1="{pad_left}" y1="{baseline}" x2="{width-pad_right}" '
        f'y2="{baseline}" stroke="#98a2b3" stroke-width="1"/>'
        f'<text x="8" y="{baseline + 4}" font-size="11" fill="#667085">0%</text>'
        + "".join(bars)
        + "".join(labels)
        + "</svg>"
    )


def render_html_report(results: dict[str, Any]) -> str:
    """Render a self-contained, actionability-first research report."""

    verdict = results["verdict"]
    summary = results["summaries"]["primary"]
    meta = results["metadata"]
    bootstrap = results["bootstrap"]
    trades = results["trades"]
    comparisons = results["summaries"]
    failures = results["failure_rules"]
    sensitivity = results["sensitivity"]
    short_robustness = results["short_overlay_robustness"]
    short_rank_correlation = results["short_overlay_rank_correlation_5d_21d"]
    short_5d = comparisons["vrc_x_ex_vrc_5d"]
    short_21d = comparisons["vrc_x_ex_vrc_21d"]

    comparison_rows = []
    labels = {
        "primary": "VRC × lagged ex-VRC fragility",
        "vrc_only": "VRC activation only",
        "fragility_only": "Lagged ex-VRC fragility crossing",
        "vrc_x_incumbent_rank": "VRC × incumbent dial rank",
        "vrc_x_incumbent_50": "VRC × incumbent dial ≥50",
    }
    for key, label in labels.items():
        item = comparisons[key]
        comparison_rows.append(
            "<tr>"
            f"<td>{escape(label)}</td><td>{item['n_trades']}</td>"
            f"<td>{_pct(item['mean_return'])}</td><td>{_pct(item['median_return'])}</td>"
            f"<td>{_pct(item['win_rate'], 0)}</td><td>{_num(item['t_stat'])}</td>"
            "</tr>"
        )

    short_horizon_rows = []
    short_labels = (
        ("vrc_only", "VRC only — no fragility gate"),
        ("primary", "Original registered 63d sizing-basis control"),
        ("vrc_x_ex_vrc_5d", "Native 5d ex-VRC overlay"),
        ("vrc_x_ex_vrc_21d", "Native 21d ex-VRC overlay"),
        ("vrc_x_ex_vrc_5d_21d_both", "5d and 21d both in upper tercile"),
        ("vrc_x_ex_vrc_5d_21d_mean", "Mean 5d/21d rank in upper tercile"),
        ("vrc_x_ex_vrc_63d_native", "Native 63d ex-VRC overlay"),
        ("vrc_x_incumbent_5d_rank", "Incumbent 5d rank — circular diagnostic"),
        ("vrc_x_incumbent_21d_rank", "Incumbent 21d rank — circular diagnostic"),
    )
    for key, label in short_labels:
        item = comparisons[key]
        short_horizon_rows.append(
            "<tr>"
            f"<td>{escape(label)}</td><td>{item['n_trades']}</td>"
            f"<td>{_pct(item['mean_return'])}</td>"
            f"<td>{_pct(item['median_return'])}</td>"
            f"<td>{_pct(item['win_rate'], 0)}</td><td>{_num(item['t_stat'])}</td>"
            "</tr>"
        )

    short_robustness_rows = []
    for key, label in (
        ("vrc_x_ex_vrc_5d", "Native 5d"),
        ("vrc_x_ex_vrc_21d", "Native 21d"),
        ("vrc_x_ex_vrc_63d_native", "Native 63d"),
    ):
        item = short_robustness[key]
        interval = item["year_block_bootstrap"]
        short_robustness_rows.append(
            "<tr>"
            f"<td>{escape(label)}</td>"
            f"<td>{_pct(item['mean_without_best_two'])}</td>"
            f"<td>{_pct(item['mean_excluding_2020'])}</td>"
            f"<td>{_pct(interval.get('ci_low'))} to {_pct(interval.get('ci_high'))}</td>"
            f"<td>{_pct(interval.get('prob_mean_le_zero'), 0)}</td>"
            "</tr>"
        )

    rule_rows = []
    for item in failures:
        status = "Pass" if item["passed"] else "Fail"
        css = "pass" if item["passed"] else "fail"
        observed = item["observed"]
        if isinstance(observed, float):
            observed_text = _pct(observed) if abs(observed) <= 1 else _num(observed)
        else:
            observed_text = "—" if observed is None else escape(str(observed))
        rule_rows.append(
            f'<tr><td>{escape(item["rule"])}</td><td class="{css}">{status}</td>'
            f"<td>{observed_text}</td></tr>"
        )

    trade_rows = []
    for row in trades.itertuples(index=False):
        trade_rows.append(
            "<tr>"
            f"<td>{pd.Timestamp(row.signal_date):%Y-%m-%d}</td>"
            f"<td>{pd.Timestamp(row.gate_date):%Y-%m-%d}</td>"
            f"<td>{pd.Timestamp(row.entry_date):%Y-%m-%d}</td>"
            f"<td>{pd.Timestamp(row.exit_date):%Y-%m-%d}</td>"
            f"<td>{row.vix:.2f}</td><td>{row.compression_pctile:.1f}</td>"
            f"<td>{row.gate_fragility_rank:.1f}</td>"
            f'<td class="{"positive" if row.net_return >= 0 else "negative"}">'
            f"{row.net_return * 100:+.2f}%</td></tr>"
        )

    core_sensitivity = sensitivity[
        sensitivity["round_trip_cost_bps"].eq(12.0)
        & sensitivity["vrc_threshold"].eq(15.0)
    ]
    sensitivity_rows = []
    for threshold in (50.0, 67.0, 80.0):
        cells = []
        for hold in (3, 5, 10):
            row = core_sensitivity[
                core_sensitivity["fragility_threshold"].eq(threshold)
                & core_sensitivity["hold_sessions"].eq(hold)
            ].iloc[0]
            value = row["mean_return"]
            css = "heat-pos" if value > 0 else "heat-neg"
            cells.append(
                f'<td class="{css}">{_pct(value)}<small>N={int(row["n_trades"])}</small></td>'
            )
        sensitivity_rows.append(
            f"<tr><th>{threshold:.0f}th</th>{''.join(cells)}</tr>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>UVXY Compression × Fragility Research</title>
<style>
:root{{--ink:#101828;--muted:#667085;--line:#d0d5dd;--paper:#f7f8fa;--card:#fff;
--red:#b42318;--red-bg:#fef3f2;--green:#067647;--green-bg:#ecfdf3;--amber:#b54708;}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);
font:15px/1.55 Inter,Segoe UI,Arial,sans-serif}} main{{max-width:1120px;margin:0 auto;padding:34px 24px 72px}}
h1{{font-size:34px;line-height:1.12;margin:8px 0 10px;letter-spacing:-.03em}} h2{{font-size:22px;margin:38px 0 12px}}
h3{{font-size:16px;margin:20px 0 8px}} p{{margin:8px 0}} .eyebrow{{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.11em}}
.hero{{background:linear-gradient(135deg,#fff 0%,#fef7f6 100%);border:1px solid #f3c7c3;border-left:6px solid var(--red);border-radius:14px;padding:24px 26px}}
.verdict{{display:inline-block;background:var(--red-bg);color:var(--red);border:1px solid #fecdca;border-radius:999px;padding:5px 10px;font-weight:700;font-size:12px;text-transform:uppercase;letter-spacing:.05em}}
.lede{{font-size:18px;max-width:900px;color:#344054}} .grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:20px}}
.metric{{background:#fff;border:1px solid var(--line);border-radius:10px;padding:14px}} .metric b{{display:block;font-size:23px;letter-spacing:-.02em}}
.metric span{{color:var(--muted);font-size:12px}} .card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px 20px;margin:14px 0}}
.gate-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}} .gate{{border:1px solid var(--line);border-radius:9px;padding:12px}}
.gate strong{{display:block}} .gate em{{font-style:normal;color:var(--red);font-weight:700;font-size:12px}}
table{{width:100%;border-collapse:collapse;background:#fff}} th,td{{text-align:right;padding:10px 11px;border-bottom:1px solid #eaecf0;vertical-align:top}}
th:first-child,td:first-child{{text-align:left}} thead th{{color:#475467;font-size:12px;text-transform:uppercase;letter-spacing:.04em;background:#f9fafb}}
.pass,.positive{{color:var(--green);font-weight:700}} .fail,.negative{{color:var(--red);font-weight:700}} .heat-pos{{background:var(--green-bg)}} .heat-neg{{background:var(--red-bg)}}
td small{{display:block;color:var(--muted);font-size:10px}} .note{{color:var(--muted);font-size:13px}} .callout{{border-left:4px solid #f79009;background:#fffaeb;padding:12px 15px;border-radius:6px}}
code{{font-family:Consolas,monospace;font-size:.92em;background:#f2f4f7;padding:2px 4px;border-radius:4px}} a{{color:#175cd3}} ul{{padding-left:21px}} .scroll{{overflow-x:auto}}
@media(max-width:800px){{.grid{{grid-template-columns:repeat(2,1fr)}}.gate-grid{{grid-template-columns:1fr}}main{{padding:22px 14px}}h1{{font-size:28px}}}}
</style>
</head>
<body><main>
<section class="hero">
  <div class="eyebrow">Research verdict · data through {meta['data_end']}</div>
  <div class="verdict">Reject / do not deploy</div>
  <h1>Shorter fragility did not rescue this VRC-activation UVXY rule</h1>
  <p class="lede">The original 63d sizing-basis rule returned {_pct(summary['mean_return'])} per trade. Rebuilding the independent ex-VRC overlay at 5d and 21d made the mean {_pct(short_5d['mean_return'])} and {_pct(short_21d['mean_return'])}, respectively, after 12 bps round-trip cost. The registered rule still failed {verdict['failed_rules']} of {verdict['total_rules']} research gates.</p>
  <div class="grid">
    <div class="metric"><b>{summary['n_trades']}</b><span>non-overlapping events</span></div>
    <div class="metric"><b>{_pct(summary['mean_return'])}</b><span>mean net return</span></div>
    <div class="metric"><b>{_pct(summary['win_rate'],0)}</b><span>win rate</span></div>
    <div class="metric"><b>{_num(summary['t_stat'])}</b><span>event-level t-stat</span></div>
  </div>
</section>

<h2>Proposed trade / actionability</h2>
<div class="card">
  <p><strong>Posture:</strong> reject as an alpha strategy; retain only as a forward paper-test specification. No live strategy-book, order-staging, or portfolio changes are justified.</p>
  <p><strong>Expression tested:</strong> long UVXY at the next open, held through the fifth post-signal session's close. The setup uses VIX compression as timing and a prior-day, equal-weight fragility composite with VRC removed as the independent risk state. The overlay was frozen at 5d, 21d, and 63d decay horizons before comparing returns.</p>
  <p><strong>Why it failed:</strong> the interaction did not turn UVXY's carry hurdle into positive expectancy. The short overlays generated only {short_5d['n_trades']} and {short_21d['n_trades']} events, were highly redundant, and have no genuinely point-in-time historical rows.</p>
</div>

<h2>Variant wedge and implementation gate</h2>
<p>The hypothesis was intuitive: a tight VIX range may contain stored convexity, while elevated non-VRC fragility may indicate vulnerability to a volatility release. This test finds no evidence that their conjunction times a tradeable VIX-futures jump.</p>
<div class="gate-grid">
  <div class="gate"><em>Not cleared</em><strong>Positive expectancy</strong><span>Primary net mean {_pct(summary['mean_return'])}; median {_pct(summary['median_return'])}.</span></div>
  <div class="gate"><em>Missing</em><strong>True out-of-sample history</strong><span>Historical dial rows are recompute vintages; qualifying PIT trades: {int(trades['genuine_pit'].sum()) if not trades.empty else 0}.</span></div>
  <div class="gate"><em>Not cleared</em><strong>Power</strong><span>{summary['n_trades']} trades versus a 30-trade minimum.</span></div>
  <div class="gate"><em>Implemented</em><strong>Basic execution model</strong><span>Close-known signal, next-open entry, adjusted OHLC, and flat costs; no quote-level fill validation.</span></div>
  <div class="gate"><em>Cleared</em><strong>No VRC double-count</strong><span>The primary fragility leg algebraically removes VRC before ranking.</span></div>
  <div class="gate"><em>Not cleared</em><strong>Robustness</strong><span>Holding-period and threshold neighbors change sign.</span></div>
</div>

<h2>Short-horizon fragility overlay</h2>
<div class="card">
  <p><strong>Construction:</strong> recompute the six registered non-VRC signals under one frozen current code vintage; hold each component at 1 while active, decay it linearly over 5, 21, or 63 sessions, equal-weight the six components, smooth once over five sessions, and rank against only the preceding 504 sessions. The signal date uses the prior close's rank.</p>
  <p><strong>Finding:</strong> the 5d and 21d ranks are {_num(short_rank_correlation)} correlated, and both short overlays lose more than VRC alone. Requiring both does not repair the expectancy. The native 63d mean is positive only because of a few outsized events: its median is {_pct(comparisons['vrc_x_ex_vrc_63d_native']['median_return'])}, its t-stat is {_num(comparisons['vrc_x_ex_vrc_63d_native']['t_stat'])}, and removing its best two trades leaves {_pct(short_robustness['vrc_x_ex_vrc_63d_native']['mean_without_best_two'])}.</p>
</div>
<div class="scroll"><table><thead><tr><th>Overlay</th><th>Trades</th><th>Mean</th><th>Median</th><th>Win rate</th><th>t-stat</th></tr></thead><tbody>{''.join(short_horizon_rows)}</tbody></table></div>
<p class="note">“Native” rows are transparent research overlays, not production dials. They are honest with respect to VRC exclusion, equal weighting, lagging, and causal ranking, but they are full-history current-vintage recomputations—not point-in-time backtests. The incumbent 5d/21d rows are shown only as circular diagnostics because their stored production dials include VRC and outcome-fitted weights. Per-trade rows for every short-horizon variant are exported in <code>uvxy_vol_alpha_short_horizon_trades.csv</code>.</p>

<h3>Outlier and regime checks</h3>
<div class="scroll"><table><thead><tr><th>Overlay</th><th>Mean without best 2</th><th>Mean excluding 2020</th><th>Year-block 90% interval</th><th>Bootstrap means ≤0</th></tr></thead><tbody>{''.join(short_robustness_rows)}</tbody></table></div>
<p class="note">All three horizon estimates are too sparse and unstable for deployment. A positive full-sample mean in one neighboring definition is not treated as model selection evidence.</p>

<h2 id="primary-result">Primary result</h2>
<div class="card">{_return_bars_svg(trades)}<p class="note">Net UVXY return by trade. The large positive events do not overcome the negative median and carry bleed.</p></div>
<div class="scroll"><table><thead><tr><th>Variant</th><th>Trades</th><th>Mean</th><th>Median</th><th>Win rate</th><th>t-stat</th></tr></thead><tbody>{''.join(comparison_rows)}</tbody></table></div>
<p class="note">Unconditional overlapping five-session UVXY mean: {_pct(results['unconditional']['mean_return'])}. Descriptive lift versus year/VIX/VIX-to-VIX3M-slope matched non-VRC dates: {_pct(results['matched_control'].get('mean_lift'))}; controls overlap, so no independent paired p-value is claimed. Year-block 90% interval: {_pct(bootstrap.get('ci_low'))} to {_pct(bootstrap.get('ci_high'))}; share of resampled means ≤ 0: {_pct(bootstrap.get('prob_mean_le_zero'),0)}.</p>

<h2>Definition sensitivity</h2>
<div class="scroll"><table><thead><tr><th>Fragility rank gate</th><th>3 sessions</th><th>5 sessions</th><th>10 sessions</th></tr></thead><tbody>{''.join(sensitivity_rows)}</tbody></table></div>
<p class="note">Mean net return at the production 15th-percentile compression threshold and 12 bps cost. Green cells are not validation: several have tiny N and were inspected only after the primary was frozen.</p>

<h2 id="failure-checks">Predeclared failure checks</h2>
<div class="scroll"><table><thead><tr><th>Rule</th><th>Status</th><th>Observed</th></tr></thead><tbody>{''.join(rule_rows)}</tbody></table></div>

<h2 id="trade-ledger">Trade ledger</h2>
<div class="scroll"><table><thead><tr><th>Signal</th><th>Fragility gate date</th><th>Entry</th><th>Exit</th><th>VIX</th><th>Compression pctile</th><th>Prior-day fragility pctile</th><th>Net return</th></tr></thead><tbody>{''.join(trade_rows)}</tbody></table></div>

<h2>Conditional action rules</h2>
<div class="card">
<ul>
  <li>Do not add this rule to <code>strategy_config.py</code> or stage orders from it.</li>
  <li>Paper-log the frozen signal only; do not tune thresholds from future misses.</li>
  <li>Re-underwrite after at least 30 non-overlapping signals and six represented years, with positive net mean after removing the best two trades.</li>
  <li>A future version needs an independent breakout trigger or VIX-futures term-structure input, preregistered before its outcomes are observed.</li>
</ul>
</div>

<h2 id="sources">Sources and limitations</h2>
<div class="card">
<p><strong>Market data:</strong> local adjusted <code>master_prices.parquet</code>, through {meta['data_end']}. Reverse splits are therefore handled, but vendor adjustments and the largest event bars still merit manual source verification before any future deployment.</p>
<p><strong>Original fragility control:</strong> local <code>rd2_fragility_simple.parquet</code>, <code>rd2_fragility.parquet</code>, and <code>signal_fire_history.parquet</code>. VRC is algebraically removed from the original simple score; parity check: {meta['ex_vrc_parity']['out_of_bounds_rows']} out-of-bounds rows. The simple series is genuinely append-only only from {meta['simple_dial_pit_start']}; earlier history is a recompute vintage.</p>
<p><strong>Short overlays:</strong> the six non-VRC component histories are recomputed from <code>rd2_spy_ohlc.parquet</code>, <code>rd2_closes.parquet</code>, and <code>rd2_sp500_closes.parquet</code> under one current definition vintage spanning {meta['short_overlay_signal_vintage_start']} through {meta['short_overlay_signal_vintage_end']}. Genuinely PIT rows: {meta['short_overlay_genuinely_pit_rows']}. VRC activations between that vintage end and the market-data end: {meta['vrc_activations_after_short_overlay_vintage']}. Residual definition lookahead still blocks an OOS claim.</p>
<p><strong>Reproducibility:</strong> the JSON summary fingerprints the explicitly loaded caches, the classification and seasonal-rank files used by the current-vintage recomputation, and the relevant local source files. Loaded-row counts are recorded for the directly loaded caches. This identifies the local research vintage but is not a lockfile for the full Python environment. VIX/VIX3M is only a volatility-slope proxy, not direct VX1/VX2 roll carry.</p>
<p><strong>Instrument:</strong> <a href="{OFFICIAL_UVXY_URL}">ProShares UVXY product page</a>. UVXY targets 1.5× the <em>daily</em> S&amp;P 500 VIX Short-Term Futures Index, not 1.5× spot VIX; {meta['uvxy_1_5x_start_date']} is the first return date under the 1.5× objective. The primary sample begins {meta['primary_start']}.</p>
<p class="callout"><strong>Research posture:</strong> historical, screen-grade falsification—not investment advice, a live signal, or authorization to trade.</p>
</div>
</main></body></html>"""


def write_outputs(results: dict[str, Any], output_dir: str | Path) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "report": output / "uvxy_vol_alpha_report.html",
        "summary": output / "uvxy_vol_alpha_summary.json",
        "trades": output / "uvxy_vol_alpha_trades.csv",
        "sensitivity": output / "uvxy_vol_alpha_sensitivity.csv",
        "leave_one_year_out": output / "uvxy_vol_alpha_leave_one_year_out.csv",
        "matched_controls": output / "uvxy_vol_alpha_matched_controls.csv",
        "short_horizon_overlays": output
        / "uvxy_vol_alpha_short_horizon_overlays.csv",
        "short_horizon_trades": output
        / "uvxy_vol_alpha_short_horizon_trades.csv",
    }
    paths["report"].write_text(render_html_report(results), encoding="utf-8")
    paths["summary"].write_text(
        json.dumps(serializable_results(results), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    results["trades"].to_csv(paths["trades"], index=False)
    results["sensitivity"].to_csv(paths["sensitivity"], index=False)
    results["leave_one_year_out"].to_csv(paths["leave_one_year_out"], index=False)
    results["matched_control_pairs"].to_csv(paths["matched_controls"], index=False)
    results["short_horizon_overlay_grid"].to_csv(
        paths["short_horizon_overlays"], index=False
    )
    results["short_horizon_trades"].to_csv(
        paths["short_horizon_trades"], index=False
    )
    return paths
