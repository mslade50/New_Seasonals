"""Deterministic, research-only funnel for the full equity universe.

The funnel deliberately stops before hypothesis testing or execution.  It
uses only bars available on or before ``asof``, gives every requested ticker
an auditable coverage verdict, ranks eligible names inside distinct research
archetypes, and emits bounded queues for a later research workflow.

There is intentionally no universal alpha score, portfolio state, position
sizing, staging, broker access, network access, or recommendation label here.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCHEMA_VERSION = "wide-opportunity-book/v0"
RESEARCH_ONLY_LABEL = "RESEARCH PRIORITY ONLY — NOT AN INVESTMENT RECOMMENDATION"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_ROOT = (PROJECT_ROOT / "artifacts").resolve()
MIN_DEFAULT_UNIVERSE_SIZE = 1_000
MIN_DEFAULT_OVERFLOW_SIZE = 500

ARCHETYPE_DESCRIPTIONS: dict[str, str] = {
    "residual_dislocation": (
        "Large market/sector-adjusted movement with confirming participation; "
        "asks whether the idiosyncratic shock persists or mean-reverts."
    ),
    "trend_acceleration": (
        "Established medium-term strength that is accelerating relative to "
        "the market; asks whether continuation survives neutral controls."
    ),
    "trend_pullback": (
        "Positive medium-term structure paired with a recent setback; asks "
        "whether pullbacks behave differently from generic weakness."
    ),
    "participation_shock": (
        "Unusual share and dollar-volume participation accompanying a price "
        "move; asks whether information arrival changes the forward path."
    ),
    "volatility_transition": (
        "A jump in short-window volatility or range relative to the prior "
        "regime; asks whether the transition predicts continuation or decay."
    ),
}


@dataclass(frozen=True)
class OpportunityConfig:
    """Configuration whose defaults produce a bounded daily research funnel."""

    asof: str | pd.Timestamp
    review_limit: int = 75
    deep_test_limit: int = 10
    audit_limit: int = 10
    audit_seed: int = 1729
    min_history: int = 63
    max_stale_sessions: int = 2
    market_ticker: str = "SPY"

    def normalized_asof(self) -> pd.Timestamp:
        value = pd.Timestamp(self.asof)
        if value.tzinfo is not None:
            value = value.tz_localize(None)
        return value.normalize()

    def validate(self) -> None:
        for name in ("review_limit", "deep_test_limit", "audit_limit"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.deep_test_limit > self.review_limit:
            raise ValueError("deep_test_limit cannot exceed review_limit")
        if self.min_history < 2:
            raise ValueError("min_history must be at least 2")
        if self.max_stale_sessions < 0:
            raise ValueError("max_stale_sessions must be non-negative")


@dataclass
class OpportunityBookResult:
    manifest: dict[str, Any]
    coverage: pd.DataFrame
    features: pd.DataFrame
    review_queue: pd.DataFrame
    deep_test_queue: pd.DataFrame
    audit_sample: pd.DataFrame


def default_universe() -> list[str]:
    """Return the liquid + overflow union, failing if fallback collapsed it."""
    from strategy_config import CSV_UNIVERSE, LIQUID_PLUS_COMMODITIES

    liquid = {_clean_ticker(t) for t in LIQUID_PLUS_COMMODITIES if _clean_ticker(t)}
    broad = {_clean_ticker(t) for t in CSV_UNIVERSE if _clean_ticker(t)}
    universe = sorted(liquid | broad)
    overflow = broad - liquid
    if (
        len(universe) < MIN_DEFAULT_UNIVERSE_SIZE
        or len(overflow) < MIN_DEFAULT_OVERFLOW_SIZE
    ):
        raise RuntimeError(
            "default broad universe is incomplete "
            f"({len(universe)} total, {len(overflow)} overflow); "
            "supply a reviewed --tickers-file instead of silently running liquid-only"
        )
    return universe


def _clean_ticker(value: Any) -> str:
    return str(value or "").strip().upper()


def _naive_dates(values: Any) -> pd.Series:
    converted = pd.to_datetime(values, errors="coerce")
    series = pd.Series(converted)
    try:
        if series.dt.tz is not None:
            series = series.dt.tz_convert(None)
    except (AttributeError, TypeError):
        pass
    return series.dt.normalize()


def _from_yfinance_multiindex(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert either yfinance MultiIndex orientation to canonical long form.

    yfinance multi-ticker downloads normally use ``(Price, Ticker)``.  Some
    saved frames reverse the levels, so identify the OHLCV level by values
    instead of assuming its position.
    """
    wanted = {"open", "high", "low", "close", "adj close", "volume"}
    matches: list[int] = []
    for level in range(raw.columns.nlevels):
        labels = {str(v).strip().lower() for v in raw.columns.get_level_values(level)}
        if "close" in labels and labels & wanted:
            matches.append(level)
    if len(matches) != 1 or raw.columns.nlevels != 2:
        raise ValueError("cannot identify Price/Ticker levels in MultiIndex prices")
    price_level = matches[0]
    ticker_level = 1 - price_level
    ticker_labels: dict[str, Any] = {}
    for original in raw.columns.get_level_values(ticker_level):
        clean = _clean_ticker(original)
        if clean:
            ticker_labels.setdefault(clean, original)
    frames: list[pd.DataFrame] = []
    for ticker, original in sorted(ticker_labels.items()):
        try:
            part = raw.xs(original, level=ticker_level, axis=1)
        except KeyError:
            continue
        if isinstance(part.columns, pd.MultiIndex):
            part.columns = part.columns.get_level_values(price_level)
        part = part.copy()
        part.columns = [str(c).strip().title() for c in part.columns]
        part["date"] = raw.index
        part["ticker"] = ticker
        frames.append(part.reset_index(drop=True))
    if not frames:
        raise ValueError("MultiIndex price frame contains no ticker panels")
    return pd.concat(frames, ignore_index=True)


def normalize_prices(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize long-form or yfinance-wide OHLCV into canonical long form.

    Missing Open/High/Low/Volume values are retained as NaN so the affected
    feature can degrade without silently excluding an otherwise useful close
    history.  Duplicate ticker/date rows keep the final observation.
    """
    if not isinstance(raw, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame")
    frame = (
        _from_yfinance_multiindex(raw)
        if isinstance(raw.columns, pd.MultiIndex)
        else raw.copy()
    )

    aliases = {str(c).strip().lower(): c for c in frame.columns}
    rename: dict[Any, str] = {}
    for canonical in ("date", "ticker", "open", "high", "low", "close", "volume"):
        if canonical in aliases:
            rename[aliases[canonical]] = canonical.title()
    frame = frame.rename(columns=rename)
    if "Date" not in frame.columns and isinstance(frame.index, pd.DatetimeIndex):
        frame = frame.copy()
        frame["Date"] = frame.index
    required = {"Date", "Ticker", "Close"}
    absent = sorted(required - set(frame.columns))
    if absent:
        raise ValueError(f"price frame missing required columns: {absent}")
    for column in ("Open", "High", "Low", "Volume"):
        if column not in frame.columns:
            frame[column] = np.nan

    out = frame[["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]].copy()
    out["Date"] = _naive_dates(out["Date"]).to_numpy()
    out["Ticker"] = out["Ticker"].map(_clean_ticker)
    for column in ("Open", "High", "Low", "Close", "Volume"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna(subset=["Date"])
    out = out[out["Ticker"] != ""]
    out = (
        out.sort_values(["Ticker", "Date"], kind="mergesort")
        .drop_duplicates(["Ticker", "Date"], keep="last")
        .reset_index(drop=True)
    )
    return out


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _pct_return(close: pd.Series, sessions: int) -> float | None:
    if len(close) <= sessions:
        return None
    before, after = _finite(close.iloc[-sessions - 1]), _finite(close.iloc[-1])
    if before is None or before <= 0 or after is None:
        return None
    return after / before - 1.0


def _annualized_vol(returns: pd.Series, sessions: int) -> float | None:
    window = returns.dropna().tail(sessions)
    if len(window) < max(3, min(sessions, 5)):
        return None
    return _finite(window.std(ddof=1) * np.sqrt(252.0))


def _relative_features(
    asset: pd.Series, benchmark: pd.Series | None
) -> dict[str, float | None]:
    empty = {
        "beta_63d": None,
        "residual_1d": None,
        "residual_5d": None,
        "residual_21d": None,
        "residual_z_63d": None,
    }
    if benchmark is None:
        return empty
    aligned = (
        pd.concat([asset.rename("asset"), benchmark.rename("bench")], axis=1)
        .dropna()
        .tail(126)
    )
    if len(aligned) < 21:
        return empty
    trailing = aligned.tail(63)
    variance = trailing["bench"].var(ddof=1)
    if not np.isfinite(variance) or variance <= 0:
        return empty
    beta = trailing["asset"].cov(trailing["bench"]) / variance
    residual = aligned["asset"] - beta * aligned["bench"]
    resid_std = residual.tail(63).std(ddof=1)
    return {
        "beta_63d": _finite(beta),
        "residual_1d": _finite(residual.iloc[-1]),
        "residual_5d": _finite(residual.tail(5).sum()),
        "residual_21d": _finite(residual.tail(21).sum()),
        "residual_z_63d": (
            _finite(residual.iloc[-1] / resid_std)
            if np.isfinite(resid_std) and resid_std > 0
            else None
        ),
    }


def _ticker_features(
    group: pd.DataFrame,
    market_returns: pd.Series | None,
    sector_returns: pd.Series | None,
    sector_label: str | None,
) -> dict[str, Any]:
    group = group.sort_values("Date", kind="mergesort")
    close = group.set_index("Date")["Close"].astype(float)
    daily = close.pct_change(fill_method=None)
    latest_close = _finite(close.iloc[-1])
    previous_close = _finite(close.iloc[-2]) if len(close) > 1 else None
    latest_open = _finite(group["Open"].iloc[-1])

    high = pd.to_numeric(group["High"], errors="coerce")
    low = pd.to_numeric(group["Low"], errors="coerce")
    previous = pd.Series(group["Close"]).shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - previous).abs(), (low - previous).abs()], axis=1
    ).max(axis=1, skipna=True)
    atr = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean().iloc[-1]

    volume = pd.to_numeric(group["Volume"], errors="coerce")
    dollar_volume = volume * pd.to_numeric(group["Close"], errors="coerce")
    volume_window = volume.tail(63).dropna()
    dollar_window = dollar_volume.tail(63).dropna()
    volume_base = volume_window.median() if not volume_window.empty else np.nan
    dollar_base = dollar_window.median() if not dollar_window.empty else np.nan
    vol5 = _annualized_vol(daily, 5)
    vol21 = _annualized_vol(daily, 21)
    vol63 = _annualized_vol(daily, 63)
    trailing_252 = close.tail(252)
    high_252 = _finite(trailing_252.max()) if len(close) >= 252 else None
    low_252 = _finite(trailing_252.min()) if len(close) >= 252 else None

    row: dict[str, Any] = {
        "latest_bar": str(pd.Timestamp(group["Date"].iloc[-1]).date()),
        "history_bars": len(group),
        "close": latest_close,
        "ret_1d": _pct_return(close, 1),
        "ret_5d": _pct_return(close, 5),
        "ret_21d": _pct_return(close, 21),
        "ret_63d": _pct_return(close, 63),
        "ret_126d": _pct_return(close, 126),
        "ret_252d": _pct_return(close, 252),
        "gap_1d": (
            latest_open / previous_close - 1.0
            if latest_open is not None and previous_close not in (None, 0)
            else None
        ),
        "atr_14": _finite(atr),
        "atr_pct": (
            _finite(atr / latest_close) if latest_close and np.isfinite(atr) else None
        ),
        "rvol_5d_ann": vol5,
        "rvol_21d_ann": vol21,
        "rvol_63d_ann": vol63,
        "vol_regime_ratio": (
            _finite(vol5 / vol63)
            if vol5 is not None and vol63 not in (None, 0)
            else None
        ),
        "volume_shock_63d": (
            _finite(volume.iloc[-1] / volume_base)
            if np.isfinite(volume_base) and volume_base > 0
            else None
        ),
        "dollar_volume_latest": _finite(dollar_volume.iloc[-1]),
        "dollar_volume_20d_mean": _finite(dollar_volume.tail(20).mean()),
        "dollar_volume_shock_63d": (
            _finite(dollar_volume.iloc[-1] / dollar_base)
            if np.isfinite(dollar_base) and dollar_base > 0
            else None
        ),
        "dist_52w_high": (
            latest_close / high_252 - 1.0 if latest_close and high_252 else None
        ),
        "dist_52w_low": (
            latest_close / low_252 - 1.0 if latest_close and low_252 else None
        ),
        "dist_sma20": None,
        "dist_sma50": None,
        "dist_sma200": None,
        "sector_label": sector_label,
    }
    for sessions in (20, 50, 200):
        if len(close) >= sessions and latest_close:
            average = _finite(close.tail(sessions).mean())
            row[f"dist_sma{sessions}"] = (
                latest_close / average - 1.0 if average else None
            )

    market = _relative_features(daily, market_returns)
    row.update({f"market_{key}": value for key, value in market.items()})
    sector = _relative_features(daily, sector_returns)
    row.update({f"sector_{key}": value for key, value in sector.items()})
    return row


def _rank_percentile(series: pd.Series, *, ascending: bool = True) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.rank(method="average", pct=True, ascending=ascending) * 100.0


def _weighted_score(
    frame: pd.DataFrame, components: list[tuple[str, float]]
) -> pd.Series:
    numerator = pd.Series(0.0, index=frame.index)
    denominator = pd.Series(0.0, index=frame.index)
    for column, weight in components:
        values = pd.to_numeric(frame[column], errors="coerce")
        valid = values.notna()
        numerator.loc[valid] += values.loc[valid] * weight
        denominator.loc[valid] += weight
    result = numerator / denominator.replace(0.0, np.nan)
    return result.where(denominator > 0)


def _add_cross_sectional_ranks(features: pd.DataFrame) -> pd.DataFrame:
    frame = features.copy()
    source_ranks: dict[str, tuple[str, bool, bool]] = {
        "ret_5d": ("ret_5d", True, False),
        "ret_21d": ("ret_21d", True, False),
        "ret_63d": ("ret_63d", True, False),
        "ret_126d": ("ret_126d", True, False),
        "dist_sma200": ("dist_sma200", True, False),
        "near_52w_high": ("dist_52w_high", True, False),
        "abs_gap": ("gap_1d", True, True),
        "abs_market_residual_1d": ("market_residual_1d", True, True),
        "abs_market_residual_5d": ("market_residual_5d", True, True),
        "market_residual_5d": ("market_residual_5d", True, False),
        "volume_shock": ("volume_shock_63d", True, False),
        "dollar_volume_shock": ("dollar_volume_shock_63d", True, False),
        "vol_regime": ("vol_regime_ratio", True, False),
        "atr_pct": ("atr_pct", True, False),
        "abs_ret_1d": ("ret_1d", True, True),
    }
    for label, (source, ascending, absolute) in source_ranks.items():
        values = pd.to_numeric(frame[source], errors="coerce")
        if absolute:
            values = values.abs()
        frame[f"xrank_{label}"] = _rank_percentile(values, ascending=ascending)

    inverse_ret5 = 100.0 - frame["xrank_ret_5d"]
    inverse_resid5 = 100.0 - frame["xrank_market_residual_5d"]
    scores = {
        "residual_dislocation": _weighted_score(
            frame.assign(
                _r5=frame["xrank_abs_market_residual_5d"],
                _r1=frame["xrank_abs_market_residual_1d"],
            ),
            [
                ("_r5", 0.40),
                ("_r1", 0.20),
                ("xrank_dollar_volume_shock", 0.25),
                ("xrank_abs_gap", 0.15),
            ],
        ),
        "trend_acceleration": _weighted_score(
            frame,
            [
                ("xrank_ret_63d", 0.30),
                ("xrank_ret_126d", 0.20),
                ("xrank_dist_sma200", 0.20),
                ("xrank_near_52w_high", 0.15),
                ("xrank_market_residual_5d", 0.15),
            ],
        ),
        "trend_pullback": _weighted_score(
            frame.assign(_inverse_ret5=inverse_ret5, _inverse_resid5=inverse_resid5),
            [
                ("xrank_ret_126d", 0.30),
                ("xrank_dist_sma200", 0.25),
                ("_inverse_ret5", 0.25),
                ("_inverse_resid5", 0.20),
            ],
        ),
        "participation_shock": _weighted_score(
            frame,
            [
                ("xrank_dollar_volume_shock", 0.35),
                ("xrank_volume_shock", 0.25),
                ("xrank_abs_market_residual_1d", 0.20),
                ("xrank_abs_gap", 0.10),
                ("xrank_vol_regime", 0.10),
            ],
        ),
        "volatility_transition": _weighted_score(
            frame,
            [
                ("xrank_vol_regime", 0.35),
                ("xrank_atr_pct", 0.20),
                ("xrank_abs_ret_1d", 0.20),
                ("xrank_abs_market_residual_5d", 0.15),
                ("xrank_dollar_volume_shock", 0.10),
            ],
        ),
    }
    for archetype, score in scores.items():
        score_col = f"archetype_{archetype}_score"
        rank_col = f"archetype_{archetype}_rank"
        percentile_col = f"archetype_{archetype}_percentile"
        frame[score_col] = score
        valid = (
            frame[["Ticker", score_col]]
            .dropna()
            .sort_values(
                [score_col, "Ticker"], ascending=[False, True], kind="mergesort"
            )
        )
        rank_map = {
            ticker: rank for rank, ticker in enumerate(valid["Ticker"], start=1)
        }
        n = len(valid)
        frame[rank_col] = frame["Ticker"].map(rank_map).astype("Int64")
        frame[percentile_col] = np.nan
        has_rank = frame[rank_col].notna()
        if n == 1:
            frame.loc[has_rank, percentile_col] = 100.0
        elif n > 1:
            frame.loc[has_rank, percentile_col] = (
                100.0 * (n - frame.loc[has_rank, rank_col].astype(float)) / (n - 1)
            )
    return frame


def _fmt_pct(value: Any) -> str:
    number = _finite(value)
    return "n/a" if number is None else f"{number * 100:+.1f}%"


def _fmt_ratio(value: Any) -> str:
    number = _finite(value)
    return "n/a" if number is None else f"{number:.2f}x"


def _why_now(row: pd.Series, archetype: str) -> str:
    if archetype == "residual_dislocation":
        return (
            f"5d market residual {_fmt_pct(row.get('market_residual_5d'))}; "
            f"1d gap {_fmt_pct(row.get('gap_1d'))}; dollar-volume shock "
            f"{_fmt_ratio(row.get('dollar_volume_shock_63d'))}."
        )
    if archetype == "trend_acceleration":
        return (
            f"63d return {_fmt_pct(row.get('ret_63d'))}; distance from "
            f"200d mean {_fmt_pct(row.get('dist_sma200'))}; 5d market "
            f"residual {_fmt_pct(row.get('market_residual_5d'))}."
        )
    if archetype == "trend_pullback":
        return (
            f"126d return {_fmt_pct(row.get('ret_126d'))} alongside a "
            f"5d move of {_fmt_pct(row.get('ret_5d'))}; distance from "
            f"200d mean {_fmt_pct(row.get('dist_sma200'))}."
        )
    if archetype == "participation_shock":
        return (
            f"Share-volume shock {_fmt_ratio(row.get('volume_shock_63d'))} "
            f"and dollar-volume shock "
            f"{_fmt_ratio(row.get('dollar_volume_shock_63d'))}; 1d market "
            f"residual {_fmt_pct(row.get('market_residual_1d'))}."
        )
    return (
        f"5d/63d volatility ratio "
        f"{_fmt_ratio(row.get('vol_regime_ratio'))}; ATR "
        f"{_fmt_pct(row.get('atr_pct'))} of price; 1d move "
        f"{_fmt_pct(row.get('ret_1d'))}."
    )


def _next_workflow(archetype: str) -> str:
    workflows = {
        "residual_dislocation": (
            "Research next: define the event without this ticker, then compare "
            "forward residual paths with day, sector and volatility controls."
        ),
        "trend_acceleration": (
            "Research next: test frozen multi-horizon continuation definitions "
            "with leave-year and leave-sector validation."
        ),
        "trend_pullback": (
            "Research next: contrast pullbacks within positive structure against "
            "matched trend and volatility controls."
        ),
        "participation_shock": (
            "Research next: separate earnings/news days, decluster episodes and "
            "test whether participation changes the forward residual path."
        ),
        "volatility_transition": (
            "Research next: condition forward paths on prior volatility regime "
            "and stress the result under realistic spread assumptions."
        ),
    }
    return workflows[archetype]


def _research_actionability(row: pd.Series, archetype: str) -> str:
    """Classify readiness for research design, never readiness to trade."""
    required = {
        "residual_dislocation": (
            "market_residual_5d",
            "gap_1d",
            "dollar_volume_shock_63d",
        ),
        "trend_acceleration": (
            "ret_63d",
            "ret_126d",
            "dist_sma200",
            "market_residual_5d",
        ),
        "trend_pullback": (
            "ret_5d",
            "ret_126d",
            "dist_sma200",
            "market_residual_5d",
        ),
        "participation_shock": (
            "volume_shock_63d",
            "dollar_volume_shock_63d",
            "market_residual_1d",
        ),
        "volatility_transition": ("vol_regime_ratio", "atr_pct", "ret_1d"),
    }[archetype]
    present = sum(_finite(row.get(field)) is not None for field in required)
    freshness = int(row.get("freshness_sessions", 999))
    if present == len(required) and freshness == 0:
        return "READY_FOR_PREREGISTRATION"
    if present >= max(1, len(required) - 1) and freshness <= 1:
        return "READY_AFTER_DATA_CHECK"
    return "DATA_REVIEW_FIRST"


def _variant_wedge(archetype: str) -> str:
    templates = {
        "residual_dislocation": (
            "Potential wedge to test: the residual move is idiosyncratic information "
            "rather than ordinary market beta or a one-day liquidity effect."
        ),
        "trend_acceleration": (
            "Potential wedge to test: acceleration across multiple horizons contains "
            "continuation information beyond generic medium-term momentum."
        ),
        "trend_pullback": (
            "Potential wedge to test: recent weakness inside positive structure has a "
            "different forward path from both generic weakness and generic momentum."
        ),
        "participation_shock": (
            "Potential wedge to test: abnormal participation changes the forward "
            "residual path rather than merely describing a known event day."
        ),
        "volatility_transition": (
            "Potential wedge to test: a volatility-regime transition predicts a "
            "distinct forward path after direction and prior-volatility controls."
        ),
    }
    return templates[archetype]


def _first_rejection_test(archetype: str) -> str:
    tests = {
        "residual_dislocation": (
            "Reject first if the event is no longer unusual after point-in-time market, "
            "sector, earnings and same-day participation controls."
        ),
        "trend_acceleration": (
            "Reject first if the apparent acceleration collapses when the horizons are "
            "lagged, sector-neutralized and compared with a plain momentum baseline."
        ),
        "trend_pullback": (
            "Reject first if the setup cannot be defined before the close or is just a "
            "relabeling of short-term reversal exposure."
        ),
        "participation_shock": (
            "Reject first if abnormal volume disappears after earnings, rebalance, split "
            "and stale-baseline exclusions."
        ),
        "volatility_transition": (
            "Reject first if the transition threshold is unstable across ordinary "
            "lookback choices or is dominated by untradeable gap observations."
        ),
    }
    return tests[archetype]


def _what_kills_it(archetype: str) -> str:
    shared = (
        "Kill if point-in-time, cost-aware forward tests fail out of sample, the effect "
        "is concentrated in one year or sector, or a simple matched baseline explains it."
    )
    if archetype == "participation_shock":
        return shared + " Also kill if removing scheduled event days removes the effect."
    if archetype == "trend_pullback":
        return shared + " Also kill if reasonable entry timing reverses the sign."
    return shared


def _what_makes_researchable(row: pd.Series, archetype: str) -> str:
    return (
        f"Freshness is {int(row.get('freshness_sessions', 999))} session(s); the "
        f"{archetype.replace('_', ' ')} definition uses frozen bar-derived fields and "
        "has an explicit neutral baseline. Investability is not assessed at this stage."
    )


def _queue_record(
    row: pd.Series, archetype: str, selection_round: int, label: str
) -> dict[str, Any]:
    alternate = []
    for other in ARCHETYPE_DESCRIPTIONS:
        if other == archetype:
            continue
        percentile = _finite(row.get(f"archetype_{other}_percentile"))
        if percentile is not None and percentile >= 80.0:
            alternate.append(other)
    return {
        "Ticker": row["Ticker"],
        "Research_Priority": label,
        "Actionability": _research_actionability(row, archetype),
        "Archetype": archetype,
        "Archetype_Rank": int(row[f"archetype_{archetype}_rank"]),
        "Archetype_Percentile": round(
            float(row[f"archetype_{archetype}_percentile"]), 2
        ),
        "Archetype_Score": round(float(row[f"archetype_{archetype}_score"]), 4),
        "Selection_Round": selection_round,
        "Alternate_Archetypes": ";".join(alternate),
        "Latest_Bar": row["latest_bar"],
        "Freshness_Sessions": int(row["freshness_sessions"]),
        "Variant_Wedge": _variant_wedge(archetype),
        "Why_Now": _why_now(row, archetype),
        "First_Rejection_Test": _first_rejection_test(archetype),
        "What_Makes_Researchable": _what_makes_researchable(row, archetype),
        "What_Kills_It": _what_kills_it(archetype),
        "Next_Workflow": _next_workflow(archetype),
        "Label": RESEARCH_ONLY_LABEL,
    }


def _round_robin_review(features: pd.DataFrame, limit: int) -> pd.DataFrame:
    columns = [
        "Ticker",
        "Research_Priority",
        "Actionability",
        "Archetype",
        "Archetype_Rank",
        "Archetype_Percentile",
        "Archetype_Score",
        "Selection_Round",
        "Alternate_Archetypes",
        "Latest_Bar",
        "Freshness_Sessions",
        "Variant_Wedge",
        "Why_Now",
        "First_Rejection_Test",
        "What_Makes_Researchable",
        "What_Kills_It",
        "Next_Workflow",
        "Label",
    ]
    if limit <= 0 or features.empty:
        return pd.DataFrame(columns=columns)
    ordered: dict[str, list[int]] = {}
    for archetype in ARCHETYPE_DESCRIPTIONS:
        score = f"archetype_{archetype}_score"
        ranked = features.dropna(subset=[score]).sort_values(
            [score, "Ticker"], ascending=[False, True], kind="mergesort"
        )
        ordered[archetype] = list(ranked.index)

    selected: set[str] = set()
    records: list[dict[str, Any]] = []
    cursors = {name: 0 for name in ordered}
    selection_round = 0
    while len(records) < min(limit, len(features)):
        selection_round += 1
        added = False
        for archetype in ARCHETYPE_DESCRIPTIONS:
            indices = ordered[archetype]
            while cursors[archetype] < len(indices):
                idx = indices[cursors[archetype]]
                cursors[archetype] += 1
                row = features.loc[idx]
                ticker = str(row["Ticker"])
                if ticker in selected:
                    continue
                selected.add(ticker)
                records.append(_queue_record(row, archetype, selection_round, "REVIEW"))
                added = True
                break
            if len(records) >= min(limit, len(features)):
                break
        if not added:
            break
    return pd.DataFrame(records, columns=columns)


def _deep_queue(review: pd.DataFrame, limit: int) -> pd.DataFrame:
    if limit <= 0 or review.empty:
        return review.head(0).copy()
    records: list[dict[str, Any]] = []
    for round_number in sorted(review["Selection_Round"].unique()):
        group = review[review["Selection_Round"] == round_number]
        for archetype in ARCHETYPE_DESCRIPTIONS:
            hit = group[group["Archetype"] == archetype]
            if hit.empty:
                continue
            record = hit.iloc[0].to_dict()
            record["Research_Priority"] = "DEEP_TEST"
            records.append(record)
            if len(records) >= limit:
                return pd.DataFrame(records, columns=review.columns)
    return pd.DataFrame(records, columns=review.columns)


def _audit_sample(
    features: pd.DataFrame, review: pd.DataFrame, config: OpportunityConfig
) -> pd.DataFrame:
    columns = [
        "Ticker",
        "Audit_Index",
        "Latest_Bar",
        "Freshness_Sessions",
        "Selection_Reason",
        "Next_Workflow",
        "Label",
    ]
    if config.audit_limit <= 0 or features.empty:
        return pd.DataFrame(columns=columns)
    review_names = set(review.get("Ticker", pd.Series(dtype=str)))
    outside = sorted(set(features["Ticker"]) - review_names)
    pool = outside if len(outside) >= config.audit_limit else sorted(features["Ticker"])
    stable_seed = int.from_bytes(
        sha256(
            f"{config.audit_seed}|{config.normalized_asof().date()}".encode()
        ).digest()[:8],
        "big",
    )
    rng = random.Random(stable_seed)
    chosen = rng.sample(pool, k=min(config.audit_limit, len(pool)))
    indexed = features.set_index("Ticker")
    records = []
    for number, ticker in enumerate(chosen, 1):
        row = indexed.loc[ticker]
        records.append(
            {
                "Ticker": ticker,
                "Audit_Index": number,
                "Latest_Bar": row["latest_bar"],
                "Freshness_Sessions": int(row["freshness_sessions"]),
                "Selection_Reason": (
                    "Seeded random coverage audit; selection is intentionally "
                    "independent of measured opportunity features."
                ),
                "Next_Workflow": (
                    "Audit next: verify feature calculations, exclusions and any "
                    "data gaps; do not promote from the audit draw alone."
                ),
                "Label": RESEARCH_ONLY_LABEL,
            }
        )
    return pd.DataFrame(records, columns=columns)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(value))
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if pd.isna(value) if not isinstance(value, (str, bytes, bool)) else False:
        return None
    return value


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [_jsonable(row) for row in frame.to_dict(orient="records")]


def build_opportunity_book(
    prices: pd.DataFrame,
    tickers: Iterable[str] | None,
    config: OpportunityConfig,
    sector_map: Mapping[str, str] | None = None,
) -> OpportunityBookResult:
    """Build the deterministic research funnel without performing any writes."""
    config.validate()
    asof = config.normalized_asof()
    universe = sorted(
        {
            _clean_ticker(t)
            for t in (tickers if tickers is not None else default_universe())
            if _clean_ticker(t)
        }
    )
    if not universe:
        raise ValueError("ticker universe is empty")
    sectors = {
        _clean_ticker(k): str(v).strip()
        for k, v in (sector_map or {}).items()
        if _clean_ticker(k)
    }

    normalized = normalize_prices(prices)
    future_rows = int((normalized["Date"] > asof).sum())
    observed = normalized[normalized["Date"] <= asof].copy()
    if observed.empty:
        raise ValueError(f"no price rows on or before {asof.date()}")
    reference_date = pd.Timestamp(observed["Date"].max()).normalize()
    sessions = pd.Index(sorted(observed["Date"].dropna().unique()))

    all_returns: dict[str, pd.Series] = {}
    all_groups = {
        ticker: group.copy() for ticker, group in observed.groupby("Ticker", sort=False)
    }
    benchmark_names = {config.market_ticker.upper()}
    benchmark_names.update(str(v).strip().upper() for v in sectors.values())
    for ticker in benchmark_names:
        group = all_groups.get(ticker)
        if group is not None:
            close = group.sort_values("Date").set_index("Date")["Close"]
            all_returns[ticker] = close.pct_change(fill_method=None)
    market_returns = all_returns.get(config.market_ticker.upper())

    coverage_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    for ticker in universe:
        group = all_groups.get(ticker)
        coverage: dict[str, Any] = {
            "Ticker": ticker,
            "Research_Status": "EXCLUDED",
            "First_Rejection": None,
            "History_Bars": 0,
            "Latest_Bar": None,
            "Freshness_Sessions": None,
            "Missing_Features": None,
        }
        if group is None or group.empty:
            coverage["First_Rejection"] = "NO_PRICE_ROWS"
            coverage_rows.append(coverage)
            continue
        group = group.sort_values("Date", kind="mergesort")
        coverage["History_Bars"] = len(group)
        latest = pd.Timestamp(group["Date"].iloc[-1]).normalize()
        coverage["Latest_Bar"] = str(latest.date())
        stale_sessions = int((sessions > latest.to_datetime64()).sum())
        coverage["Freshness_Sessions"] = stale_sessions
        if len(group) < config.min_history:
            coverage["First_Rejection"] = "INSUFFICIENT_HISTORY"
            coverage_rows.append(coverage)
            continue
        latest_close = _finite(group["Close"].iloc[-1])
        if latest_close is None or latest_close <= 0:
            coverage["First_Rejection"] = "INVALID_LAST_CLOSE"
            coverage_rows.append(coverage)
            continue
        if stale_sessions > config.max_stale_sessions:
            coverage["First_Rejection"] = "STALE_PRICE"
            coverage_rows.append(coverage)
            continue

        sector_label = sectors.get(ticker)
        sector_reference = _clean_ticker(sector_label) if sector_label else ""
        row = _ticker_features(
            group,
            market_returns,
            all_returns.get(sector_reference),
            sector_label,
        )
        row["Ticker"] = ticker
        row["freshness_sessions"] = stale_sessions
        missing = sorted(
            k
            for k, value in row.items()
            if k not in {"Ticker", "sector_label"} and value is None
        )
        coverage["Research_Status"] = "ELIGIBLE"
        coverage["Missing_Features"] = ";".join(missing)
        coverage_rows.append(coverage)
        feature_rows.append(row)

    coverage_frame = (
        pd.DataFrame(coverage_rows)
        .sort_values("Ticker", kind="mergesort")
        .reset_index(drop=True)
    )
    features = pd.DataFrame(feature_rows)
    if not features.empty:
        features = features.sort_values("Ticker", kind="mergesort").reset_index(
            drop=True
        )

        # When a sector map contains category labels rather than tradable
        # benchmark tickers, provide a point-in-time cross-sectional residual.
        # A direct benchmark series, when available, remains in sector_* fields.
        for horizon in (1, 5, 21):
            source = f"ret_{horizon}d"
            grouped_median = features.groupby("sector_label", dropna=True)[
                source
            ].transform("median")
            features[f"sector_cross_section_residual_{horizon}d"] = (
                features[source] - grouped_median
            )
        features = _add_cross_sectional_ranks(features)

    review = _round_robin_review(features, config.review_limit)
    deep = _deep_queue(review, min(config.deep_test_limit, len(review)))
    audit = _audit_sample(features, review, config)

    exclusions = (
        coverage_frame["First_Rejection"].dropna().value_counts().sort_index().to_dict()
    )
    freshness = (
        coverage_frame["Latest_Bar"]
        .fillna("NO_BAR")
        .value_counts()
        .sort_index()
        .to_dict()
    )
    archetype_summary: dict[str, Any] = {}
    for archetype, description in ARCHETYPE_DESCRIPTIONS.items():
        subset = review[review.get("Archetype", pd.Series(dtype=str)) == archetype]
        archetype_summary[archetype] = {
            "description": description,
            "review_count": len(subset),
            "top_tickers": list(subset["Ticker"].head(5)),
        }

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "asof": str(asof.date()),
        "research_only": True,
        "no_order": True,
        "production_writes": False,
        "automatic_promotion": False,
        "label": RESEARCH_ONLY_LABEL,
        "purpose": (
            "Allocate research attention across a broad universe; outputs are "
            "not signals, recommendations, portfolio instructions or executable actions."
        ),
        "source": {
            "local_prices_only": True,
            "price_reference_bar": str(reference_date.date()),
            "rows_on_or_before_asof": len(observed),
            "future_rows_discarded": future_rows,
            "market_ticker": config.market_ticker.upper(),
            "sector_map_entries": len(sectors),
        },
        "config": {
            "review_limit": config.review_limit,
            "deep_test_limit": config.deep_test_limit,
            "audit_limit": config.audit_limit,
            "audit_seed": config.audit_seed,
            "min_history": config.min_history,
            "max_stale_sessions": config.max_stale_sessions,
        },
        "coverage": {
            "requested_count": len(universe),
            "universe_sha256": sha256("\n".join(universe).encode()).hexdigest(),
            "eligible_count": int(
                (coverage_frame["Research_Status"] == "ELIGIBLE").sum()
            ),
            "excluded_count": int(
                (coverage_frame["Research_Status"] == "EXCLUDED").sum()
            ),
            "first_rejection_counts": exclusions,
            "latest_bar_counts": freshness,
            "first_rejection_field": "First_Rejection",
            "coverage_file": "coverage.csv",
        },
        "selection": {
            "method": (
                "Separate archetype scores and deterministic round-robin selection; "
                "there is no universal score. Ties resolve by ticker ascending."
            ),
            "review_count": len(review),
            "deep_test_count": len(deep),
            "audit_count": len(audit),
            "audit_method": (
                "Seeded random sample, preferentially outside the review queue; "
                "independent of opportunity features."
            ),
        },
        "archetypes": archetype_summary,
        "review_queue": _records(review),
        "deep_test_queue": _records(deep),
        "audit_sample": _records(audit),
        "next_workflow": (
            "A separate research process may preregister and test selected hypotheses. "
            "No row is authorized for staging, publishing or execution."
        ),
        "auditability": {
            "features_file": "features.csv",
            "review_file": "review_queue.csv",
            "deep_test_file": "deep_test_queue.csv",
            "audit_file": "audit_sample.csv",
            "rank_fields": [
                f"archetype_{name}_{suffix}"
                for name in ARCHETYPE_DESCRIPTIONS
                for suffix in ("score", "rank", "percentile")
            ],
        },
    }
    return OpportunityBookResult(
        manifest=_jsonable(manifest),
        coverage=coverage_frame,
        features=features,
        review_queue=review,
        deep_test_queue=deep,
        audit_sample=audit,
    )


def _html_table(
    frame: pd.DataFrame, columns: list[str], limit: int | None = None
) -> str:
    if frame.empty:
        return '<p class="empty">No rows.</p>'
    shown = frame.head(limit) if limit else frame
    head = "".join(f"<th>{escape(c.replace('_', ' '))}</th>" for c in columns)
    body = []
    for _, row in shown.iterrows():
        cells = "".join(
            f"<td>{escape(str(row.get(c, '') if pd.notna(row.get(c, '')) else ''))}</td>"
            for c in columns
        )
        body.append(f"<tr>{cells}</tr>")
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def _candidate_cards(frame: pd.DataFrame) -> str:
    if frame.empty:
        return '<p class="empty">No rows.</p>'
    fields = [
        ("Variant wedge", "Variant_Wedge"),
        ("Why now", "Why_Now"),
        ("First rejection", "First_Rejection_Test"),
        ("What makes it researchable", "What_Makes_Researchable"),
        ("What kills it", "What_Kills_It"),
        ("Next workflow", "Next_Workflow"),
    ]
    cards: list[str] = []
    for _, row in frame.iterrows():
        detail = "".join(
            f"<dt>{escape(label)}</dt><dd>{escape(str(row.get(column, '')))}</dd>"
            for label, column in fields
        )
        cards.append(
            '<article class="candidate">'
            f'<div class="candidate-head"><h3>{escape(str(row.get("Ticker", "")))}</h3>'
            f'<span>{escape(str(row.get("Archetype", "")).replace("_", " "))}</span></div>'
            f'<div class="readiness">{escape(str(row.get("Actionability", "")))}</div>'
            f"<dl>{detail}</dl></article>"
        )
    return f'<section class="candidate-grid">{"".join(cards)}</section>'


def render_html(result: OpportunityBookResult) -> str:
    manifest = result.manifest
    coverage = manifest["coverage"]
    selection = manifest["selection"]
    arch_cards = "".join(
        f"<article><h3>{escape(name.replace('_', ' ').title())}</h3>"
        f'<div class="count">{info["review_count"]} queued</div>'
        f"<p>{escape(info['description'])}</p>"
        f"<small>Top: {escape(', '.join(info['top_tickers']) or 'none')}</small></article>"
        for name, info in manifest["archetypes"].items()
    )
    exclusion_rows = pd.DataFrame(
        [
            {"First_Rejection": key, "Count": value}
            for key, value in coverage["first_rejection_counts"].items()
        ]
    )
    review_columns = [
        "Ticker",
        "Research_Priority",
        "Actionability",
        "Archetype",
        "Archetype_Rank",
        "Archetype_Percentile",
        "Variant_Wedge",
        "Why_Now",
        "First_Rejection_Test",
        "Next_Workflow",
    ]
    audit_columns = ["Ticker", "Audit_Index", "Selection_Reason", "Next_Workflow"]
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Wide Opportunity Book — {escape(manifest["asof"])}</title>
<style>
:root{{--ink:#172033;--muted:#667085;--paper:#f5f7fb;--card:#fff;--line:#dce2eb;--navy:#173b65;--teal:#087f8c;--amber:#a66400}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:14px/1.5 Inter,Segoe UI,Arial,sans-serif}}
header{{background:linear-gradient(120deg,#122b49,#175d68);color:white;padding:42px max(5vw,28px) 36px}}
header .eyebrow{{letter-spacing:.14em;text-transform:uppercase;font-weight:700;color:#8de0dc}}h1{{font-size:clamp(30px,5vw,52px);line-height:1.05;margin:8px 0 12px}}
header p{{max-width:850px;font-size:16px;color:#d9e9ee}}main{{max-width:1500px;margin:auto;padding:28px max(3vw,20px) 60px}}
.warning{{background:#fff3d8;border:1px solid #e5be6a;border-left:6px solid var(--amber);padding:16px 18px;border-radius:8px;margin-bottom:22px;font-weight:650}}
.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:18px 0 28px}}.stat,.archetypes article{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px;box-shadow:0 5px 15px #263b5710}}
.stat strong{{display:block;font-size:28px;color:var(--navy)}}.stat span{{color:var(--muted)}}h2{{font-size:24px;margin:34px 0 14px}}
.archetypes{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px}}.archetypes h3{{margin:0 0 8px;color:var(--navy)}}.archetypes p{{color:var(--muted)}}.count{{font-weight:750;color:var(--teal)}}
.candidate-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:14px}}.candidate{{background:#fff;border:1px solid var(--line);border-radius:12px;padding:18px;box-shadow:0 5px 15px #263b5710}}.candidate-head{{display:flex;justify-content:space-between;gap:12px;align-items:baseline}}.candidate-head h3{{font-size:24px;margin:0;color:var(--navy)}}.candidate-head span{{text-transform:capitalize;color:var(--muted)}}.readiness{{display:inline-block;margin:10px 0 12px;padding:4px 8px;border-radius:999px;background:#e5f4f2;color:#08666c;font-size:12px;font-weight:750}}dl{{margin:0}}dt{{margin-top:10px;font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:#667085;font-weight:750}}dd{{margin:2px 0 0}}
.table-wrap{{overflow:auto;background:white;border:1px solid var(--line);border-radius:10px}}table{{border-collapse:collapse;width:100%;min-width:900px}}th{{background:#eaf0f6;text-align:left;color:#344054;position:sticky;top:0}}th,td{{padding:10px 12px;border-bottom:1px solid #edf0f4;vertical-align:top}}tr:hover td{{background:#f9fbfd}}td:nth-child(1){{font-weight:750}}
.empty,footer{{color:var(--muted)}}code{{background:#edf1f5;padding:2px 5px;border-radius:4px}}footer{{margin-top:36px;border-top:1px solid var(--line);padding-top:18px}}
</style></head><body>
<header><div class="eyebrow">Deterministic broad-universe research funnel</div><h1>Wide Opportunity Book</h1><p>As of {escape(manifest["asof"])} · price reference {escape(manifest["source"]["price_reference_bar"])}. Every requested ticker receives a coverage verdict; eligible names compete only within transparent research archetypes.</p></header>
<main><div class="warning">{escape(RESEARCH_ONLY_LABEL)}. This local report allocates research attention only and contains no sizing, staging, portfolio or executable instructions.</div>
<section class="stats"><div class="stat"><strong>{coverage["requested_count"]}</strong><span>requested tickers</span></div><div class="stat"><strong>{coverage["eligible_count"]}</strong><span>eligible</span></div><div class="stat"><strong>{coverage["excluded_count"]}</strong><span>excluded</span></div><div class="stat"><strong>{selection["review_count"]}</strong><span>review queue</span></div><div class="stat"><strong>{selection["deep_test_count"]}</strong><span>deep-test queue</span></div><div class="stat"><strong>{selection["audit_count"]}</strong><span>random audits</span></div></section>
<h2>Archetype map</h2><section class="archetypes">{arch_cards}</section>
<h2>Deep-test research cards</h2>{_candidate_cards(result.deep_test_queue)}
<h2>Review queue</h2>{_html_table(result.review_queue, review_columns)}
<h2>Seeded coverage audit</h2>{_html_table(result.audit_sample, audit_columns)}
<h2>First exclusions</h2>{_html_table(exclusion_rows, ["First_Rejection", "Count"])}
<footer>Full per-name coverage, feature values, component ranks, archetype scores and deterministic tie-breaks are supplied beside this report in CSV and JSON. Future rows discarded: {manifest["source"]["future_rows_discarded"]}.</footer></main></body></html>"""


def write_opportunity_book(
    result: OpportunityBookResult,
    output_dir: str | Path,
    *,
    artifacts_root: str | Path = ARTIFACTS_ROOT,
) -> dict[str, Path]:
    """Write a complete bundle inside the explicitly allowed artifacts root."""
    output = Path(output_dir).expanduser().resolve()
    root = Path(artifacts_root).expanduser().resolve()
    try:
        output.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"artifact output must stay under {root}: {output}") from exc
    if output.exists() and not output.is_dir():
        raise FileExistsError(f"artifact output is not a directory: {output}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty artifact directory: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "manifest": output / "opportunity_book.json",
        "coverage": output / "coverage.csv",
        "features": output / "features.csv",
        "review": output / "review_queue.csv",
        "deep_test": output / "deep_test_queue.csv",
        "audit": output / "audit_sample.csv",
        "html": output / "index.html",
    }
    result.coverage.to_csv(paths["coverage"], index=False)
    result.features.to_csv(paths["features"], index=False)
    result.review_queue.to_csv(paths["review"], index=False)
    result.deep_test_queue.to_csv(paths["deep_test"], index=False)
    result.audit_sample.to_csv(paths["audit"], index=False)
    paths["html"].write_text(render_html(result), encoding="utf-8")
    # A manifest is the completion marker. Publish it only after every payload
    # succeeds so a partial directory never advertises a complete run.
    paths["manifest"].write_text(
        json.dumps(result.manifest, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return paths
