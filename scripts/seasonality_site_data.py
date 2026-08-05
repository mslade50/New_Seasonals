"""Build the static, read-only price snapshot used by the private-site
seasonality lab.

The source parquet is only read.  Output is written beneath the caller's
build directory as a small manifest plus one compact binary payload per ticker.
No network clients, credentials, R2 helpers, or production data writers are
imported here.
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ("ticker", "date", "High", "Low", "Close")
THESIS_HORIZONS = (5, 10, 21)
THESIS_SAME_CYCLE_WEIGHT = 0.70
THESIS_RECENCY_HALF_LIFE = 20.0
THESIS_DIRECTION_PRIOR_STRENGTH = 4.0


def _ticker_id(ticker: str) -> str:
    """Return a reversible filename-safe identifier for a market symbol."""
    raw = base64.urlsafe_b64encode(ticker.encode("utf-8")).decode("ascii")
    return raw.rstrip("=")


def _write_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"), ensure_ascii=False)


def _write_binary(frame: pd.DataFrame, path: Path) -> None:
    """Write SLB1: magic + count + int32 dates + float32 close + float32 ATR.

    Dates are days since 1970-01-01 and all arrays are little-endian.  The
    columnar 12-bytes/session representation is less than half the uncompressed
    JSON size while retaining the source parquet's practical precision.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    days = ((frame["date"].values.astype("datetime64[D]") - np.datetime64("1970-01-01"))
            .astype("<i4"))
    closes = frame["Close"].to_numpy(dtype="<f4", copy=True)
    atrs = frame["ATR"].to_numpy(dtype="<f4", copy=True)
    count = np.asarray([len(frame)], dtype="<u4")
    with path.open("wb") as handle:
        handle.write(b"SLB1")
        handle.write(count.tobytes())
        handle.write(days.tobytes())
        handle.write(closes.tobytes())
        handle.write(atrs.tobytes())


def prepare_ticker_frame(frame: pd.DataFrame, atr_window: int = 14) -> pd.DataFrame:
    """Normalize one ticker and reproduce ``pages/user_input.py`` ATR exactly.

    That page uses a simple 14-session rolling mean of true range.  This is
    intentionally different from the Wilder ATR used by the trade sizer.
    """
    out = frame.loc[:, ["date", "High", "Low", "Close"]].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for column in ("High", "Low", "Close"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = (out.dropna(subset=["date", "High", "Low", "Close"])
           .sort_values("date")
           .drop_duplicates("date", keep="last")
           .reset_index(drop=True))
    if out.empty:
        return out

    previous_close = out["Close"].shift(1)
    upper = pd.concat([out["High"], previous_close], axis=1).max(axis=1)
    lower = pd.concat([out["Low"], previous_close], axis=1).min(axis=1)
    true_range = upper - lower
    out["ATR"] = true_range.rolling(window=atr_window).mean()
    return out


def _weighted_quantile(values, weights, quantile: float) -> float | None:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return None
    values, weights = values[valid], weights[valid]
    order = np.argsort(values, kind="mergesort")
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights)
    cutoff = float(np.clip(quantile, 0.0, 1.0)) * cumulative[-1]
    return float(values[min(np.searchsorted(cumulative, cutoff, side="left"), len(values) - 1)])


def _weighted_mean(values, weights) -> float | None:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return None
    return float(np.average(values[valid], weights=weights[valid]))


def _shrink_probability(raw_probability: float, effective_n: float,
                        prior_strength: float = THESIS_DIRECTION_PRIOR_STRENGTH) -> float:
    """Shrink a weighted empirical hit rate toward 50%.

    The four-observation symmetric prior is intentionally mild.  It keeps a
    six-year presidential-cycle sample from masquerading as a precise option
    probability while preserving the ordering of genuine seasonal differences.
    """
    n = max(float(effective_n), 0.0)
    strength = max(float(prior_strength), 0.0)
    return float((float(raw_probability) * n + 0.5 * strength) / (n + strength))


def _seasonal_observations(frame: pd.DataFrame, asof: pd.Timestamp,
                           horizon: int, day_tolerance: int = 2) -> list[dict]:
    """One historical forward path per prior year near today's trading day."""
    if frame.empty:
        return []
    work = frame.copy().reset_index(drop=True)
    work["year"] = work["date"].dt.year.astype(int)
    work["day"] = work.groupby("year").cumcount() + 1
    current = work[(work["year"] == int(asof.year)) & (work["date"] <= asof)]
    if current.empty:
        return []
    target_day = int(current["day"].iloc[-1])
    close = work["Close"].to_numpy(float)
    high = work["High"].to_numpy(float)
    low = work["Low"].to_numpy(float)
    atr = work["ATR"].to_numpy(float)
    years = work["year"].to_numpy(int)
    days = work["day"].to_numpy(int)
    out = []
    for year in sorted(set(years)):
        if year >= int(asof.year):
            continue
        positions = np.flatnonzero(years == year)
        if positions.size == 0:
            continue
        distance = np.abs(days[positions] - target_day)
        best = int(positions[int(np.argmin(distance))])
        if int(distance.min()) > int(day_tolerance) or best + int(horizon) >= len(work):
            continue
        start = float(close[best])
        start_atr = float(atr[best])
        if not np.isfinite(start) or start <= 0 or not np.isfinite(start_atr) or start_atr <= 0:
            continue
        end = best + int(horizon)
        end_close = float(close[end])
        future_high = high[best + 1:end + 1]
        future_low = low[best + 1:end + 1]
        if not np.isfinite(end_close) or not np.isfinite(future_high).any() or not np.isfinite(future_low).any():
            continue
        path_close = close[best:end + 1]
        log_returns = np.diff(np.log(path_close[path_close > 0]))
        realized_vol = (float(np.std(log_returns, ddof=1) * np.sqrt(252))
                        if len(log_returns) >= 2 else None)
        terminal = end_close / start - 1.0
        out.append({
            "year": int(year),
            "same_cycle": int(year) % 4 == int(asof.year) % 4,
            "terminal": float(terminal),
            "terminal_atr": float((end_close - start) / start_atr),
            "max_up": float(np.nanmax(future_high) / start - 1.0),
            "max_down": float(np.nanmin(future_low) / start - 1.0),
            "realized_vol": realized_vol,
        })
    return out


def build_weighted_seasonal_distribution(
    prepared: pd.DataFrame,
    *,
    asof=None,
    horizons=THESIS_HORIZONS,
    same_cycle_weight: float = THESIS_SAME_CYCLE_WEIGHT,
    half_life: float = THESIS_RECENCY_HALF_LIFE,
) -> dict | None:
    """Build disjoint 70/30 same-cycle/other-cycle forward distributions.

    Recency decay is normalized *within* each cohort, so the total probability
    mass remains exactly 70% same-cycle and 30% other-cycle.  This deliberately
    differs from the legacy 75% cycle + 25% all-years rank, whose all-years leg
    contains the same-cycle observations a second time.
    """
    if prepared is None or prepared.empty:
        return None
    frame = prepared.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if frame.empty:
        return None
    anchor = pd.Timestamp(asof).normalize() if asof is not None else frame["date"].iloc[-1].normalize()
    same_share = float(np.clip(same_cycle_weight, 0.0, 1.0))
    horizon_payload = {}

    for horizon in horizons:
        observations = _seasonal_observations(frame, anchor, int(horizon))
        same_idx = [i for i, row in enumerate(observations) if row["same_cycle"]]
        other_idx = [i for i, row in enumerate(observations) if not row["same_cycle"]]
        # A presidential-cycle distribution is not auto-actionable until both
        # disjoint cohorts have a minimally useful sample.
        eligible = len(same_idx) >= 3 and len(other_idx) >= 5
        if not eligible:
            horizon_payload[str(int(horizon))] = {
                "eligible": False,
                "n_same_cycle": len(same_idx),
                "n_other_cycle": len(other_idx),
                "first_rejection": "Needs at least 3 same-cycle and 5 other-cycle historical windows.",
            }
            continue

        raw_decay = np.asarray([
            1.0 if not half_life or half_life <= 0 else
            0.5 ** ((int(anchor.year) - int(row["year"])) / float(half_life))
            for row in observations
        ], dtype=float)
        weights = np.zeros(len(observations), dtype=float)
        same_total = raw_decay[same_idx].sum()
        other_total = raw_decay[other_idx].sum()
        weights[same_idx] = same_share * raw_decay[same_idx] / same_total
        weights[other_idx] = (1.0 - same_share) * raw_decay[other_idx] / other_total
        effective_n = float(1.0 / np.square(weights).sum())

        terminal = np.asarray([row["terminal"] for row in observations], dtype=float)
        terminal_atr = np.asarray([row["terminal_atr"] for row in observations], dtype=float)
        max_up = np.asarray([row["max_up"] for row in observations], dtype=float)
        max_down = np.asarray([row["max_down"] for row in observations], dtype=float)
        realized_vol = np.asarray([
            row["realized_vol"] if row["realized_vol"] is not None else np.nan
            for row in observations
        ], dtype=float)

        raw_p_up = float(weights[terminal > 0].sum())
        p_up = _shrink_probability(raw_p_up, effective_n)
        up_mask, down_mask = terminal > 0, terminal < 0
        up_median = _weighted_quantile(terminal[up_mask], weights[up_mask], 0.5)
        down_median = _weighted_quantile(terminal[down_mask], weights[down_mask], 0.5)
        bull_target = up_median if up_median is not None else _weighted_quantile(terminal, weights, 0.75)
        bear_target = down_median if down_median is not None else _weighted_quantile(terminal, weights, 0.25)

        bull_touch_raw = float(weights[max_up >= bull_target].sum()) if bull_target is not None else None
        bear_touch_raw = float(weights[max_down <= bear_target].sum()) if bear_target is not None else None
        bull_no_touch = max_up < bull_target if bull_target is not None else np.zeros(len(weights), dtype=bool)
        bear_no_touch = max_down > bear_target if bear_target is not None else np.zeros(len(weights), dtype=bool)
        confidence = "moderate" if effective_n >= 8 and len(same_idx) >= 5 else "low"

        def q(values, level):
            value = _weighted_quantile(values, weights, level)
            return round(value, 6) if value is not None else None

        horizon_payload[str(int(horizon))] = {
            "eligible": True,
            "n_same_cycle": len(same_idx),
            "n_other_cycle": len(other_idx),
            "effective_n": round(effective_n, 2),
            "same_cycle_weight": round(float(weights[same_idx].sum()), 4),
            "other_cycle_weight": round(float(weights[other_idx].sum()), 4),
            "confidence": confidence,
            "p_up_raw": round(raw_p_up, 4),
            "p_up": round(p_up, 4),
            "mean_return": round(_weighted_mean(terminal, weights), 6),
            "q10": q(terminal, 0.10),
            "q25": q(terminal, 0.25),
            "median": q(terminal, 0.50),
            "q75": q(terminal, 0.75),
            "q90": q(terminal, 0.90),
            "mean_atr": round(_weighted_mean(terminal_atr, weights), 4),
            "median_atr": round(_weighted_quantile(terminal_atr, weights, 0.50), 4),
            "forecast_rv": round(_weighted_quantile(realized_vol, weights, 0.50), 4),
            "bull": {
                "target_return": round(float(bull_target), 6) if bull_target is not None else None,
                "terminal_probability": round(p_up, 4),
                "touch_probability": (round(_shrink_probability(bull_touch_raw, effective_n), 4)
                                      if bull_touch_raw is not None else None),
                "no_touch_return": (round(_weighted_quantile(terminal[bull_no_touch], weights[bull_no_touch], 0.5), 6)
                                    if bull_no_touch.any() else q(terminal, 0.25)),
            },
            "bear": {
                "target_return": round(float(bear_target), 6) if bear_target is not None else None,
                "terminal_probability": round(1.0 - p_up, 4),
                "touch_probability": (round(_shrink_probability(bear_touch_raw, effective_n), 4)
                                      if bear_touch_raw is not None else None),
                "no_touch_return": (round(_weighted_quantile(terminal[bear_no_touch], weights[bear_no_touch], 0.5), 6)
                                    if bear_no_touch.any() else q(terminal, 0.75)),
            },
            "first_rejection": ("Low effective sample after cohort and recency weighting."
                                if confidence == "low" else
                                "Historical seasonality can be overridden by current catalysts and regime shifts."),
        }

    return {
        "asof": anchor.strftime("%Y-%m-%d"),
        "cycle": int(anchor.year % 4),
        "horizons": horizon_payload,
    }


def export_seasonality_snapshot(
    source: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    min_year: int = 2000,
    atr_window: int = 14,
) -> dict:
    """Export static per-ticker inputs without modifying ``source``.

    Existing output files are overwritten in place, but stale files are not
    removed.  The manifest is authoritative, so an old unreferenced payload
    can never become selectable in the UI.
    """
    source_path = Path(source)
    output_path = Path(output_dir)
    prices_path = output_path / "t"

    prices = pd.read_parquet(source_path, columns=list(REQUIRED_COLUMNS))
    missing = [column for column in REQUIRED_COLUMNS if column not in prices.columns]
    if missing:
        raise ValueError(f"seasonality source is missing columns: {', '.join(missing)}")

    prices["ticker"] = prices["ticker"].astype(str).str.upper().str.strip()
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    prices = prices[(prices["ticker"] != "") & prices["date"].notna()]
    prices = prices[prices["date"].dt.year >= int(min_year)]
    if prices.empty:
        raise ValueError("seasonality source has no usable rows")

    ticker_meta: dict[str, dict] = {}
    ticker_theses: dict[str, dict] = {}
    total_rows = 0
    global_asof: pd.Timestamp | None = None

    for ticker, group in prices.groupby("ticker", sort=True):
        prepared = prepare_ticker_frame(group, atr_window=atr_window)
        if prepared.empty:
            continue
        ticker_id = _ticker_id(ticker)
        _write_binary(prepared, prices_path / f"{ticker_id}.bin")

        start = prepared["date"].iloc[0]
        end = prepared["date"].iloc[-1]
        count = int(len(prepared))
        ticker_meta[ticker] = {
            "id": ticker_id,
            "start": start.strftime("%Y-%m-%d"),
            "end": end.strftime("%Y-%m-%d"),
            "n": count,
            "file": f"t/{ticker_id}.bin",
        }
        thesis = build_weighted_seasonal_distribution(prepared, asof=end)
        if thesis is not None:
            ticker_theses[ticker] = thesis
        total_rows += count
        if global_asof is None or end > global_asof:
            global_asof = end

    if not ticker_meta:
        raise ValueError("seasonality source has no exportable tickers")

    manifest = {
        "version": 1,
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "asof": global_asof.strftime("%Y-%m-%d") if global_asof is not None else None,
        "history_floor": f"{int(min_year):04d}-01-01",
        "price_basis": "adjusted OHLCV from master_prices.parquet",
        "encoding": "SLB1 columnar little-endian: int32 epoch-day, float32 close, float32 ATR",
        "atr": {
            "window": int(atr_window),
            "method": "simple rolling mean of true range",
        },
        "ticker_count": len(ticker_meta),
        "row_count": total_rows,
        "thesis_file": "theses.json",
        "tickers": ticker_meta,
    }
    theses = {
        "version": 1,
        "asof": global_asof.strftime("%Y-%m-%d") if global_asof is not None else None,
        "methodology": {
            "cohorts": "70% same presidential-cycle years + 30% other-cycle years; disjoint cohorts",
            "same_cycle_weight": THESIS_SAME_CYCLE_WEIGHT,
            "other_cycle_weight": round(1.0 - THESIS_SAME_CYCLE_WEIGHT, 2),
            "recency_half_life_years": THESIS_RECENCY_HALF_LIFE,
            "probability_shrinkage": "symmetric four-observation prior toward 50%",
            "basis": "adjusted OHLC; one matching trading-day window per prior year",
            "horizons_td": list(THESIS_HORIZONS),
            "status": "candidate prior, not a recommendation or option-price forecast",
        },
        "tickers": ticker_theses,
    }
    _write_json(theses, output_path / "theses.json")
    _write_json(manifest, output_path / "manifest.json")
    return manifest


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Build private-site seasonality inputs locally")
    parser.add_argument("--source", default=root / "data" / "master_prices.parquet")
    parser.add_argument("--out", default=root / "dist" / "data" / "seasonality")
    parser.add_argument("--min-year", type=int, default=2000)
    args = parser.parse_args()
    manifest = export_seasonality_snapshot(args.source, args.out, min_year=args.min_year)
    print(
        "seasonality snapshot: "
        f"{manifest['ticker_count']:,} tickers, {manifest['row_count']:,} rows, "
        f"as of {manifest['asof']}"
    )


if __name__ == "__main__":
    main()
