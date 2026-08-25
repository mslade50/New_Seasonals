"""Build the private site's compact nightly SPY-beta table.

The canonical master-price cache stores adjusted OHLCV.  This builder reads
only the scanner universe's ``ticker``, ``date``, and adjusted ``Close``
columns, computes close-to-close returns, and writes ``data/betas.json`` for
the browser.  It does not download prices or interact with any broker.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strategy_config import ACCOUNT_VALUE, CSV_UNIVERSE, LIQUID_PLUS_COMMODITIES


DEFAULT_PRICES = ROOT / "data" / "master_prices.parquet"
DEFAULT_OUTPUT = ROOT / "data" / "betas.json"
MIN_OBSERVATIONS = 20
METHOD = "OLS slope of daily adjusted close returns vs SPY; null if < 20 obs"
WINDOWS = (63, 252)


def beta_universe() -> set[str]:
    """Return every symbol that either scanner tier can put in the book."""
    return {
        str(ticker).strip().upper()
        for ticker in (
            *CSV_UNIVERSE,
            *LIQUID_PLUS_COMMODITIES,
            "SPY",
            "QQQ",
            "IWM",
            "DIA",
        )
        if str(ticker).strip()
    }


def load_prices(path: Path, universe: Iterable[str]) -> pd.DataFrame:
    """Read only the needed columns and tickers from the adjusted price cache."""
    tickers = sorted({str(ticker).strip().upper() for ticker in universe})
    if "SPY" not in tickers:
        tickers.append("SPY")
        tickers.sort()
    if not path.is_file():
        raise FileNotFoundError(f"master price cache is missing: {path}")
    return pd.read_parquet(
        path,
        columns=["ticker", "date", "Close"],
        filters=[("ticker", "in", tickers)],
    )


def _normalise_prices(prices: pd.DataFrame) -> pd.DataFrame:
    required = {"ticker", "date", "Close"}
    missing = sorted(required - set(prices.columns))
    if missing:
        raise ValueError(f"price frame is missing required columns: {', '.join(missing)}")

    frame = prices.loc[:, ["ticker", "date", "Close"]].copy()
    frame["ticker"] = frame["ticker"].astype(str).str.strip().str.upper()
    frame["date"] = (
        pd.to_datetime(frame["date"], errors="coerce", utc=True)
        .dt.tz_convert(None)
        .dt.normalize()
    )
    frame["Close"] = pd.to_numeric(frame["Close"], errors="coerce")
    frame = frame.dropna(subset=["ticker", "date", "Close"])
    frame = frame[frame["ticker"].ne("") & frame["Close"].gt(0)]
    frame = (
        frame.sort_values(["ticker", "date"])
        .drop_duplicates(["ticker", "date"], keep="last")
        .reset_index(drop=True)
    )
    return frame


def _fit_beta(pairs: pd.DataFrame) -> tuple[float | None, float | None, int]:
    pairs = pairs.dropna(subset=["spy", "ticker"])
    n_obs = int(len(pairs))
    if n_obs < MIN_OBSERVATIONS:
        return None, None, n_obs

    x = pairs["spy"].to_numpy(dtype=float)
    y = pairs["ticker"].to_numpy(dtype=float)
    centred_x = x - x.mean()
    denominator = float(np.dot(centred_x, centred_x))
    if not np.isfinite(denominator) or denominator <= np.finfo(float).eps:
        return None, None, n_obs

    beta = float(np.dot(centred_x, y - y.mean()) / denominator)
    alpha = float(y.mean() - beta * x.mean())
    residuals = y - (alpha + beta * x)
    idio_vol = float(np.std(residuals, ddof=1))
    if not np.isfinite(beta) or not np.isfinite(idio_vol):
        return None, None, n_obs
    return round(beta, 6), round(idio_vol, 8), n_obs


def _generated_utc(value: dt.datetime | str | None) -> str:
    if value is None:
        value = dt.datetime.now(dt.timezone.utc)
    if isinstance(value, str):
        return value
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.timezone.utc)
    return value.astimezone(dt.timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def build_beta_payload(
    prices: pd.DataFrame,
    *,
    universe: Iterable[str] | None = None,
    account_value: float = ACCOUNT_VALUE,
    generated_utc: dt.datetime | str | None = None,
) -> dict:
    """Compute 63d/252d OLS betas from adjusted close-to-close returns."""
    tickers = beta_universe() if universe is None else {
        str(ticker).strip().upper() for ticker in universe if str(ticker).strip()
    }
    tickers.add("SPY")
    frame = _normalise_prices(prices)
    frame = frame[frame["ticker"].isin(tickers)]

    spy_rows = frame[frame["ticker"].eq("SPY")].sort_values("date")
    if len(spy_rows) < 2:
        raise ValueError("SPY requires at least two adjusted closes")
    spy_last_row = spy_rows.iloc[-1]
    spy_closes = spy_rows.set_index("date")["Close"]
    spy_returns = spy_closes.pct_change(fill_method=None).rename("spy")
    spy_returns = spy_returns.replace([np.inf, -np.inf], np.nan)
    spy_return_dates = spy_returns.dropna().index

    closes_by_ticker = {
        ticker: group.set_index("date")["Close"]
        for ticker, group in frame.groupby("ticker", sort=False)
    }
    records: dict[str, dict] = {}
    for ticker in sorted(tickers):
        ticker_closes = closes_by_ticker.get(ticker)
        # Reindex closes to SPY's trading calendar before taking returns.  A
        # missing ticker bar therefore invalidates both that session and the
        # following return instead of pairing a multi-session ticker move with
        # a one-session SPY move.
        ticker_returns = (
            ticker_closes.reindex(spy_closes.index)
            .pct_change(fill_method=None)
            .replace([np.inf, -np.inf], np.nan)
            .rename("ticker")
            if ticker_closes is not None
            else None
        )
        joined = (
            pd.concat([spy_returns, ticker_returns], axis=1)
            if ticker_returns is not None
            else pd.DataFrame(columns=["spy", "ticker"], dtype=float)
        )

        fits: dict[int, tuple[float | None, float | None, int]] = {}
        for window in WINDOWS:
            dates = spy_return_dates[-window:]
            fits[window] = _fit_beta(joined.reindex(dates))

        beta63, idio_vol63, n63 = fits[63]
        beta252, _idio_vol252, n252 = fits[252]
        records[ticker] = {
            "beta63": beta63,
            "beta252": beta252,
            "idio_vol63": idio_vol63,
            "n63": n63,
            "n252": n252,
        }

    return {
        "asof": pd.Timestamp(spy_last_row["date"]).strftime("%Y-%m-%d"),
        "generated_utc": _generated_utc(generated_utc),
        "method": METHOD,
        "spy_last": round(float(spy_last_row["Close"]), 6),
        "account_value": float(account_value),
        "tickers": records,
    }


def write_payload(payload: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prices", type=Path, default=DEFAULT_PRICES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    universe = beta_universe()
    prices = load_prices(args.prices, universe)
    payload = build_beta_payload(prices, universe=universe)
    write_payload(payload, args.output)
    print(
        f"Wrote {args.output} ({len(payload['tickers'])} tickers, "
        f"SPY as of {payload['asof']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
