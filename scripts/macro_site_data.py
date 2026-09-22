"""Build the static payload behind the private-site Macro Seasonality tab.

Replicates the table half of ``pages/macro_seasonality.py`` at build time:
per macro ticker, the last price, MA-extension percentile ranks (5/20/50/200
sessions over a trailing 2-year window of adjusted closes) and the ATR
seasonal ranks looked up as-of today from atr_seasonal_ranks.parquet. Macro-only
symbols absent from that strategy artifact use the same annual rank calculation
on the frozen master-price history; the canonical rank file is never changed. The
chart half is rendered in the browser from the same per-ticker binaries the
Seasonality Lab uses, so this module only stamps each row with its bin path.

Both source parquets are only read.  No network clients or writers here.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from macro_universe import SECTOR_ETFS, TICKER_INFO, get_ticker_label
from scripts.seasonality_site_data import _ticker_id

MA_WINDOWS = (5, 20, 50, 200)
SZNL_WINDOWS = (5, 10, 21, 63, 126, 252)
SORT_WINDOWS = (5, 10, 21, 63)
LOOKBACK_SESSIONS = 504  # matches the page's period="2y" yfinance pull


def percentile_rank(series: pd.Series, value: float) -> float | None:
    s = series.dropna().values
    if s.size == 0 or value is None or not np.isfinite(value):
        return None
    return float((s <= value).sum() / s.size * 100.0)


def extension_ranks(close: pd.Series) -> dict:
    """Price + percentile rank of today's distance from each MA, over the
    distances observed across the whole trailing window (mirrors
    ``load_sector_metrics``)."""
    out: dict = {"price": float(close.iloc[-1])}
    for window in MA_WINDOWS:
        ma = close.rolling(window).mean()
        dist = (close - ma) / ma * 100.0
        valid = dist.dropna()
        latest = float(valid.iloc[-1]) if not valid.empty else None
        out[f"r{window}"] = percentile_rank(dist, latest)
    return out


def load_sznl_asof(ranks_path: str | os.PathLike[str], asof: pd.Timestamp) -> dict[str, dict]:
    """{ticker: {'s5': rank, ...}} as-of the given date (last value <= asof)."""
    frame = pd.read_parquet(ranks_path)
    frame["Date"] = pd.to_datetime(frame["Date"]).dt.normalize()
    frame = frame[frame["Date"] <= asof.normalize()]
    frame = frame.sort_values("Date").groupby("ticker").tail(1)
    cols = {f"atr_sznl_{w}d": f"s{w}" for w in SZNL_WINDOWS}
    out: dict[str, dict] = {}
    for row in frame.itertuples(index=False):
        vals = {"sznl_asof": row.Date.strftime("%Y-%m-%d"), "sznl_source": "canonical"}
        for raw, key in cols.items():
            value = getattr(row, raw, None)
            vals[key] = float(value) if value is not None and np.isfinite(value) else None
        out[row.ticker] = vals
    return out


def rank_session(asof: pd.Timestamp) -> pd.Series:
    """Last canonical rank-calendar session, including year-boundary holidays."""
    from build_atr_seasonal_ranks import generate_trading_dates

    for year in (asof.year, asof.year - 1):
        calendar = generate_trading_dates(year)
        eligible = calendar[calendar["Date"] <= asof.normalize()]
        if not eligible.empty:
            return eligible.iloc[-1]
    raise ValueError(f"No seasonal rank session on or before {asof}")


def macro_only_ranks(group: pd.DataFrame, session: pd.Series) -> dict:
    """Same math and rounding as the canonical builder, using full history."""
    from build_atr_seasonal_ranks import compute_ranks_for_year, prepare_ticker_data

    frame = group.set_index("date")[["High", "Low", "Close"]]
    frame = frame.apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Macro seasonal history contains invalid OHLC values")
    prepared = prepare_ticker_data(frame)
    annual = None if prepared is None else compute_ranks_for_year(prepared, session["Date"].year)
    if annual is None:
        raise ValueError("Macro seasonal ranks require at least three prior calendar years")
    values = annual.loc[int(session["day_count"])].round(1)
    return {
        **{f"s{w}": float(values[f"atr_sznl_{w}d"]) for w in SZNL_WINDOWS},
        "sznl_asof": session["Date"].strftime("%Y-%m-%d"),
        "sznl_source": "macro_price_history",
    }


def validate_macro_rank_coverage(payload: dict) -> None:
    """Block incomplete serialized tables rather than silently shipping dashes."""
    rows = payload.get("rows") or []
    if not rows or payload.get("sznl_available") is not True:
        raise ValueError("Macro seasonal ranks are unavailable")
    requested = payload.get("sznl_requested_asof")
    if not requested:
        raise ValueError("Macro seasonal rank request date is missing")
    expected = rank_session(pd.Timestamp(requested))["Date"].strftime("%Y-%m-%d")
    if payload.get("sznl_asof") != expected:
        raise ValueError("Macro seasonal rank date does not match the requested session")
    invalid = []
    for row in rows:
        values = [row.get(f"s{w}") for w in SZNL_WINDOWS]
        if row.get("sznl_asof") != expected or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not np.isfinite(value) or not 0 <= value <= 100 for value in values
        ):
            invalid.append(str(row.get("ticker", "?")))
    if invalid:
        raise ValueError(f"Macro seasonal ranks incomplete or stale for: {', '.join(invalid)}")


def sort_key(row: dict) -> float:
    devs = [abs(row[f"s{w}"] - 50.0) for w in SORT_WINDOWS if row.get(f"s{w}") is not None]
    return max(devs) if devs else -1.0


def export_macro_snapshot(
    prices_path: str | os.PathLike[str],
    ranks_path: str | os.PathLike[str],
    out_path: str | os.PathLike[str],
    *,
    asof: pd.Timestamp | None = None,
) -> dict:
    asof = pd.Timestamp(dt.date.today()) if asof is None else pd.Timestamp(asof)
    session = rank_session(asof)
    tickers = sorted(set(SECTOR_ETFS))

    prices = pd.read_parquet(prices_path, columns=["ticker", "date", "High", "Low", "Close"])
    prices["ticker"] = prices["ticker"].astype(str).str.upper().str.strip()
    prices = prices[prices["ticker"].isin(tickers)]
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    prices["Close"] = pd.to_numeric(prices["Close"], errors="coerce")
    prices = prices.dropna(subset=["date", "Close"])

    sznl = {}
    sznl_ok = False
    if ranks_path and Path(ranks_path).exists():
        try:
            sznl = load_sznl_asof(ranks_path, asof)
            sznl_ok = True
        except Exception as e:
            print(f"  macro: seasonal ranks unavailable ({e})")

    rows = []
    price_asof = None
    from strategy_config import CSV_UNIVERSE, LIQUID_PLUS_COMMODITIES
    strategy_tickers = set(CSV_UNIVERSE) | set(LIQUID_PLUS_COMMODITIES)
    for ticker in tickers:
        info = TICKER_INFO.get(ticker, ("", ""))
        row: dict = {"ticker": ticker, "name": info[0], "ibkr": info[1],
                     "price": None}
        row["chart_label"] = get_ticker_label(ticker)
        for window in MA_WINDOWS:
            row[f"r{window}"] = None
        group = (prices[prices["ticker"] == ticker]
                 .sort_values("date")
                 .drop_duplicates("date", keep="last"))
        # Index tickers master_prices doesn't carry (^DJT, ^SOX, intl carets)
        # would render as dash-only rows with no chart — drop them. Non-caret
        # ETFs stay as table-only rows: their absence is a cache gap to fix,
        # not a ticker yfinance can't serve.
        if ticker.startswith("^") and group.empty:
            continue
        if not group.empty:
            close = group.set_index("date")["Close"].tail(LOOKBACK_SESSIONS)
            row.update(extension_ranks(close))
            row["file"] = f"t/{_ticker_id(ticker)}.bin"
            last = group["date"].iloc[-1]
            if price_asof is None or last > price_asof:
                price_asof = last
        # Never repair missing strategy ranks with a separate calculation.
        # Those remain a canonical-input failure caught by the deployment gate.
        if sznl_ok and ticker not in sznl and ticker not in strategy_tickers and not group.empty:
            try:
                sznl[ticker] = macro_only_ranks(group, session)
            except ValueError as exc:
                raise ValueError(f"Macro seasonal ranks failed for {ticker}: {exc}") from exc
        for window in SZNL_WINDOWS:
            row[f"s{window}"] = sznl.get(ticker, {}).get(f"s{window}")
        for key in ("sznl_asof", "sznl_source"):
            row[key] = sznl.get(ticker, {}).get(key)
        rows.append(row)

    rows.sort(key=sort_key, reverse=True)
    payload = {
        "version": 1,
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "asof": price_asof.strftime("%Y-%m-%d") if price_asof is not None else None,
        "sznl_requested_asof": asof.strftime("%Y-%m-%d"),
        "sznl_asof": session["Date"].strftime("%Y-%m-%d"),
        "sznl_available": sznl_ok,
        "rows": rows,
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"), ensure_ascii=False)
    return payload


def main() -> None:
    root = Path(_ROOT)
    parser = argparse.ArgumentParser(description="Build private-site macro seasonality payload")
    parser.add_argument("--prices", default=root / "data" / "master_prices.parquet")
    parser.add_argument("--ranks", default=root / "atr_seasonal_ranks.parquet")
    parser.add_argument("--out", default=root / "dist" / "data" / "seasonality" / "macro.json")
    args = parser.parse_args()
    payload = export_macro_snapshot(args.prices, args.ranks, args.out)
    with_prices = sum(1 for r in payload["rows"] if r.get("price") is not None)
    print(f"macro payload: {len(payload['rows'])} tickers ({with_prices} with prices), "
          f"sznl_available={payload['sznl_available']}")


if __name__ == "__main__":
    main()
