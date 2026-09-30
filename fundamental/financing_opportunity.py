"""Transparent research filters joining trading conditions to reported cash runway.

No offering probabilities, current-cash estimates, or trading instructions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import pandas as pd


VERSION = "financing-opportunity.v1"


@dataclass(frozen=True)
class ScreenPolicy:
    min_price: float = 3.0
    min_dollar_volume_20: float = 5_000_000
    min_history: int = 120
    sustained_return_20: float = .15
    sustained_return_60: float = .25
    sustained_relative_60: float = .10
    fresh_return_5: float = .10
    fresh_gap_5: float = .08
    fresh_relative_volume: float = 1.5
    max_runway_months: float = 24
    max_balance_age_days: int = 150

    def to_dict(self):
        return asdict(self)


def finite(value):
    try:
        return float(value) if math.isfinite(float(value)) else None
    except (TypeError, ValueError):
        return None


def ticker_bars(raw, ticker):
    """Handle yfinance (Price,Ticker), (Ticker,Price), and flat columns."""
    if isinstance(raw.columns, pd.MultiIndex):
        levels = [i for i in range(raw.columns.nlevels) if ticker in raw.columns.get_level_values(i)]
        if not levels:
            return pd.DataFrame()
        frame = raw.xs(ticker, level=levels[0], axis=1).copy()
    else:
        frame = raw.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    frame.columns = [str(c).strip() for c in frame.columns]
    frame.index = pd.to_datetime(frame.index).tz_localize(None).normalize()
    return frame.sort_index()


def price_metrics(frame, benchmark, *, session, policy=ScreenPolicy()):
    """Use completed sessions only; missing sessions never become zero returns.

    Adjusted close measures total returns/trend. Yahoo Close x Volume measures
    dollar turnover; Open/prior Close measures split-adjusted opening gaps.
    Relative volume uses the PRECEDING 20 sessions, excluding the measured day.
    """
    result = dict(price_status="unavailable", price_reasons=[], setups=[])
    required = {"Open", "Close", "Adj Close", "Volume"}
    if frame.empty or not required <= set(frame.columns):
        result["price_reasons"] = ["OHLCV/adjusted-close history unavailable"]
        return result
    if frame.index.duplicated().any():
        result["price_reasons"] = ["Duplicate price sessions"]
        return result
    cutoff = pd.Timestamp(session).normalize()
    d = frame.loc[frame.index <= cutoff].copy()
    d = d.dropna(subset=list(required))
    if d.empty:
        result["price_reasons"] = ["No completed-session prices"]
        return result
    d = d.apply(pd.to_numeric, errors="coerce")
    result.update(price_as_of=d.index[-1].date().isoformat(), history_sessions=len(d),
                  close=finite(d.Close.iloc[-1]))
    if d.index[-1] != cutoff:
        result["price_reasons"].append("Latest completed session missing")
    if len(d) < policy.min_history:
        result["price_reasons"].append("Insufficient price history")
    if (d[["Open", "Close", "Adj Close"]] <= 0).any().any() or (d.Volume < 0).any():
        result["price_reasons"].append("Invalid price/volume observation")
    # Missing interior sessions can turn 60-row returns into a different horizon.
    expected = benchmark.index[benchmark.index <= cutoff][-policy.min_history:]
    if len(expected) < policy.min_history or not expected.isin(d.index).all():
        result["price_reasons"].append("Incomplete recent session coverage")
    if result["price_reasons"]:
        return result
    adjusted = d["Adj Close"]
    result["dollar_volume_20"] = finite((d.Close * d.Volume).iloc[-20:].mean())
    result["volume_ratio"] = finite(d.Volume.iloc[-1] / d.Volume.iloc[-21:-1].mean())
    rv = d.Volume / d.Volume.shift(1).rolling(20).mean()
    result["max_volume_ratio_5"] = finite(rv.iloc[-5:].max())
    gaps = d.Open / d.Close.shift(1) - 1
    result["gap_latest"] = finite(gaps.iloc[-1])
    result["max_gap_5"] = finite(gaps.iloc[-5:].max())
    # Reverse splits can change the security dramatically; retain a visible flag.
    result["split_recent"] = bool("Stock Splits" in d and d["Stock Splits"].iloc[-60:].fillna(0).ne(0).any())
    for n in (5, 20, 60):
        start, end = d.index[-n-1], d.index[-1]
        result[f"return_{n}"] = finite(adjusted.iloc[-1] / adjusted.iloc[-n-1] - 1)
        if start not in benchmark.index or end not in benchmark.index:
            result["price_reasons"].append("Benchmark dates missing")
            return result
        base = benchmark.loc[end, "Adj Close"] / benchmark.loc[start, "Adj Close"] - 1
        result[f"relative_{n}"] = finite(result[f"return_{n}"] - base)
    result["above_sma50"] = bool(adjusted.iloc[-1] >= adjusted.iloc[-50:].mean())
    result["distance_high_252"] = finite(adjusted.iloc[-1] / adjusted.iloc[-252:].max() - 1)
    result["high_window_sessions"] = min(len(d), 252)
    result["price_status"] = "complete"
    if result["close"] < policy.min_price:
        result["price_reasons"].append("Below price floor")
    if result["dollar_volume_20"] < policy.min_dollar_volume_20:
        result["price_reasons"].append("Below sustained dollar-volume floor")
    result["tradable_filter"] = not result["price_reasons"]
    sustained = (result["return_20"] >= policy.sustained_return_20
        and result["return_60"] >= policy.sustained_return_60
        and result["relative_60"] >= policy.sustained_relative_60
        and result["above_sma50"])
    fresh = ((result["return_5"] >= policy.fresh_return_5 or result["max_gap_5"] >= policy.fresh_gap_5)
        and (result["max_volume_ratio_5"] or 0) >= policy.fresh_relative_volume)
    if result["tradable_filter"]:
        if sustained:
            result["setups"].append("Sustained strength")
        if fresh:
            result["setups"].append("Fresh rally")
    return result


def join_funding(row, financial, policy=ScreenPolicy()):
    """Join without promoting missing/stale facts or silently estimating cash today."""
    row = dict(row, financial=financial, funding_status="Not checked", candidate=False)
    if not financial:
        return row
    if financial.get("status") != "calculated":
        row["funding_status"] = "Financial coverage gap"
    elif financial.get("balance_age_days", 9999) > policy.max_balance_age_days:
        row["funding_status"] = "Stale financials"
    else:
        operating = finite(financial.get("runway_6m"))
        capex = finite(financial.get("runway_with_capex_6m"))
        if operating is not None and operating <= policy.max_runway_months:
            row["funding_status"] = "Operating runway ≤24m"
            row["candidate"] = bool(row.get("setups"))
        elif capex is not None and capex <= policy.max_runway_months:
            row["funding_status"] = "Capex-inclusive runway ≤24m"
            row["candidate"] = bool(row.get("setups"))
        elif financial.get("monthly_burn_6m") == 0:
            row["funding_status"] = "No operating burn observed"
        else:
            row["funding_status"] = "Longer reported runway"
    return row


def choose_financial_queue(rows, limit):
    """Alternate cohorts; strength by 20d return, fresh rallies by 5d move/gap."""
    queues = [[r for r in rows if label in r.get("setups", [])]
              for label in ("Sustained strength", "Fresh rally")]
    queues[0].sort(key=lambda r: (-(r.get("return_20") or 0), r["ticker"]))
    queues[1].sort(key=lambda r: (-max(r.get("return_5") or 0, r.get("max_gap_5") or 0), r["ticker"]))
    selected = []
    seen = set()
    for i in range(max((len(q) for q in queues), default=0)):
        for q in queues:
            if i < len(q) and q[i]["ticker"] not in seen:
                selected.append(q[i]["ticker"])
                seen.add(q[i]["ticker"])
                if len(selected) == limit:
                    return selected
    return selected
