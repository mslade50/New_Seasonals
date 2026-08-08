"""Broad, research-only public-equity universe discovery and queue selection."""

from __future__ import annotations

from datetime import date
from typing import Iterable

import numpy as np
import pandas as pd

from .config import BROAD_UNIVERSE_POLICY, BroadUniversePolicy
from .metrics import compute_trend_metrics


SPECIALIST_SECTORS = {
    "Financial Services": "financials_specialist",
    "Real Estate": "real_estate_specialist",
}
SPECIALIST_INDUSTRY_TERMS = {
    "biotechnology": "biotech_pipeline_specialist",
    "shell companies": "special_situation",
    "blank checks": "special_situation",
}
SIZE_ORDER = {"small": 0, "mid": 1, "large": 2, "mega": 3, "unknown": 4}


def normalize_screener_rows(payload: list[dict], *, as_of: str | date) -> pd.DataFrame:
    """Normalize FMP's broad company screener without treating it as a thesis."""
    if not payload:
        return pd.DataFrame()
    raw = pd.json_normalize(payload, sep="__")
    aliases = {
        "symbol": "ticker",
        "companyName": "company_name",
        "marketCap": "market_cap",
        "volume": "screener_volume",
        "price": "screener_price",
    }
    frame = raw.rename(columns=aliases)
    # The screener supplies both a verbose ``exchange`` value and the stable
    # short code used by our gates.  Assign explicitly to avoid duplicate
    # column names after normalization.
    if "exchangeShortName" in raw.columns:
        frame["exchange"] = raw["exchangeShortName"]
    for column in (
        "ticker", "company_name", "market_cap", "sector", "industry",
        "exchange", "country", "screener_volume", "screener_price",
        "isEtf", "isFund", "isActivelyTrading",
    ):
        if column not in frame.columns:
            frame[column] = np.nan
    frame["ticker"] = (
        frame["ticker"].astype(str).str.upper().str.strip().str.replace(".", "-", regex=False)
    )
    for column in ("market_cap", "screener_volume", "screener_price"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["as_of"] = str(as_of)[:10]
    frame["source_name"] = "Financial Modeling Prep company screener"
    frame["source_label"] = "fact_provider_standardized"
    keep = [
        "ticker", "company_name", "market_cap", "sector", "industry",
        "exchange", "country", "screener_volume", "screener_price",
        "isEtf", "isFund", "isActivelyTrading", "as_of", "source_name",
        "source_label",
    ]
    return frame[keep].drop_duplicates("ticker", keep="first").reset_index(drop=True)


def _market_cap_band(value: float) -> str:
    if not np.isfinite(value):
        return "unknown"
    if value < 2_000_000_000:
        return "small"
    if value < 10_000_000_000:
        return "mid"
    if value < 200_000_000_000:
        return "large"
    return "mega"


def _research_lane(sector: object, industry: object) -> str:
    sector_text = str(sector or "")
    if sector_text in SPECIALIST_SECTORS:
        return SPECIALIST_SECTORS[sector_text]
    industry_text = str(industry or "").lower()
    for term, lane in SPECIALIST_INDUSTRY_TERMS.items():
        if term in industry_text:
            return lane
    return "standard_company"


def _trend_state(row: pd.Series) -> str:
    needed = (
        row.get("above_sma200"), row.get("sma200_slope_20d"),
        row.get("return_12_1"), row.get("relative_return_12_1"),
    )
    if any(pd.isna(value) for value in needed):
        return "UNKNOWN"
    if bool(needed[0]) and needed[1] > 0 and needed[2] > 0 and needed[3] >= 0:
        return "GREEN"
    if bool(needed[0]) or (needed[2] > 0 and needed[1] >= 0):
        return "AMBER"
    return "RED"


def build_broad_universe(
    screener: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    as_of: str | date,
    policy: BroadUniversePolicy = BROAD_UNIVERSE_POLICY,
    fundamental_tickers: Iterable[str] = (),
    sec_tickers: Iterable[str] = (),
) -> pd.DataFrame:
    """Build the full discovery funnel while preserving specialist lanes."""
    if screener.empty:
        return pd.DataFrame()
    frame = screener.copy()
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    trend = compute_trend_metrics(prices, frame["ticker"], as_of=as_of)
    if not trend.empty:
        frame = frame.merge(trend, on="ticker", how="left")
    for column in (
        "price", "sma200", "sma200_slope_20d", "return_12_1",
        "dollar_volume_63d", "price_history_days", "relative_return_12_1",
        "above_sma200",
    ):
        if column not in frame.columns:
            frame[column] = np.nan

    bool_default = pd.Series(False, index=frame.index)
    is_etf = frame.get("isEtf", bool_default).fillna(False).astype(bool)
    is_fund = frame.get("isFund", bool_default).fillna(False).astype(bool)
    active = frame.get("isActivelyTrading", pd.Series(True, index=frame.index)).fillna(True).astype(bool)
    exchange_ok = frame["exchange"].astype(str).str.upper().isin(policy.primary_exchanges)
    liquid = pd.to_numeric(frame["dollar_volume_63d"], errors="coerce") >= policy.min_dollar_volume_63d
    history_ok = pd.to_numeric(frame["price_history_days"], errors="coerce") >= policy.min_price_history_days
    price_ok = pd.to_numeric(frame["price"], errors="coerce") >= policy.min_price
    cap_ok = pd.to_numeric(frame["market_cap"], errors="coerce") >= policy.min_market_cap
    frame["research_eligible"] = (
        ~is_etf & ~is_fund & active & exchange_ok & liquid & history_ok & price_ok & cap_ok
    )

    reasons: list[str] = []
    for idx, row in frame.iterrows():
        reason = ""
        if bool(is_etf.loc[idx]) or bool(is_fund.loc[idx]):
            reason = "Fund or ETF; the sleeve researches operating companies."
        elif not bool(active.loc[idx]):
            reason = "Security is not actively trading."
        elif not bool(exchange_ok.loc[idx]):
            reason = "Listing is outside the primary US exchange scope."
        elif pd.isna(row.get("market_cap")) or row.get("market_cap", 0) < policy.min_market_cap:
            reason = "Market capitalization is below the broad research floor."
        elif pd.isna(row.get("price")):
            reason = "Local adjusted price history is not yet available."
        elif row.get("price", 0) < policy.min_price:
            reason = "Price is below the broad research floor."
        elif row.get("price_history_days", 0) < policy.min_price_history_days:
            reason = "Fewer than 252 adjusted daily bars are available."
        elif pd.isna(row.get("dollar_volume_63d")) or row.get("dollar_volume_63d", 0) < policy.min_dollar_volume_63d:
            reason = "Dollar liquidity is below the broad research floor."
        reasons.append(reason)
    frame["eligibility_reason"] = reasons
    frame["market_cap_band"] = frame["market_cap"].map(_market_cap_band)
    frame["research_lane"] = [
        _research_lane(sector, industry)
        for sector, industry in zip(frame["sector"], frame["industry"])
    ]
    frame["trend_state"] = frame.apply(_trend_state, axis=1)
    fundamental_set = {str(t).upper() for t in fundamental_tickers}
    sec_set = {str(t).upper() for t in sec_tickers}
    frame["fundamental_covered"] = frame["ticker"].isin(fundamental_set)
    frame["sec_covered"] = frame["ticker"].isin(sec_set)
    frame["funnel_stage"] = np.select(
        [
            frame["research_eligible"] & frame["research_lane"].eq("standard_company"),
            frame["research_eligible"],
        ],
        ["standard_enrichment_queue", "specialist_research_queue"],
        default="monitor_or_exclude",
    )
    frame["live_actions_enabled"] = False
    return frame.sort_values(
        ["research_eligible", "market_cap"], ascending=[False, False]
    ).reset_index(drop=True)


def select_balanced_enrichment_batch(
    universe: pd.DataFrame,
    limit: int,
    *,
    exclude_tickers: Iterable[str] = (),
    include_specialists: bool = False,
) -> list[str]:
    """Select a deterministic sector/size-balanced statement-enrichment batch."""
    if universe.empty or limit <= 0:
        return []
    excluded = {str(t).upper() for t in exclude_tickers}
    eligible = universe[
        universe["research_eligible"].fillna(False)
        & ~universe["ticker"].astype(str).str.upper().isin(excluded)
    ].copy()
    if not include_specialists:
        eligible = eligible[eligible["research_lane"].eq("standard_company")]
    if eligible.empty:
        return []
    eligible["_size_order"] = eligible["market_cap_band"].map(SIZE_ORDER).fillna(4)
    eligible["_stratum_rank"] = (
        eligible.groupby(["market_cap_band", "sector"], dropna=False)["dollar_volume_63d"]
        .rank(method="first", ascending=False)
    )
    eligible = eligible.sort_values(
        ["_stratum_rank", "_size_order", "sector", "dollar_volume_63d", "ticker"],
        ascending=[True, True, True, False, True],
    )
    return eligible["ticker"].astype(str).head(limit).tolist()


def summarize_universe(universe: pd.DataFrame) -> dict:
    if universe.empty:
        return {
            "discovered": 0, "research_eligible": 0, "standard_queue": 0,
            "specialist_queue": 0, "fundamental_covered": 0, "sec_covered": 0,
            "by_size": {}, "by_lane": {},
        }
    eligible = universe[universe["research_eligible"].fillna(False)]
    return {
        "discovered": int(len(universe)),
        "research_eligible": int(len(eligible)),
        "standard_queue": int(eligible["research_lane"].eq("standard_company").sum()),
        "specialist_queue": int((~eligible["research_lane"].eq("standard_company")).sum()),
        "fundamental_covered": int(universe["fundamental_covered"].fillna(False).sum()),
        "sec_covered": int(universe["sec_covered"].fillna(False).sum()),
        "by_size": {
            str(k): int(v) for k, v in eligible["market_cap_band"].value_counts().items()
        },
        "by_lane": {
            str(k): int(v) for k, v in eligible["research_lane"].value_counts().items()
        },
    }
