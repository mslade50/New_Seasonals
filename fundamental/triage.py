"""Cross-sectional research prioritization with explicit non-trading gates."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from .config import LOWER_IS_BETTER, SCORE_WEIGHTS, UNIVERSE_POLICY
from .research_process import apply_research_routes
from .schemas import validate_candidate_frame


SPECIALIST_LANES = {
    "financials_specialist",
    "real_estate_specialist",
    "biotech_pipeline_specialist",
}


def _research_lane(row: pd.Series) -> str:
    lane = str(row.get("research_lane") or "").strip()
    if lane and lane.lower() != "nan":
        return lane
    sector = str(row.get("sector") or "")
    industry = str(row.get("industry") or "").lower()
    if sector == "Financial Services":
        return "financials_specialist"
    if sector == "Real Estate":
        return "real_estate_specialist"
    if "biotechnology" in industry:
        return "biotech_pipeline_specialist"
    if "shell companies" in industry or "blank checks" in industry:
        return "special_situation"
    return "standard_company"


def _rank_metric(frame: pd.DataFrame, metric: str, lower_is_better: bool) -> pd.Series:
    values = pd.to_numeric(frame[metric], errors="coerce")
    global_rank = values.rank(pct=True, method="average") * 100.0
    if lower_is_better:
        global_rank = 100.0 - global_rank
    result = global_rank.copy()
    sectors = frame.get("sector", pd.Series("UNKNOWN", index=frame.index)).fillna("UNKNOWN")
    for _, idx in sectors.groupby(sectors).groups.items():
        if len(idx) < 5:
            continue
        local = values.loc[idx].rank(pct=True, method="average") * 100.0
        if lower_is_better:
            local = 100.0 - local
        result.loc[idx] = local
    return result


def _trend_state(row: pd.Series) -> str:
    needed = (
        row.get("price"),
        row.get("sma200"),
        row.get("sma200_slope_20d"),
        row.get("return_12_1"),
        row.get("relative_return_12_1"),
    )
    if any(pd.isna(x) for x in needed):
        return "UNKNOWN"
    distance = float(needed[0]) / float(needed[1]) - 1.0 if float(needed[1]) else np.nan
    if distance > 0 and needed[2] > 0 and needed[3] > 0 and needed[4] >= 0:
        return "GREEN"
    if distance < -0.05 and needed[2] < 0 and needed[4] < 0:
        return "RED"
    return "AMBER"


def _first_rejection(row: pd.Series) -> str:
    if row.get("hard_exclusion_reason"):
        return str(row["hard_exclusion_reason"])
    lane = _research_lane(row)
    if lane == "financials_specialist":
        return "Baseline covered; banks and financials require capital, credit, and book-value underwriting before ranking."
    if lane == "real_estate_specialist":
        return "Baseline covered; REITs require AFFO, asset value, occupancy, and maturity analysis before ranking."
    if lane == "biotech_pipeline_specialist":
        return "Baseline covered; biotech requires pipeline, probability, cash-runway, and dilution analysis before ranking."
    if row.get("score_coverage_pct", 0) < 70:
        return "Insufficient comparable financial history; normalization may change the rank."
    if row.get("trend_state") == "RED":
        return "Price trend is damaged; no full-size entry until the 200-day trend recovers."
    if pd.notna(row.get("net_debt_to_ebitda")) and row.get("net_debt_to_ebitda") > 3:
        return "Leverage is the first downside question; stress normalized cash flow and maturities."
    if pd.notna(row.get("share_count_cagr_3y")) and row.get("share_count_cagr_3y") > 0.03:
        return "Per-share dilution is running above 3% annually."
    if not bool(row.get("latest_fcf_positive", False)):
        return "Latest free cash flow is not positive."
    return "The screen cannot prove a variant view; expectations and valuation need a full underwrite."


def _why_now(row: pd.Series) -> str:
    if _research_lane(row) in SPECIALIST_LANES:
        return "Specialist baseline complete; retain in its dedicated underwriting lane."
    pieces = []
    if row.get("research_score", 0) >= 80:
        pieces.append("top-decile-style fundamental composite")
    elif row.get("research_score", 0) >= 65:
        pieces.append("above-average fundamental composite")
    if row.get("trend_state") == "GREEN":
        pieces.append("confirming 200-day and relative trend")
    if pd.notna(row.get("revenue_growth_change")) and row.get("revenue_growth_change") > 0:
        pieces.append("revenue growth acceleration")
    if pd.notna(row.get("fcf_margin_change")) and row.get("fcf_margin_change") > 0:
        pieces.append("improving free-cash-flow margin")
    return "; ".join(pieces).capitalize() + "." if pieces else "No decisive change signal; retain as a screen flag only."


def score_candidates(
    metrics: pd.DataFrame,
    trend: pd.DataFrame,
    *,
    as_of: str | date,
) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    frame = metrics.copy()
    if not trend.empty:
        frame = frame.merge(trend, on="ticker", how="left")
    frame["research_lane"] = frame.apply(_research_lane, axis=1)
    standard = frame["research_lane"].eq("standard_company")
    for metric in SCORE_WEIGHTS:
        if metric not in frame.columns:
            frame[metric] = np.nan
        frame[f"rank_{metric}"] = np.nan
        if standard.any():
            frame.loc[standard, f"rank_{metric}"] = _rank_metric(
                frame.loc[standard], metric, metric in LOWER_IS_BETTER
            )

    weighted = pd.Series(0.0, index=frame.index)
    covered = pd.Series(0.0, index=frame.index)
    for metric, weight in SCORE_WEIGHTS.items():
        rank = frame[f"rank_{metric}"]
        present = rank.notna()
        weighted = weighted + rank.fillna(0.0) * weight
        covered = covered + present.astype(float) * weight
    frame["score_coverage_pct"] = covered
    frame["research_score"] = (weighted / covered.replace(0, np.nan)).round(1)
    frame["trend_state"] = frame.apply(_trend_state, axis=1)

    # One issuer can have multiple listed common-share classes.  Research the
    # issuer once and default to its most liquid class unless a later security
    # memo establishes a reason to prefer another class.
    duplicate_nonprimary: set[int] = set()
    if "issuer_cik" in frame.columns:
        valid_cik = frame["issuer_cik"].notna() & frame["issuer_cik"].astype(str).ne("")
        for _, idx in frame[valid_cik].groupby(frame.loc[valid_cik, "issuer_cik"].astype(str)).groups.items():
            if len(idx) < 2:
                continue
            liquidity = pd.to_numeric(frame.loc[idx, "dollar_volume_63d"], errors="coerce")
            primary = liquidity.fillna(-np.inf).idxmax()
            duplicate_nonprimary.update(set(idx) - {primary})

    reasons = []
    for row_idx, row in frame.iterrows():
        reason = ""
        industry = str(row.get("industry") or "").lower()
        sector = str(row.get("sector") or "")
        if row_idx in duplicate_nonprimary:
            reason = "Duplicate issuer share class; the more liquid listed class is the phase-one research security."
        elif int(row.get("statement_periods") or 0) < UNIVERSE_POLICY.min_annual_periods:
            reason = "Fewer than four comparable annual statement periods are available."
        elif row.get("research_lane") == "special_situation":
            reason = "Special situations require event-specific underwriting outside this fundamental sleeve."
        elif pd.isna(row.get("market_cap")) or row.get("market_cap", 0) < UNIVERSE_POLICY.min_market_cap:
            reason = "Market capitalization is below the phase-one universe floor."
        elif pd.isna(row.get("price")):
            reason = "Adjusted price history is unavailable; trend and liquidity cannot be verified."
        elif row.get("price", 0) < UNIVERSE_POLICY.min_price:
            reason = "Price is below the phase-one universe floor."
        elif pd.isna(row.get("dollar_volume_63d")) or row.get("dollar_volume_63d", 0) < UNIVERSE_POLICY.min_dollar_volume_63d:
            reason = "Dollar liquidity is below the phase-one universe floor."
        reasons.append(reason)
    frame["hard_exclusion_reason"] = reasons

    def priority(row: pd.Series) -> str:
        if row["hard_exclusion_reason"]:
            return "Reject"
        if row.get("research_lane") in SPECIALIST_LANES:
            return "C - screen flag only"
        if row["score_coverage_pct"] < 70:
            return "C - screen flag only"
        if row["research_score"] >= 80 and row["trend_state"] == "GREEN":
            return "A - immediate research candidate"
        if row["research_score"] >= 65:
            return "B - watchlist / needs trigger"
        return "C - screen flag only"

    frame["research_priority"] = frame.apply(priority, axis=1)
    frame["source_posture"] = np.where(
        pd.to_numeric(frame.get("sec_rows", 0), errors="coerce").fillna(0) > 0,
        "SEC package present; reported facts are not yet line-by-line reconciled",
        "Preliminary provider-standardized data; SEC tie-out pending",
    )
    frame["actionability"] = "Research priority only — not approved for capital"
    frame["variant_wedge"] = "Unproven: a screen cannot establish what the market has mispriced"
    frame["why_now"] = frame.apply(_why_now, axis=1)
    frame["first_rejection"] = frame.apply(_first_rejection, axis=1)
    frame["what_makes_investable"] = (
        "Filed-fact tie-out, a falsifiable company thesis, reverse-DCF expectations, "
        "bear/base/bull valuation, and portfolio-fit approval"
    )
    frame["what_kills_it"] = (
        "Thesis-pillar break, accounting or governance failure, unsafe leverage, "
        "or risk/reward that no longer clears the portfolio hurdle"
    )
    def next_workflow(row: pd.Series) -> str:
        lane = row.get("research_lane")
        if lane == "financials_specialist":
            return "Financials scorecard → capital and credit review → company tearsheet"
        if lane == "real_estate_specialist":
            return "REIT scorecard → AFFO/NAV and maturity review → company tearsheet"
        if lane == "biotech_pipeline_specialist":
            return "Biotech scorecard → pipeline and cash-runway review → event analysis"
        return "Company tearsheet → filing normalization → initiating coverage / thesis tracker"

    frame["next_workflow"] = frame.apply(next_workflow, axis=1)
    frame["implementation_readiness"] = (
        "Not implementation-ready — no live action path exists in phase one"
    )
    frame["as_of"] = str(as_of)[:10]

    order = {
        "A - immediate research candidate": 0,
        "B - watchlist / needs trigger": 1,
        "C - screen flag only": 2,
        "Reject": 3,
    }
    frame["_priority_order"] = frame["research_priority"].map(order)
    frame = frame.sort_values(["_priority_order", "research_score"], ascending=[True, False])
    frame = frame.drop(columns="_priority_order").reset_index(drop=True)
    frame = apply_research_routes(frame)
    validate_candidate_frame(frame)
    return frame
