"""Deterministic accounting, valuation, and trend metrics for candidate triage."""

from __future__ import annotations

from datetime import date
from typing import Iterable

import numpy as np
import pandas as pd


def _numeric(frame: pd.DataFrame, aliases: Iterable[str]) -> pd.Series:
    for name in aliases:
        if name in frame.columns:
            return pd.to_numeric(frame[name], errors="coerce")
    return pd.Series(np.nan, index=frame.index, dtype=float)


def _safe_div(numerator, denominator):
    with np.errstate(divide="ignore", invalid="ignore"):
        value = numerator / denominator
    if isinstance(value, pd.Series):
        return value.replace([np.inf, -np.inf], np.nan)
    return float(value) if np.isfinite(value) else np.nan


def _cagr(last: float, first: float, years: int) -> float:
    if years <= 0 or not np.isfinite(last) or not np.isfinite(first) or last <= 0 or first <= 0:
        return np.nan
    return (last / first) ** (1.0 / years) - 1.0


def _endpoint_rows(bundle: pd.DataFrame, endpoint: str) -> pd.DataFrame:
    if bundle.empty or "endpoint" not in bundle.columns:
        return pd.DataFrame()
    out = bundle[bundle["endpoint"].eq(endpoint)].copy()
    if out.empty:
        return out
    if "date" in out.columns:
        parsed = pd.to_datetime(out["date"], errors="coerce")
        # Profile rows inherit a null ``date`` column when heterogeneous FMP
        # endpoints are concatenated.  Keep those undated metadata rows.
        if parsed.notna().any():
            out["date"] = parsed
            out = out.dropna(subset=["date"]).sort_values("date")
            out = out.drop_duplicates("date", keep="last")
    return out


def _statement_table(bundle: pd.DataFrame) -> pd.DataFrame:
    income = _endpoint_rows(bundle, "income-statement")
    balance = _endpoint_rows(bundle, "balance-sheet-statement")
    cash = _endpoint_rows(bundle, "cash-flow-statement")
    if income.empty:
        return pd.DataFrame()

    inc = pd.DataFrame({
        "date": income["date"],
        "revenue": _numeric(income, ("revenue",)),
        "gross_profit": _numeric(income, ("grossProfit", "gross_profit")),
        "operating_income": _numeric(income, ("operatingIncome", "operating_income")),
        "net_income": _numeric(income, ("netIncome", "net_income")),
        "income_before_tax": _numeric(income, ("incomeBeforeTax", "income_before_tax")),
        "income_tax": _numeric(income, ("incomeTaxExpense", "income_tax_expense")),
        "ebitda": _numeric(income, ("ebitda", "EBITDA")),
        "diluted_shares": _numeric(
            income,
            ("weightedAverageShsOutDil", "weightedAverageSharesDiluted", "weighted_average_shares_diluted"),
        ),
    })

    if not balance.empty:
        bal = pd.DataFrame({
            "date": balance["date"],
            "total_assets": _numeric(balance, ("totalAssets", "total_assets")),
            "total_debt": _numeric(balance, ("totalDebt", "total_debt")),
            "cash": _numeric(
                balance,
                ("cashAndShortTermInvestments", "cashAndCashEquivalents", "cash_and_short_term_investments"),
            ),
            "equity": _numeric(
                balance,
                ("totalStockholdersEquity", "totalShareholdersEquity", "total_stockholders_equity"),
            ),
        })
        inc = inc.merge(bal, on="date", how="left")
    else:
        for col in ("total_assets", "total_debt", "cash", "equity"):
            inc[col] = np.nan

    if not cash.empty:
        cfs = pd.DataFrame({
            "date": cash["date"],
            "operating_cash_flow": _numeric(cash, ("operatingCashFlow", "netCashProvidedByOperatingActivities")),
            "capital_expenditure": _numeric(cash, ("capitalExpenditure", "capital_expenditure")),
            "free_cash_flow": _numeric(cash, ("freeCashFlow", "free_cash_flow")),
            "stock_based_compensation": _numeric(cash, ("stockBasedCompensation", "stock_based_compensation")),
            "common_stock_repurchased": _numeric(cash, ("commonStockRepurchased", "common_stock_repurchased")),
            "common_stock_issued": _numeric(cash, ("commonStockIssued", "common_stock_issued")),
        })
        inc = inc.merge(cfs, on="date", how="left")
    else:
        for col in (
            "operating_cash_flow", "capital_expenditure", "free_cash_flow",
            "stock_based_compensation", "common_stock_repurchased", "common_stock_issued",
        ):
            inc[col] = np.nan

    # Some vendor rows omit freeCashFlow even when CFO and capex are present.
    computed_fcf = inc["operating_cash_flow"] + inc["capital_expenditure"]
    inc["free_cash_flow"] = inc["free_cash_flow"].fillna(computed_fcf)
    return inc.sort_values("date").reset_index(drop=True)


def calculate_ticker_metrics(
    bundle: pd.DataFrame,
    *,
    ticker: str,
    market_cap: float | None = None,
    company_name: str | None = None,
    sector: str | None = None,
    industry: str | None = None,
    research_lane: str | None = None,
    sec_rows: int = 0,
) -> dict:
    statements = _statement_table(bundle)
    result = {
        "ticker": ticker.upper(),
        "company_name": company_name,
        "sector": sector,
        "industry": industry,
        "research_lane": research_lane or "standard_company",
        "market_cap": pd.to_numeric(market_cap, errors="coerce"),
        "statement_periods": len(statements),
        "sec_rows": int(sec_rows),
    }
    if statements.empty:
        return result

    s = statements.tail(5).copy()
    latest = s.iloc[-1]
    prior = s.iloc[-2] if len(s) >= 2 else pd.Series(dtype=float)

    tax_rate = _safe_div(s["income_tax"], s["income_before_tax"]).clip(0.0, 0.35).fillna(0.21)
    s["nopat"] = s["operating_income"] * (1.0 - tax_rate)
    s["invested_capital"] = s["total_debt"] + s["equity"] - s["cash"]
    s["avg_invested_capital"] = (s["invested_capital"] + s["invested_capital"].shift(1)) / 2.0
    # Negative/zero invested capital is common after sustained repurchases and
    # makes the conventional ROIC denominator economically uninterpretable.
    # Treat it as unavailable rather than publishing a precise but misleading
    # negative return; asset-light quality remains visible in cash margins,
    # gross profitability, conversion, and per-share measures.
    meaningful_capital = s["avg_invested_capital"].where(s["avg_invested_capital"] > 0)
    s["roic"] = _safe_div(s["nopat"], meaningful_capital)
    s["gross_margin"] = _safe_div(s["gross_profit"], s["revenue"])
    s["fcf_margin"] = _safe_div(s["free_cash_flow"], s["revenue"])

    lookback_idx = max(0, len(s) - 4)
    years = max(0, int(round((s.iloc[-1]["date"] - s.iloc[lookback_idx]["date"]).days / 365.25)))
    delta_investment = s.iloc[-1]["invested_capital"] - s.iloc[lookback_idx]["invested_capital"]
    incremental_roic = _safe_div(
        s.iloc[-1]["nopat"] - s.iloc[lookback_idx]["nopat"], delta_investment
    ) if delta_investment > 0 else np.nan

    revenue_growth = s["revenue"].pct_change()
    fcf_margin = s["fcf_margin"]
    market_cap_value = result["market_cap"]
    if not np.isfinite(market_cap_value) or market_cap_value <= 0:
        market_cap_value = np.nan

    accepted = pd.to_datetime(
        bundle.loc[bundle["endpoint"].isin({"income-statement", "balance-sheet-statement", "cash-flow-statement"}), "accepted_at"]
        if "accepted_at" in bundle.columns else pd.Series(dtype="datetime64[ns]"),
        utc=True,
        errors="coerce",
    )
    latest_accepted = accepted.max() if not accepted.dropna().empty else pd.NaT

    result.update({
        "latest_fiscal_date": latest["date"],
        "latest_accepted_at": latest_accepted,
        "revenue": latest["revenue"],
        "free_cash_flow": latest["free_cash_flow"],
        "net_income": latest["net_income"],
        "ebitda": latest["ebitda"],
        "roic": s["roic"].iloc[-1],
        "incremental_roic": incremental_roic,
        "gross_profitability": _safe_div(latest["gross_profit"], latest["total_assets"]),
        "gross_margin_stability": s["gross_margin"].std(ddof=0),
        "revenue_cagr_3y": _cagr(s.iloc[-1]["revenue"], s.iloc[lookback_idx]["revenue"], years),
        "fcf_margin": fcf_margin.iloc[-1],
        "cash_conversion": _safe_div(latest["operating_cash_flow"], latest["net_income"])
            if latest["net_income"] > 0 else np.nan,
        "accrual_ratio": _safe_div(
            latest["net_income"] - latest["operating_cash_flow"], latest["total_assets"]
        ),
        "sbc_to_revenue": _safe_div(latest["stock_based_compensation"], latest["revenue"]),
        "share_count_cagr_3y": _cagr(
            latest["diluted_shares"], s.iloc[lookback_idx]["diluted_shares"], years
        ),
        "net_debt_to_ebitda": _safe_div(latest["total_debt"] - latest["cash"], latest["ebitda"])
            if latest["ebitda"] > 0 else np.nan,
        "fcf_positive_years": float((s["free_cash_flow"] > 0).sum()),
        "fcf_yield": _safe_div(latest["free_cash_flow"], market_cap_value),
        "earnings_yield": _safe_div(latest["net_income"], market_cap_value),
        "revenue_growth_change": (
            revenue_growth.iloc[-1] - revenue_growth.iloc[-2]
            if len(revenue_growth.dropna()) >= 2 else np.nan
        ),
        "fcf_margin_change": (
            fcf_margin.iloc[-1] - fcf_margin.iloc[-2]
            if len(fcf_margin.dropna()) >= 2 else np.nan
        ),
        "latest_gross_margin": s["gross_margin"].iloc[-1],
        "latest_revenue_growth": revenue_growth.iloc[-1] if len(revenue_growth) >= 2 else np.nan,
        "latest_fcf_positive": bool(latest["free_cash_flow"] > 0) if pd.notna(latest["free_cash_flow"]) else False,
    })
    return result


def build_metric_frame(
    fmp_snapshot: pd.DataFrame,
    symbol_master: pd.DataFrame,
    sec_snapshot: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if fmp_snapshot.empty:
        return pd.DataFrame()
    symbols = symbol_master.copy()
    symbols["ticker"] = symbols["ticker"].astype(str).str.upper()
    symbol_rows = symbols.drop_duplicates("ticker").set_index("ticker").to_dict("index")
    sec_counts = {}
    if sec_snapshot is not None and not sec_snapshot.empty and "ticker" in sec_snapshot.columns:
        sec_counts = sec_snapshot.groupby(sec_snapshot["ticker"].astype(str).str.upper()).size().to_dict()

    rows = []
    for ticker, bundle in fmp_snapshot.groupby(fmp_snapshot["ticker"].astype(str).str.upper()):
        info = symbol_rows.get(ticker, {})
        profile = _endpoint_rows(bundle, "profile")
        p = profile.iloc[-1] if not profile.empty else pd.Series(dtype=object)
        market_cap = info.get("market_cap")
        if pd.isna(market_cap) if market_cap is not None else True:
            market_cap = p.get("marketCap", p.get("mktCap"))
        rows.append(calculate_ticker_metrics(
            bundle,
            ticker=ticker,
            market_cap=market_cap,
            company_name=info.get("company_name") or p.get("companyName"),
            sector=info.get("sector") or p.get("sector"),
            industry=info.get("industry") or p.get("industry"),
            research_lane=info.get("research_lane") or "standard_company",
            sec_rows=int(sec_counts.get(ticker, 0)),
        ) | {
            "issuer_cik": p.get("cik"),
            "isin": p.get("isin"),
            "cusip": p.get("cusip"),
            "exchange": info.get("exchange") or p.get("exchange"),
        })
    return pd.DataFrame(rows)


def compute_trend_metrics(
    prices: pd.DataFrame,
    tickers: Iterable[str],
    *,
    as_of: str | date | None = None,
    benchmark: str = "SPY",
) -> pd.DataFrame:
    if prices.empty:
        return pd.DataFrame()
    wanted = {str(t).upper() for t in tickers} | {benchmark.upper()}
    px = prices.copy()
    px["ticker"] = px["ticker"].astype(str).str.upper()
    px["date"] = pd.to_datetime(px["date"], errors="coerce")
    px = px[px["ticker"].isin(wanted)].dropna(subset=["date", "Close"])
    if as_of is not None:
        px = px[px["date"] <= pd.Timestamp(as_of)]

    def one(group: pd.DataFrame) -> dict:
        group = group.sort_values("date").drop_duplicates("date", keep="last")
        close = pd.to_numeric(group["Close"], errors="coerce")
        volume = pd.to_numeric(group.get("Volume", pd.Series(np.nan, index=group.index)), errors="coerce")
        sma200 = close.rolling(200, min_periods=200).mean()
        return {
            "ticker": str(group["ticker"].iloc[-1]),
            "price_as_of": group["date"].iloc[-1],
            "price": close.iloc[-1],
            "sma200": sma200.iloc[-1] if len(sma200) else np.nan,
            "sma200_slope_20d": _safe_div(sma200.iloc[-1], sma200.iloc[-21]) - 1.0
                if len(sma200) >= 21 and pd.notna(sma200.iloc[-21]) else np.nan,
            "return_12_1": _safe_div(close.iloc[-21], close.iloc[-252]) - 1.0
                if len(close) >= 252 else np.nan,
            "dollar_volume_63d": (close * volume).tail(63).mean(),
            "price_history_days": int(close.notna().sum()),
        }

    rows = [one(g) for _, g in px.groupby("ticker") if not g.empty]
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    benchmark_row = out[out["ticker"].eq(benchmark.upper())]
    benchmark_return = benchmark_row["return_12_1"].iloc[-1] if not benchmark_row.empty else np.nan
    out["relative_return_12_1"] = out["return_12_1"] - benchmark_return
    out["above_sma200"] = out["price"] > out["sma200"]
    return out[out["ticker"].isin({str(t).upper() for t in tickers})].reset_index(drop=True)
