"""Compact, source-backed issuer baselines for advanced research candidates."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ADVANCE_PRIORITIES = {
    "A - immediate research candidate",
    "B - watchlist / needs trigger",
}


def _esc(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    return html.escape(str(value))


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _pct(value: Any, digits: int = 1) -> str:
    number = _number(value)
    return "—" if number is None else f"{number * 100:.{digits}f}%"


def _money(value: Any) -> str:
    number = _number(value)
    if number is None:
        return "—"
    magnitude = abs(number)
    if magnitude >= 1e12:
        return f"${number / 1e12:,.2f}T"
    if magnitude >= 1e9:
        return f"${number / 1e9:,.1f}B"
    if magnitude >= 1e6:
        return f"${number / 1e6:,.1f}M"
    return f"${number:,.0f}"


def _multiple(value: Any) -> str:
    number = _number(value)
    return "—" if number is None else f"{number:,.1f}×"


def _date(value: Any) -> str:
    timestamp = pd.to_datetime(value, errors="coerce", utc=True)
    return "—" if pd.isna(timestamp) else str(timestamp.date())


def _endpoint(frame: pd.DataFrame, ticker: str, endpoint: str) -> pd.DataFrame:
    if frame.empty or "ticker" not in frame.columns or "endpoint" not in frame.columns:
        return pd.DataFrame()
    rows = frame[
        frame["ticker"].astype(str).str.upper().eq(ticker.upper())
        & frame["endpoint"].astype(str).eq(endpoint)
    ].copy()
    if "date" in rows.columns:
        rows["date"] = pd.to_datetime(rows["date"], errors="coerce")
        rows = rows.sort_values("date")
    return rows


def _latest(rows: pd.DataFrame) -> pd.Series:
    return rows.iloc[-1] if not rows.empty else pd.Series(dtype=object)


def _prior(rows: pd.DataFrame) -> pd.Series:
    return rows.iloc[-2] if len(rows) >= 2 else pd.Series(dtype=object)


def _growth(current: Any, previous: Any) -> float | None:
    current_number = _number(current)
    previous_number = _number(previous)
    if current_number is None or previous_number in (None, 0):
        return None
    return current_number / previous_number - 1.0


def _margin(numerator: Any, denominator: Any) -> float | None:
    numerator_number = _number(numerator)
    denominator_number = _number(denominator)
    if numerator_number is None or denominator_number in (None, 0):
        return None
    return numerator_number / denominator_number


def _plain_description(profile: pd.Series) -> str:
    description = str(profile.get("description") or "").strip()
    if not description:
        return "A current business description has not yet been sourced."
    if len(description) > 520:
        description = description[:517].rsplit(" ", 1)[0] + "…"
    return description


def build_tearsheet_context(
    candidate: pd.Series,
    fmp: pd.DataFrame,
    sec: pd.DataFrame,
) -> dict[str, Any]:
    """Build a factual issuer baseline without promoting the screen to a thesis."""
    ticker = str(candidate.get("ticker") or "").upper()
    report_as_of = pd.to_datetime(candidate.get("as_of"), errors="coerce")
    price_as_of = pd.to_datetime(candidate.get("price_as_of"), errors="coerce")
    price_age_bd: int | None = None
    if not pd.isna(report_as_of) and not pd.isna(price_as_of):
        price_age_bd = int(np.busday_count(price_as_of.date(), report_as_of.date()))
    price_is_current = price_age_bd is not None and price_age_bd <= 1
    profile = _latest(_endpoint(fmp, ticker, "profile"))
    income_rows = _endpoint(fmp, ticker, "income-statement")
    cash_rows = _endpoint(fmp, ticker, "cash-flow-statement")
    balance_rows = _endpoint(fmp, ticker, "balance-sheet-statement")
    metric_rows = _endpoint(fmp, ticker, "key-metrics")
    ratio_rows = _endpoint(fmp, ticker, "ratios")
    estimate_rows = _endpoint(fmp, ticker, "analyst-estimates")

    income = _latest(income_rows)
    income_prior = _prior(income_rows)
    cash = _latest(cash_rows)
    cash_prior = _prior(cash_rows)
    balance = _latest(balance_rows)
    metrics = _latest(metric_rows)
    ratios = _latest(ratio_rows)

    latest_period = pd.to_datetime(income.get("date"), errors="coerce")
    future_estimates = estimate_rows.copy()
    if not future_estimates.empty and not pd.isna(latest_period):
        future_estimates = future_estimates[future_estimates["date"] > latest_period]
    next_estimate = future_estimates.iloc[0] if not future_estimates.empty else pd.Series(dtype=object)

    revenue_growth = _growth(income.get("revenue"), income_prior.get("revenue"))
    operating_margin = _margin(income.get("operatingIncome"), income.get("revenue"))
    prior_operating_margin = _margin(
        income_prior.get("operatingIncome"), income_prior.get("revenue")
    )
    fcf_margin = _margin(cash.get("freeCashFlow"), income.get("revenue"))
    prior_fcf_margin = _margin(cash_prior.get("freeCashFlow"), income_prior.get("revenue"))
    share_growth = _growth(
        income.get("weightedAverageShsOutDil"), income_prior.get("weightedAverageShsOutDil")
    )
    consensus_revenue_growth = _growth(next_estimate.get("revenueAvg"), income.get("revenue"))
    consensus_eps_growth = _growth(next_estimate.get("epsAvg"), income.get("epsDiluted"))

    trend = str(candidate.get("trend_state") or "UNKNOWN")
    if trend == "GREEN":
        core_question = (
            "Can reported cash compounding and the next estimate path outrun expectations "
            "enough to justify the current trailing valuation?"
        )
    elif trend == "AMBER":
        core_question = (
            "Can operating evidence strengthen before mixed long-term price confirmation "
            "turns into fundamental deterioration?"
        )
    else:
        core_question = (
            "Is weak price confirmation a temporary expectations reset, or an early signal "
            "that the historical economics are no longer durable?"
        )

    read = (
        f"{candidate.get('company_name') or profile.get('companyName') or ticker} is a "
        f"{candidate.get('research_priority', 'screened')} name with a {candidate.get('research_score', '—')} "
        f"composite score and {trend.lower()} trend confirmation. The screen identifies a research priority, "
        "not a demonstrated mispricing: the variant view, forward valuation, and catalyst path remain unproven."
    )
    if not price_is_current:
        read += " The price and trend snapshot is stale and must be refreshed before timing conclusions are used."

    drivers = [
        {
            "metric": "Revenue",
            "period": _date(income.get("date")),
            "value": _money(income.get("revenue")),
            "change": _pct(revenue_growth),
            "source": "S2",
            "evidence": "Provider-standardized reported result",
            "confidence": "Medium",
        },
        {
            "metric": "Operating margin",
            "period": _date(income.get("date")),
            "value": _pct(operating_margin),
            "change": "—" if operating_margin is None or prior_operating_margin is None
            else f"{(operating_margin - prior_operating_margin) * 100:+.1f} pts",
            "source": "S2",
            "evidence": "Derived from provider-standardized statements",
            "confidence": "Medium",
        },
        {
            "metric": "Free-cash-flow margin",
            "period": _date(income.get("date")),
            "value": _pct(fcf_margin),
            "change": "—" if fcf_margin is None or prior_fcf_margin is None
            else f"{(fcf_margin - prior_fcf_margin) * 100:+.1f} pts",
            "source": "S2",
            "evidence": "Derived from provider-standardized statements",
            "confidence": "Medium",
        },
        {
            "metric": "Diluted share count",
            "period": _date(income.get("date")),
            "value": f"{_number(income.get('weightedAverageShsOutDil')) / 1e9:,.2f}B"
            if _number(income.get("weightedAverageShsOutDil")) is not None else "—",
            "change": _pct(share_growth),
            "source": "S2",
            "evidence": "Provider-standardized reported result",
            "confidence": "Medium",
        },
    ]

    valuation = [
        ("FCF yield", _pct(candidate.get("fcf_yield")), "Latest FCF ÷ current market cap", "Derived"),
        ("Earnings yield", _pct(candidate.get("earnings_yield")), "Latest net income ÷ current market cap", "Derived"),
        ("Trailing P/E", _multiple(ratios.get("priceToEarningsRatio")), _date(ratios.get("date")), "Provider"),
        ("Trailing EV / EBITDA", _multiple(metrics.get("evToEBITDA")), _date(metrics.get("date")), "Provider"),
        ("Net debt / EBITDA", _multiple(candidate.get("net_debt_to_ebitda")), _date(balance.get("date")), "Derived"),
    ]

    triggers = []
    if not next_estimate.empty:
        triggers.append({
            "title": "Next fiscal expectations bar",
            "detail": (
                f"For {_date(next_estimate.get('date'))}, provider consensus is "
                f"{_money(next_estimate.get('revenueAvg'))} revenue ({_pct(consensus_revenue_growth)} vs. latest reported) "
                f"and EPS {_number(next_estimate.get('epsAvg')):,.2f} ({_pct(consensus_eps_growth)} vs. latest reported)"
                if _number(next_estimate.get("epsAvg")) is not None
                else f"For {_date(next_estimate.get('date'))}, provider consensus revenue is {_money(next_estimate.get('revenueAvg'))}."
            ),
            "source": "S5",
        })
    else:
        triggers.append({
            "title": "Next fiscal expectations bar",
            "detail": "Current consensus for the next unreported fiscal period is not available in the local source set.",
            "source": "Gap",
        })
    triggers.extend([
        {
            "title": "Fundamental proof",
            "detail": (
                "The next filing must preserve positive FCF and support the historical revenue, margin, "
                "and per-share direction shown above."
            ),
            "source": "S2 / S3",
        },
        {
            "title": "Trend proof",
            "detail": (
                f"Current state is {trend}. A GREEN state requires price above a rising 200-day average, "
                "positive 12–1 momentum, and non-negative relative momentum versus SPY."
            ),
            "source": "S4",
        },
        {
            "title": "Valuation proof",
            "detail": "Build a reverse DCF and peer range before claiming the market is mispricing the company.",
            "source": "Gap",
        },
    ])

    risks = []
    if trend != "GREEN":
        risks.append(f"{trend.title()} trend confirmation makes timing and downside path unresolved.")
    if revenue_growth is not None and revenue_growth < 0:
        risks.append(f"Latest annual revenue declined {_pct(abs(revenue_growth))}.")
    if fcf_margin is not None and prior_fcf_margin is not None and fcf_margin < prior_fcf_margin:
        risks.append(f"Latest FCF margin contracted {(prior_fcf_margin - fcf_margin) * 100:.1f} points.")
    if _number(candidate.get("net_debt_to_ebitda")) is not None and float(candidate["net_debt_to_ebitda"]) > 2.5:
        risks.append(f"Net debt / EBITDA is {_multiple(candidate.get('net_debt_to_ebitda'))}; maturities and normalized cash flow need stress testing.")
    if _number(candidate.get("roic")) is None:
        risks.append("Conventional ROIC is not meaningful on the current invested-capital base; use normalized unit and cash economics.")
    if _number(candidate.get("fcf_yield")) is not None and float(candidate["fcf_yield"]) < 0.025:
        risks.append("A sub-2.5% trailing FCF yield leaves the stock reliant on durable growth and/or a sustained premium multiple.")
    if not risks:
        risks.append("The screen has not established a downside distribution; cyclicality, competition, and multiple compression still require underwriting.")

    ticker_sec = sec[
        sec.get("ticker", pd.Series(dtype=str)).astype(str).str.upper().eq(ticker)
    ].copy() if not sec.empty and "ticker" in sec.columns else pd.DataFrame()
    latest_sec = pd.Series(dtype=object)
    if not ticker_sec.empty:
        ticker_sec["accepted_at"] = pd.to_datetime(ticker_sec.get("accepted_at"), errors="coerce", utc=True)
        regular_forms = ticker_sec[
            ticker_sec.get("form", pd.Series(index=ticker_sec.index, dtype=str))
            .astype(str).isin({"10-K", "10-Q", "20-F", "40-F"})
        ]
        source_rows = regular_forms if not regular_forms.empty else ticker_sec
        accepted = source_rows["accepted_at"].dropna()
        latest_sec = source_rows.loc[accepted.idxmax()] if not accepted.empty else source_rows.iloc[-1]

    profile_fetched = profile.get("fetched_at")
    statement_fetched = income.get("fetched_at")
    estimate_fetched = next_estimate.get("fetched_at")
    sources = [
        {
            "id": "S1", "name": "FMP company profile", "type": "Provider-standardized",
            "as_of": _date(profile_fetched), "period": "Current profile",
            "location": str(profile.get("source_url") or "FMP stable/profile"),
            "freshness": "Current", "notes": "Business description and issuer metadata; not a primary filing.",
        },
        {
            "id": "S2", "name": "FMP financial statements and ratios", "type": "Provider-standardized",
            "as_of": _date(statement_fetched), "period": _date(income.get("date")),
            "location": "FMP stable statement endpoints",
            "freshness": "Current", "notes": f"Latest statement acceptance timestamp: {_date(income.get('accepted_at'))}.",
        },
        {
            "id": "S3", "name": "SEC EDGAR Companyfacts", "type": "Primary filing facts",
            "as_of": _date(latest_sec.get("accepted_at")),
            "period": str(latest_sec.get("form") or "—"),
            "location": str(latest_sec.get("source_url") or "SEC Companyfacts"),
            "freshness": "Current" if not latest_sec.empty else "Missing",
            "notes": (
                f"{len(ticker_sec):,} filed fact rows locally archived; metrics are not yet line-by-line tag reconciled."
                if not ticker_sec.empty else "Filed-fact package not available."
            ),
        },
        {
            "id": "S4", "name": "Adjusted research price cache", "type": "Derived market data",
            "as_of": _date(candidate.get("price_as_of")), "period": "Latest close / trailing history",
            "location": "Local master and overflow price caches",
            "freshness": "Current" if price_is_current else "Stale",
            "notes": (
                "Used for price, 200-day trend, 12–1 momentum, and liquidity."
                if price_is_current else
                f"Last bar is {price_age_bd if price_age_bd is not None else 'an unknown number of'} business days old; refresh required."
            ),
        },
        {
            "id": "S5", "name": "FMP analyst estimates", "type": "Provider consensus",
            "as_of": _date(estimate_fetched), "period": _date(next_estimate.get("date")),
            "location": str(next_estimate.get("source_url") or "FMP stable/analyst-estimates"),
            "freshness": "Current" if not next_estimate.empty else "Missing",
            "notes": "Snapshot history starts with this project; no historical estimate-revision series yet.",
        },
    ]

    return {
        "ticker": ticker,
        "company_name": candidate.get("company_name") or profile.get("companyName") or ticker,
        "sector": candidate.get("sector") or profile.get("sector"),
        "industry": candidate.get("industry") or profile.get("industry"),
        "as_of": str(candidate.get("as_of") or candidate.get("price_as_of") or "")[:10],
        "priority": candidate.get("research_priority"),
        "score": candidate.get("research_score"),
        "trend": trend,
        "read": read,
        "core_question": core_question,
        "description": _plain_description(profile),
        "tiles": [
            ("Price", _money(candidate.get("price")), f"S4 · {_date(candidate.get('price_as_of'))}"),
            ("Market cap", _money(candidate.get("market_cap")), "Current universe metadata"),
            ("Revenue growth", _pct(revenue_growth), f"S2 · {_date(income.get('date'))}"),
            ("FCF margin", _pct(fcf_margin), f"S2 · {_date(cash.get('date'))}"),
            ("FCF yield", _pct(candidate.get("fcf_yield")), "Derived · current market cap"),
        ],
        "drivers": drivers,
        "valuation": valuation,
        "triggers": triggers,
        "risks": risks,
        "gaps": ([
            f"Market price and trend data is stale as of {_date(candidate.get('price_as_of'))}; refresh before using timing signals."
        ] if not price_is_current else []) + [
            "No filing narrative, footnotes, segment normalization, or management guidance has been extracted yet.",
            "No earnings transcript, investor presentation, or guidance-change history is in the local evidence set.",
            "No reverse DCF, peer valuation, or explicit market-implied expectations analysis has been completed.",
            "Ownership, short interest, borrow, crowding, factor exposure, and passive-flow relevance are not sourced.",
            "The current universe and historical price set remain survivorship-biased until delisted securities are added.",
        ],
        "sources": sources,
        "next_route": (
            "Initiating coverage: extract the latest 10-K/10-Q narrative and segments, normalize KPIs, "
            "build a reverse DCF and peer frame, then convert only a source-backed variant view into a thesis tracker."
        ),
        "cash": balance.get("cashAndShortTermInvestments"),
        "debt": balance.get("totalDebt"),
    }


def render_tearsheet(context: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tiles = "".join(
        f'<div class="tile"><span>{_esc(label)}</span><b>{_esc(value)}</b><small>{_esc(note)}</small></div>'
        for label, value, note in context["tiles"]
    )
    driver_rows = "".join(
        "<tr>"
        f"<td><b>{_esc(row['metric'])}</b></td><td>{_esc(row['period'])}</td>"
        f"<td class='num'>{_esc(row['value'])}</td><td class='num'>{_esc(row['change'])}</td>"
        f"<td>{_esc(row['source'])}</td><td>{_esc(row['evidence'])}</td><td>{_esc(row['confidence'])}</td>"
        "</tr>" for row in context["drivers"]
    )
    valuation_rows = "".join(
        f"<tr><td><b>{_esc(name)}</b></td><td class='num'>{_esc(value)}</td>"
        f"<td>{_esc(period)}</td><td>{_esc(evidence)}</td></tr>"
        for name, value, period, evidence in context["valuation"]
    )
    trigger_rows = "".join(
        f"<div class='trigger'><b>{_esc(item['title'])}</b><p>{_esc(item['detail'])}</p>"
        f"<small>{_esc(item['source'])}</small></div>" for item in context["triggers"]
    )
    risk_items = "".join(f"<li>{_esc(item)}</li>" for item in context["risks"])
    gap_items = "".join(f"<li>{_esc(item)}</li>" for item in context["gaps"])
    source_rows = "".join(
        "<tr>"
        f"<td><b>{_esc(row['id'])}</b></td><td>{_esc(row['name'])}<small>{_esc(row['type'])}</small></td>"
        f"<td>{_esc(row['as_of'])}</td><td>{_esc(row['period'])}</td><td>{_esc(row['freshness'])}</td>"
        f"<td>{_esc(row['notes'])}</td>"
        "</tr>" for row in context["sources"]
    )
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_esc(context['ticker'])} — Fundamental research baseline</title>
<style>
:root{{--bg:#0b1020;--panel:#121a2d;--panel2:#17233b;--text:#eaf0fa;--muted:#9cabc0;--line:#293954;--blue:#72b7ff;--green:#67dca8;--amber:#ffd070;--red:#ff9999}}
*{{box-sizing:border-box}}body{{margin:0;background:linear-gradient(180deg,#09101d,#0d1628 42%,#0a1020);color:var(--text);font:14px/1.5 Inter,Segoe UI,Arial,sans-serif}}main{{max-width:1260px;margin:auto;padding:30px}}a{{color:var(--blue)}}
.eyebrow{{color:var(--muted);font-size:12px;letter-spacing:.8px;text-transform:uppercase}}h1{{font-size:34px;line-height:1.08;margin:6px 0}}h2{{font-size:20px;margin:30px 0 12px}}h3{{font-size:14px;margin:0 0 5px}}.muted,small{{color:var(--muted)}}
.warning{{margin:18px 0;padding:12px 15px;background:#332a18;border:1px solid #98742c;border-radius:10px;color:#ffdc8a;font-weight:650}}.hero{{display:grid;grid-template-columns:1.45fr .75fr;gap:14px}}
.panel,.tile,.trigger{{background:rgba(18,26,45,.95);border:1px solid var(--line);border-radius:12px}}.panel{{padding:18px}}.read{{font-size:17px;line-height:1.55;margin:0}}.question{{font-size:17px;font-weight:700;color:#cae2ff;margin:7px 0 0}}
.tiles{{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin:14px 0}}.tile{{padding:13px 14px}}.tile span,.tile small{{display:block}}.tile span{{font-size:11px;text-transform:uppercase;color:var(--muted);letter-spacing:.5px}}.tile b{{display:block;font-size:22px;margin:3px 0}}
.two{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}.triggers{{display:grid;grid-template-columns:1fr 1fr;gap:10px}}.trigger{{padding:13px;background:var(--panel2)}}.trigger p{{margin:5px 0 8px}}table{{border-collapse:collapse;width:100%}}th{{text-align:left;color:#b8c5d8;font-size:11px;text-transform:uppercase;letter-spacing:.45px;border-bottom:1px solid var(--line);padding:9px}}td{{padding:10px 9px;border-bottom:1px solid #202d44;vertical-align:top}}td.num{{text-align:right;font-variant-numeric:tabular-nums}}td small{{display:block;margin-top:2px}}ul{{margin:6px 0;padding-left:20px}}li{{margin:6px 0}}.route{{border-left:4px solid var(--blue);background:var(--panel2);padding:14px 16px;border-radius:8px}}footer{{padding:24px 0;color:var(--muted);font-size:12px}}
@media(max-width:850px){{main{{padding:18px}}.hero,.two,.triggers,.tiles{{grid-template-columns:1fr}}.table-wrap{{overflow:auto}}table{{min-width:760px}}}}
</style></head><body><main>
<div class="eyebrow">Fundamental sleeve · issuer baseline · {_esc(context['as_of'])}</div>
<h1>{_esc(context['ticker'])} <span class="muted">{_esc(context['company_name'])}</span></h1>
<div class="muted">{_esc(context['sector'])} · {_esc(context['industry'])} · {_esc(context['priority'])} · Score {_esc(context['score'])} · {_esc(context['trend'])} trend</div>
<div class="warning">Research baseline only. This page does not recommend a security, allocate capital, or enable an order.</div>
<section class="hero"><div class="panel"><div class="eyebrow">Investor read</div><p class="read">{_esc(context['read'])}</p></div>
<div class="panel"><div class="eyebrow">Core research question</div><p class="question">{_esc(context['core_question'])}</p></div></section>
<section class="tiles">{tiles}</section>
<section class="panel"><h3>Business baseline</h3><p>{_esc(context['description'])}</p><small>Provider-standardized company profile (S1); verify against the latest filing before use in a thesis.</small></section>
<h2>Earnings drivers</h2><section class="panel table-wrap"><table><thead><tr><th>Metric</th><th>Period</th><th>Value</th><th>YoY / Δ</th><th>Source</th><th>Evidence</th><th>Confidence</th></tr></thead><tbody>{driver_rows}</tbody></table></section>
<section class="two"><div><h2>Trailing Valuation Snapshot</h2><div class="panel"><table><thead><tr><th>Metric</th><th>Value</th><th>Period / basis</th><th>Evidence</th></tr></thead><tbody>{valuation_rows}</tbody></table><p class="muted">Cash {_money(context['cash'])}; debt {_money(context['debt'])}. Historical and derived measures only—no forward value conclusion is supported yet.</p></div></div>
<div><h2>Research risks</h2><div class="panel"><ul>{risk_items}</ul></div></div></section>
<h2>Proof points and triggers</h2><section class="triggers">{trigger_rows}</section>
<section class="two"><div><h2>Material evidence gaps</h2><div class="panel"><ul>{gap_items}</ul></div></div>
<div><h2>Next analytical route</h2><div class="route">{_esc(context['next_route'])}</div></div></section>
<h2>Source ledger</h2><section class="panel table-wrap"><table><thead><tr><th>ID</th><th>Source</th><th>As of</th><th>Period</th><th>Freshness</th><th>Notes</th></tr></thead><tbody>{source_rows}</tbody></table></section>
<footer><a href="../fundamental_daily.html">← Back to broad research funnel</a> · No live-action path is present.</footer>
</main></body></html>"""
    output_path.write_text(page, encoding="utf-8")
    return output_path


def build_tearsheet_pack(
    candidates: pd.DataFrame,
    fmp: pd.DataFrame,
    sec: pd.DataFrame,
    output_dir: Path,
    *,
    max_names: int = 10,
) -> dict[str, str]:
    """Render one standalone issuer baseline for each advanced screen name."""
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = candidates[candidates["research_priority"].isin(ADVANCE_PRIORITIES)].head(max_names)
    links: dict[str, str] = {}
    manifest: list[dict[str, Any]] = []
    for _, candidate in selected.iterrows():
        context = build_tearsheet_context(candidate, fmp, sec)
        ticker = context["ticker"]
        path = render_tearsheet(context, output_dir / f"{ticker}.html")
        links[ticker] = f"tearsheets/{path.name}"
        manifest.append({
            "ticker": ticker,
            "as_of": context["as_of"],
            "priority": context["priority"],
            "path": str(path),
            "live_actions_enabled": False,
        })
    (output_dir / "manifest.json").write_text(
        json.dumps({"tearsheets": manifest, "live_actions_enabled": False}, indent=2),
        encoding="utf-8",
    )
    return links
