"""Free listing and Yahoo adapters; incomplete coverage is explicit, never eligible.

Nasdaq listings establish listed security identity, not domicile or liquidity.
Yahoo is an unofficial secondary source and remains separately labelled.
"""
from __future__ import annotations

from io import StringIO
import math
import re

import pandas as pd


def parse_listing_directory(text, *, nasdaq):
    lines = text.splitlines()
    stamp = next((s for s in lines if s.startswith("File Creation Time:")), None)
    if stamp is None:
        raise ValueError("listing directory has no creation timestamp")
    created = re.search(r"(\d{8})(\d{2}:\d{2})", stamp)
    if not created:
        raise ValueError("unrecognized listing creation timestamp")
    as_of = pd.to_datetime(created[1] + created[2], format="%m%d%Y%H:%M").tz_localize("America/New_York")
    frame = pd.read_csv(StringIO("\n".join(s for s in lines if not s.startswith("File Creation Time:"))), sep="|", dtype=str, keep_default_na=False)
    required = {"Symbol" if nasdaq else "ACT Symbol", "Security Name", "ETF", "Test Issue"}
    if not required <= set(frame.columns):
        raise ValueError("listing directory schema changed")
    rows = []
    for item in frame.to_dict("records"):
        ticker = item["Symbol" if nasdaq else "ACT Symbol"].strip().replace(".", "-")
        name = item["Security Name"]
        reasons = []
        if item["Test Issue"] != "N": reasons.append("test_or_unknown_issue")
        if item["ETF"] != "N": reasons.append("etf_or_unknown_type")
        if nasdaq and item.get("Financial Status") != "N": reasons.append("financial_status_not_normal")
        if nasdaq and item.get("NextShares") != "N": reasons.append("nextshares_or_unknown")
        exchange = "NASDAQ" if nasdaq else {"N": "NYSE", "A": "AMEX"}.get(item.get("Exchange"))
        if not exchange: reasons.append("outside_exchange_scope")
        if not re.fullmatch(r"[A-Z]{1,5}(?:-[AB])?", ticker): reasons.append("unsupported_symbol")
        if re.search(r"\b(preferred|preference|warrants?|rights?|units?|notes?|bonds?|depositary)\b", name, re.I):
            reasons.append("non_common_or_depositary")
        if not re.search(r"\b(common|ordinary|capital)\s+(stock|shares?)\b", name, re.I):
            reasons.append("common_stock_not_verified")
        rows.append(dict(ticker=ticker, exchange=exchange, company_name=name,
            listing_as_of=as_of, listing_eligible=not reasons,
            listing_reasons=";".join(reasons), source="Nasdaq Trader Symbol Directory"))
    result = pd.DataFrame(rows)
    if result["ticker"].duplicated().any():
        raise ValueError("duplicate normalized symbols in listing directory")
    return result


def symbol_candidate(listing, info, *, fetched_at, price_floor=3, volume_floor=300_000):
    now = pd.Timestamp(fetched_at)
    if now.tzinfo is None:
        raise ValueError("fetched_at needs timezone")
    row = dict(listing)
    reasons = row.get("listing_reasons", "").split(";") if not row.get("listing_eligible") else []
    age = now.tz_convert("UTC") - pd.Timestamp(row["listing_as_of"]).tz_convert("UTC")
    if not pd.Timedelta(0) <= age <= pd.Timedelta(days=4): reasons.append("stale_listing")
    if str(info.get("symbol", "")).replace(".", "-") != row["ticker"]: reasons.append("metadata_identity_mismatch")
    if info.get("country") != "United States": reasons.append("us_domicile_not_verified")
    if info.get("quoteType") != "EQUITY": reasons.append("equity_type_not_verified")
    for field, key in [("price", "regularMarketPrice"), ("avg_volume", "averageVolume"), ("market_cap", "marketCap")]:
        row[field] = pd.to_numeric(info.get(key), errors="coerce")
        if not math.isfinite(row[field]) or row[field] <= 0: reasons.append(f"missing_{field}")
    quote_time = pd.to_datetime(info.get("regularMarketTime"), unit="s", utc=True, errors="coerce")
    if pd.isna(quote_time) or not pd.Timedelta(0) <= now.tz_convert("UTC")-quote_time <= pd.Timedelta(days=4):
        reasons.append("stale_quote")
    if row["price"] <= price_floor: reasons.append("price_filter")
    if row["avg_volume"] <= volume_floor: reasons.append("volume_filter")
    row.update(sector=info.get("sector"), industry=info.get("industry"),
               as_of=now.tz_convert("America/New_York").tz_localize(None).normalize(),
               metadata_source="Yahoo Finance via yfinance", eligible=not reasons,
               eligibility_reasons=";".join(filter(None, reasons)))
    return row


def share_structure(info):
    values = {"float_shares": info.get("floatShares"), "outstanding_shares": info.get("sharesOutstanding")}
    if any(v is None or not math.isfinite(float(v)) or float(v) <= 0 for v in values.values()):
        raise ValueError("Yahoo share structure incomplete")
    if values["float_shares"] > values["outstanding_shares"] * 1.01:
        raise ValueError("float exceeds shares outstanding; review share class/ADR basis")
    return {**values, "source": "Yahoo Finance via yfinance", "vintage": "current_snapshot"}


def normalize_yahoo_news(payload, *, ticker, company_name, as_of):
    """Exclude unrelated feed items; never assign the requested ticker to all hits."""
    cutoff = pd.Timestamp(as_of)
    if cutoff.tzinfo is None:
        raise ValueError("as_of needs timezone")
    token = re.sub(r"\s+(inc\.?|corp\.?|corporation|company)\b.*$", "", company_name, flags=re.I).strip()
    rows = []
    for item in payload:
        content = item.get("content", item)
        title = str(content.get("title", ""))
        body = str(content.get("summary") or content.get("description") or "")
        # Feed membership alone does not prove issuer relevance.
        linked = content.get("finance", {}).get("stockTickers", [])
        tickers = {str(v.get("symbol", "")) if isinstance(v, dict) else str(v) for v in linked}
        relevant = ticker in tickers or re.search(r"(?<!\w)" + re.escape(ticker) + r"(?!\w)", title + " " + body)
        relevant = relevant or (len(token) >= 4 and token.lower() in (title + " " + body).lower())
        stamp = pd.to_datetime(content.get("pubDate"), utc=True, errors="coerce")
        url = (content.get("canonicalUrl") or {}).get("url", "")
        if not relevant or not title or pd.isna(stamp) or stamp > cutoff or not url.startswith("https://"):
            continue
        rows.append(dict(symbol=ticker, publishedDate=stamp.isoformat(), title=title, text=body,
            url=url, publisher=(content.get("provider") or {}).get("displayName", "Yahoo Finance"),
            source_scope="secondary_business_news", _endpoint="yfinance_news"))
    return rows


def reported_earnings_dates(frame, *, as_of):
    """Secondary evidence only; zero EPS counts, estimated past dates do not."""
    cutoff = pd.Timestamp(as_of)
    if cutoff.tzinfo is None: raise ValueError("as_of needs timezone")
    rows = []
    for stamp, row in frame.iterrows():
        stamp = pd.Timestamp(stamp)
        if stamp.tzinfo is None: raise ValueError("earnings date lacks timezone")
        actual = pd.to_numeric(row.get("Reported EPS"), errors="coerce")
        if stamp > cutoff or not math.isfinite(actual): continue
        rows.append(dict(date=stamp.tz_convert("America/New_York").date().isoformat(),
            eps_actual=float(actual), source="Yahoo Finance via yfinance",
            evidence_status="secondary_reported_requires_primary_confirmation"))
    return rows
