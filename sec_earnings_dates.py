"""Primary-source confirmation of elapsed earnings dates from SEC EDGAR.

An issuer that announces results files an 8-K with Item 2.02 ("Results of
Operations and Financial Condition"). Its ``reportDate`` is the date of the
earliest event reported, i.e. the announcement. This replaces the FMP actuals
check that used to confirm an Alpha expectation had actually happened.

Only dates are confirmed; no EPS/revenue values are produced. Foreign filers
(6-K), issuers that skip the 8-K and SEC outages simply yield no proof. The
caller keeps those events as unverified rather than stopping publication.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any

import pandas as pd

from trading_calendar import TRADING_DAY

SEC_WWW = "https://www.sec.gov"
CONFIRMATION_SOURCE = "sec_8k_item_2_02"
# How far before the expected date an 8-K may be dated and still confirm it.
EARLY_WINDOW = pd.Timedelta(days=10)
# An 8-K is due within four business days of the event it reports.
MAX_FILING_LAG = pd.Timedelta(days=6)
# Items that accompany a plain results release (Reg FD / exhibits).
RESULTS_ONLY_ITEMS = {"2.02", "7.01", "9.01"}
LOOKBACK_TRADING_DAYS = 10


def confirmation_targets(prior: pd.DataFrame, as_of) -> pd.DataFrame:
    """Unconfirmed provider expectations dated in the recent window or today."""
    from earnings_calendar_provider import UNVERIFIED_ELAPSED
    as_of = pd.Timestamp(as_of).normalize()
    if not {"event_status", "fiscalDateEnding"}.issubset(prior.columns):
        return pd.DataFrame(columns=["ticker", "date", "fiscalDateEnding"])
    mask = (prior.event_status.isin(["expected", UNVERIFIED_ELAPSED])
            & prior.date.between(as_of - LOOKBACK_TRADING_DAYS * TRADING_DAY, as_of)
            & prior.fiscalDateEnding.notna() & prior.fiscalDateEnding.astype(str).ne(""))
    targets = prior.loc[mask, ["ticker", "date", "fiscalDateEnding"]].sort_values("date")
    # One proof per fiscal period; the reconciler supersedes every row of it.
    return targets.drop_duplicates(["ticker", "fiscalDateEnding"], keep="last").reset_index(drop=True)


def user_agent() -> str | None:
    for name in ("SEC_USER_AGENT", "FUNDAMENTAL_SEC_USER_AGENT"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return None


def _announcement_date(items: set[str], report_date, accepted_at):
    """``reportDate`` is the EARLIEST event in the filing. When an earnings 8-K
    also reports an unrelated event (e.g. an earlier bylaw change, Item 5.03),
    that date is not the release; fall back to the filing's acceptance day,
    since results 8-Ks are furnished alongside the release."""
    report_date = pd.to_datetime(report_date, errors="coerce")
    if items <= RESULTS_ONLY_ITEMS:
        return report_date
    accepted = pd.to_datetime(accepted_at, utc=True, errors="coerce")
    if pd.isna(accepted):
        return pd.NaT
    return accepted.tz_convert("America/New_York").tz_localize(None).normalize()


def _item_202_filings(submissions: dict, cik: int) -> list[dict]:
    recent = (submissions.get("filings") or {}).get("recent") or {}
    rows = []
    for i, form in enumerate(recent.get("form") or []):
        items = {item.strip() for item in str((recent.get("items") or [""] * (i + 1))[i] or "").split(",") if item.strip()}
        if form != "8-K" or "2.02" not in items:
            continue
        accession = str(recent["accessionNumber"][i])
        rows.append(dict(
            accession=accession,
            report_date=_announcement_date(items, recent["reportDate"][i], recent["acceptanceDateTime"][i]),
            accepted_at=str(recent["acceptanceDateTime"][i]),
            source_url=(f"{SEC_WWW}/Archives/edgar/data/{cik}/{accession.replace('-', '')}/"
                        f"{recent['primaryDocument'][i]}"),
            record={key: (recent.get(key) or [None] * (i + 1))[i] for key in
                    ("accessionNumber", "form", "items", "reportDate", "filingDate", "acceptanceDateTime",
                     "primaryDocument")},
        ))
    return rows


def match_filing(event, filings: list[dict], as_of) -> dict | None:
    """Pick the Item 2.02 filing closest to the expected date, if plausible."""
    expected = pd.Timestamp(event.date).normalize()
    fiscal = pd.to_datetime(event.fiscalDateEnding, errors="coerce")
    as_of = pd.Timestamp(as_of).normalize()
    best = None
    for filing in filings:
        announced = filing["report_date"]
        accepted = pd.to_datetime(filing["accepted_at"], utc=True, errors="coerce")
        if pd.isna(announced) or pd.isna(accepted) or pd.isna(fiscal):
            continue
        accepted_day = accepted.tz_convert("America/New_York").tz_localize(None).normalize()
        if not (fiscal <= announced <= accepted_day <= as_of and expected - EARLY_WINDOW <= announced
                and accepted_day - announced <= MAX_FILING_LAG):
            continue
        if best is None or abs(announced - expected) < abs(best["report_date"] - expected):
            best = filing
    return best


def collect_sec_confirmations(targets: pd.DataFrame, as_of, client=None) -> tuple[pd.DataFrame, dict]:
    """Return SEC date proofs for ``targets`` plus a receipt-safe summary.

    Never raises for missing credentials or per-issuer lookup failures; those
    events simply stay unconfirmed.
    """
    summary: dict[str, Any] = dict(requested=len(targets), confirmed=0, unmatched=[], lookup_errors=[], available=True)
    if targets.empty:
        return pd.DataFrame(), summary
    if client is None:
        agent = user_agent()
        if agent is None:
            summary.update(available=False, reason="SEC_USER_AGENT/FUNDAMENTAL_SEC_USER_AGENT not set")
            return pd.DataFrame(), summary
        from fundamental.sec import SECClient
        client = SECClient(agent)
    try:
        ciks = client.ticker_map()
    except Exception as exc:  # SEC outage: publish with events left unverified.
        summary.update(available=False, reason=f"ticker map unavailable ({type(exc).__name__})")
        return pd.DataFrame(), summary
    captured = pd.Timestamp.now(tz="UTC").isoformat()
    rows, used, cache = [], set(), {}
    for event in targets.sort_values(["ticker", "date"]).itertuples():
        ticker = str(event.ticker).upper()
        cik = ciks.get(ticker.replace(".", "-"))
        if cik is None:
            summary["unmatched"].append(ticker)
            continue
        if cik not in cache:
            try:
                cache[cik] = _item_202_filings(client.submissions(cik), cik)
            except Exception as exc:
                summary["lookup_errors"].append(f"{ticker}: {type(exc).__name__}")
                cache[cik] = None
        if cache[cik] is None:
            continue
        filing = match_filing(event, [f for f in cache[cik] if f["accession"] not in used], as_of)
        if filing is None:
            summary["unmatched"].append(ticker)
            continue
        used.add(filing["accession"])
        payload = json.dumps(filing["record"], sort_keys=True, default=str).encode()
        rows.append(dict(ticker=ticker, date=filing["report_date"], fiscalDateEnding=str(event.fiscalDateEnding),
                         announcement_confirmed=True, source_url=filing["source_url"],
                         accepted_at=filing["accepted_at"], captured_at=captured,
                         payload_digest=hashlib.sha256(payload).hexdigest(),
                         confirmation_source=CONFIRMATION_SOURCE))
    summary["confirmed"] = len(rows)
    summary["unmatched"] = sorted(set(summary["unmatched"]))
    return pd.DataFrame(rows), summary
