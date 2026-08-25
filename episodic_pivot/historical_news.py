"""Point-in-time evidence ledger for historical Episodic Pivot candidates.

The labeler is deliberately blind to forward returns.  SEC acceptance times are
the primary timing source; FMP stock news is a secondary discovery archive whose
timezone-naive timestamps can never, by themselves, prove pre-open causality.
Empty search results mean unresolved coverage, not "no catalyst".
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, timezone
from datetime import time as wall_time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import exchange_calendars
import pandas as pd
import requests

from .news import _contains_non_negated_phrase, normalize_url

SEC_DATA = "https://data.sec.gov"
SEC_ARCHIVES = "https://www.sec.gov/Archives/edgar/data"
FMP_STABLE = "https://financialmodelingprep.com/stable"
_NY = ZoneInfo("America/New_York")
_UTC = timezone.utc
_XNYS = exchange_calendars.get_calendar(
    "XNYS",
    start="1990-01-01",
    end="2040-12-31",
)
_SEC_CHUNK = re.compile(r"^CIK\d+-submissions-\d+\.json$")
_LABEL_SCHEMA_VERSION = "v6"

BLINDED_EVENT_COLUMNS = (
    "ticker",
    "date",
    "previous_session",
    "previous_close",
    "event_open",
    "gap_pct",
    "prior_addv_63",
    "prior_atr_14",
    "prior_atr_pct_14",
    "prior_atr_window_clean",
    "prior_atr_calendar_complete",
    "prior_63d_return_pct",
    "earnings_date_match",
    "prior_window_clean",
    "basis_review_cleared",
    "event_half_double_review_required",
    "sample_period",
    "holdout_status",
)

_CATEGORY_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "EARNINGS_GUIDANCE",
        (
            "financial results",
            "quarterly results",
            "earnings per share",
            "earnings release",
            "earnings",
            "reports first quarter",
            "reports second quarter",
            "reports third quarter",
            "reports fourth quarter",
            "revenue guidance",
            "raises guidance",
            "raised guidance",
            "lowers guidance",
            "lowered guidance",
            "full-year outlook",
            "fiscal year outlook",
        ),
    ),
    (
        "FINANCING_DILUTION",
        (
            "public offering",
            "registered direct",
            "at-the-market offering",
            "secondary offering",
            "convertible notes",
            "shelf registration",
            "prospectus supplement",
            "equity line",
        ),
    ),
    (
        "M_AND_A_STRATEGIC",
        (
            "to acquire",
            "will acquire",
            "merger agreement",
            "definitive agreement",
            "strategic alternatives",
            "tender offer",
            "business combination",
        ),
    ),
    (
        "REGULATORY_CLINICAL",
        (
            "fda approval",
            "fda approves",
            "complete response letter",
            "clinical trial",
            "primary endpoint",
            "phase 1",
            "phase 2",
            "phase 3",
            "clinical hold",
            "regulatory approval",
        ),
    ),
    (
        "PRODUCT_CUSTOMER_CONTRACT",
        (
            "contract award",
            "awarded a contract",
            "purchase order",
            "multi-year contract",
            "commercial launch",
            "new product",
            "launches",
            "customer agreement",
            "strategic partnership",
        ),
    ),
    (
        "LEGAL_INVESTIGATION",
        (
            "investigation",
            "subpoena",
            "department of justice",
            "sec investigation",
            "sec probe",
            "sec charges",
            "charged by the sec",
            "lawsuit",
            "settlement",
            "patent litigation",
        ),
    ),
    (
        "DISTRESS_RESTRUCTURING",
        (
            "chapter 11",
            "bankruptcy",
            "going concern",
            "restructuring",
            "debt exchange",
            "forbearance",
            "covenant default",
        ),
    ),
    (
        "MANAGEMENT_GOVERNANCE",
        (
            "appoints chief executive",
            "chief executive officer resigns",
            "ceo resigns",
            "board of directors",
            "management transition",
        ),
    ),
    (
        "ANALYST_ACTION",
        ("upgraded to", "downgraded to", "price target", "initiates coverage"),
    ),
    (
        "CORPORATE_ACTION",
        (
            "reverse stock split",
            "stock split",
            "special dividend",
            "share repurchase",
            "exchange offer",
        ),
    ),
)

_CATEGORY_PRIORITY = (
    "EARNINGS_GUIDANCE",
    "REGULATORY_CLINICAL",
    "M_AND_A_STRATEGIC",
    "PRODUCT_CUSTOMER_CONTRACT",
    "FINANCING_DILUTION",
    "DISTRESS_RESTRUCTURING",
    "LEGAL_INVESTIGATION",
    "MANAGEMENT_GOVERNANCE",
    "CORPORATE_ACTION",
    "ANALYST_ACTION",
    "MATERIAL_AGREEMENT_UNCLASSIFIED",
    "OTHER_MATERIAL_FILING",
    "UNCLASSIFIED_COVERAGE",
)

_POSITIVE_TRAJECTORY_PHRASES = (
    "raises guidance",
    "raised guidance",
    "boosts outlook",
    "beats estimates",
    "beat estimates",
    "top estimates",
    "above consensus",
    "record revenue",
    "record sales",
    "met the primary endpoint",
    "positive topline",
    "fda approval",
    "lifts clinical hold",
    "lifted clinical hold",
    "removes clinical hold",
    "removed clinical hold",
    "resume testing",
    "resumes testing",
    "resume trial",
    "resumes trial",
    "contract award",
)
_ADVERSE_TRAJECTORY_PHRASES = (
    "lowers guidance",
    "lowered guidance",
    "cuts guidance",
    "withdrew guidance",
    "misses estimates",
    "missed estimates",
    "failed the primary endpoint",
    "clinical hold",
    "complete response letter",
    "public offering",
    "registered direct",
    "at-the-market offering",
    "reverse stock split",
    "chapter 11",
    "bankruptcy",
    "going concern",
    "sec charges",
    "charged by the sec",
    "criminal charges",
    "fraud charges",
)
_STRUCTURAL_ADVERSE_TYPES = {
    "FINANCING_DILUTION",
    "DISTRESS_RESTRUCTURING",
    "LEGAL_INVESTIGATION",
}

_LOW_SIGNAL_FMP_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "LOW_SIGNAL_HOLDINGS_UPDATE",
        (
            "reduces position in",
            "lowers holdings in",
            "cuts holdings in",
            "raises holdings in",
            "boosts stake in",
            "acquires shares of",
            "purchases shares of",
            "sells shares of",
            "new position in",
        ),
    ),
    (
        "LOW_SIGNAL_LEGAL_SOLICITATION",
        (
            "investigation reminder",
            "investigation alert",
            "shareholder alert",
            "encourages investors",
            "losses of $",
            "class action lawsuit against",
            "class action lawsuit and",
            "lead plaintiff deadline",
            "upcoming deadline",
            "investor deadline",
            "shareholder deadline",
            "opportunity to lead",
            "securities fraud lawsuit",
            "recover your losses",
            "seeks recovery for investors",
        ),
    ),
)

_POINT_IN_TIME_IDENTIFIER_QUALITIES = {"POINT_IN_TIME_TICKER_CIK_VALIDATED"}

_MATERIAL_SEC_FORMS = {
    "8-K",
    "6-K",
    "10-Q",
    "10-K",
    "20-F",
    "40-F",
    "S-1",
    "S-3",
    "F-1",
    "F-3",
    "NT 10-Q",
    "NT 10-K",
    "SC 13D",
    "SC 13D/A",
    "SC TO-T",
    "SC TO-I",
}


@dataclass(frozen=True)
class EventWindow:
    event_id: str
    ticker: str
    previous_close_at: pd.Timestamp
    event_open_at: pd.Timestamp
    event_close_at: pd.Timestamp


def _iso_now() -> str:
    return datetime.now(_UTC).isoformat().replace("+00:00", "Z")


def _json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _event_id(ticker: str, event_date: object) -> str:
    day = pd.Timestamp(event_date).date().isoformat()
    raw = f"{str(ticker).upper().strip()}|{day}".encode()
    return "EPH-" + hashlib.sha256(raw).hexdigest()[:20].upper()


def blind_events(events: pd.DataFrame) -> pd.DataFrame:
    """Return a stable candidate file with no event-day/forward outcome columns."""

    required = {"ticker", "date", "previous_session"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"historical candidates missing {sorted(missing)}")
    columns = [column for column in BLINDED_EVENT_COLUMNS if column in events]
    blinded = events.loc[:, columns].copy()
    blinded["ticker"] = blinded["ticker"].astype(str).str.upper().str.strip()
    for column in ("date", "previous_session"):
        blinded[column] = pd.to_datetime(blinded[column], errors="raise").dt.normalize()
    blinded.insert(
        0,
        "event_id",
        [
            _event_id(ticker, day)
            for ticker, day in zip(blinded["ticker"], blinded["date"])
        ],
    )
    if blinded["event_id"].duplicated().any():
        raise ValueError("historical event identifiers are not unique")
    return blinded.sort_values(["date", "ticker"]).reset_index(drop=True)


def event_window(row: pd.Series | dict[str, Any]) -> EventWindow:
    ticker = str(row["ticker"]).upper()
    event_day = pd.Timestamp(row["date"]).date()
    previous_day = pd.Timestamp(row["previous_session"]).date()
    previous_session = pd.Timestamp(previous_day)
    event_session = pd.Timestamp(event_day)
    if not _XNYS.is_session(previous_session) or not _XNYS.is_session(event_session):
        raise ValueError("event window contains a non-XNYS session date")
    previous_close = pd.Timestamp(_XNYS.session_close(previous_session)).tz_convert(
        "UTC"
    )
    event_open = pd.Timestamp(_XNYS.session_open(event_session)).tz_convert("UTC")
    event_close = pd.Timestamp(_XNYS.session_close(event_session)).tz_convert("UTC")
    return EventWindow(
        event_id=str(row["event_id"]),
        ticker=ticker,
        previous_close_at=previous_close,
        event_open_at=event_open,
        event_close_at=event_close,
    )


def classify_event_text(
    text: str,
    *,
    form: str = "",
    items: str = "",
) -> tuple[str, ...]:
    """Classify a filing/headline without using market outcomes."""

    probe = " ".join(str(text or "").lower().split())
    labels = [
        label
        for label, phrases in _CATEGORY_PATTERNS
        if any(phrase in probe for phrase in phrases)
    ]
    upper_form = str(form or "").upper().strip()
    item_set = {item.strip() for item in str(items or "").split(",") if item.strip()}
    if upper_form in {"10-K", "10-Q", "20-F", "40-F"} or "2.02" in item_set:
        labels.append("EARNINGS_GUIDANCE")
    if upper_form in {"S-1", "S-3", "F-1", "F-3"} or upper_form.startswith("424B"):
        labels.append("FINANCING_DILUTION")
    if "1.01" in item_set:
        labels.append("MATERIAL_AGREEMENT_UNCLASSIFIED")
    if "2.01" in item_set:
        labels.append("M_AND_A_STRATEGIC")
    if "3.02" in item_set:
        labels.append("FINANCING_DILUTION")
    if "4.02" in item_set:
        labels.append("LEGAL_INVESTIGATION")
    if "5.02" in item_set:
        labels.append("MANAGEMENT_GOVERNANCE")
    if upper_form.startswith("SC 13D"):
        labels.append("M_AND_A_STRATEGIC")
    if not labels and upper_form in {"8-K", "6-K"}:
        labels.append("OTHER_MATERIAL_FILING")
    if not labels:
        labels.append("UNCLASSIFIED_COVERAGE")
    return tuple(label for label in _CATEGORY_PRIORITY if label in set(labels))


def classify_trajectory(
    text: str,
    event_types: Iterable[str],
    *,
    infer_structural_adverse: bool = True,
) -> str:
    probe = " ".join(str(text or "").lower().split())
    types = set(event_types)
    positive = any(
        _contains_non_negated_phrase(probe, phrase)
        for phrase in _POSITIVE_TRAJECTORY_PHRASES
    )
    resolved_clinical_hold = any(
        _contains_non_negated_phrase(probe, phrase)
        for phrase in (
            "lifts clinical hold",
            "lifted clinical hold",
            "removes clinical hold",
            "removed clinical hold",
            "resume testing",
            "resumes testing",
            "resume trial",
            "resumes trial",
        )
    )
    resolved_legal_charge = bool(
        re.search(
            r"(?:sec charges|charged by the sec|criminal charges|fraud charges)"
            r".{0,80}\b(?:dismissed|dropped|withdrawn|vacated|cleared|acquitted)\b",
            probe,
        )
    )
    adverse = (
        infer_structural_adverse and bool(types & _STRUCTURAL_ADVERSE_TYPES)
    ) or any(
        _contains_non_negated_phrase(probe, phrase)
        and not (phrase == "clinical hold" and resolved_clinical_hold)
        and not (
            phrase
            in {"sec charges", "charged by the sec", "criminal charges", "fraud charges"}
            and resolved_legal_charge
        )
        for phrase in _ADVERSE_TRAJECTORY_PHRASES
    )
    if positive and adverse:
        return "MIXED_TRAJECTORY"
    if adverse:
        return "ADVERSE_OR_DILUTIVE"
    if positive:
        return "POSITIVE_TRAJECTORY"
    return "TRAJECTORY_UNRESOLVED"


def classify_fmp_source_quality(title: str, body: str, *, publisher: str = "") -> str:
    """Flag discovery items that are context, not candidate catalyst evidence."""

    probe = " ".join(f"{title} {body}".lower().split())
    publisher_probe = " ".join(str(publisher or "").lower().split())
    legal_template = any(
        phrase in probe
        for phrase in (
            "class action",
            "lead plaintiff",
            "upcoming deadline",
            "investor deadline",
            "shareholder deadline",
            "opportunity to lead",
            "securities fraud lawsuit",
            "recover your losses",
            "recover losses",
            "seeks recovery for investors",
            "securities fraud investigation",
            "securities law violations",
            "class action lawsuit has been filed",
        )
    )
    legal_audience = any(
        phrase in probe
        for phrase in ("shareholder", "stockholder", "investor", "purchaser")
    )
    legal_call_to_action = any(
        phrase in probe
        for phrase in (
            "contact ",
            "deadline to join",
            "join class action",
            "join the class action",
            "suffered a loss",
            "lost money",
            "potential recovery",
            "seek compensation",
            "secure counsel",
            "appointment as lead plaintiff",
            "lawsuit submission form",
            "free consultation",
            "encouraged to contact",
            "click here to learn more",
            "learn more about the investigation",
            "learn more about this investigation",
            "#classaction",
            "classaction",
        )
    )
    named_law_solicitor = any(
        phrase in probe
        for phrase in (
            "levi & korsinsky",
            "kahn swick & foti",
            "schall law firm",
            "gross law firm",
            "pomerantz law firm",
            "bronstein, gewirtz",
            "rosen, a leading law firm",
            "attorney advertising",
            "claims filer",
            "claimsfiler",
            "holzer & holzer",
            "holzer and holzer",
            "wohl & fruchter",
            "wohl and fruchter",
            "scott+scott",
            "scott + scott",
            "scott and scott",
            "faruqi & faruqi",
            "faruqi and faruqi",
        )
    )
    law_firm_identity = any(
        phrase in probe
        for phrase in (
            " law firm",
            " law offices",
            " law group",
            " law launches",
            " llp announces",
            "kirby mcinerney",
            "block & leviton",
            "gainey mckenna & egleston",
            "bfa law",
        )
    )
    law_firm_publisher = any(
        phrase in publisher_probe
        for phrase in ("law firm", "law offices", "law group", "law llp")
    )
    if legal_template and (
        legal_audience
        or legal_call_to_action
        or named_law_solicitor
        or law_firm_identity
        or law_firm_publisher
    ):
        return "LOW_SIGNAL_LEGAL_SOLICITATION"
    if legal_call_to_action and (
        legal_audience
        or named_law_solicitor
        or law_firm_identity
        or law_firm_publisher
    ):
        return "LOW_SIGNAL_LEGAL_SOLICITATION"
    if (named_law_solicitor or law_firm_identity) and any(
        phrase in probe
        for phrase in (
            "investigation",
            "investigate",
            "investigates",
            "investigating",
            "lawsuit",
            "class action",
        )
    ):
        return "LOW_SIGNAL_LEGAL_SOLICITATION"
    if (
        "investigation" in probe
        and "shareholder" in probe
        and any(
            phrase in probe
            for phrase in (
                "on behalf",
                "contact",
                "lost money",
                "losses",
                "affected by fraud",
            )
        )
    ):
        return "LOW_SIGNAL_LEGAL_SOLICITATION"
    for label, phrases in _LOW_SIGNAL_FMP_PATTERNS:
        if any(phrase in probe for phrase in phrases):
            return label
    return "STANDARD_DISCOVERY"


def _parallel_records(table: dict[str, Any]) -> list[dict[str, Any]]:
    columns = {key: value for key, value in table.items() if isinstance(value, list)}
    row_count = max((len(values) for values in columns.values()), default=0)
    return [
        {
            key: values[index] if index < len(values) else None
            for key, values in columns.items()
        }
        for index in range(row_count)
    ]


def normalize_sec_submissions(
    payloads: Iterable[tuple[str, dict[str, Any]]],
    *,
    ticker: str,
    cik: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source_file, payload in payloads:
        table = (payload.get("filings") or {}).get("recent") or {}
        if not table and isinstance(payload.get("accessionNumber"), list):
            table = payload
        for raw in _parallel_records(table):
            accession = str(raw.get("accessionNumber") or "").strip()
            if not accession:
                continue
            accepted_raw = str(raw.get("acceptanceDateTime") or "").strip()
            accepted_at = pd.to_datetime(accepted_raw, utc=True, errors="coerce")
            accepted_quality = "MISSING"
            if pd.notna(accepted_at):
                local = accepted_at.tz_convert(_NY)
                accepted_quality = (
                    "DATE_ONLY_OR_SYNTHETIC_MIDNIGHT"
                    if local.time() == wall_time(0, 0)
                    else "SEC_ACCEPTANCE_EXACT"
                )
            accession_flat = accession.replace("-", "")
            primary_document = str(raw.get("primaryDocument") or "").strip()
            filing_url = (
                f"{SEC_ARCHIVES}/{int(cik)}/{accession_flat}/{primary_document}"
                if primary_document
                else f"{SEC_ARCHIVES}/{int(cik)}/{accession_flat}/{accession}.txt"
            )
            categories = classify_event_text(
                f"{raw.get('primaryDocDescription') or ''} {raw.get('items') or ''}",
                form=str(raw.get("form") or ""),
                items=str(raw.get("items") or ""),
            )
            trajectory = classify_trajectory(
                f"{raw.get('primaryDocDescription') or ''} {raw.get('items') or ''}",
                categories,
            )
            rows.append(
                {
                    "ticker": ticker.upper(),
                    "cik": int(cik),
                    "accession_number": accession,
                    "form": raw.get("form"),
                    "filing_date": pd.to_datetime(
                        raw.get("filingDate"), errors="coerce"
                    ),
                    "report_date": pd.to_datetime(
                        raw.get("reportDate"), errors="coerce"
                    ),
                    "accepted_at": accepted_at,
                    "accepted_at_raw": accepted_raw,
                    "accepted_at_quality": accepted_quality,
                    "items": raw.get("items"),
                    "primary_document": primary_document,
                    "primary_doc_description": raw.get("primaryDocDescription"),
                    "url": filing_url,
                    "source_file": source_file,
                    "source_provider": "SEC_EDGAR",
                    "source_label": "fact_source_reported",
                    "event_types": "|".join(categories),
                    "trajectory_signal": trajectory,
                    "source_cluster_id": "SEC:" + accession,
                }
            )
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows).drop_duplicates("accession_number", keep="first")
    return frame.sort_values(["filing_date", "accepted_at"], na_position="last")


def normalize_fmp_news(
    payload: list[dict[str, Any]],
    *,
    ticker: str,
    company_name: str = "",
) -> pd.DataFrame:
    rows = []
    for raw in payload:
        title = str(raw.get("title") or "").strip()
        body = str(raw.get("text") or "").strip()
        raw_url = str(raw.get("url") or "").strip()
        url = normalize_url(raw_url) if raw_url else ""
        published_raw = str(raw.get("publishedDate") or "").strip()
        published_naive = pd.to_datetime(published_raw, errors="coerce")
        fingerprint_text = re.sub(r"[^a-z0-9]+", " ", f"{title} {body}".lower()).strip()
        fingerprint = hashlib.sha256(
            fingerprint_text[:4000].encode("utf-8")
        ).hexdigest()
        publisher = str(raw.get("publisher") or raw.get("site") or "")
        source_quality = classify_fmp_source_quality(
            title,
            body,
            publisher=publisher,
        )
        issuer_context = _issuer_bound_context(
            title,
            body,
            ticker=ticker,
            company_name=company_name,
        )
        categories = classify_event_text(issuer_context)
        issuer_relevant = bool(issuer_context)
        rows.append(
            {
                "ticker": ticker.upper(),
                "provider_symbol": str(raw.get("symbol") or "").upper(),
                "title": title,
                "text": body,
                "url": url,
                "publisher": publisher,
                "published_at_raw": published_raw,
                "published_date": (
                    published_naive.normalize() if pd.notna(published_naive) else pd.NaT
                ),
                "published_at": pd.NaT,
                "published_at_quality": "PROVIDER_TIMEZONE_UNKNOWN",
                "source_provider": "FMP_STOCK_NEWS",
                "source_label": "fact_provider_standardized",
                "source_quality": source_quality,
                "event_types": "|".join(categories),
                # FMP publishedDate has no timezone and same-day stories may be
                # reactions to the gap. Historical secondary direction is
                # therefore disabled rather than treated as a causal label.
                "trajectory_signal": "TRAJECTORY_UNRESOLVED",
                "source_cluster_id": "STORY:" + fingerprint,
                "issuer_relevant": issuer_relevant,
            }
        )
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    return frame.drop_duplicates(["source_cluster_id"], keep="first")


def _mentions_issuer(text: str, *, ticker: str, company_name: str) -> bool:
    lower = " ".join(re.findall(r"[a-z0-9]+", str(text).lower()))
    symbol = re.escape(str(ticker).lower())
    if re.search(
        rf"(?:\${symbol}\b|(?:nasdaq|nyse|amex)\s+{symbol}\b|\b{symbol}\s+(?:stock|shares)\b)",
        str(text).lower(),
    ):
        return True
    company = re.sub(
        r"\b(?:incorporated|corporation|company|holdings?|limited|plc|inc|corp|co|ltd)\b",
        " ",
        str(company_name).lower(),
    )
    tokens = [
        token for token in re.findall(r"[a-z0-9]+", company) if token not in {"the"}
    ]
    if not tokens:
        return False
    # Two-token cores reduce false positives for generic words such as Energy;
    # a distinctive one-token issuer name is matched on word boundaries.
    core = " ".join(tokens[:2] if len(tokens) >= 2 else tokens)
    return bool(re.search(rf"\b{re.escape(core)}\b", lower))


def _issuer_bound_context(
    title: str,
    body: str,
    *,
    ticker: str,
    company_name: str,
) -> str:
    """Keep only sentences bound to the candidate and explicit pronoun follow-ups."""

    combined = f"{title}. {body}"
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|[\r\n]+", combined)
        if sentence.strip()
    ]
    windows: list[str] = []
    for index, sentence in enumerate(sentences):
        # Keep issuer-bearing clauses, not an entire multi-company sentence.
        # This prevents a candidate beneficiary from inheriting another
        # company's bankruptcy, regulatory failure, offering, or acquisition.
        clauses = [
            clause.strip(" ,")
            for clause in re.split(
                r"\s+(?:on\s+(?:a\s+)?report(?:s)?\s+(?:of|that)|"
                r"after\s+(?:a\s+)?report(?:s)?\s+(?:of|that)|while|whereas)\s+|[;]",
                sentence,
                flags=re.IGNORECASE,
            )
            if clause.strip(" ,")
        ]
        issuer_clauses = [
            clause
            for clause in clauses
            if _mentions_issuer(clause, ticker=ticker, company_name=company_name)
        ]
        if not issuer_clauses:
            continue
        resolved = issuer_clauses
        if index + 1 < len(sentences) and re.match(
            r"^(?:the\s+company|it|management|the\s+board)\b",
            sentences[index + 1],
            flags=re.IGNORECASE,
        ):
            resolved.append(sentences[index + 1])
        window = " ".join(resolved)
        if window not in windows:
            windows.append(window)
    return " ".join(windows)


def bind_sec_evidence(filings: pd.DataFrame, event: pd.Series) -> pd.DataFrame:
    if filings.empty:
        return pd.DataFrame()
    window = event_window(event)
    event_day = pd.Timestamp(event["date"]).normalize()
    previous_day = pd.Timestamp(event["previous_session"]).normalize()
    accepted_day = (
        filings["accepted_at"].dt.tz_convert(_NY).dt.tz_localize(None).dt.normalize()
    )
    filing_day = pd.to_datetime(filings["filing_date"], errors="coerce").dt.normalize()
    forms = filings["form"].fillna("").astype(str).str.upper().str.strip()
    material_form = forms.isin(_MATERIAL_SEC_FORMS) | forms.str.startswith("424B")
    candidates = filings.loc[
        material_form
        & (
            accepted_day.isin([previous_day, event_day])
            | filing_day.isin([previous_day, event_day])
        )
    ].copy()
    if candidates.empty:
        return candidates
    candidates.insert(0, "event_id", window.event_id)
    candidates["conservative_public_at"] = candidates["accepted_at"] + pd.Timedelta(
        minutes=3
    )
    candidates["timing_status"] = "TIMING_UNRESOLVED"
    exact = candidates["accepted_at_quality"].eq("SEC_ACCEPTANCE_EXACT")
    public = candidates["conservative_public_at"]
    candidates.loc[
        exact & public.gt(window.previous_close_at) & public.le(window.event_open_at),
        "timing_status",
    ] = "PREOPEN_SEC_ASSUMED_PUBLIC"
    candidates.loc[
        exact & public.gt(window.event_open_at) & public.le(window.event_close_at),
        "timing_status",
    ] = "POST_OPEN_CONTEXT"
    candidates.loc[exact & public.le(window.previous_close_at), "timing_status"] = (
        "STALE_PRIOR_DISCLOSURE"
    )
    candidates.loc[exact & public.gt(window.event_close_at), "timing_status"] = (
        "AFTER_EVENT_SESSION"
    )
    return candidates


def bind_fmp_evidence(news: pd.DataFrame, event: pd.Series) -> pd.DataFrame:
    if news.empty:
        return pd.DataFrame()
    event_day = pd.Timestamp(event["date"]).normalize()
    previous_day = pd.Timestamp(event["previous_session"]).normalize()
    relevant = news.get("issuer_relevant", pd.Series(True, index=news.index)).fillna(
        False
    )
    candidates = news.loc[
        relevant
        & news["published_date"].between(previous_day - pd.Timedelta(days=1), event_day)
    ].copy()
    if candidates.empty:
        return candidates
    candidates.insert(0, "event_id", str(event["event_id"]))
    candidates["timing_status"] = "TIMING_UNRESOLVED"
    candidates.loc[candidates["published_date"].lt(previous_day), "timing_status"] = (
        "STALE_PRIOR_DISCLOSURE"
    )
    return candidates


def _category_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    for packed in frame.get("event_types", pd.Series(dtype=str)).fillna(""):
        for category in str(packed).split("|"):
            if category:
                counts[category] = counts.get(category, 0) + 1
    return counts


def _primary_category(frame: pd.DataFrame, *, fallback: str) -> str:
    counts = _category_counts(frame)
    return next(
        (category for category in _CATEGORY_PRIORITY if counts.get(category)),
        fallback,
    )


def _trajectory_posture(frame: pd.DataFrame) -> str:
    signals = set(frame.get("trajectory_signal", pd.Series(dtype=str)).dropna()) - {
        "TRAJECTORY_UNRESOLVED"
    }
    if len(signals) > 1 or "MIXED_TRAJECTORY" in signals:
        return "MIXED_TRAJECTORY"
    if signals:
        return next(iter(signals))
    return "TRAJECTORY_UNRESOLVED"


def summarize_event_evidence(
    events: pd.DataFrame,
    evidence: pd.DataFrame,
    provider_status: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    status_by_event = (
        {
            event_id: group
            for event_id, group in provider_status.groupby("event_id", sort=False)
        }
        if not provider_status.empty
        else {}
    )
    evidence_by_event = (
        {
            event_id: group
            for event_id, group in evidence.groupby("event_id", sort=False)
        }
        if not evidence.empty
        else {}
    )
    for event in events.itertuples(index=False):
        group = evidence_by_event.get(event.event_id, pd.DataFrame())
        statuses = status_by_event.get(event.event_id, pd.DataFrame())
        decision_group = group.loc[
            ~group.get("timing_status", pd.Series(dtype=str)).isin(
                {
                    "STALE_PRIOR_DISCLOSURE",
                    "POST_OPEN_CONTEXT",
                    "AFTER_EVENT_SESSION",
                }
            )
            & ~group.get(
                "source_quality",
                pd.Series("STANDARD_DISCOVERY", index=group.index),
            )
            .fillna("STANDARD_DISCOVERY")
            .astype(str)
            .str.startswith("LOW_SIGNAL_")
        ]
        preopen_sec = decision_group.loc[
            decision_group.get("source_provider", pd.Series(dtype=str)).eq("SEC_EDGAR")
            & decision_group.get("timing_status", pd.Series(dtype=str)).eq(
                "PREOPEN_SEC_ASSUMED_PUBLIC"
            )
        ]
        validated_preopen_sec = preopen_sec.loc[
            preopen_sec.get(
                "identifier_quality",
                pd.Series("", index=preopen_sec.index, dtype=str),
            ).isin(_POINT_IN_TIME_IDENTIFIER_QUALITIES)
        ]
        secondary_context = decision_group.loc[
            decision_group.get("source_provider", pd.Series(dtype=str)).eq(
                "FMP_STOCK_NEWS"
            )
        ]
        preopen_types = {
            category
            for packed in preopen_sec.get("event_types", pd.Series(dtype=str)).fillna(
                ""
            )
            for category in str(packed).split("|")
            if category
        }
        specific_preopen = preopen_types - {
            "MATERIAL_AGREEMENT_UNCLASSIFIED",
            "OTHER_MATERIAL_FILING",
            "UNCLASSIFIED_COVERAGE",
        }
        validated_types = {
            category
            for packed in validated_preopen_sec.get(
                "event_types", pd.Series(dtype=str)
            ).fillna("")
            for category in str(packed).split("|")
            if category
        }
        specific_validated = validated_types - {
            "MATERIAL_AGREEMENT_UNCLASSIFIED",
            "OTHER_MATERIAL_FILING",
            "UNCLASSIFIED_COVERAGE",
        }
        if specific_validated:
            evidence_posture = "PRIMARY_PREOPEN_SEC_ASSUMED_PUBLIC_CLASSIFIED"
        elif not validated_preopen_sec.empty:
            evidence_posture = "PRIMARY_PREOPEN_SEC_ASSUMED_PUBLIC_UNCLASSIFIED"
        elif specific_preopen:
            evidence_posture = (
                "PREOPEN_SEC_ASSUMED_PUBLIC_IDENTITY_UNRESOLVED_CLASSIFIED"
            )
        elif not preopen_sec.empty:
            evidence_posture = (
                "PREOPEN_SEC_ASSUMED_PUBLIC_IDENTITY_UNRESOLVED_UNCLASSIFIED"
            )
        elif not decision_group.empty:
            evidence_posture = "TIMING_UNRESOLVED"
        elif not group.empty:
            evidence_posture = "CONTEXT_ONLY_NOT_CAUSAL"
        elif (
            not statuses.empty
            and statuses["status"].astype(str).str.startswith("ERROR").all()
        ):
            evidence_posture = "PROVIDER_ERROR"
        else:
            evidence_posture = "COVERAGE_UNRESOLVED"
        if not validated_preopen_sec.empty:
            primary_type = _primary_category(
                validated_preopen_sec,
                fallback="UNCLASSIFIED_PRIMARY_DISCLOSURE",
            )
        elif not preopen_sec.empty:
            primary_type = "IDENTITY_UNRESOLVED"
        elif not secondary_context.empty:
            primary_type = "TIMING_UNRESOLVED"
        elif not group.empty:
            primary_type = "CONTEXT_ONLY"
        else:
            primary_type = "COVERAGE_UNRESOLVED"
        trajectory_posture = _trajectory_posture(validated_preopen_sec)
        preopen_sec_event_type = _primary_category(
            preopen_sec,
            fallback="NO_PREOPEN_SEC_EVIDENCE",
        )
        secondary_context_event_type = _primary_category(
            secondary_context,
            fallback="NO_SECONDARY_CONTEXT",
        )
        category_counts = _category_counts(decision_group)
        rows.append(
            {
                "event_id": event.event_id,
                "ticker": event.ticker,
                "date": event.date,
                "evidence_posture": evidence_posture,
                "primary_event_type": primary_type,
                "trajectory_posture": trajectory_posture,
                "preopen_sec_event_type": preopen_sec_event_type,
                "preopen_sec_trajectory_posture": _trajectory_posture(preopen_sec),
                "secondary_context_event_type": secondary_context_event_type,
                "secondary_context_trajectory_posture": _trajectory_posture(
                    secondary_context
                ),
                "unique_source_clusters": int(
                    decision_group.get(
                        "source_cluster_id", pd.Series(dtype=str)
                    ).nunique()
                ),
                "sec_filings": int(
                    decision_group.get("source_provider", pd.Series(dtype=str))
                    .eq("SEC_EDGAR")
                    .sum()
                ),
                "fmp_articles": int(
                    decision_group.get("source_provider", pd.Series(dtype=str))
                    .eq("FMP_STOCK_NEWS")
                    .sum()
                ),
                "unique_source_clusters_raw": int(
                    group.get("source_cluster_id", pd.Series(dtype=str)).nunique()
                ),
                "sec_filings_raw": int(
                    group.get("source_provider", pd.Series(dtype=str))
                    .eq("SEC_EDGAR")
                    .sum()
                ),
                "fmp_articles_raw": int(
                    group.get("source_provider", pd.Series(dtype=str))
                    .eq("FMP_STOCK_NEWS")
                    .sum()
                ),
                "provider_attempts": len(statuses),
                "provider_errors": int(
                    statuses.get("status", pd.Series(dtype=str))
                    .astype(str)
                    .str.startswith("ERROR")
                    .sum()
                ),
                "event_types": "|".join(
                    category
                    for category in _CATEGORY_PRIORITY
                    if category_counts.get(category)
                ),
            }
        )
    return pd.DataFrame(rows)


class SECSubmissionArchive:
    def __init__(
        self,
        user_agent: str,
        cache_root: str | Path,
        *,
        timeout: int = 30,
        sleep_seconds: float = 0.12,
        cache_only: bool = False,
        session: requests.Session | None = None,
    ):
        if not user_agent.strip():
            raise ValueError("SEC User-Agent is required")
        self.user_agent = user_agent.strip()
        self.cache_root = Path(cache_root)
        self.timeout = timeout
        self.sleep_seconds = max(0.11, float(sleep_seconds))
        self.cache_only = bool(cache_only)
        self.session = session or requests.Session()

    def _cached_json(self, url: str, path: Path) -> tuple[dict[str, Any], str, str]:
        if path.exists():
            envelope = json.loads(path.read_text(encoding="utf-8"))
            return envelope["payload"], str(envelope["retrieved_at"]), "CACHE_HIT"
        if self.cache_only:
            raise RuntimeError("SEC_CACHE_MISS")
        response = self.session.get(
            url,
            headers={
                "User-Agent": self.user_agent,
                "Accept-Encoding": "gzip, deflate",
            },
            timeout=self.timeout,
        )
        if response.status_code != 200:
            raise RuntimeError(f"SEC_HTTP_{response.status_code}")
        payload = response.json()
        if not isinstance(payload, dict):
            raise TypeError("SEC_NON_OBJECT_PAYLOAD")
        retrieved_at = _iso_now()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"source_url": url, "retrieved_at": retrieved_at, "payload": payload},
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        time.sleep(self.sleep_seconds)
        return payload, retrieved_at, "FETCHED"

    def filings(
        self,
        *,
        ticker: str,
        cik: int,
        start: date,
        end: date,
    ) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
        cik_text = f"{int(cik):010d}"
        main_name = f"CIK{cik_text}.json"
        main_url = f"{SEC_DATA}/submissions/{main_name}"
        main, retrieved_at, cache_status = self._cached_json(
            main_url, self.cache_root / "sec" / main_name
        )
        payloads: list[tuple[str, dict[str, Any]]] = [(main_name, main)]
        records = [
            {
                "provider": "SEC_EDGAR",
                "resource": main_name,
                "status": cache_status,
                "retrieved_at": retrieved_at,
                "raw_payload_sha256": _json_sha256(main),
            }
        ]
        for meta in (main.get("filings") or {}).get("files") or []:
            name = str(meta.get("name") or "")
            if not _SEC_CHUNK.fullmatch(name) or Path(name).name != name:
                continue
            filing_from = pd.to_datetime(meta.get("filingFrom"), errors="coerce")
            filing_to = pd.to_datetime(meta.get("filingTo"), errors="coerce")
            if (
                pd.notna(filing_from)
                and pd.notna(filing_to)
                and (filing_to.date() < start or filing_from.date() > end)
            ):
                continue
            url = f"{SEC_DATA}/submissions/{name}"
            payload, chunk_retrieved, chunk_status = self._cached_json(
                url, self.cache_root / "sec" / name
            )
            payloads.append((name, payload))
            records.append(
                {
                    "provider": "SEC_EDGAR",
                    "resource": name,
                    "status": chunk_status,
                    "retrieved_at": chunk_retrieved,
                    "raw_payload_sha256": _json_sha256(payload),
                }
            )
        bundle_digest = _json_sha256(
            [(name, _json_sha256(payload)) for name, payload in payloads]
        )
        safe_ticker = re.sub(r"[^A-Z0-9._-]", "_", ticker.upper())
        normalized_path = (
            self.cache_root
            / "sec-normalized"
            / (
                f"{_LABEL_SCHEMA_VERSION}_{safe_ticker}_CIK{cik_text}_"
                f"{start.isoformat()}_{end.isoformat()}_{bundle_digest[:16]}.parquet"
            )
        )
        if normalized_path.exists():
            return pd.read_parquet(normalized_path), records
        normalized = normalize_sec_submissions(payloads, ticker=ticker, cik=cik)
        normalized_path.parent.mkdir(parents=True, exist_ok=True)
        normalized.to_parquet(normalized_path, index=False)
        return normalized, records


class FMPStockNewsArchive:
    def __init__(
        self,
        api_key: str,
        cache_root: str | Path,
        *,
        timeout: int = 30,
        sleep_seconds: float = 0.12,
        max_pages: int = 5,
        page_limit: int = 250,
        cache_only: bool = False,
        session: requests.Session | None = None,
    ):
        if not api_key.strip():
            raise ValueError("FMP API key is required")
        self.api_key = api_key.strip()
        self.cache_root = Path(cache_root)
        self.timeout = timeout
        self.sleep_seconds = max(0.0, float(sleep_seconds))
        self.max_pages = max(1, int(max_pages))
        self.page_limit = max(1, min(250, int(page_limit)))
        self.cache_only = bool(cache_only)
        self.session = session or requests.Session()

    def news(
        self,
        *,
        ticker: str,
        start: date,
        end: date,
        company_name: str = "",
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        safe_ticker = re.sub(r"[^A-Z0-9._-]", "_", ticker.upper())
        path = (
            self.cache_root
            / "fmp-stock-news"
            / safe_ticker
            / f"{start.isoformat()}_{end.isoformat()}.json"
        )
        if path.exists():
            envelope = json.loads(path.read_text(encoding="utf-8"))
            payload = envelope["payload"]
            status = "CACHE_HIT"
            retrieved_at = str(envelope["retrieved_at"])
        else:
            if self.cache_only:
                raise RuntimeError("FMP_CACHE_MISS")
            payload: list[dict[str, Any]] = []
            for page in range(self.max_pages):
                response = self.session.get(
                    f"{FMP_STABLE}/news/stock",
                    params={
                        "symbols": ticker.upper(),
                        "from": start.isoformat(),
                        "to": end.isoformat(),
                        "page": page,
                        "limit": self.page_limit,
                        "apikey": self.api_key,
                    },
                    timeout=self.timeout,
                )
                if response.status_code != 200:
                    raise RuntimeError(f"FMP_HTTP_{response.status_code}")
                page_rows = response.json()
                if not isinstance(page_rows, list):
                    raise TypeError("FMP_NON_LIST_PAYLOAD")
                payload.extend(row for row in page_rows if isinstance(row, dict))
                time.sleep(self.sleep_seconds)
                if len(page_rows) < self.page_limit:
                    break
            retrieved_at = _iso_now()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "source_url": f"{FMP_STABLE}/news/stock",
                        "retrieved_at": retrieved_at,
                        "payload": payload,
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            status = "FETCHED"
        return normalize_fmp_news(payload, ticker=ticker, company_name=company_name), {
            "provider": "FMP_STOCK_NEWS",
            "resource": f"{ticker}:{start}:{end}",
            "status": status,
            "retrieved_at": retrieved_at,
            "rows": len(payload),
            "raw_payload_sha256": _json_sha256(payload),
        }
