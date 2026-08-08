"""SEC EDGAR submissions and XBRL company-facts adapter.

SEC is the filing source of truth.  The adapter joins accession numbers in
companyfacts to the filing acceptance timestamp in submissions so historical
research can exclude facts that were not yet public.
"""

from __future__ import annotations

import os
import time
from datetime import date
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

from .config import ROOT
from .storage import archive_json, iso_utc, snapshot_part_path, write_immutable_parquet


SEC_DATA = "https://data.sec.gov"
SEC_WWW = "https://www.sec.gov"


def load_user_agent() -> str:
    load_dotenv(ROOT / ".env", override=False)
    value = os.environ.get("FUNDAMENTAL_SEC_USER_AGENT", "").strip()
    if not value:
        raise RuntimeError(
            "FUNDAMENTAL_SEC_USER_AGENT is required (for example: "
            "'New Seasonals research your-email@example.com')"
        )
    return value


class SECClient:
    def __init__(self, user_agent: str | None = None, *, timeout: int = 30,
                 sleep_seconds: float = 0.12, session=None):
        self.user_agent = user_agent or load_user_agent()
        self.timeout = timeout
        self.sleep_seconds = max(0.11, sleep_seconds)
        self.session = session or requests.Session()

    def get_json(self, url: str) -> dict:
        response = self.session.get(
            url,
            headers={"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        time.sleep(self.sleep_seconds)
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError(f"SEC returned {type(payload).__name__} for {url}")
        return payload

    def ticker_map(self) -> dict[str, int]:
        payload = self.get_json(f"{SEC_WWW}/files/company_tickers.json")
        return {
            str(row["ticker"]).upper().replace(".", "-"): int(row["cik_str"])
            for row in payload.values()
            if isinstance(row, dict) and row.get("ticker") and row.get("cik_str")
        }

    def submissions(self, cik: int) -> dict:
        return self.get_json(f"{SEC_DATA}/submissions/CIK{cik:010d}.json")

    def companyfacts(self, cik: int) -> dict:
        return self.get_json(f"{SEC_DATA}/api/xbrl/companyfacts/CIK{cik:010d}.json")


def acceptance_map(submissions: dict) -> dict[str, str]:
    recent = ((submissions.get("filings") or {}).get("recent") or {})
    accessions = recent.get("accessionNumber") or []
    accepted = recent.get("acceptanceDateTime") or []
    return {str(a): str(t) for a, t in zip(accessions, accepted) if a and t}


def normalize_companyfacts(
    payload: dict,
    submissions: dict,
    *,
    ticker: str,
    cik: int,
    snapshot_as_of: str | date,
    digest: str,
    fetched_at: str,
) -> pd.DataFrame:
    accepted_by_accn = acceptance_map(submissions)
    rows: list[dict[str, Any]] = []
    for taxonomy, concepts in (payload.get("facts") or {}).items():
        for tag, concept in (concepts or {}).items():
            for unit, observations in (concept.get("units") or {}).items():
                for obs in observations or []:
                    accn = obs.get("accn")
                    rows.append({
                        "ticker": ticker.upper(),
                        "cik": int(cik),
                        "taxonomy": taxonomy,
                        "tag": tag,
                        "label": concept.get("label"),
                        "description": concept.get("description"),
                        "unit": unit,
                        "value": obs.get("val"),
                        "start": obs.get("start"),
                        "end": obs.get("end"),
                        "fiscal_year": obs.get("fy"),
                        "fiscal_period": obs.get("fp"),
                        "form": obs.get("form"),
                        "filed": obs.get("filed"),
                        "frame": obs.get("frame"),
                        "accession_number": accn,
                        "accepted_at": accepted_by_accn.get(str(accn)) if accn else None,
                        "source_name": "SEC EDGAR XBRL Company Facts",
                        "source_label": "fact_source_reported",
                        "source_url": f"{SEC_DATA}/api/xbrl/companyfacts/CIK{cik:010d}.json",
                        "fetched_at": fetched_at,
                        "snapshot_as_of": str(snapshot_as_of)[:10],
                        "payload_digest": digest,
                    })
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["accepted_at"] = pd.to_datetime(frame["accepted_at"], utc=True, errors="coerce")
    return frame


def fetch_companyfacts_snapshot(
    client: SECClient, ticker: str, cik: int, snapshot_as_of: str | date
) -> tuple[pd.DataFrame, dict]:
    part = snapshot_part_path("sec", snapshot_as_of, ticker, "companyfacts")
    if part.exists():
        frame = pd.read_parquet(part)
        return frame, {
            "ticker": ticker.upper(),
            "cik": int(cik),
            "rows": len(frame),
            "facts_digest": (
                str(frame["payload_digest"].dropna().iloc[0])
                if "payload_digest" in frame.columns and frame["payload_digest"].notna().any()
                else None
            ),
            "submissions_digest": None,
            "facts_path": None,
            "submissions_path": None,
            "snapshot_path": str(part),
            "fetched_at": (
                str(frame["fetched_at"].dropna().iloc[0])
                if "fetched_at" in frame.columns and frame["fetched_at"].notna().any()
                else iso_utc()
            ),
            "reused_frozen_part": True,
        }
    submissions = client.submissions(cik)
    facts = client.companyfacts(cik)
    submissions_path, submissions_digest = archive_json(submissions, "sec", "submissions", str(cik))
    facts_path, facts_digest = archive_json(facts, "sec", "companyfacts", str(cik))
    fetched_at = iso_utc()
    frame = normalize_companyfacts(
        facts,
        submissions,
        ticker=ticker,
        cik=cik,
        snapshot_as_of=snapshot_as_of,
        digest=facts_digest,
        fetched_at=fetched_at,
    )
    if not frame.empty:
        write_immutable_parquet(frame, part)
    return frame, {
        "ticker": ticker.upper(),
        "cik": int(cik),
        "rows": len(frame),
        "facts_digest": facts_digest,
        "submissions_digest": submissions_digest,
        "facts_path": str(facts_path),
        "submissions_path": str(submissions_path),
        "snapshot_path": str(part) if not frame.empty else None,
        "fetched_at": fetched_at,
        "reused_frozen_part": False,
    }
