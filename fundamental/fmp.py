"""Financial Modeling Prep adapter with immutable payload archiving."""

from __future__ import annotations

import os
import time
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

from .config import FMP_ENDPOINTS, ROOT
from .storage import (
    archive_json,
    iso_utc,
    payload_digest as semantic_digest,
    snapshot_part_path,
    write_immutable_parquet,
)


BASE_URL = "https://financialmodelingprep.com/stable"
DEFAULT_TIMEOUT = 30
COLLECTION_ENDPOINTS = {"company-screener"}

PROFILE_STABLE_FIELDS = (
    "symbol", "companyName", "currency", "cik", "isin", "cusip",
    "exchange", "exchangeShortName", "industry", "sector", "country",
    "website", "description", "ceo", "fullTimeEmployees", "ipoDate",
    "isActivelyTrading", "isEtf", "isFund", "isAdr",
)


def load_api_key() -> str:
    load_dotenv(ROOT / ".env", override=False)
    key = os.environ.get("FMP_API_KEY", "").strip()
    if not key:
        raise RuntimeError("FMP_API_KEY is required in the environment or project .env")
    return key


class FMPClient:
    def __init__(self, api_key: str | None = None, *, timeout: int = DEFAULT_TIMEOUT,
                 retries: int = 3, sleep_seconds: float = 0.12, session=None):
        self.api_key = api_key or load_api_key()
        self.timeout = timeout
        self.retries = retries
        self.sleep_seconds = sleep_seconds
        self.session = session or requests.Session()

    def fetch(self, endpoint: str, symbol: str, **params: Any) -> list[dict]:
        if endpoint not in FMP_ENDPOINTS:
            raise ValueError(f"unsupported FMP endpoint: {endpoint}")
        query = {"symbol": symbol.upper(), "apikey": self.api_key, **params}
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                response = self.session.get(
                    f"{BASE_URL}/{endpoint}", params=query, timeout=self.timeout
                )
                if response.status_code == 429:
                    time.sleep(2 ** attempt)
                    continue
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, list):
                    raise RuntimeError(f"FMP {endpoint} returned {type(payload).__name__}")
                time.sleep(self.sleep_seconds)
                return payload
            except (requests.RequestException, ValueError, RuntimeError) as exc:
                last_error = exc
                if attempt < self.retries - 1:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"FMP {endpoint} failed for {symbol}: {last_error}")

    def fetch_collection(self, endpoint: str, **params: Any) -> list[dict]:
        """Fetch a supported non-symbol collection such as the stock screener."""
        if endpoint not in COLLECTION_ENDPOINTS:
            raise ValueError(f"unsupported FMP collection endpoint: {endpoint}")
        query = {"apikey": self.api_key, **params}
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                response = self.session.get(
                    f"{BASE_URL}/{endpoint}", params=query, timeout=self.timeout
                )
                if response.status_code == 429:
                    time.sleep(2 ** attempt)
                    continue
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, list):
                    raise RuntimeError(f"FMP {endpoint} returned {type(payload).__name__}")
                time.sleep(self.sleep_seconds)
                return payload
            except (requests.RequestException, ValueError, RuntimeError) as exc:
                last_error = exc
                if attempt < self.retries - 1:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"FMP {endpoint} failed: {last_error}")


def _endpoint_params(endpoint: str) -> dict:
    if endpoint == "profile":
        return {}
    if endpoint == "analyst-estimates":
        return {"period": "annual", "limit": 12, "page": 0}
    return {"period": "annual", "limit": 10}


def normalize_rows(
    payload: list[dict],
    *,
    ticker: str,
    endpoint: str,
    snapshot_as_of: str | date,
    digest: str,
    fetched_at: str,
) -> pd.DataFrame:
    if not payload:
        return pd.DataFrame()
    frame = pd.json_normalize(payload, sep="__")
    if endpoint == "profile":
        # /profile mixes stable issuer identity with intraday price, market cap,
        # beta, range, and volume.  Those volatile fields belong to dated market
        # sources and would make a daily metadata snapshot non-idempotent.
        stable = [column for column in PROFILE_STABLE_FIELDS if column in frame.columns]
        frame = frame[stable].copy()
    normalized_digest = semantic_digest(frame.where(pd.notna(frame), None).to_dict("records"))
    frame.insert(0, "ticker", ticker.upper())
    frame.insert(1, "endpoint", endpoint)
    frame["source_name"] = "Financial Modeling Prep"
    frame["source_label"] = (
        "estimate_consensus" if endpoint == "analyst-estimates" else "fact_provider_standardized"
    )
    frame["source_url"] = f"{BASE_URL}/{endpoint}"
    frame["fetched_at"] = fetched_at
    frame["snapshot_as_of"] = str(snapshot_as_of)[:10]
    frame["payload_digest"] = normalized_digest
    frame["raw_payload_digest"] = digest

    accepted = None
    for column in ("acceptedDate", "accepted_date", "filingDate", "filing_date"):
        if column in frame.columns:
            candidate = pd.to_datetime(frame[column], utc=True, errors="coerce")
            accepted = candidate if accepted is None else accepted.fillna(candidate)
    frame["accepted_at"] = accepted if accepted is not None else pd.NaT
    return frame


def fetch_ticker_bundle(
    client: FMPClient,
    ticker: str,
    snapshot_as_of: str | date,
    *,
    endpoints: tuple[str, ...] = FMP_ENDPOINTS,
) -> tuple[pd.DataFrame, list[dict]]:
    """Fetch, archive, normalize, and snapshot all requested FMP datasets."""
    frames: list[pd.DataFrame] = []
    records: list[dict] = []
    fetched_at = iso_utc()
    for endpoint in endpoints:
        part = snapshot_part_path("fmp", snapshot_as_of, ticker, endpoint)
        if part.exists():
            frame = pd.read_parquet(part)
            frames.append(frame)
            records.append({
                "ticker": ticker.upper(),
                "endpoint": endpoint,
                "rows": len(frame),
                "payload_digest": (
                    str(frame["payload_digest"].dropna().iloc[0])
                    if "payload_digest" in frame.columns and frame["payload_digest"].notna().any()
                    else None
                ),
                "raw_path": None,
                "snapshot_path": str(part),
                "fetched_at": (
                    str(frame["fetched_at"].dropna().iloc[0])
                    if "fetched_at" in frame.columns and frame["fetched_at"].notna().any()
                    else fetched_at
                ),
                "reused_frozen_part": True,
            })
            continue
        payload = client.fetch(endpoint, ticker, **_endpoint_params(endpoint))
        raw_path, digest = archive_json(payload, "fmp", endpoint, ticker)
        frame = normalize_rows(
            payload,
            ticker=ticker,
            endpoint=endpoint,
            snapshot_as_of=snapshot_as_of,
            digest=digest,
            fetched_at=fetched_at,
        )
        if not frame.empty:
            write_immutable_parquet(frame, part)
            frames.append(frame)
            row_count = len(frame)
        else:
            part = None
            row_count = 0
        records.append({
            "ticker": ticker.upper(),
            "endpoint": endpoint,
            "rows": row_count,
            "payload_digest": digest,
            "raw_path": str(raw_path),
            "snapshot_path": str(part) if part else None,
            "fetched_at": fetched_at,
            "reused_frozen_part": False,
        })
    combined = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    return combined, records
