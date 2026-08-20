"""Contract for the historical-only survivorship price surface.

The artifact is intentionally separate from ``master_prices.parquet``.  It is
used by research/backtest consumers only; the live scanner never loads it.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


SURVIVORSHIP_CONTRACT_VERSION = "major-index-removals-2020-v1"
SURVIVORSHIP_BASIS = (
    "dividend-adjusted OHLCV with explicit reviewed terminal-value marks"
)
SURVIVORSHIP_SCOPE_START = "2020-01-01"
SURVIVORSHIP_REQUIRED_TICKERS_V1 = frozenset({
    "ABMD", "ADS", "ANSS", "ATVI", "AVB", "CMA", "CTLT", "CTRA",
    "CTXS", "DAY", "DFS", "DISH", "DRE", "EA", "FRC", "GPS", "HBI",
    "HES", "HFC", "HOLX", "IPG", "JNPR", "JWN", "K", "KSU", "MRO",
    "NLSN", "PXD", "SEE", "SGEN", "SIVB", "SPLK", "TWTR", "WBA",
    "WRK", "XEC",
})
SURVIVORSHIP_REQUIRED_COLUMNS = (
    "ticker", "date", "Open", "High", "Low", "Close", "Volume",
)
SURVIVORSHIP_REQUIRED_TERMINAL_MARKS = {
    "SIVB": {
        "date": "2023-03-10",
        "price": 0.01,
        "reason": "FDIC closure / listed trading halt; conservative common-equity terminal value",
    },
    "FRC": {
        "date": "2023-05-01",
        "price": 0.01,
        "reason": "FDIC receivership; conservative common-equity terminal value",
    },
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: str | Path) -> dict:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - contract boundary must fail loud
        raise ValueError(f"survivorship manifest is unreadable: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("survivorship manifest must be a JSON object")
    return payload


def validate_survivorship_artifact(
    artifact_path: str | Path,
    manifest_path: str | Path,
    *,
    verify_digest: bool = True,
) -> dict:
    """Validate the artifact and return its manifest.

    The manifest must report complete coverage for the declared 2020+ major
    index-removal catalog.  Older/smaller-company survivorship is deliberately
    outside this v1 contract and must not be implied by callers.
    """
    artifact = Path(artifact_path)
    manifest_file = Path(manifest_path)
    if not artifact.is_file() or artifact.stat().st_size == 0:
        raise ValueError(f"survivorship artifact is missing or empty: {artifact}")
    if not manifest_file.is_file() or manifest_file.stat().st_size == 0:
        raise ValueError(f"survivorship manifest is missing or empty: {manifest_file}")

    manifest = load_manifest(manifest_file)
    if manifest.get("contract_version") != SURVIVORSHIP_CONTRACT_VERSION:
        raise ValueError(
            "unsupported survivorship contract version: "
            f"{manifest.get('contract_version')!r} "
            f"(expected {SURVIVORSHIP_CONTRACT_VERSION!r})"
        )
    if manifest.get("basis") != SURVIVORSHIP_BASIS:
        raise ValueError(
            f"unsupported survivorship basis: {manifest.get('basis')!r}"
        )
    if str(manifest.get("scope_start")) != SURVIVORSHIP_SCOPE_START:
        raise ValueError(
            f"unsupported survivorship scope_start: {manifest.get('scope_start')!r}"
        )
    unresolved = sorted({str(t).upper() for t in manifest.get("unresolved_required", [])})
    if unresolved:
        raise ValueError(
            "survivorship catalog is incomplete for its required scope: "
            f"{unresolved[:20]}"
        )

    frame = pd.read_parquet(artifact)
    missing = [column for column in SURVIVORSHIP_REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"survivorship artifact missing columns: {missing}")
    if frame.empty:
        raise ValueError("survivorship artifact contains no rows")

    tickers = frame["ticker"].astype("string").str.upper().str.strip()
    dates = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    if tickers.isna().any() or tickers.eq("").any():
        raise ValueError("survivorship artifact contains null/blank tickers")
    if dates.isna().any():
        raise ValueError("survivorship artifact contains invalid dates")
    if pd.DataFrame({"ticker": tickers, "date": dates}).duplicated().any():
        raise ValueError("survivorship artifact contains duplicate ticker/date rows")

    values = frame[["Open", "High", "Low", "Close", "Volume"]].apply(
        pd.to_numeric, errors="coerce"
    )
    numeric = values.to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("survivorship artifact contains non-finite OHLCV")
    if (values[["Open", "High", "Low", "Close"]] <= 0).any().any():
        raise ValueError("survivorship artifact contains non-positive prices")
    if (values["Volume"] < 0).any():
        raise ValueError("survivorship artifact contains negative volume")
    if (values["High"] + 1e-8 < values[["Open", "Low", "Close"]].max(axis=1)).any():
        raise ValueError("survivorship artifact contains invalid highs")
    if (values["Low"] - 1e-8 > values[["Open", "High", "Close"]].min(axis=1)).any():
        raise ValueError("survivorship artifact contains invalid lows")

    artifact_tickers = sorted(set(tickers.tolist()))
    declared_artifact = sorted(
        {str(t).upper().strip() for t in manifest.get("artifact_tickers", [])}
    )
    if artifact_tickers != declared_artifact:
        raise ValueError("survivorship artifact ticker set does not match its manifest")
    if int(manifest.get("artifact_rows", -1)) != len(frame):
        raise ValueError("survivorship artifact row count does not match its manifest")

    required = {str(t).upper().strip() for t in manifest.get("required_tickers", [])}
    if required != set(SURVIVORSHIP_REQUIRED_TICKERS_V1):
        missing = sorted(set(SURVIVORSHIP_REQUIRED_TICKERS_V1) - required)
        unexpected = sorted(required - set(SURVIVORSHIP_REQUIRED_TICKERS_V1))
        raise ValueError(
            "survivorship manifest does not match the reviewed v1 catalog "
            f"(missing={missing[:20]}, unexpected={unexpected[:20]})"
        )
    terminal_marks = manifest.get("terminal_marks", [])
    marked_tickers = [str(mark.get("ticker") or "").upper().strip() for mark in terminal_marks]
    if len(marked_tickers) != len(set(marked_tickers)):
        raise ValueError("survivorship manifest contains duplicate terminal marks")
    for ticker, expected in SURVIVORSHIP_REQUIRED_TERMINAL_MARKS.items():
        if ticker not in required:
            continue
        matches = [mark for mark in terminal_marks if str(mark.get("ticker") or "").upper().strip() == ticker]
        if len(matches) != 1:
            raise ValueError(f"required terminal mark is missing for {ticker}")
        observed_date = pd.to_datetime(matches[0].get("date"), errors="coerce")
        observed_price = pd.to_numeric(matches[0].get("price"), errors="coerce")
        if (
            pd.isna(observed_date)
            or pd.Timestamp(observed_date).normalize() != pd.Timestamp(expected["date"])
            or not np.isfinite(observed_price)
            or abs(float(observed_price) - float(expected["price"])) > 1e-8
        ):
            raise ValueError(f"required terminal mark contract mismatch for {ticker}")

    for mark in terminal_marks:
        ticker = str(mark.get("ticker") or "").upper().strip()
        date = pd.to_datetime(mark.get("date"), errors="coerce")
        price = pd.to_numeric(mark.get("price"), errors="coerce")
        if not ticker or pd.isna(date) or not np.isfinite(price) or float(price) <= 0:
            raise ValueError("survivorship manifest contains an invalid terminal mark")
        row = frame.loc[(tickers == ticker) & (dates == pd.Timestamp(date).normalize())]
        if len(row) != 1:
            raise ValueError(f"terminal mark is missing or duplicated for {ticker}")
        observed = row.iloc[0]
        if any(abs(float(observed[column]) - float(price)) > 1e-8 for column in ("Open", "High", "Low", "Close")):
            raise ValueError(f"terminal mark price mismatch for {ticker}")
        if float(observed["Volume"]) != 0.0:
            raise ValueError(f"terminal mark volume must be zero for {ticker}")

    existing = {
        str(t).upper().strip()
        for t in manifest.get("covered_by_primary_sources", [])
    }
    if required != set(artifact_tickers) | existing:
        raise ValueError(
            "survivorship required coverage is not exactly artifact + primary-source coverage"
        )
    if verify_digest:
        declared_sha = str(manifest.get("artifact_sha256") or "")
        actual_sha = sha256_file(artifact)
        if not declared_sha or declared_sha != actual_sha:
            raise ValueError("survivorship artifact SHA-256 does not match its manifest")
    return manifest
