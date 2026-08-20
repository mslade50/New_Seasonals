"""Build the historical-only price surface for major 2020+ index removals.

This closes the repo's concrete survivorship gap without putting dead symbols
into the live scanner.  Discovery uses FMP's delisted-company catalog crossed
with historical S&P 500, Nasdaq-100, and Dow constituent removals.  Prices use
FMP's dividend-adjusted daily OHLCV endpoint.

The output is intentionally separate from ``master_prices.parquet`` and has no
implicit upload path.  Production publication is performed only by the
reviewed GitHub Actions repair workflow after validation.
"""
from __future__ import annotations

import argparse
import datetime as dt
import difflib
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from survivorship_contract import (  # noqa: E402
    SURVIVORSHIP_BASIS,
    SURVIVORSHIP_CONTRACT_VERSION,
    SURVIVORSHIP_REQUIRED_TICKERS_V1,
    SURVIVORSHIP_REQUIRED_TERMINAL_MARKS,
    SURVIVORSHIP_SCOPE_START,
    sha256_file,
    validate_survivorship_artifact,
)


FMP_BASE = "https://financialmodelingprep.com/stable"
DELISTED_ENDPOINT = "delisted-companies"
PRICE_ENDPOINT = "historical-price-eod/dividend-adjusted"
INDEX_ENDPOINTS = {
    "SP500": "historical-sp500-constituent",
    "NASDAQ100": "historical-nasdaq-constituent",
    "DOW30": "historical-dowjones-constituent",
}
US_EXCHANGES = {"NASDAQ", "NYSE", "AMEX"}
REQUEST_TIMEOUT = 45
MAX_RETRIES = 4
PAGE_SIZE = 100
MAX_DELISTED_PAGES = 250
MIN_PRICE_ROWS = 252
IDENTITY_THRESHOLD = 0.65

DEFAULT_OUTPUT = ROOT / "data" / "survivorship_prices.parquet"
DEFAULT_MANIFEST = ROOT / "data" / "survivorship_prices.meta.json"
DEFAULT_MASTER = ROOT / "data" / "master_prices.parquet"
DEFAULT_OVERFLOW = ROOT / "data" / "overflow_prices.parquet"

# Same company, renamed around the removal/ticker transition.  These are
# explicit because fuzzy matching must otherwise reject rather than guess.
IDENTITY_ALIASES = {
    "ADS": ("alliance data systems", "bread financial"),
    "HFC": ("hollyfrontier", "hf sinclair"),
}

# FMP's listed history stops at the halt/delisting and therefore omits the
# economic wipeout that makes these two failures essential to a survivorship
# repair.  A one-cent liquidation mark is an explicit conservative research
# assumption, not a claimed exchange print.  It prevents the engine from
# valuing a held SIVB position forever at its pre-failure $106 close.
TERMINAL_VALUE_POLICY = SURVIVORSHIP_REQUIRED_TERMINAL_MARKS

_NAME_STOPWORDS = {
    "class", "co", "common", "company", "corp", "corporation", "group",
    "holding", "holdings", "inc", "incorporated", "limited", "ltd", "plc",
    "stock", "the", "delisted",
}


def load_api_key() -> str:
    key = os.environ.get("FMP_API_KEY")
    if key:
        return key
    env_path = ROOT / ".env"
    if env_path.is_file():
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            name, sep, value = raw.partition("=")
            if sep and name.strip() == "FMP_API_KEY":
                key = value.strip().strip('"').strip("'")
                break
    if not key:
        raise RuntimeError("FMP_API_KEY is required")
    return key


def _request_list(
    session: requests.Session,
    endpoint: str,
    api_key: str,
    **params,
) -> list:
    params = {**params, "apikey": api_key}
    for attempt in range(MAX_RETRIES):
        try:
            response = session.get(
                f"{FMP_BASE}/{endpoint}", params=params, timeout=REQUEST_TIMEOUT
            )
            if response.status_code == 200:
                payload = response.json()
                if not isinstance(payload, list):
                    raise RuntimeError(f"{endpoint}: unexpected response shape")
                return payload
            if response.status_code == 429 or response.status_code >= 500:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
                    continue
            raise RuntimeError(f"{endpoint}: HTTP {response.status_code}")
        except requests.RequestException as exc:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"{endpoint}: request failed: {exc}") from exc
    raise RuntimeError(f"{endpoint}: retry budget exhausted")


def fetch_delisted_catalog(session: requests.Session, api_key: str) -> list:
    rows: list[dict] = []
    for page in range(MAX_DELISTED_PAGES):
        batch = _request_list(
            session, DELISTED_ENDPOINT, api_key, page=page, limit=PAGE_SIZE
        )
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < PAGE_SIZE:
            break
        time.sleep(0.04)
    if not rows:
        raise RuntimeError("FMP delisted-company catalog is empty")
    return rows


def fetch_index_changes(session: requests.Session, api_key: str) -> list:
    rows: list[dict] = []
    for index_name, endpoint in INDEX_ENDPOINTS.items():
        payload = _request_list(session, endpoint, api_key)
        rows.extend({**row, "index": index_name} for row in payload)
    if not rows:
        raise RuntimeError("FMP historical-index removal catalog is empty")
    return rows


def _normal_name(value: object) -> str:
    tokens = re.sub(r"[^a-z0-9 ]", " ", str(value or "").lower()).split()
    return " ".join(token for token in tokens if token not in _NAME_STOPWORDS)


def identity_score(index_name: object, delisted_name: object) -> float:
    left, right = _normal_name(index_name), _normal_name(delisted_name)
    if not left or not right:
        return 0.0
    return difflib.SequenceMatcher(None, left, right).ratio()


def _explicit_identity_alias(ticker: str, index_name: object, delisted_name: object) -> bool:
    alias = IDENTITY_ALIASES.get(ticker)
    if not alias:
        return False
    left, right = _normal_name(index_name), _normal_name(delisted_name)
    return alias[0] in left and alias[1] in right


def build_catalog(
    delisted_rows: list[dict],
    index_rows: list[dict],
    *,
    as_of: str | dt.date | pd.Timestamp,
    scope_start: str = SURVIVORSHIP_SCOPE_START,
) -> tuple[list[dict], list[dict]]:
    """Return identity-matched catalog rows and rejected same-symbol matches."""
    as_of_ts = pd.Timestamp(as_of).normalize()
    start_ts = pd.Timestamp(scope_start).normalize()
    removals: dict[str, list[dict]] = {}
    for row in index_rows:
        ticker = str(row.get("removedTicker") or "").upper().strip().replace(".", "-")
        removal_date = pd.to_datetime(row.get("date"), errors="coerce")
        if not ticker or pd.isna(removal_date):
            continue
        removal_date = pd.Timestamp(removal_date).normalize()
        if not (start_ts <= removal_date <= as_of_ts):
            continue
        removals.setdefault(ticker, []).append({**row, "_removal_date": removal_date})

    delisted_by_ticker: dict[str, list[dict]] = {}
    for row in delisted_rows:
        if str(row.get("exchange") or "").upper().strip() not in US_EXCHANGES:
            continue
        ticker = str(row.get("symbol") or "").upper().strip().replace(".", "-")
        delisted_date = pd.to_datetime(row.get("delistedDate"), errors="coerce")
        if not ticker or pd.isna(delisted_date):
            continue
        delisted_date = pd.Timestamp(delisted_date).normalize()
        if delisted_date <= as_of_ts:
            delisted_by_ticker.setdefault(ticker, []).append(
                {**row, "_delisted_date": delisted_date}
            )

    catalog: list[dict] = []
    rejected: list[dict] = []
    for ticker, ticker_removals in sorted(removals.items()):
        candidates = delisted_by_ticker.get(ticker, [])
        best = None
        for removal in ticker_removals:
            removal_date = removal["_removal_date"]
            for company in candidates:
                # Index providers usually remove a company on/just after its
                # final trade.  A ten-day lead accommodates announcement and
                # constituent-effective-date conventions without accepting a
                # prior symbol incarnation.
                if company["_delisted_date"] < removal_date - pd.Timedelta(days=10):
                    continue
                score = identity_score(
                    removal.get("removedSecurity"), company.get("companyName")
                )
                alias = _explicit_identity_alias(
                    ticker, removal.get("removedSecurity"), company.get("companyName")
                )
                rank = (1 if alias else 0, score, -abs((company["_delisted_date"] - removal_date).days))
                if best is None or rank > best[0]:
                    best = (rank, removal, company, score, alias)
        if best is None or (best[3] < IDENTITY_THRESHOLD and not best[4]):
            if candidates:
                rejected.append({
                    "ticker": ticker,
                    "removed_names": sorted({str(x.get("removedSecurity") or "") for x in ticker_removals}),
                    "delisted_names": sorted({str(x.get("companyName") or "") for x in candidates}),
                    "reason": "same ticker but company identity did not match",
                })
            continue

        _, removal, company, score, alias = best
        all_indices = sorted({str(x.get("index")) for x in ticker_removals})
        all_removal_dates = sorted(x["_removal_date"] for x in ticker_removals)
        catalog.append({
            "ticker": ticker,
            "company_name": company.get("companyName"),
            "removed_security": removal.get("removedSecurity"),
            "exchange": company.get("exchange"),
            "ipo_date": company.get("ipoDate"),
            "delisted_date": company["_delisted_date"].date().isoformat(),
            "first_index_removal": all_removal_dates[0].date().isoformat(),
            "indices": all_indices,
            "identity_score": round(float(score), 6),
            "identity_alias": bool(alias),
        })
    return catalog, rejected


def _source_stats(paths: list[Path]) -> dict[str, dict]:
    stats: dict[str, dict] = {}
    for path in paths:
        if not path.is_file() or path.stat().st_size == 0:
            continue
        frame = pd.read_parquet(path, columns=["ticker", "date"])
        frame["ticker"] = frame["ticker"].astype(str).str.upper().str.strip().str.replace(".", "-", regex=False)
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
        for ticker, group in frame.dropna(subset=["date"]).groupby("ticker"):
            prior = stats.get(ticker)
            record = {
                "rows": int(len(group)),
                "first_date": group["date"].min(),
                "last_date": group["date"].max(),
                "source": str(path),
            }
            if prior is None or record["rows"] > prior["rows"]:
                stats[ticker] = record
    return stats


def _primary_source_covers(stats: dict | None, delisted_date: str) -> bool:
    if not stats or int(stats["rows"]) < MIN_PRICE_ROWS:
        return False
    end = pd.Timestamp(delisted_date)
    return pd.Timestamp(stats["last_date"]) >= end - pd.Timedelta(days=15)


def fetch_price_history(
    session: requests.Session,
    api_key: str,
    ticker: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    rows: list[dict] = []
    cursor = pd.Timestamp(start)
    finish = pd.Timestamp(end)
    while cursor <= finish:
        chunk_end = min(cursor + pd.DateOffset(years=7) - pd.Timedelta(days=1), finish)
        rows.extend(_request_list(
            session,
            PRICE_ENDPOINT,
            api_key,
            symbol=ticker,
            **{"from": cursor.date().isoformat(), "to": chunk_end.date().isoformat()},
        ))
        cursor = chunk_end + pd.Timedelta(days=1)
        time.sleep(0.04)
    if not rows:
        return pd.DataFrame(columns=["ticker", "date", "Open", "High", "Low", "Close", "Volume"])
    frame = pd.DataFrame(rows).rename(columns={
        "adjOpen": "Open",
        "adjHigh": "High",
        "adjLow": "Low",
        "adjClose": "Close",
        "volume": "Volume",
    })
    needed = ["date", "Open", "High", "Low", "Close", "Volume"]
    missing = [column for column in needed if column not in frame.columns]
    if missing:
        raise ValueError(f"{ticker}: FMP price response missing {missing}")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["ticker"] = ticker
    return (
        frame[["ticker", *needed]]
        .drop_duplicates(["ticker", "date"], keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )


def apply_terminal_value_policy(frame: pd.DataFrame, ticker: str) -> tuple[pd.DataFrame, dict | None]:
    policy = TERMINAL_VALUE_POLICY.get(ticker)
    if not policy:
        return frame, None
    terminal_date = pd.Timestamp(policy["date"]).normalize()
    terminal_price = float(policy["price"])
    out = frame.loc[frame["date"] != terminal_date].copy()
    mark = pd.DataFrame([{
        "ticker": ticker,
        "date": terminal_date,
        "Open": terminal_price,
        "High": terminal_price,
        "Low": terminal_price,
        "Close": terminal_price,
        "Volume": 0.0,
    }])
    out = pd.concat([out, mark], ignore_index=True).sort_values("date").reset_index(drop=True)
    return out, {"ticker": ticker, **policy}


def validate_price_frame(
    frame: pd.DataFrame,
    ticker: str,
    delisted_date: str,
    terminal_mark: dict | None = None,
) -> dict:
    if len(frame) < MIN_PRICE_ROWS:
        raise ValueError(f"{ticker}: only {len(frame)} dividend-adjusted rows")
    values = frame[["Open", "High", "Low", "Close", "Volume"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError(f"{ticker}: non-finite OHLCV")
    if (values[["Open", "High", "Low", "Close"]] <= 0).any().any():
        raise ValueError(f"{ticker}: non-positive adjusted price")
    if (values["Volume"] < 0).any():
        raise ValueError(f"{ticker}: negative volume")
    if (values["High"] + 1e-8 < values[["Open", "Low", "Close"]].max(axis=1)).any():
        raise ValueError(f"{ticker}: high below another OHLC field")
    if (values["Low"] - 1e-8 > values[["Open", "High", "Close"]].min(axis=1)).any():
        raise ValueError(f"{ticker}: low above another OHLC field")
    if frame["date"].isna().any():
        raise ValueError(f"{ticker}: invalid dates")
    allowed_end = pd.Timestamp(
        terminal_mark["date"] if terminal_mark else delisted_date
    )
    if frame["date"].max() > allowed_end:
        raise ValueError(f"{ticker}: price history extends beyond declared delisting")
    return {
        "rows": int(len(frame)),
        "first_date": frame["date"].min().date().isoformat(),
        "last_date": frame["date"].max().date().isoformat(),
    }


def build(
    *,
    output: Path,
    manifest_path: Path,
    master_source: Path,
    overflow_source: Path,
    as_of: str,
    api_key: str,
    delisted_rows: list[dict] | None = None,
    index_rows: list[dict] | None = None,
) -> dict:
    session = requests.Session()
    delisted_rows = delisted_rows if delisted_rows is not None else fetch_delisted_catalog(session, api_key)
    index_rows = index_rows if index_rows is not None else fetch_index_changes(session, api_key)
    catalog, identity_rejections = build_catalog(delisted_rows, index_rows, as_of=as_of)
    if not catalog:
        raise RuntimeError("identity-matched 2020+ major-removal catalog is empty")
    discovered = {item["ticker"] for item in catalog}
    expected = set(SURVIVORSHIP_REQUIRED_TICKERS_V1)
    if discovered != expected:
        raise RuntimeError(
            "discovered catalog does not match reviewed v1 scope "
            f"(missing={sorted(expected - discovered)}, "
            f"unexpected={sorted(discovered - expected)}); review and version "
            "the catalog instead of publishing a silent scope change"
        )

    source_paths = [master_source, overflow_source]
    source_stats = _source_stats(source_paths)
    frames: list[pd.DataFrame] = []
    covered_by_primary: list[str] = []
    price_audit: list[dict] = []
    unresolved: dict[str, str] = {}
    terminal_marks: list[dict] = []

    for number, item in enumerate(catalog, start=1):
        ticker = item["ticker"]
        if (
            ticker not in TERMINAL_VALUE_POLICY
            and _primary_source_covers(source_stats.get(ticker), item["delisted_date"])
        ):
            covered_by_primary.append(ticker)
            print(f"[{number:02d}/{len(catalog)}] {ticker}: covered by primary R2 prices")
            continue
        ipo = pd.to_datetime(item.get("ipo_date"), errors="coerce")
        start = pd.Timestamp("2000-01-01") if pd.isna(ipo) else max(
            pd.Timestamp("2000-01-01"), pd.Timestamp(ipo) - pd.Timedelta(days=5)
        )
        try:
            frame = fetch_price_history(
                session, api_key, ticker, start.date().isoformat(), item["delisted_date"]
            )
            frame, terminal_mark = apply_terminal_value_policy(frame, ticker)
            audit = validate_price_frame(
                frame, ticker, item["delisted_date"], terminal_mark
            )
            frames.append(frame)
            price_audit.append({"ticker": ticker, **audit})
            if terminal_mark:
                terminal_marks.append(terminal_mark)
            print(f"[{number:02d}/{len(catalog)}] {ticker}: fetched {len(frame):,} rows")
        except Exception as exc:  # noqa: BLE001 - aggregate complete failure report
            unresolved[ticker] = str(exc)
            print(f"[{number:02d}/{len(catalog)}] ERROR {ticker}: {exc}")

    if unresolved:
        raise RuntimeError(
            "required 2020+ survivorship coverage is incomplete: "
            + json.dumps(unresolved, sort_keys=True)
        )

    artifact = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["ticker", "date", "Open", "High", "Low", "Close", "Volume"]
    )
    artifact = artifact.sort_values(["ticker", "date"]).reset_index(drop=True)
    for column in ["Open", "High", "Low", "Close", "Volume"]:
        artifact[column] = pd.to_numeric(artifact[column], errors="raise").astype("float32")
    output.parent.mkdir(parents=True, exist_ok=True)
    artifact.to_parquet(output, index=False)

    required_tickers = sorted(item["ticker"] for item in catalog)
    payload = {
        "contract_version": SURVIVORSHIP_CONTRACT_VERSION,
        "basis": SURVIVORSHIP_BASIS,
        "scope_start": SURVIVORSHIP_SCOPE_START,
        "scope": (
            "Identity-matched delisted/renamed constituents removed from the "
            "S&P 500, Nasdaq-100, or Dow since 2020; not the full US delisted universe"
        ),
        "as_of": str(pd.Timestamp(as_of).date()),
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_endpoints": {
            "delisted": f"{FMP_BASE}/{DELISTED_ENDPOINT}",
            "prices": f"{FMP_BASE}/{PRICE_ENDPOINT}",
            "indices": [f"{FMP_BASE}/{endpoint}" for endpoint in INDEX_ENDPOINTS.values()],
        },
        "required_tickers": required_tickers,
        "covered_by_primary_sources": sorted(covered_by_primary),
        "artifact_tickers": sorted(artifact["ticker"].unique().tolist()),
        "artifact_rows": int(len(artifact)),
        "artifact_sha256": sha256_file(output),
        "unresolved_required": [],
        "catalog": catalog,
        "identity_rejections": identity_rejections,
        "price_audit": sorted(price_audit, key=lambda row: row["ticker"]),
        "terminal_marks": sorted(terminal_marks, key=lambda row: row["ticker"]),
        "primary_source_sha256": {
            str(path): sha256_file(path)
            for path in source_paths
            if path.is_file() and path.stat().st_size > 0
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    validate_survivorship_artifact(output, manifest_path)
    print(
        f"Validated {len(required_tickers)} required tickers: "
        f"{len(covered_by_primary)} primary + {artifact['ticker'].nunique()} historical-only"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--master-source", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--overflow-source", type=Path, default=DEFAULT_OVERFLOW)
    parser.add_argument("--as-of", default=dt.date.today().isoformat())
    args = parser.parse_args()
    build(
        output=args.output,
        manifest_path=args.manifest,
        master_source=args.master_source,
        overflow_source=args.overflow_source,
        as_of=args.as_of,
        api_key=load_api_key(),
    )


if __name__ == "__main__":
    main()
