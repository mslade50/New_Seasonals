"""Point-in-time-aware U.S. economic release history helpers.

The generated cache lives at ``data/macro_release_history.parquet`` and is
built by ``scripts/build_macro_releases.py``.  FMP supplies historical
calendar rows with actual, estimate, and previous values.  Rows backfilled
after their release are explicitly labelled ``vendor_historical_snapshot``;
only rows first observed on their release date qualify as ``live_capture``.

The module deliberately calls positive/negative differences ``above`` and
``below`` consensus.  "Beat" is ambiguous across indicators: a higher NFP is
usually stronger growth, while a higher CPI is usually an inflationary miss.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
PARQUET_PATH = ROOT / "data" / "macro_release_history.parquet"
SOURCE = "fmp:economic-calendar"


# FMP has used duplicate names for the same CPI release.  These aliases give
# downstream studies one stable event id while the priority map chooses a
# single provider row when both synonyms are present.
EVENT_ALIASES = {
    "Non Farm Payrolls": "nfp",
    "Unemployment Rate": "unemployment_rate",
    "Average Hourly Earnings MoM": "average_hourly_earnings_mom",
    "Average Hourly Earnings YoY": "average_hourly_earnings_yoy",
    "Average Weekly Hours": "average_weekly_hours",
    "Nonfarm Payrolls Private": "private_payrolls",
    "Government Payrolls": "government_payrolls",
    "Manufacturing Payrolls": "manufacturing_payrolls",
    "CPI MoM": "cpi_mom",
    "Inflation Rate MoM": "cpi_mom",
    "CPI YoY": "cpi_yoy",
    "Inflation Rate YoY": "cpi_yoy",
    "Core CPI MoM": "core_cpi_mom",
    "Core Inflation Rate MoM": "core_cpi_mom",
    "Core CPI YoY": "core_cpi_yoy",
    "Core Inflation Rate YoY": "core_cpi_yoy",
    "Producer Price Index MoM": "ppi_mom",
    "Producer Price Index YoY": "ppi_yoy",
    "Core PPI MoM": "core_ppi_mom",
    "Core PPI YoY": "core_ppi_yoy",
    "PCE Price Index MoM": "pce_mom",
    "PCE Price Index YoY": "pce_yoy",
    "Core PCE Price Index MoM": "core_pce_mom",
    "Core PCE Price Index YoY": "core_pce_yoy",
    "Retail Sales MoM": "retail_sales_mom",
    "Retail Sales Ex Autos MoM": "retail_sales_ex_autos_mom",
    "Initial Jobless Claims": "initial_jobless_claims",
    "Continuing Jobless Claims": "continuing_jobless_claims",
    "JOLTs Job Openings": "jolts_job_openings",
    "ADP Employment Change": "adp_employment_change",
    "ISM Manufacturing PMI": "ism_manufacturing_pmi",
    "ISM Services PMI": "ism_services_pmi",
    "GDP Growth Rate QoQ": "gdp_qoq",
    "GDP Growth Rate QoQ 2nd Est": "gdp_qoq_second_estimate",
    "GDP Growth Rate QoQ 3rd Est": "gdp_qoq_third_estimate",
}

PROVIDER_NAME_PRIORITY = {
    # FMP's CPI-labelled synonyms are timestamped six hours early through
    # much of 2021-23.  The Inflation Rate rows carry the correct 08:30 ET
    # release time and the same actual/estimate values.
    "CPI MoM": 10,
    "Inflation Rate MoM": 20,
    "CPI YoY": 10,
    "Inflation Rate YoY": 20,
    "Core CPI MoM": 10,
    "Core Inflation Rate MoM": 20,
    "Core CPI YoY": 10,
    "Core Inflation Rate YoY": 20,
}

NUMERIC_COLUMNS = [
    "actual", "consensus", "previous", "change", "change_percentage",
    "surprise",
]

EXPECTED_COLUMNS = [
    "release_key", "release_ts_utc", "release_date", "time_et", "country",
    "currency", "event_id", "event_name", "provider_event",
    "reference_period", "actual", "consensus", "previous", "unit",
    "impact", "change", "change_percentage", "surprise",
    "surprise_label", "source", "vintage_quality", "first_seen_at_utc",
    "last_seen_at_utc",
]

BLS_CALENDAR_FAMILY = {
    "nfp": "nfp",
    "unemployment_rate": "nfp",
    "average_hourly_earnings_mom": "nfp",
    "average_hourly_earnings_yoy": "nfp",
    "average_weekly_hours": "nfp",
    "private_payrolls": "nfp",
    "government_payrolls": "nfp",
    "manufacturing_payrolls": "nfp",
    "cpi_mom": "cpi",
    "cpi_yoy": "cpi",
    "core_cpi_mom": "cpi",
    "core_cpi_yoy": "cpi",
    "ppi_mom": "ppi",
    "ppi_yoy": "ppi",
    "core_ppi_mom": "ppi",
    "core_ppi_yoy": "ppi",
}


def split_event_name(provider_event: str) -> tuple[str, str]:
    """Return FMP's event base name and trailing parenthetical reference."""
    text = str(provider_event or "").strip()
    match = re.match(r"^(.*?)\s*\(([^()]*)\)\s*$", text)
    if not match:
        return text, ""
    return match.group(1).strip(), match.group(2).strip()


def slugify_event(event_name: str) -> str:
    if event_name in EVENT_ALIASES:
        return EVENT_ALIASES[event_name]
    slug = re.sub(r"[^a-z0-9]+", "_", event_name.lower()).strip("_")
    return slug or "unknown_event"


def _release_key(row: pd.Series) -> str:
    raw = "|".join([
        str(row["country"]),
        str(row["event_id"]),
        str(row["reference_period"]),
        pd.Timestamp(row["release_ts_utc"]).isoformat(),
    ])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


def normalize_fmp_rows(
    rows: list[dict],
    fetched_at: pd.Timestamp | str | None = None,
    country: str = "US",
) -> pd.DataFrame:
    """Normalize raw FMP calendar rows into the stable release schema."""
    fetched = pd.Timestamp.now(tz="UTC") if fetched_at is None else pd.Timestamp(fetched_at)
    if fetched.tzinfo is None:
        fetched = fetched.tz_localize("UTC")
    else:
        fetched = fetched.tz_convert("UTC")

    records = []
    wanted = country.upper()
    for raw in rows:
        if str(raw.get("country", "")).upper() != wanted:
            continue
        provider_event = str(raw.get("event", "")).strip()
        if not provider_event:
            continue
        event_name, reference_period = split_event_name(provider_event)
        ts = pd.to_datetime(raw.get("date"), errors="coerce", utc=True)
        if pd.isna(ts):
            continue
        et = ts.tz_convert("America/New_York")
        records.append({
            "release_ts_utc": ts,
            "release_date": et.tz_localize(None).normalize(),
            "time_et": et.strftime("%H:%M"),
            "country": wanted,
            "currency": raw.get("currency"),
            "event_id": slugify_event(event_name),
            "event_name": event_name,
            "provider_event": provider_event,
            "reference_period": reference_period,
            "actual": raw.get("actual"),
            "consensus": raw.get("estimate"),
            "previous": raw.get("previous"),
            "unit": raw.get("unit"),
            "impact": raw.get("impact"),
            "change": raw.get("change"),
            "change_percentage": raw.get("changePercentage"),
            "source": SOURCE,
            "first_seen_at_utc": fetched,
            "last_seen_at_utc": fetched,
            "_provider_priority": PROVIDER_NAME_PRIORITY.get(event_name, 0),
        })

    if not records:
        return pd.DataFrame(columns=EXPECTED_COLUMNS)

    df = pd.DataFrame(records)
    for col in ["actual", "consensus", "previous", "change", "change_percentage"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["surprise"] = df["actual"] - df["consensus"]
    df["surprise_label"] = np.select(
        [df["surprise"] > 0, df["surprise"] < 0, df["surprise"].eq(0)],
        ["above", "below", "inline"],
        default="unknown",
    )
    fetched_et_date = fetched.tz_convert("America/New_York").tz_localize(None).normalize()
    df["vintage_quality"] = np.select(
        [df["release_date"] < fetched_et_date, df["release_date"] == fetched_et_date],
        ["vendor_historical_snapshot", "live_capture"],
        default="scheduled_snapshot",
    )
    df["release_key"] = df.apply(_release_key, axis=1)

    # Collapse provider synonyms such as CPI MoM / Inflation Rate MoM.  Keep
    # the explicitly CPI-labelled row where both exist; before 2020 the
    # Inflation Rate name remains because it is the only row available.
    # Use the ET release date rather than FMP's timestamp in this key.  One of
    # the duplicate CPI aliases is six hours early in 2021-23, so timestamp-
    # level deduplication would preserve both rows.
    dedupe_key = ["country", "event_id", "reference_period", "release_date"]
    df = (df.sort_values(dedupe_key + ["_provider_priority"])
            .drop_duplicates(dedupe_key, keep="last")
            .drop(columns="_provider_priority")
            .sort_values(["release_ts_utc", "event_id"])
            .reset_index(drop=True))
    return df[EXPECTED_COLUMNS]


def align_bls_release_calendar(df: pd.DataFrame, calendar_df: pd.DataFrame) -> pd.DataFrame:
    """Align BLS-family rows to the repo's verified release date/time calendar.

    FMP has several historical timezone errors and also labels the annual NFP
    benchmark revision as a regular payroll print.  Rows matching the local
    BLS release calendar inherit its ET time; off-calendar rows get a distinct
    ``*_off_calendar`` event id so ordinary event studies cannot ingest them.
    """
    if df.empty:
        return df.copy()
    required = {"date", "event", "time_et"}
    missing = required - set(calendar_df.columns)
    if missing:
        raise ValueError(f"macro calendar missing columns: {sorted(missing)}")

    out = df.copy()
    cal = calendar_df.copy()
    cal["date"] = pd.to_datetime(cal["date"]).dt.normalize()
    lookup = {
        (str(row.event), pd.Timestamp(row.date).normalize()): str(row.time_et)
        for row in cal.itertuples()
        if str(row.event) in {"nfp", "cpi", "ppi"} and str(row.time_et)
    }

    for idx, row in out.iterrows():
        family = BLS_CALENDAR_FAMILY.get(str(row["event_id"]))
        if family is None:
            continue
        release_date = pd.Timestamp(row["release_date"]).normalize()
        time_et = lookup.get((family, release_date))
        if not time_et:
            out.at[idx, "event_id"] = f"{row['event_id']}_off_calendar"
            continue
        local = pd.Timestamp(f"{release_date:%Y-%m-%d} {time_et}").tz_localize(
            "America/New_York"
        )
        out.at[idx, "time_et"] = time_et
        out.at[idx, "release_ts_utc"] = local.tz_convert("UTC")

    out["release_ts_utc"] = pd.to_datetime(out["release_ts_utc"], utc=True)
    out["release_key"] = out.apply(_release_key, axis=1)
    out = (out.drop_duplicates("release_key", keep="last")
              .sort_values(["release_ts_utc", "event_id"])
              .reset_index(drop=True))
    return out[EXPECTED_COLUMNS]


def merge_release_history(existing: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
    """Merge a refresh without rewriting already captured released values.

    Once an existing row has a non-null actual, its actual/consensus/previous
    triplet is frozen.  This prevents a provider-side historical rewrite from
    silently changing an event-study classification.  Missing fields can still
    be filled, and unreleased rows can update normally.
    """
    if existing is None or existing.empty:
        return fresh.sort_values(["release_ts_utc", "event_id"]).reset_index(drop=True)
    if fresh is None or fresh.empty:
        return existing.sort_values(["release_ts_utc", "event_id"]).reset_index(drop=True)

    old = existing.copy().set_index("release_key", drop=False)
    new = fresh.copy().set_index("release_key", drop=False)
    shared = old.index.intersection(new.index)
    frozen_cols = [
        "actual", "consensus", "previous", "change", "change_percentage",
        "surprise", "surprise_label",
    ]

    for key in shared:
        old_row = old.loc[key]
        new_row = new.loc[key]
        if isinstance(old_row, pd.DataFrame) or isinstance(new_row, pd.DataFrame):
            raise ValueError(f"duplicate release_key during merge: {key}")
        merged = new_row.copy()
        merged["first_seen_at_utc"] = old_row["first_seen_at_utc"]
        if pd.notna(old_row.get("actual")):
            for col in frozen_cols:
                merged[col] = old_row.get(col)
            merged["vintage_quality"] = old_row.get("vintage_quality")
        else:
            for col in frozen_cols:
                if pd.isna(merged.get(col)) and pd.notna(old_row.get(col)):
                    merged[col] = old_row.get(col)
        new.loc[key] = merged

    only_old = old.loc[old.index.difference(new.index)]
    out = pd.concat([only_old, new], ignore_index=True)
    out = (out.drop_duplicates("release_key", keep="last")
              .sort_values(["release_ts_utc", "event_id"])
              .reset_index(drop=True))
    return out[EXPECTED_COLUMNS]


def load_macro_releases(
    events: list[str] | None = None,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    require_surprise: bool = False,
    path: str | Path = PARQUET_PATH,
) -> pd.DataFrame:
    """Load the normalized cache with optional event/date filters."""
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(
            f"macro release history not found at {target}; run "
            "python scripts/build_macro_releases.py --full"
        )
    df = pd.read_parquet(target)
    df["release_ts_utc"] = pd.to_datetime(df["release_ts_utc"], utc=True)
    df["release_date"] = pd.to_datetime(df["release_date"]).dt.normalize()
    if events is not None:
        df = df[df["event_id"].isin(events)]
    if start is not None:
        df = df[df["release_date"] >= pd.Timestamp(start).normalize()]
    if end is not None:
        df = df[df["release_date"] <= pd.Timestamp(end).normalize()]
    if require_surprise:
        df = df[df["actual"].notna() & df["consensus"].notna()]
    return df.reset_index(drop=True)
