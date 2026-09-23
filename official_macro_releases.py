"""Official economic observations, with release time and data vintage kept separate.

These adapters never synthesize a consensus forecast. The BLS API is a current
vintage, not an archive of the originally announced numbers. Parsers reject
ambiguous dates/series instead of assigning a plausible release date.
"""
from __future__ import annotations

import hashlib
import math
import re
import xml.etree.ElementTree as ET
from io import StringIO

import pandas as pd
from bs4 import BeautifulSoup

from macro_releases import EXPECTED_COLUMNS, _release_key

MONTH_DATE = r"[A-Za-z]+\s+\d{1,2},\s*\d{4}"
BLS_SERIES = {
    "cpi_mom": ("CUSR0000SA0", "cpi", "pct1", "%", "SA"),
    "cpi_yoy": ("CUUR0000SA0", "cpi", "pct12", "%", "NSA"),
    "core_cpi_mom": ("CUSR0000SA0L1E", "cpi", "pct1", "%", "SA"),
    "core_cpi_yoy": ("CUUR0000SA0L1E", "cpi", "pct12", "%", "NSA"),
    "nfp": ("CES0000000001", "nfp", "diff1", "K", "SA"),
    "unemployment_rate": ("LNS14000000", "nfp", "level", "%", "SA"),
    "ppi_mom": ("WPSFD4", "ppi", "pct1", "%", "SA"),
    "ppi_yoy": ("WPUFD4", "ppi", "pct12", "%", "NSA"),
    "core_ppi_mom": ("WPSFD49104", "ppi", "pct1", "%", "SA"),
    "core_ppi_yoy": ("WPUFD49104", "ppi", "pct12", "%", "NSA"),
    "average_hourly_earnings_mom": ("CES0500000003", "nfp", "pct1", "%", "SA"),
    "average_hourly_earnings_yoy": ("CES0500000003", "nfp", "pct12", "%", "SA"),
    "average_weekly_hours": ("CES0500000002", "nfp", "level", "Hours", "SA"),
    "private_payrolls": ("CES0500000001", "nfp", "diff1", "K", "SA"),
    "government_payrolls": ("CES9000000001", "nfp", "diff1", "K", "SA"),
    "manufacturing_payrolls": ("CES3000000001", "nfp", "diff1", "K", "SA"),
}


def utc(value):
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        raise ValueError("an explicit timezone is required")
    return stamp.tz_convert("UTC")


def release_time(day, clock="08:30"):
    return pd.Timestamp(f"{pd.Timestamp(day).date()} {clock}").tz_localize(
        "America/New_York", ambiguous="raise", nonexistent="raise"
    ).tz_convert("UTC")


def observation(event, value, period, released, *, source, fetched_at,
                digest, unit="%", adjustment="SA", vintage="official_release_snapshot",
                previous=None, previous_basis=None):
    released, fetched = utc(released), utc(fetched_at)
    if released > fetched:
        raise ValueError("cannot publish an actual before its release")
    if not period or not source.startswith("https://") or not digest:
        raise ValueError("period, source URL and source digest are required")
    if not math.isfinite(float(value)):
        raise ValueError("actual must be finite")
    local = released.tz_convert("America/New_York")
    row = dict.fromkeys(EXPECTED_COLUMNS)
    row.update(
        release_ts_utc=released, release_date=local.tz_localize(None).normalize(),
        time_et=local.strftime("%H:%M"), country="US", currency="USD",
        event_id=event, event_name=event.replace("_", " "), provider_event=event,
        reference_period=period, actual=float(value), previous=previous, unit=unit,
        source=source, vintage_quality=vintage, first_seen_at_utc=fetched,
        last_seen_at_utc=fetched, adjustment=adjustment, payload_digest=digest,
        previous_basis=previous_basis,
    )
    row["release_key"] = _release_key(pd.Series(row))
    # Different captures of the same release remain independently auditable.
    row["observation_key"] = hashlib.sha256(
        f"{row['release_key']}|{digest}|{value}|{vintage}".encode()
    ).hexdigest()
    return row


def calendar_match(calendar, family, period, fetched_at):
    periods = pd.to_datetime(calendar["ref_period"], format="%B %Y", errors="coerce").dt.to_period("M")
    match = calendar[(calendar["event"] == family) & (periods == period)]
    match = match[match["source"].astype(str).str.startswith("bls:")]
    if len(match) != 1 or pd.isna(match.iloc[0]["time_et"]):
        raise ValueError(f"no unique official release timestamp for {family} {period}")
    row = match.iloc[0]
    released = release_time(row["date"], row["time_et"])
    if released > utc(fetched_at):
        raise ValueError(f"{family} {period} has not been released")
    family_rows = calendar[(calendar["event"] == family)
                           & calendar["source"].astype(str).str.startswith("bls:")]
    due = [release_time(r.date, r.time_et) for r in family_rows.itertuples()
           if pd.notna(r.time_et) and release_time(r.date, r.time_et) <= utc(fetched_at)]
    if due and released < max(due):
        raise ValueError(f"{family} missed a newer scheduled release")
    return released


def parse_bls_api(payload, calendar, *, fetched_at, digest):
    if payload.get("status") != "REQUEST_SUCCEEDED":
        raise ValueError("BLS API did not succeed")
    series = {}
    for item in payload.get("Results", {}).get("series", []):
        values = {}
        for d in item.get("data", []):
            if re.fullmatch(r"M(0[1-9]|1[0-2])", d.get("period", "")):
                value = pd.to_numeric(d.get("value"), errors="coerce")
                values[pd.Period(f"{d['year']}-{d['period'][1:]}", freq="M")] = value
        series[item["seriesID"]] = values
    rows, gaps = [], []
    for event, (code, family, transform, unit, adjustment) in BLS_SERIES.items():
        values = series.get(code, {})
        if not values:
            gaps.append(f"{event}: missing series {code}")
            continue
        period = max(values)  # A missing latest value must not roll back silently.
        value = values[period]
        lag = 12 if transform == "pct12" else 1
        prior = values.get(period - lag)
        if transform != "level":
            if prior is None or not math.isfinite(prior) or prior == 0:
                gaps.append(f"{event}: missing comparison period {period-lag}")
                continue
            value = value - prior if transform == "diff1" else (value / prior - 1) * 100
        if not math.isfinite(value):
            gaps.append(f"{event}: latest observation unavailable")
            continue
        try:
            released = calendar_match(calendar, family, period, fetched_at)
            rows.append(observation(
                event, round(value, 1), str(period), released, source=
                f"https://api.bls.gov/publicAPI/v2/timeseries/data/{code}",
                fetched_at=fetched_at, digest=digest, unit=unit, adjustment=adjustment,
                vintage="official_api_latest_vintage",
            ))
        except ValueError as exc:
            gaps.append(str(exc))
    return rows, gaps


def parse_bls_rss(raw, calendar, *, fetched_at, digest):
    root = ET.fromstring(raw)
    mapping = {
        "Consumer Price Index (CPI)": ("cpi_mom", "cpi", "%", 1),
        "Unemployment Rate": ("unemployment_rate", "nfp", "%", 1),
        "Payroll Employment": ("nfp", "nfp", "K", 1000),
        "Producer Price Index - Final Demand": ("ppi_mom", "ppi", "%", 1),
    }
    rows = []
    for description in root.findall("./channel/item/description"):
        for paragraph in BeautifulSoup(description.text or "", "html.parser").find_all("p"):
            text = paragraph.get_text(" ", strip=True)
            name = text.split(":", 1)[0].strip()
            if name not in mapping:
                continue
            span = paragraph.find("span", class_="data")
            if span is None:
                raise ValueError(f"BLS RSS missing value: {name}")
            m = re.fullmatch(r"([+\-]?[\d,.]+)%?(?:\(p\))?\s+in\s+([A-Za-z]+ \d{4})",
                             span.get_text(" ", strip=True))
            if not m:
                raise ValueError(f"unrecognized BLS RSS value: {name}")
            event, family, unit, scale = mapping[name]
            period = pd.Period(pd.to_datetime(m[2]), freq="M")
            rows.append(observation(
                event, float(m[1].replace(",", "")) / scale, str(period),
                calendar_match(calendar, family, period, fetched_at),
                source="https://www.bls.gov/feed/bls_latest.rss", fetched_at=fetched_at,
                digest=digest, unit=unit, vintage="official_latest_headline_snapshot",
            ))
    if len(rows) != len(mapping):
        raise ValueError("BLS RSS lacks one or more required headline series")
    return rows


def _release_text(raw):
    soup = BeautifulSoup(raw, "html.parser")
    main = soup.find("article") or soup.find("main") or soup
    return re.sub(r"\s+", " ", main.get_text(" ", strip=True))


def _match(pattern, text):
    match = re.search(pattern, text, re.I)
    if not match:
        raise ValueError(f"official release format changed: {pattern}")
    return match


def _signed(direction, value):
    return float(value) * (-1 if direction.lower() in {"decreased", "down", "fell"} else 1)


def parse_bea(raw, kind, *, source, fetched_at, digest):
    text = _release_text(raw)
    date = _match(r"EMBARGOED UNTIL RELEASE AT 8:30 a\.m\. E[DS]T, [A-Za-z]+, (" + MONTH_DATE + ")", text)[1]
    released = release_time(date)
    common = dict(source=source, fetched_at=fetched_at, digest=digest)
    if kind == "gdp":
        m = _match(r"Real gross domestic product \(GDP\) (increased|decreased) at an annual rate of ([\d.]+) percent in the (first|second|third|fourth) quarter of (\d{4})", text)
        quarter = ["first", "second", "third", "fourth"].index(m[3].lower()) + 1
        estimate = _match(r"GDP \((Advance|Second|Third) Estimate\)", text)[1].lower()
        event = "gdp_qoq" + (f"_{estimate}_estimate" if estimate != "advance" else "")
        return [observation(event, _signed(m[1], m[2]), f"{m[4]}Q{quarter}",
                            released, adjustment="SAAR", **common)]
    if kind != "pce":
        raise ValueError(f"unsupported BEA release {kind}")
    period = pd.Period(pd.to_datetime(_match(r"Personal Income and Outlays, ([A-Za-z]+ \d{4})", text)[1]), freq="M")
    rows = []
    # Monthly values come from the release's explicitly labelled comparison table.
    table = next((t for t in pd.read_html(StringIO(raw))
                  if t.astype(str).apply(lambda c: c.str.contains("PCE price index", regex=False)).any().any()), None)
    if table is None:
        raise ValueError("BEA PCE monthly table missing")
    expected_months = [(period - 1).strftime("%B"), period.strftime("%B")]
    column_months = [str(v).strip() for v in table.columns[1:3]]
    first_row_months = [str(v).strip() for v in table.iloc[0, 1:3]]
    if column_months != expected_months and first_row_months != expected_months:
        raise ValueError("BEA monthly table periods do not match the release title")
    for label, event in [("PCE price index", "pce_mom"),
                         ("PCE price index excluding food and energy", "core_pce_mom")]:
        match = table[table.iloc[:, 0].astype(str).str.strip().eq(label)]
        if len(match) != 1:
            raise ValueError(f"missing/ambiguous BEA table row {label}")
        prior, value = pd.to_numeric(match.iloc[0, 1:3], errors="raise")
        rows.append(observation(event, value, str(period), released, previous=float(prior),
                                previous_basis="prior_month_as_revised_in_this_release", **common))
    m = _match(r"From the same month one year ago, the PCE price index.*?(increased|decreased) ([\d.]+) percent.*?Excluding food and energy, the PCE price index (increased|decreased) ([\d.]+) percent from one year ago", text)
    for event, direction, value in [("pce_yoy", m[1], m[2]), ("core_pce_yoy", m[3], m[4])]:
        rows.append(observation(event, _signed(direction, value), str(period), released, **common))
    return rows


def parse_retail(text, *, fetched_at, digest):
    text = re.sub(r"\s+", " ", text)
    day = _match(r"FOR RELEASE AT 8:30 AM E[DS]T, [A-Za-z]+, (" + MONTH_DATE + ")", text)[1]
    period = pd.Period(pd.to_datetime(_match(r"ADVANCE MONTHLY SALES FOR RETAIL AND FOOD SERVICES, ([A-Za-z]+ \d{4})", text)[1]), freq="M")
    match = _match(r"\$[\d,.]+ billion,? (up|down) ([\d.]+) percent.*?from the previous month", text)
    return [observation("retail_sales_mom", _signed(match[1], match[2]), str(period),
                        release_time(day), source="https://www.census.gov/retail/marts/www/marts_current.pdf",
                        fetched_at=fetched_at, digest=digest)]


def parse_claims(text, *, fetched_at, digest):
    text = re.sub(r"\s+", " ", text)
    day = _match(r"EMBARGOED UNTIL 8:30 A\.M\. \(Eastern\) [A-Za-z]+, (" + MONTH_DATE + ")", text)[1]
    year = pd.Timestamp(day).year
    rows = []
    patterns = {
        "initial_jobless_claims": r"week ending ([A-Za-z]+ \d{1,2}), the advance figure for seasonally adjusted initial claims was ([\d,]+)",
        "continuing_jobless_claims": r"advance number for seasonally adjusted insured unemployment during the week ending ([A-Za-z]+ \d{1,2}) was ([\d,]+)",
    }
    for event, pattern in patterns.items():
        match = _match(pattern, text)
        period = pd.Timestamp(f"{match[1]}, {year}")
        if period > pd.Timestamp(day):
            period = pd.Timestamp(f"{match[1]}, {year-1}")
        if not 0 <= (pd.Timestamp(day) - period).days <= 21:
            raise ValueError("claims observation week is inconsistent with release date")
        rows.append(observation(event, float(match[2].replace(",", "")) / 1000,
                    str(period.date()), release_time(day), source="https://www.dol.gov/ui/data.pdf",
                    fetched_at=fetched_at, digest=digest, unit="K"))
    return rows


def next_release(raw, kind, *, source):
    """Only explicit next-release announcements, never inferred month/week rules."""
    text = _release_text(raw) if kind in {"gdp", "pce"} else re.sub(r"\s+", " ", raw)
    if kind == "retail":
        match = _match(r"scheduled for release on (" + MONTH_DATE + r") at 8:30 a\.m\. E[DS]T", text)
    else:
        match = _match(r"Next release:\s*(" + MONTH_DATE + r"),? at 8:30 a\.m\. E[DS]T", text)
    return dict(event=kind, release_ts_utc=release_time(match[1]), source=source,
                schedule_basis="explicit_next_release_notice")


def current_capture(rows):
    """Prefer released headlines over recalculated API values; retain both in archive."""
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    priority = {"official_release_snapshot": 0, "official_latest_headline_snapshot": 1,
                "official_api_latest_vintage": 2}
    frame["_priority"] = frame["vintage_quality"].map(priority).fillna(9)
    return frame.sort_values("_priority").drop_duplicates("release_key").drop(columns="_priority")


def merge_official_history(existing, fresh):
    """Keep previously captured values AND their original provenance intact.

    FMP reference labels differ (e.g. Aug versus 2026-08); event/date matching
    avoids double events at the migration boundary. No forecast gets copied
    onto a new official observation. Revisions live in the capture archive.
    """
    if existing.empty:
        return fresh.copy()
    old = existing.copy()
    old["release_date"] = pd.to_datetime(old["release_date"]).dt.normalize()
    incoming = fresh.copy()
    incoming["release_date"] = pd.to_datetime(incoming["release_date"]).dt.normalize()
    keys = ["country", "event_id", "release_date"]
    duplicate_old = old.loc[old.duplicated(keys, keep=False), keys]
    incoming_keys = set(map(tuple, incoming[keys].to_numpy()))
    if any(tuple(r) in incoming_keys for r in duplicate_old.to_numpy()) or incoming.duplicated(keys).any():
        raise ValueError("ambiguous event/date in macro history")
    populated = set(map(tuple, old.loc[old["actual"].notna(), keys].to_numpy()))
    incoming = incoming[[tuple(row) not in populated for row in incoming[keys].to_numpy()]]
    replaceable = set(map(tuple, incoming[keys].to_numpy()))
    old = old[[tuple(row) not in replaceable for row in old[keys].to_numpy()]]
    if incoming.empty:
        return old.sort_values(["release_ts_utc", "event_id"])
    if old.empty:
        return incoming.sort_values(["release_ts_utc", "event_id"])
    return pd.concat([old, incoming], ignore_index=True).sort_values(["release_ts_utc", "event_id"])
