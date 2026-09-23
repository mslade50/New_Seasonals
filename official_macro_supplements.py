"""Issuer releases and official schedule checks for the macro replacement."""
from __future__ import annotations

import re
from urllib.parse import urljoin

import pandas as pd
from bs4 import BeautifulSoup

from official_macro_releases import (
    MONTH_DATE, _match, _release_text, observation, release_time, utc,
)

ISM_INDEX = "https://www.prnewswire.com/news/institute-for-supply-management/"
ADP_INDEX = "https://mediacenter.adp.com/press-releases?l=100"
CLAIMS_SCHEDULE = "https://oui.doleta.gov/unemploy/archive.asp"
JOLTS_SERIES = "JTS000000000000000JOL"


def discover_release(raw, kind):
    """Choose the newest explicitly dated issuer release, never the first link."""
    soup = BeautifulSoup(raw, "html.parser")
    found = []
    for link in soup.find_all("a", href=True):
        label = link.get_text(" ", strip=True)
        if kind == "adp":
            if not re.search(r"ADP National Employment Report: Private.?Sector Employment", label, re.I):
                continue
            url = urljoin(ADP_INDEX, link["href"])
            match = re.match(r"https://mediacenter\.adp\.com/(\d{4}-\d{2}-\d{2})-", url)
            if match:
                found.append((pd.Timestamp(match[1]), url.split("#")[0]))
        else:
            match = re.search(rf"{kind} PMI.*?;\s*([A-Za-z]+ \d{{4}}) ISM", label, re.I)
            url = urljoin(ISM_INDEX, link["href"])
            if match and url.startswith("https://www.prnewswire.com/news-releases/"):
                found.append((pd.Timestamp(match[1]), url))
    if not found:
        raise ValueError(f"issuer index has no {kind} release")
    newest = max(d for d, _ in found)
    urls = {u for d, u in found if d == newest}
    if len(urls) != 1:
        raise ValueError(f"ambiguous newest {kind} release")
    return urls.pop()


def parse_ism(raw, kind, *, source, fetched_at, digest):
    if kind not in {"manufacturing", "services"}:
        raise ValueError("unknown ISM series")
    if not source.startswith("https://www.prnewswire.com/news-releases/"):
        raise ValueError("unexpected ISM publisher")
    soup = BeautifulSoup(raw, "html.parser")
    title = soup.find("h1")
    if title is None:
        raise ValueError("ISM title missing")
    m = _match(rf"{kind} PMI.*?at ([\d.]+)%;\s*([A-Za-z]+ \d{{4}}) ISM", title.get_text(" ", strip=True))
    period = pd.Period(pd.to_datetime(m[2]), freq="M")
    date = soup.find("meta", attrs={"name": "date"})
    if date is None:
        raise ValueError("ISM publication timestamp missing")
    stamp = utc(date["content"])
    text = _release_text(raw)
    if not re.search(r"News provided by\s+Institute for Supply Management", text, re.I):
        raise ValueError("release is not supplied by ISM")
    if pd.Period(stamp.tz_convert("America/New_York").date(), freq="M") != period + 1:
        raise ValueError("ISM reference month does not match release month")
    row = observation(f"ism_{kind}_pmi", m[1], str(period), stamp,
                      source=source, fetched_at=fetched_at, digest=digest, unit="Index")
    next_day = _match(r"featuring [A-Za-z]+ \d{4} data will be released at 10:00 a\.m\. ET on [A-Za-z]+, (" + MONTH_DATE + ")", text)[1]
    schedule = dict(event=f"ism_{kind}_pmi", release_ts_utc=release_time(next_day, "10:00"),
                    source=source, schedule_basis="explicit_next_release_notice")
    return [row], schedule


def parse_adp(raw, *, source, fetched_at, digest):
    if not source.startswith("https://mediacenter.adp.com/"):
        raise ValueError("unexpected ADP publisher")
    soup = BeautifulSoup(raw, "html.parser")
    title = soup.find("meta", attrs={"property": "og:title"})
    if title is None:
        raise ValueError("ADP title missing")
    m = _match(r"ADP National Employment Report: Private.?Sector Employment (Increased by|Decreased by|Shed) ([\d,]+) Jobs in ([A-Za-z]+)", title["content"])
    stamp = _match(r"ITEMDATE:\s*(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}):\d{2} E[DS]T", raw)
    released = release_time(stamp[1], stamp[2])
    # Reference month is the month preceding publication, including January.
    period = pd.Period(pd.Timestamp(stamp[1]), freq="M") - 1
    if period.strftime("%B").lower() != m[3].lower():
        raise ValueError("ADP reference month does not match publication")
    value = float(m[2].replace(",", "")) / 1000
    if m[1].lower() != "increased by":
        value = -value
    row = observation("adp_employment_change", value, str(period), released,
                      source=source, fetched_at=fetched_at, digest=digest, unit="K")
    next_day = _match(r"will be released on (" + MONTH_DATE + r") at 8:15 a\.m\. ET", _release_text(raw))[1]
    return [row], dict(event="adp_employment_change", release_ts_utc=release_time(next_day, "08:15"),
                       source=source, schedule_basis="explicit_next_release_notice")


def parse_retail_ex_autos(text, *, fetched_at, digest):
    text = re.sub(r"\s+", " ", text)
    day = _match(r"FOR RELEASE AT 8:30 AM E[DS]T, [A-Za-z]+, (" + MONTH_DATE + ")", text)[1]
    period = pd.Period(pd.to_datetime(_match(r"ADVANCE MONTHLY SALES FOR RETAIL AND FOOD SERVICES, ([A-Za-z]+ \d{4})", text)[1]), freq="M")
    table = _match(r"Table 2\.\s*Estimated Change in Monthly Sales(.*?)Table 3\.", text)[1]
    current, prior = period.strftime("%b. %Y"), (period-1).strftime("%b. %Y")
    if current + " Advance" not in table or prior + " Preliminary" not in table:
        raise ValueError("retail table periods do not match headline")
    value = _match(r"Total \(excl\. motor vehicle & parts\)[\s.\u2026]*([+-]?[\d.]+)", table)[1]
    return [observation("retail_sales_ex_autos_mom", value, str(period), release_time(day),
                        source="https://www.census.gov/retail/marts/www/marts_current.pdf",
                        fetched_at=fetched_at, digest=digest)]


def claims_release_schedule(raw, *, as_of):
    """Use DOL's published weekly rule AND its explicit holiday exceptions."""
    text = _release_text(raw)
    _match(r"published each week on Thursday morning at 8:30am EST", text)
    updated = pd.Timestamp(_match(r"Updated:\s*(" + MONTH_DATE + ")", text)[1])
    today = utc(as_of).tz_convert("America/New_York").tz_localize(None).normalize()
    if updated.year != today.year or updated > today:
        raise ValueError("DOL schedule does not cover the current year")
    section = _match(r"Release Date Release Time (.*?)\d{4} January", text)[1]
    exceptions = [pd.Timestamp(s) for s in re.findall(r"[A-Za-z]+, (" + MONTH_DATE + r")\s+8:30 AM E[DS]T", section)]
    dates = set(pd.date_range(today-pd.Timedelta(days=14), today+pd.Timedelta(days=14), freq="W-THU"))
    for day in exceptions:
        # Each holiday row replaces the regular Thursday of the same week.
        thursday = day + pd.Timedelta(days=3-day.weekday())
        dates.discard(thursday)
        if today-pd.Timedelta(days=14) <= day <= today+pd.Timedelta(days=14):
            dates.add(day)
    stamps = sorted(release_time(d) for d in dates)
    due = max(d for d in stamps if d <= utc(as_of))
    upcoming = min(d for d in stamps if d > utc(as_of))
    return due, dict(event="jobless_claims", release_ts_utc=upcoming,
                     source=CLAIMS_SCHEDULE, schedule_basis="official_weekly_rule_with_holiday_exceptions")


def parse_nyfed_jolts_calendar(raw, month, *, source):
    """Read actual calendar cells; don't borrow another event's clock time."""
    soup = BeautifulSoup(raw, "html.parser")
    text = soup.get_text(" ", strip=True)
    _match(r"(?:all Eastern Time|All times are (?:U\.S\. )?Eastern)", text)
    month = pd.Period(month, freq="M")
    _match(month.strftime("%B") + r"\s+" + str(month.year), text)
    rows = []
    for link in soup.find_all("a"):
        if link.get_text(" ", strip=True) != "JOLTS":
            continue
        cell = link.find_parent("td")
        if cell is None:
            raise ValueError("JOLTS link is not in a calendar cell")
        day = int(_match(r"^\s*(\d{1,2})\b", cell.get_text(" ", strip=True))[1])
        after = cell.get_text(" ", strip=True).split("JOLTS", 1)[1]
        clock = _match(r"^\s*\((\d{1,2}:\d{2})\)", after)[1]
        rows.append(dict(event="jolts", release_ts_utc=release_time(f"{month}-{day:02d}", clock),
                         source=source, schedule_basis="nyfed_official_calendar"))
    return rows


def parse_jolts_api(payload, schedules, *, fetched_at, digest):
    if payload.get("status") != "REQUEST_SUCCEEDED":
        raise ValueError("BLS API did not succeed")
    series = [s for s in payload.get("Results", {}).get("series", []) if s["seriesID"] == JOLTS_SERIES]
    if len(series) != 1:
        raise ValueError("JOLTS API series missing or ambiguous")
    values = {pd.Period(f"{d['year']}-{d['period'][1:]}", freq="M"): d["value"]
              for d in series[0]["data"] if re.fullmatch(r"M(0[1-9]|1[0-2])", d["period"])}
    if not values:
        raise ValueError("JOLTS API has no monthly observations")
    now = utc(fetched_at)
    due = [s for s in schedules if s["release_ts_utc"] <= now]
    upcoming = [s for s in schedules if s["release_ts_utc"] > now]
    if not due or not upcoming:
        raise ValueError("JOLTS calendar lacks a due or upcoming release")
    release = max(due, key=lambda s: s["release_ts_utc"])
    period = max(values)
    # The Fed calendar omits reference months. Require a uniquely plausible
    # monthly period (a 26-day window, shorter than any month). Delayed releases
    # outside this window fail closed instead of guessing a period association.
    lag = (release["release_ts_utc"].tz_convert("America/New_York").date() - period.end_time.date()).days
    if not 20 <= lag <= 45:
        raise ValueError("JOLTS reference month is stale or needs explicit delayed-release mapping")
    row = observation("jolts_job_openings", values[period], str(period), release["release_ts_utc"],
                      source=f"https://api.bls.gov/publicAPI/v2/timeseries/data/{JOLTS_SERIES}",
                      fetched_at=fetched_at, digest=digest, unit="K", vintage="official_api_latest_vintage")
    row["release_time_source"] = release["source"]
    row["reference_mapping_basis"] = "latest_due_release_with_20_to_45_day_reference_lag_gate"
    return [row], min(upcoming, key=lambda s: s["release_ts_utc"])
