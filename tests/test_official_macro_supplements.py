import pandas as pd
import pytest

from official_macro_supplements import (discover_release, parse_ism, parse_adp,
    parse_retail_ex_autos, claims_release_schedule, parse_nyfed_jolts_calendar,
    parse_jolts_api, JOLTS_SERIES)

META = dict(fetched_at="2026-09-23T18:00:00Z", digest="test-digest")
ISM = "https://www.prnewswire.com/news-releases/example.html"
ADP = "https://mediacenter.adp.com/example"


def ism():
    return '''<meta name="date" content="2026-09-01T10:00:00-04:00">
    <h1>Manufacturing PMI at 54.6%; August 2026 ISM Report</h1>
    News provided by Institute for Supply Management
    Report featuring September 2026 data will be released at 10:00 a.m. ET on Thursday, October 1, 2026.'''


def adp():
    return '''<meta property="og:title" content="ADP National Employment Report: Private-Sector Employment Increased by 38,000 Jobs in August">
    <!-- ITEMDATE: 2026-09-02 08:15:00 EDT -->
    Report will be released on September 30, 2026 at 8:15 a.m. ET.'''


def test_ism_issuer_timestamp_and_next_notice():
    rows, schedule = parse_ism(ism(), "manufacturing", source=ISM, **META)
    assert rows[0]["actual"] == 54.6 and rows[0]["time_et"] == "10:00"
    assert rows[0]["consensus"] is None
    assert schedule["release_ts_utc"] == pd.Timestamp("2026-10-01T14:00:00Z")


@pytest.mark.parametrize("raw", ["captcha", ism().replace("Institute for Supply Management", "Someone Else"),
    ism().replace("August 2026", "June 2026"), ism().replace('2026-09-01T10:00:00-04:00', '2026-10-01T10:00:00-04:00')])
def test_ism_rejects_untrusted_or_inconsistent_release(raw):
    with pytest.raises(ValueError): parse_ism(raw, "manufacturing", source=ISM, **META)


@pytest.mark.parametrize("direction,amount,expected", [("Increased by", "38,000", 38), ("Shed", "32,000", -32), ("Decreased by", "0", 0)])
def test_adp_units_sign_and_publication_time(direction, amount, expected):
    raw = adp().replace("Increased by 38,000", direction + " " + amount)
    rows, schedule = parse_adp(raw, source=ADP, **META)
    assert rows[0]["actual"] == expected and rows[0]["unit"] == "K"
    assert rows[0]["release_ts_utc"] == pd.Timestamp("2026-09-02T12:15:00Z")
    assert schedule["release_ts_utc"] == pd.Timestamp("2026-09-30T12:15:00Z")


def test_adp_reference_month_and_explicit_time_required():
    for raw in [adp().replace("in August", "in September"), adp().replace("ITEMDATE:", "DATE:")]:
        with pytest.raises(ValueError): parse_adp(raw, source=ADP, **META)


def test_discovery_uses_newest_period_not_link_order():
    raw = '''<a href="/news-releases/old.html">Manufacturing PMI at 55.6%; July 2026 ISM Report</a>
    <a href="/news-releases/new.html">Manufacturing PMI at 54.6%; August 2026 ISM Report</a>'''
    assert discover_release(raw, "manufacturing").endswith("/new.html")
    with pytest.raises(ValueError):
        discover_release(raw + raw.replace("new.html", "conflict.html"), "manufacturing")


def test_retail_only_percent_change_table_and_verified_columns():
    raw = '''FOR RELEASE AT 8:30 AM EDT, WEDNESDAY, SEPTEMBER 16, 2026
    ADVANCE MONTHLY SALES FOR RETAIL AND FOOD SERVICES, AUGUST 2026
    Table 1. Total (excl. motor vehicle & parts) 631568
    Table 2. Estimated Change in Monthly Sales Aug. 2026 Advance Jul. 2026 Preliminary
    Total (excl. motor vehicle & parts) ... -1.4 6.9 -0.2 5.8
    Table 3. Total (excl. motor vehicle & parts) 1.3'''
    assert parse_retail_ex_autos(raw, **META)[0]["actual"] == -1.4
    with pytest.raises(ValueError): parse_retail_ex_autos(raw.replace("Aug. 2026 Advance", "Jul. 2026 Advance"), **META)


def claims():
    return '''Publication Schedule: published each week on Thursday morning at 8:30am EST.
    Release Date Release Time Wednesday, November 25, 2026 8:30 AM EST
    2025 January Calendar Updated: September 11, 2026'''


def test_claims_thanksgiving_exception_and_dst():
    due, upcoming = claims_release_schedule(claims(), as_of="2026-11-26T15:00:00Z")
    assert due == pd.Timestamp("2026-11-25T13:30:00Z")
    assert upcoming["release_ts_utc"] == pd.Timestamp("2026-12-03T13:30:00Z")
    due, upcoming = claims_release_schedule(claims(), as_of="2026-09-23T18:00:00Z")
    assert due == pd.Timestamp("2026-09-17T12:30:00Z")
    assert upcoming["release_ts_utc"] == pd.Timestamp("2026-09-24T12:30:00Z")


def test_claims_before_release_and_stale_schedule():
    due, _ = claims_release_schedule(claims(), as_of="2026-11-25T13:00:00Z")
    assert due == pd.Timestamp("2026-11-19T13:30:00Z")
    with pytest.raises(ValueError): claims_release_schedule(claims(), as_of="2027-01-07T15:00:00Z")


def jolts_calendar():
    return '''September 2026 (all Eastern Time)
    <table><td><div>01<br/><a>JOLTS</a><br/>(10:00)<a>Other event</a>(10:30)</div></td>
    <td><div>29<br/><a>Other event</a>(08:30)<a>JOLTS</a><br/>(10:00)</div></td></table>'''


def test_jolts_reads_correct_clock_and_distinct_monthly_releases():
    schedules = parse_nyfed_jolts_calendar(jolts_calendar(), "2026-09", source="https://www.newyorkfed.org/calendar")
    assert [s["release_ts_utc"] for s in schedules] == [pd.Timestamp("2026-09-01T14:00:00Z"), pd.Timestamp("2026-09-29T14:00:00Z")]
    payload = {"status": "REQUEST_SUCCEEDED", "Results": {"series": [
        {"seriesID": JOLTS_SERIES, "data": [{"year": "2026", "period": "M07", "value": "7130"}]}]}}
    rows, upcoming = parse_jolts_api(payload, schedules, **META)
    assert rows[0]["actual"] == 7130 and rows[0]["reference_period"] == "2026-07"
    assert rows[0]["release_time_source"] == "https://www.newyorkfed.org/calendar"
    assert upcoming["release_ts_utc"] == pd.Timestamp("2026-09-29T14:00:00Z")
    payload["Results"]["series"][0]["data"][0]["period"] = "M06"
    with pytest.raises(ValueError, match="stale"):
        parse_jolts_api(payload, schedules, **META)


def test_jolts_calendar_wrong_month_or_missing_time_fails():
    with pytest.raises(ValueError):
        parse_nyfed_jolts_calendar(jolts_calendar(), "2026-10", source="https://www.newyorkfed.org/calendar")
    with pytest.raises(ValueError):
        parse_nyfed_jolts_calendar(jolts_calendar().replace("(10:00)", "TBA"), "2026-09", source="https://www.newyorkfed.org/calendar")


def test_jolts_does_not_attach_stale_values_to_new_release():
    schedules = parse_nyfed_jolts_calendar(jolts_calendar(), "2026-09", source="https://www.newyorkfed.org/calendar")
    schedules += [dict(release_ts_utc=pd.Timestamp("2026-11-03T15:00:00Z"), source="https://www.newyorkfed.org/calendar")]
    payload = {"status": "REQUEST_SUCCEEDED", "Results": {"series": [
        {"seriesID": JOLTS_SERIES, "data": [{"year": "2026", "period": "M07", "value": "7130"}]}]}}
    with pytest.raises(ValueError, match="stale"):
        parse_jolts_api(payload, schedules, fetched_at="2026-09-29T15:00:00Z", digest="test")
