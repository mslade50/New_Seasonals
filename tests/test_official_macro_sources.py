import pandas as pd
import pytest

from official_macro_releases import (observation, release_time, parse_bls_api,
    parse_bls_rss, parse_claims, parse_retail, parse_bea, current_capture,
    merge_official_history, calendar_match, next_release)

NOW = "2026-09-22T15:00:00Z"
META = dict(fetched_at=NOW, digest="abc")


def calendar():
    return pd.DataFrame([dict(date="2026-09-11", event="cpi", ref_period="August 2026", time_et="08:30", source="bls:current:cpi"),
        dict(date="2026-09-04", event="nfp", ref_period="August 2026", time_et="08:30", source="bls:current:empsit"),
        dict(date="2026-09-10", event="ppi", ref_period="August 2026", time_et="08:30", source="bls:current:ppi")])


def row(value=0, **kwargs):
    return observation("cpi_mom", value, "2026-08", release_time("2026-09-11"),
                       source="https://www.bls.gov/example", **(META | kwargs))


def test_release_dst_and_zero_no_forecast():
    assert release_time("2026-01-09").hour == 13
    assert release_time("2026-09-11").hour == 12
    assert row()["actual"] == 0
    assert row()["consensus"] is None and row()["surprise"] is None


@pytest.mark.parametrize("changes", [{"fetched_at": "2026-09-01T00:00:00Z"}, {"fetched_at": "2026-09-22"}, {"digest": ""}])
def test_invalid_capture_fails(changes):
    with pytest.raises(ValueError): row(**changes)


def test_bls_transform_and_missing_month_not_row_shift():
    def payload(data): return {"status": "REQUEST_SUCCEEDED", "Results": {"series": [{"seriesID": "CUSR0000SA0", "data": data}]}}
    data = [dict(year="2026", period="M08", value="102"), dict(year="2026", period="M07", value="100")]
    rows, gaps = parse_bls_api(payload(data), calendar(), **META)
    assert rows[0]["actual"] == 2
    assert rows[0]["vintage_quality"] == "official_api_latest_vintage"
    data[1]["period"] = "M06"
    rows, gaps = parse_bls_api(payload(data), calendar(), **META)
    assert not rows and any("comparison period" in g for g in gaps)
    data[0]["value"] = "-"
    rows, _ = parse_bls_api(payload(data), calendar(), **META)
    assert not rows


def test_bls_http_success_is_not_api_success():
    with pytest.raises(ValueError): parse_bls_api({"status": "REQUEST_NOT_PROCESSED"}, calendar(), **META)


def test_calendar_requires_unique_verified_date():
    with pytest.raises(ValueError): calendar_match(pd.concat([calendar(), calendar()]), "cpi", pd.Period("2026-08"), NOW)
    cal = calendar(); cal["source"] = "computed"
    with pytest.raises(ValueError): calendar_match(cal, "cpi", pd.Period("2026-08"), NOW)
    with pytest.raises(ValueError): calendar_match(calendar(), "cpi", pd.Period("2026-08"), "2026-09-11T12:00:00Z")
    extra = dict(date="2026-09-20", event="cpi", ref_period="September 2026", time_et="08:30", source="bls:current:cpi")
    with pytest.raises(ValueError, match="missed"):
        calendar_match(pd.concat([calendar(), pd.DataFrame([extra])]), "cpi", pd.Period("2026-08"), NOW)


def test_explicit_next_release_not_inferred_from_cadence():
    result = next_release("<main>Next release: September 30, 2026, at 8:30 a.m. EDT</main>", "pce", source="https://www.bea.gov/")
    assert result["release_ts_utc"] == pd.Timestamp("2026-09-30T12:30:00Z")
    with pytest.raises(ValueError): next_release("monthly release", "pce", source="https://www.bea.gov/")


def test_rss_signed_zero_payroll_units():
    names = [("Consumer Price Index (CPI)", "+0.0%"), ("Unemployment Rate", "4.1%"),
             ("Payroll Employment", "-2,000(p)"), ("Producer Price Index - Final Demand", "-0.1%(p)")]
    html = "".join(f'<p>{name}:<span class="data">{value} in Aug 2026</span></p>' for name, value in names)
    raw = f"<rss><channel><item><description><![CDATA[{html}]]></description></item></channel></rss>"
    rows = parse_bls_rss(raw, calendar(), **META)
    assert next(r for r in rows if r["event_id"] == "nfp")["actual"] == -2
    assert rows[0]["actual"] == 0
    with pytest.raises(ValueError): parse_bls_rss(raw.replace("Aug 2026", "unknown"), calendar(), **META)


def test_claims_seasonality_and_prior_year_week():
    text = """EMBARGOED UNTIL 8:30 A.M. (Eastern) Thursday, January 8, 2026
    In the week ending January 3, the advance figure for seasonally adjusted initial claims was 210,000.
    The advance number for seasonally adjusted insured unemployment during the week ending December 27 was 1,800,000.
    Unadjusted claims were 400,000."""
    rows = parse_claims(text, **META)
    assert [r["actual"] for r in rows] == [210, 1800]
    assert rows[1]["reference_period"] == "2025-12-27"
    with pytest.raises(ValueError): parse_claims(text.replace("December 27", "November 1"), **META)


def test_retail_sign_and_reference_not_release_month():
    text = """FOR RELEASE AT 8:30 AM EDT, WEDNESDAY, SEPTEMBER 16, 2026
    ADVANCE MONTHLY SALES FOR RETAIL AND FOOD SERVICES, AUGUST 2026
    were $773.9 billion, down 1.2 percent (plus/minus 0.4) from the previous month."""
    rows = parse_retail(text, **META)
    assert rows[0]["actual"] == -1.2 and rows[0]["reference_period"] == "2026-08"
    with pytest.raises(ValueError): parse_retail("Access denied", **META)


def test_gdp_annualization_estimate_separate():
    raw = """<main>EMBARGOED UNTIL RELEASE AT 8:30 a.m. EDT, Wednesday, August 26, 2026
    GDP (Second Estimate) Real gross domestic product (GDP) decreased at an annual rate of 1.5 percent in the second quarter of 2026</main>"""
    result = parse_bea(raw, "gdp", source="https://www.bea.gov/news/example", **META)[0]
    assert result["actual"] == -1.5
    assert result["adjustment"] == "SAAR" and result["event_id"] == "gdp_qoq_second_estimate"


def test_pce_table_months_zero_and_revised_previous():
    raw = """<main>EMBARGOED UNTIL RELEASE AT 8:30 a.m. EDT, Wednesday, August 26, 2026
    Personal Income and Outlays, July 2026
    <table><thead><tr><th>Measure</th><th>June</th><th>July</th></tr></thead>
    <tr><td>PCE price index</td><td>-0.1</td><td>0.0</td></tr>
    <tr><td>PCE price index excluding food and energy</td><td>0.1</td><td>0.2</td></tr></table>
    From the same month one year ago, the PCE price index for July increased 3.7 percent.
    Excluding food and energy, the PCE price index increased 3.3 percent from one year ago.</main>"""
    rows = parse_bea(raw, "pce", source="https://www.bea.gov/news/example", **META)
    assert rows[0]["actual"] == 0 and rows[0]["previous"] == -0.1
    assert rows[0]["previous_basis"] == "prior_month_as_revised_in_this_release"
    with pytest.raises(ValueError, match="periods"):
        parse_bea(raw.replace("<th>July", "<th>August"), "pce", source="https://www.bea.gov/news/example", **META)


def test_prefer_headline_keep_api_vintage_independent():
    api = row(0.1, vintage="official_api_latest_vintage")
    headline = row(0.2, vintage="official_latest_headline_snapshot", digest="def")
    candidate = current_capture([api, headline])
    assert len(candidate) == 1 and candidate.iloc[0].actual == 0.2
    assert api["observation_key"] != headline["observation_key"]


def test_merge_preserves_original_values_and_provenance():
    original = row(0.1); original.update(reference_period="Aug", release_key="legacy", source="FMP", consensus=0.2)
    result = merge_official_history(pd.DataFrame([original]), pd.DataFrame([row(0.3)]))
    assert len(result) == 1
    assert result.iloc[0].actual == 0.1 and result.iloc[0].source == "FMP"
    original["actual"] = None
    result = merge_official_history(pd.DataFrame([original]), pd.DataFrame([row(0.3)]))
    assert result.iloc[0].actual == 0.3 and pd.isna(result.iloc[0].consensus)


def test_ambiguous_target_merge_fails_unrelated_duplicates_preserved():
    with pytest.raises(ValueError): merge_official_history(pd.DataFrame([row(), row()]), pd.DataFrame([row(0.3)]))
    old = row(); old["event_id"] = "unrelated"
    result = merge_official_history(pd.DataFrame([old, old]), pd.DataFrame([row(0.3)]))
    assert len(result) == 3
