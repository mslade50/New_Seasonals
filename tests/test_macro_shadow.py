import pandas as pd

from scripts.compare_macro_shadow import compare_actuals, compare_schedules, comparable_value


def official(event="initial_jobless_claims", unit="K", value=196):
    return pd.DataFrame([dict(event_id=event, release_date=pd.Timestamp("2026-09-17"),
        release_ts_utc=pd.Timestamp("2026-09-17T12:30:00Z"), actual=value, unit=unit,
        reference_period="2026-09-12", vintage_quality="official_release_snapshot", source="https://www.dol.gov/ui/data.pdf")])


def control(**changes):
    row=dict(event_id="initial_jobless_claims", release_date=pd.Timestamp("2026-09-17"),
        release_ts_utc=pd.Timestamp("2026-09-17T12:30:00Z"), actual=196, unit="K", reference_period=None)
    return pd.DataFrame([row | changes])


def compare(fmp, frame=None, fetched="2026-09-23T15:00:00Z"):
    return compare_actuals(official() if frame is None else frame, fmp, fmp_fetched_at=fetched).iloc[0]


def test_units_normalized_only_when_explicit():
    assert compare(control(actual=.196, unit="M")).status == "match"
    assert compare(control(actual=196000, unit=None)).status == "incomparable_units"
    assert comparable_value(54.6, None, "Index", "ism_manufacturing_pmi") == 54.6


def test_zero_missing_and_values_are_distinct():
    assert compare(control(actual=0), official(value=0)).status == "match"
    assert compare(control(actual=None)).status == "missing_fmp_actual"
    assert compare(control(actual=197)).status == "value_difference"


def test_time_difference_not_hidden_by_same_value():
    result = compare(control(release_ts_utc=pd.Timestamp("2026-09-17T13:30:00Z")))
    assert result.status == "time_difference" and result.value_match
    assert not result.time_match


def test_duplicate_control_not_counted_as_match():
    assert compare(pd.concat([control(), control()])).status == "ambiguous_fmp_release"


def test_wrong_date_and_older_control_are_not_failures_of_official_source():
    result = compare(control(release_date=pd.Timestamp("2026-09-16")))
    assert result.status == "missing_fmp_release"
    assert result.nearest_fmp_date == "2026-09-16"
    assert compare(control(), fetched="2026-09-17T12:00:00Z").status == "control_predates_release"


def test_decimal_rounding_does_not_hide_material_differences():
    assert compare(control(actual=.30000000001, unit="%"), official(unit="%", value=.3)).status == "match"
    assert compare(control(actual=.4, unit="%"), official(unit="%", value=.3)).status == "value_difference"


def test_schedule_matching_includes_missing_and_conflicting_clocks():
    expected=dict(event="jobless_claims", release_ts_utc=pd.Timestamp("2026-09-24T12:30:00Z"), source="https://www.dol.gov")
    fmp=control(release_date=pd.Timestamp("2026-09-24"), release_ts_utc=expected["release_ts_utc"])
    args=dict(now="2026-09-23T15:00:00Z", horizon="2026-10-07T23:59:00Z")
    assert compare_schedules([expected], fmp, **args)[0]["status"] == "match"
    fmp.loc[0,"release_ts_utc"] += pd.Timedelta(hours=1)
    assert compare_schedules([expected], fmp, **args)[0]["status"] == "time_difference"
    assert compare_schedules([expected], fmp.iloc[:0], **args)[0]["status"] == "missing_fmp_schedule"


def test_schedule_outside_horizon_is_not_scored():
    expected=dict(event="retail", release_ts_utc=pd.Timestamp("2026-10-15T12:30:00Z"), source="https://www.census.gov")
    assert compare_schedules([expected],control(),now="2026-09-23T15:00:00Z",horizon="2026-10-07T23:59:00Z") == []


def test_hours_named_measure_and_gdp_stage_alias():
    assert comparable_value(34.4, None, "Hours", "average_weekly_hours") == 34.4
    assert comparable_value(34.4, None, "Hours", "other_event") is None
    frame = official("gdp_qoq_second_estimate", "%", 1.5)
    assert compare(control(event_id="gdp_qoq", unit="%", actual=1.5), frame).status == "match"
    assert compare(control(event_id="gdp_qoq", unit="%", actual=1.5,
                           release_date=pd.Timestamp("2026-09-16")), frame).status == "missing_fmp_release"
