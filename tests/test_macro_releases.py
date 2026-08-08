"""Guards for the economic actual/consensus history pipeline."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from macro_releases import (  # noqa: E402
    align_bls_release_calendar,
    load_macro_releases,
    merge_release_history,
    normalize_fmp_rows,
    split_event_name,
)
from scripts.build_macro_releases import month_windows  # noqa: E402


def _row(event, date, actual, estimate, previous=0, unit="%", impact="High"):
    return {
        "date": date,
        "country": "US",
        "event": event,
        "currency": "USD",
        "previous": previous,
        "estimate": estimate,
        "actual": actual,
        "change": None,
        "changePercentage": None,
        "impact": impact,
        "unit": unit,
    }


def test_split_event_name_preserves_reference_period():
    assert split_event_name("Non Farm Payrolls (Jul)") == ("Non Farm Payrolls", "Jul")
    assert split_event_name("FOMC Press Conference") == ("FOMC Press Conference", "")


def test_normalize_aliases_dedupes_and_computes_surprise():
    rows = [
        _row("Inflation Rate MoM (Dec)", "2020-01-14 13:30:00", 0.2, 0.3),
        _row("CPI MoM (Dec)", "2020-01-14 13:30:00", 0.2, 0.3),
        _row("Non Farm Payrolls (Dec)", "2020-01-10 13:30:00", 145, 164,
             previous=256, unit="K"),
        _row("CPI MoM (Dec)", "2020-01-14 13:30:00", 99, 99),
    ]
    # Last duplicate provider row wins before alias collapse; this exercises
    # the stable key as well as the CPI/Inflation Rate preference.
    rows.pop()
    df = normalize_fmp_rows(rows, fetched_at="2020-02-01T00:00:00Z")

    assert list(df["event_id"]) == ["nfp", "cpi_mom"]
    assert df["release_key"].is_unique
    cpi = df[df["event_id"] == "cpi_mom"].iloc[0]
    assert cpi["provider_event"] == "Inflation Rate MoM (Dec)"
    assert cpi["release_date"] == pd.Timestamp("2020-01-14")
    assert cpi["time_et"] == "08:30"
    assert np.isclose(cpi["surprise"], -0.1)
    assert cpi["surprise_label"] == "below"
    assert cpi["vintage_quality"] == "vendor_historical_snapshot"


def test_same_day_row_is_labelled_live_capture():
    df = normalize_fmp_rows(
        [_row("Non Farm Payrolls (Jul)", "2026-08-07 12:30:00", -23, 80,
              previous=20, unit="K")],
        fetched_at="2026-08-07T21:00:00Z",
    )
    assert df.loc[0, "vintage_quality"] == "live_capture"
    assert df.loc[0, "surprise"] == -103
    assert df.loc[0, "surprise_label"] == "below"


def test_bls_calendar_alignment_fixes_time_and_quarantines_extra_event():
    df = normalize_fmp_rows([
        _row("Non Farm Payrolls (Mar)", "2014-04-04 13:30:00", 192, 200,
             unit="K"),
        _row("Non Farm Payrolls (Mar)", "2024-08-21 14:00:00", -818, None,
             unit="K"),
    ], fetched_at="2026-08-07T22:00:00Z")
    calendar = pd.DataFrame({
        "date": ["2014-04-04"],
        "event": ["nfp"],
        "time_et": ["08:30"],
    })
    aligned = align_bls_release_calendar(df, calendar)

    regular = aligned[aligned["release_date"] == pd.Timestamp("2014-04-04")].iloc[0]
    assert regular["time_et"] == "08:30"
    assert regular["release_ts_utc"] == pd.Timestamp("2014-04-04T12:30:00Z")
    benchmark = aligned[aligned["release_date"] == pd.Timestamp("2024-08-21")].iloc[0]
    assert benchmark["event_id"] == "nfp_off_calendar"


def test_merge_freezes_released_actual_and_consensus():
    old = normalize_fmp_rows(
        [_row("Non Farm Payrolls (Jul)", "2026-08-07 12:30:00", -23, 80,
              previous=20, unit="K")],
        fetched_at="2026-08-07T21:00:00Z",
    )
    revised = normalize_fmp_rows(
        [_row("Non Farm Payrolls (Jul)", "2026-08-07 12:30:00", -10, 75,
              previous=25, unit="K")],
        fetched_at="2026-08-08T21:00:00Z",
    )
    merged = merge_release_history(old, revised)

    assert len(merged) == 1
    assert merged.loc[0, "actual"] == -23
    assert merged.loc[0, "consensus"] == 80
    assert merged.loc[0, "previous"] == 20
    assert merged.loc[0, "surprise"] == -103
    assert merged.loc[0, "first_seen_at_utc"] == old.loc[0, "first_seen_at_utc"]
    assert merged.loc[0, "last_seen_at_utc"] == revised.loc[0, "last_seen_at_utc"]


def test_merge_fills_previously_missing_release_values():
    scheduled = normalize_fmp_rows(
        [_row("CPI MoM (Jul)", "2026-08-12 12:30:00", None, 0.2,
              previous=0.3)],
        fetched_at="2026-08-11T21:00:00Z",
    )
    released = normalize_fmp_rows(
        [_row("CPI MoM (Jul)", "2026-08-12 12:30:00", 0.1, 0.2,
              previous=0.3)],
        fetched_at="2026-08-12T21:00:00Z",
    )
    merged = merge_release_history(scheduled, released)
    assert merged.loc[0, "actual"] == 0.1
    assert merged.loc[0, "surprise_label"] == "below"
    assert merged.loc[0, "vintage_quality"] == "live_capture"


def test_month_windows_do_not_expose_fmp_long_range_truncation():
    windows = list(month_windows(pd.Timestamp("2020-01-15"), pd.Timestamp("2020-03-02")))
    assert windows == [
        (pd.Timestamp("2020-01-15"), pd.Timestamp("2020-01-31")),
        (pd.Timestamp("2020-02-01"), pd.Timestamp("2020-02-29")),
        (pd.Timestamp("2020-03-01"), pd.Timestamp("2020-03-02")),
    ]


def test_loader_filters_event_and_surprise(tmp_path):
    df = normalize_fmp_rows([
        _row("Non Farm Payrolls (Dec)", "2020-01-10 13:30:00", 145, 164,
             unit="K"),
        _row("FOMC Press Conference", "2020-01-29 19:30:00", None, None),
    ], fetched_at="2020-02-01T00:00:00Z")
    path = tmp_path / "macro.parquet"
    df.to_parquet(path, index=False)

    loaded = load_macro_releases(
        events=["nfp"], start="2020-01-01", end="2020-01-31",
        require_surprise=True, path=path,
    )
    assert len(loaded) == 1
    assert loaded.loc[0, "event_id"] == "nfp"
