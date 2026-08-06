"""Guards for data/macro_events.csv + macro_calendar helpers."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from macro_calendar import (EVENT_TYPES, event_dates, event_window_mask,
                            load_macro_events, next_event, td_offset)


@pytest.fixture(scope="module")
def cal() -> pd.DataFrame:
    return load_macro_events()


def test_schema_and_span(cal):
    assert list(cal.columns) == ["date", "event", "detail", "ref_period",
                                 "time_et", "source"]
    assert set(cal["event"]) == set(EVENT_TYPES)
    assert cal["date"].min() <= pd.Timestamp("2000-01-10")
    assert cal["date"].max() >= pd.Timestamp("2026-12-01")
    assert not cal.duplicated(subset=["date", "event"]).any()
    assert cal["date"].is_monotonic_increasing


KNOWN_DATES = [
    # (event, date) — externally verified anchors
    ("fomc_decision", "2000-05-16"),
    ("fomc_decision", "2004-06-30"),      # first hike of the 2004 cycle
    ("fomc_decision", "2008-12-16"),      # ZIRP
    ("fomc_decision", "2019-07-31"),
    ("fomc_decision", "2022-06-15"),      # 75bp
    ("fomc_decision", "2025-12-10"),      # NOT 12-17 (risk_dashboard typo)
    ("fomc_intermeeting", "2008-01-22"),
    ("fomc_intermeeting", "2020-03-15"),
    ("cpi", "2021-11-10"),                # the 6.2% print
    ("cpi", "2022-09-13"),                # the -4.3% SPX day
    ("cpi", "2013-10-30"),                # shutdown-delayed Sept CPI
    ("cpi", "2025-10-24"),                # shutdown-delayed Sept CPI
    ("nfp", "2008-12-05"),
    ("nfp", "2013-10-22"),                # shutdown-delayed Sept NFP
    ("nfp", "2023-01-06"),
    ("nfp", "2025-11-20"),                # shutdown-delayed Sept NFP
    ("jackson_hole", "2010-08-27"),       # Bernanke QE2 hint
    ("jackson_hole", "2020-08-27"),       # AIT framework (Thursday)
    ("jackson_hole", "2022-08-26"),       # Powell "pain"
    ("jackson_hole", "2026-08-28"),
    ("opex", "2020-03-20"),
    ("quad_witching", "2021-12-17"),
    ("election", "2016-11-08"),
    ("election", "2020-11-03"),
]


@pytest.mark.parametrize("event,day", KNOWN_DATES)
def test_known_dates_present(cal, event, day):
    sub = cal[(cal["event"] == event) & (cal["date"] == pd.Timestamp(day))]
    assert len(sub) == 1, f"{event} {day} missing"


def test_canceled_oct_2025_releases_absent(cal):
    for ev in ("cpi", "nfp", "ppi"):
        sub = cal[(cal["event"] == ev)
                  & (cal["ref_period"] == "October 2025")]
        assert sub.empty, f"{ev} for October 2025 was canceled, row exists"


def test_fomc_counts_per_year(cal):
    dec = cal[cal["event"] == "fomc_decision"]
    for y in range(2000, 2027):
        n = (dec["date"].dt.year == y).sum()
        expected = 7 if y == 2020 else 8   # March 2020 meeting cancelled
        assert n == expected, f"{y}: {n} scheduled decisions"


def test_bls_release_counts(cal):
    for ev in ("cpi", "nfp"):
        sub = cal[cal["event"] == ev]
        for y in range(2000, 2025):
            n = (sub["date"].dt.year == y).sum()
            assert 11 <= n <= 13, f"{ev} {y}: {n}"


def test_bls_releases_are_weekdays(cal):
    bls = cal[cal["event"].isin(["cpi", "nfp", "ppi"])]
    assert (bls["date"].dt.weekday < 5).all()


def test_jackson_hole_anchor_weekday(cal):
    jh = cal[cal["event"] == "jackson_hole"]
    for row in jh.itertuples():
        expected = 3 if row.date.year == 2020 else 4   # Thu in 2020, else Fri
        assert row.date.weekday() == expected, row.date


def test_td_offset_basic():
    # Sessions around the 2022-06-15 FOMC (no holidays in the window)
    idx = pd.bdate_range("2022-06-06", "2022-06-24")
    idx = idx[idx != pd.Timestamp("2022-06-20")]      # Juneteenth holiday
    off = td_offset(idx, "fomc_decision")
    assert off.loc["2022-06-15"] == 0
    assert off.loc["2022-06-14"] == -1
    assert off.loc["2022-06-10"] == -3
    assert off.loc["2022-06-16"] == 1


def test_td_offset_weekend_event_maps_forward():
    # 2020-03-15 intermeeting cut was a Sunday -> Monday 03-16 is day 0
    idx = pd.bdate_range("2020-03-09", "2020-03-20")
    off = td_offset(idx, "fomc_intermeeting")
    assert off.loc["2020-03-16"] == 0


def test_event_window_mask_boundaries():
    idx = pd.bdate_range("2022-06-06", "2022-06-24")
    idx = idx[idx != pd.Timestamp("2022-06-20")]
    m = event_window_mask(idx, "fomc_decision", before_td=2, after_td=1)
    assert not m.loc["2022-06-10"]
    assert m.loc["2022-06-13"]
    assert m.loc["2022-06-15"]
    assert m.loc["2022-06-16"]
    assert not m.loc["2022-06-17"]


def test_next_event():
    nxt = next_event("2026-08-06", "jackson_hole")
    assert nxt == pd.Timestamp("2026-08-28")


def test_event_dates_unique_sorted():
    for ev in EVENT_TYPES:
        d = event_dates(ev)
        assert d.is_monotonic_increasing and d.is_unique
