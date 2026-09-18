"""cboe_putcall: parser, incremental-merge, and freshness-assertion guards.

The parser regex targets CBOE's escaped-JSON markup; a site redesign makes it
return {} for every day, which the workflow's --assert-fresh-bd turns into a
red run instead of a silent green no-op. These tests freeze that contract.
"""
import datetime as dt
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cboe_putcall as cp


def _pair(name: str, value: str) -> str:
    return f'\\"name\\":\\"{name}\\",\\"value\\":\\"{value}\\"'


def test_parse_body_maps_known_fields_only():
    body = "junk" + _pair("TOTAL PUT/CALL RATIO", "1.01") + "," + \
        _pair("EQUITY PUT/CALL RATIO", "0.55") + "," + \
        _pair("SOME OTHER RATIO", "9.99") + "tail"
    assert cp._parse_body(body) == {"total": 1.01, "equity": 0.55}


def test_parse_body_skips_unparseable_values():
    assert cp._parse_body(_pair("EQUITY PUT/CALL RATIO", "n/a")) == {}


def test_parse_body_empty_page():
    assert cp._parse_body("<html>maintenance</html>") == {}


def test_backfill_skips_cached_dates_and_appends(tmp_path, monkeypatch):
    monkeypatch.setattr(cp, "CACHE_PATH", str(tmp_path / "pc.parquet"))
    seed = pd.DataFrame({"equity": [0.50]},
                        index=pd.DatetimeIndex([pd.Timestamp("2026-08-03")],
                                               name="date"))
    cp._save(seed)

    fetched: list[dt.date] = []

    def fake_fetch(d, **kwargs):
        fetched.append(d)
        return {"equity": 0.60}

    monkeypatch.setattr(cp, "_fetch_day", fake_fetch)
    df = cp.backfill("2026-08-03", "2026-08-04", sleep_between=0)

    assert fetched == [dt.date(2026, 8, 4)]          # cached day not refetched
    assert df.loc["2026-08-03", "equity"] == 0.50    # existing row untouched
    assert df.loc["2026-08-04", "equity"] == 0.60
    assert df.index.is_monotonic_increasing and not df.index.duplicated().any()


def test_session_guard_rejects_non_sessions():
    # 2025-01-09: NYSE closed for the Carter day of mourning. CBOE still
    # served a page and the scraper cached equity 0.00 from it.
    assert "special closure" in cp.session_reject_reason(dt.date(2025, 1, 9))
    assert "holiday" in cp.session_reject_reason(dt.date(2025, 4, 18))   # Good Friday
    assert "holiday" in cp.session_reject_reason(dt.date(2025, 12, 25))  # Christmas
    assert "holiday" in cp.session_reject_reason(dt.date(2022, 6, 20))   # Juneteenth observed
    assert "weekend" in cp.session_reject_reason(dt.date(2025, 4, 19))   # Saturday
    assert "weekend" in cp.session_reject_reason(dt.date(2025, 4, 20))   # Sunday
    assert "closure" in cp.session_reject_reason(dt.date(2012, 10, 29))  # Sandy


def test_session_guard_accepts_federal_only_holidays():
    # Columbus Day and Veterans Day are federal holidays the NYSE trades
    # through, which is why USFederalHolidayCalendar is not used as-is.
    assert cp.session_reject_reason(dt.date(2025, 10, 13)) is None  # Columbus Day
    assert cp.session_reject_reason(dt.date(2025, 11, 11)) is None  # Veterans Day
    assert cp.session_reject_reason(dt.date(2021, 12, 31)) is None  # Sat New Year, NYSE open
    assert cp.session_reject_reason(dt.date(2026, 9, 17)) is None   # plain session


def test_row_guard_equity_band():
    good = dt.date(2026, 9, 17)
    assert cp.row_rejection_reason(good, {"equity": 0.62}) is None
    # A real panic print has to survive: 2.40 on 2022-12-28 is the series max.
    assert cp.row_rejection_reason(good, {"equity": 2.40}) is None
    assert cp.row_rejection_reason(good, {"equity": 0.32}) is None  # series min
    assert "<= 0" in cp.row_rejection_reason(good, {"equity": 0.0})
    assert "<= 0" in cp.row_rejection_reason(good, {"equity": -0.5})
    assert "missing" in cp.row_rejection_reason(good, {"total": 1.0})
    assert "missing" in cp.row_rejection_reason(good, {"equity": float("nan")})
    assert "outside" in cp.row_rejection_reason(good, {"equity": 0.01})
    assert "outside" in cp.row_rejection_reason(good, {"equity": 62.0})


def test_purge_drops_polluted_rows_and_keeps_the_rest():
    idx = pd.DatetimeIndex([pd.Timestamp("2025-01-08"),
                            pd.Timestamp("2025-01-09"),   # closed, equity 0.00
                            pd.Timestamp("2025-01-10"),
                            pd.Timestamp("2025-01-13")], name="date")
    df = pd.DataFrame({"equity": [0.60, 0.00, 0.70, 0.60],
                       "total": [0.95, 1.73, 0.97, 0.95]}, index=idx)
    clean, dropped = cp.purge_invalid(df)

    assert [d for d, _ in dropped] == [dt.date(2025, 1, 9)]
    assert len(clean) == 3
    assert pd.Timestamp("2025-01-09") not in clean.index
    assert clean.index.name == "date"
    assert list(clean.columns) == ["equity", "total"]
    assert clean.equals(df.drop(index=[pd.Timestamp("2025-01-09")]))
    # A clean frame is returned untouched (no copy-and-drop churn).
    again, dropped_again = cp.purge_invalid(clean)
    assert dropped_again == [] and again.equals(clean)


def test_backfill_rejects_a_bad_fresh_row(tmp_path, monkeypatch):
    monkeypatch.setattr(cp, "CACHE_PATH", str(tmp_path / "pc.parquet"))
    seed = pd.DataFrame({"equity": [0.50]},
                        index=pd.DatetimeIndex([pd.Timestamp("2026-09-16")],
                                               name="date"))
    cp._save(seed)
    monkeypatch.setattr(cp, "_fetch_day", lambda d, **kw: {"equity": 0.0})

    df = cp.backfill("2026-09-16", "2026-09-17", sleep_between=0)
    assert pd.Timestamp("2026-09-17") not in df.index   # zero print refused
    assert df.loc["2026-09-16", "equity"] == 0.50


def test_backfill_purges_a_polluted_cache_on_load(tmp_path, monkeypatch):
    monkeypatch.setattr(cp, "CACHE_PATH", str(tmp_path / "pc.parquet"))
    idx = pd.DatetimeIndex([pd.Timestamp("2025-01-08"),
                            pd.Timestamp("2025-01-09"),
                            pd.Timestamp("2025-01-10")], name="date")
    cp._save(pd.DataFrame({"equity": [0.60, 0.00, 0.70]}, index=idx))
    monkeypatch.setattr(cp, "_fetch_day", lambda d, **kw: None)

    df = cp.backfill("2025-01-08", "2025-01-10", sleep_between=0)
    assert pd.Timestamp("2025-01-09") not in df.index
    # self-healed on disk, not just in memory
    assert pd.Timestamp("2025-01-09") not in cp._read_cache().index
    assert len(cp._read_cache()) == 2


def test_committed_cache_holds_only_valid_sessions():
    df = pd.read_parquet(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "cboe_putcall.parquet"))
    df.index = pd.to_datetime(df.index)
    bad_dates = [d.date() for d in df.index if cp.session_reject_reason(d)]
    assert bad_dates == [], f"non-session rows in the committed cache: {bad_dates}"
    assert not (df["equity"] <= 0).any()
    assert df["equity"].between(cp.EQUITY_MIN, cp.EQUITY_MAX).all()
    assert pd.Timestamp("2025-01-09") not in df.index
    # 2024-01-10's 1.55 spike is a real print left in place on purpose.
    assert df.loc[pd.Timestamp("2024-01-10"), "equity"] == 1.55


def test_freshness_age_bdays():
    df = pd.DataFrame({"equity": [0.5]},
                      index=pd.DatetimeIndex([pd.Timestamp("2026-07-31")]))  # Fri
    assert cp.freshness_age_bdays(df, asof=dt.date(2026, 8, 3)) == 1   # Mon
    assert cp.freshness_age_bdays(df, asof=dt.date(2026, 8, 4)) == 2   # steady-state + holiday tolerance
    assert cp.freshness_age_bdays(df, asof=dt.date(2026, 8, 5)) == 3   # workflow threshold breach
    assert cp.freshness_age_bdays(pd.DataFrame()) is None
