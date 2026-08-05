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


def test_freshness_age_bdays():
    df = pd.DataFrame({"equity": [0.5]},
                      index=pd.DatetimeIndex([pd.Timestamp("2026-07-31")]))  # Fri
    assert cp.freshness_age_bdays(df, asof=dt.date(2026, 8, 3)) == 1   # Mon
    assert cp.freshness_age_bdays(df, asof=dt.date(2026, 8, 4)) == 2   # steady-state + holiday tolerance
    assert cp.freshness_age_bdays(df, asof=dt.date(2026, 8, 5)) == 3   # workflow threshold breach
    assert cp.freshness_age_bdays(pd.DataFrame()) is None
