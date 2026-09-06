"""Acquisition gates for the unattended intraday updater."""

import sys

import pandas as pd

import cache_io
from scripts import update_intraday_yfinance as updater


def _fresh_frame(ts="2026-09-04 15:45"):
    return pd.DataFrame({"ts": [pd.Timestamp(ts)]})


def _run_main(monkeypatch, files, fetch, merge, *, upload=False):
    monkeypatch.setattr(updater, "_existing_files", lambda *_args: files)
    monkeypatch.setattr(updater, "_fetch_one", fetch)
    monkeypatch.setattr(updater, "_merge_and_write", merge)
    monkeypatch.setattr(updater.time, "sleep", lambda *_args: None)
    argv = ["update_intraday_yfinance.py"]
    if upload:
        argv.append("--upload")
    monkeypatch.setattr(sys, "argv", argv)
    return updater.main()


def test_all_empty_fails_before_metadata_or_upload(monkeypatch):
    metadata_calls = []
    upload_calls = []
    monkeypatch.setattr(
        updater, "_rebuild_meta", lambda *_args: metadata_calls.append(True)
    )
    monkeypatch.setattr(
        cache_io,
        "upload_from_local",
        lambda *args, **_kwargs: upload_calls.append(args) or True,
    )
    monkeypatch.setattr(cache_io, "is_configured", lambda: True)

    rc = _run_main(
        monkeypatch,
        [("SPY", "missing-spy.parquet"), ("QQQ", "missing-qqq.parquet")],
        lambda *_args: pd.DataFrame(),
        lambda ticker, _path, _fresh: {
            "ticker": ticker,
            "added": 0,
            "status": "no-fresh",
        },
        upload=True,
    )

    assert rc == 1
    assert metadata_calls == []
    assert upload_calls == []


def test_nonempty_zero_add_requires_current_acquisition(monkeypatch):
    metadata_calls = []
    monkeypatch.setattr(
        updater, "_rebuild_meta", lambda *_args: metadata_calls.append(True)
    )
    monkeypatch.setattr(updater, "_is_current_acquisition", lambda *_args: False)

    rc = _run_main(
        monkeypatch,
        [("SPY", "missing-spy.parquet")],
        lambda *_args: _fresh_frame("2026-08-01 15:45"),
        lambda ticker, _path, fresh: {
            "ticker": ticker,
            "added": 0,
            "status": "ok",
            "fresh_last_ts": fresh["ts"].max(),
        },
    )

    assert rc == 1
    assert metadata_calls == []


def test_current_nonempty_zero_add_can_rebuild_metadata(monkeypatch):
    metadata_calls = []
    monkeypatch.setattr(
        updater,
        "_rebuild_meta",
        lambda *_args: metadata_calls.append(True) or pd.DataFrame({"ticker": ["SPY"]}),
    )
    monkeypatch.setattr(updater, "_is_current_acquisition", lambda *_args: True)

    rc = _run_main(
        monkeypatch,
        [("SPY", "missing-spy.parquet")],
        lambda *_args: _fresh_frame(),
        lambda ticker, _path, fresh: {
            "ticker": ticker,
            "added": 0,
            "status": "ok",
            "fresh_last_ts": fresh["ts"].max(),
        },
    )

    assert rc == 0
    assert metadata_calls == [True]


def test_partial_acquisition_is_reported_without_an_invented_threshold(
    monkeypatch, capsys
):
    monkeypatch.setattr(
        updater,
        "_rebuild_meta",
        lambda *_args: pd.DataFrame({"ticker": ["SPY"]}),
    )

    def fetch(ticker, _period):
        return _fresh_frame() if ticker == "SPY" else pd.DataFrame()

    def merge(ticker, _path, fresh):
        if fresh.empty:
            return {"ticker": ticker, "added": 0, "status": "no-fresh"}
        return {
            "ticker": ticker,
            "added": 1,
            "status": "ok",
            "fresh_last_ts": fresh["ts"].max(),
        }

    rc = _run_main(
        monkeypatch,
        [("SPY", "missing-spy.parquet"), ("QQQ", "missing-qqq.parquet")],
        fetch,
        merge,
    )

    output = capsys.readouterr().out
    assert rc == 0
    assert "partial intraday acquisition: 1/2" in output
    assert "QQQ" in output


def test_currentness_uses_latest_required_nyse_session():
    postclose = pd.Timestamp("2026-09-04 17:10", tz=updater.ET)
    assert updater._is_current_acquisition("2026-09-04 15:45", postclose)
    assert not updater._is_current_acquisition("2026-09-03 15:45", postclose)

    saturday = pd.Timestamp("2026-09-05 12:00", tz=updater.ET)
    good_friday = pd.Timestamp("2026-04-03 12:00", tz=updater.ET)
    assert updater._is_current_acquisition("2026-09-04 15:45", saturday)
    assert updater._is_current_acquisition("2026-04-02 15:45", good_friday)
