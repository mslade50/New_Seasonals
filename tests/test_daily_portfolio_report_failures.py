"""Process-status contracts for the unattended portfolio report."""

import pandas as pd
import pytest

import daily_portfolio_report as report


def _must_not_run(name):
    def fail(*_args, **_kwargs):
        raise AssertionError(f"{name} ran after an upstream failure")

    return fail


def _valid_backtest_result():
    return (
        pd.DataFrame({"Ticker": ["SPY"]}),
        pd.Series([750_000.0], index=[pd.Timestamp("2026-09-04")]),
        pd.Series([0.0], index=[pd.Timestamp("2026-09-04")]),
        {"SPY": pd.DataFrame()},
        750_000.0,
    )


@pytest.mark.parametrize("failure", ["backtest", "chart", "export"])
def test_early_report_failure_is_nonzero_and_has_no_downstream_side_effects(
    monkeypatch, failure
):
    monkeypatch.setenv("LOCAL_AUTOMATION_STRICT", "1")
    monkeypatch.setattr(
        report, "write_portfolio_to_sheet", _must_not_run("Sheets write")
    )
    monkeypatch.setattr(
        report, "send_portfolio_email", _must_not_run("email send")
    )

    if failure == "backtest":
        monkeypatch.setattr(
            report,
            "run_12month_backtest",
            lambda **_kwargs: (None, None, None, None, None),
        )
        monkeypatch.setattr(
            report, "create_portfolio_chart", _must_not_run("chart build")
        )
        monkeypatch.setattr(
            report, "save_chart_as_png", _must_not_run("chart export")
        )
    elif failure == "chart":
        monkeypatch.setattr(
            report, "run_12month_backtest", lambda **_kwargs: _valid_backtest_result()
        )
        monkeypatch.setattr(report, "create_portfolio_chart", lambda *_args: None)
        monkeypatch.setattr(
            report, "save_chart_as_png", _must_not_run("chart export")
        )
    else:
        monkeypatch.setattr(
            report, "run_12month_backtest", lambda **_kwargs: _valid_backtest_result()
        )
        monkeypatch.setattr(report, "create_portfolio_chart", lambda *_args: object())
        monkeypatch.setattr(report, "save_chart_as_png", lambda *_args, **_kwargs: None)

    assert report.main() == 1


def test_module_entrypoint_propagates_main_status():
    source = report.__loader__.get_source(report.__name__)
    assert 'raise SystemExit(main())' in source
