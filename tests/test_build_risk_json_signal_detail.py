import json

import pandas as pd

from scripts.build_risk_json import _build_signal_detail


def _periods(history):
    active = history[history]
    return [] if active.empty else [(active.index[0], active.index[-1])]


def test_signal_detail_uses_shared_dates_and_rounds_metric_values():
    dates = pd.date_range("2026-01-05", periods=4, freq="B")
    signals = {
        "Distribution Dominance": {
            "summary": "D/A ratio: 4.4 (threshold: 3.75)",
            "da_ratio": pd.Series(
                [3.111, 4.444, 9.876],
                index=[dates[0], dates[2], dates[-1] + pd.Timedelta(days=10)],
            ),
            "signal_history": pd.Series([False, True, True, False], index=dates),
        },
        "Pre-FOMC Rally": {
            "summary": "Next FOMC: Jan 28",
            "signal_history": pd.Series([False, False, True, False], index=dates),
        },
    }

    detail = _build_signal_detail(signals, dates, _periods)

    da = detail["Distribution Dominance"]
    assert da["metric"]["values"] == [3.11, None, 4.44, None]
    assert da["metric"]["thresholds"] == [
        {"value": 3.75, "label": "Fire", "operator": ">"},
        {"value": 6.0, "label": "Elevated", "operator": ">"},
    ]
    assert da["current"] == {
        "value": 9.88,
        "summary": "D/A ratio: 4.4 (threshold: 3.75)",
    }
    assert da["periods"] == [["2026-01-06", "2026-01-07"]]

    fomc = detail["Pre-FOMC Rally"]
    assert fomc["metric"] is None
    assert fomc["current"] == {"value": None, "summary": "Next FOMC: Jan 28"}
    assert fomc["periods"] == [["2026-01-07", "2026-01-07"]]

    # The compact block must remain strict JSON (no NaN/Infinity leakage).
    json.dumps(detail, allow_nan=False)


def test_malformed_period_history_does_not_drop_metric_detail():
    dates = pd.date_range("2026-02-02", periods=2, freq="B")
    signals = {
        "Dispersion": {
            "summary": "Composite pctile: 90th",
            "composite_pctile": pd.Series([85.04, 90.06], index=dates),
            "signal_history": object(),
        }
    }

    def broken_periods(_history):
        raise TypeError("bad history")

    detail = _build_signal_detail(signals, dates, broken_periods)["Dispersion"]

    assert detail["periods"] == []
    assert detail["metric"]["values"] == [85.0, 90.1]
    assert detail["current"]["value"] == 90.1
