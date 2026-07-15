"""Focused tests for the interactive backtester's raw xsec metric builder."""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt


_SOURCE_PATH = Path(__file__).parents[1] / "pages" / "backtester.py"
_TREE = ast.parse(_SOURCE_PATH.read_text(encoding="utf-8"))
_WANTED = {"_metric_series", "build_xsec_metric_matrices"}
_NODES = [
    node for node in _TREE.body
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in _WANTED
]
_NS = {"pd": pd}
exec(compile(ast.Module(body=_NODES, type_ignores=[]), str(_SOURCE_PATH), "exec"), _NS)

_metric_series = _NS["_metric_series"]
build_xsec_metric_matrices = _NS["build_xsec_metric_matrices"]


def _frame(scale=1.0, periods=320):
    idx = pd.bdate_range("2024-01-02", periods=periods)
    step = np.arange(periods, dtype=float)
    close = scale * (50.0 + 0.08 * step + 2.5 * np.sin(step / 11.0))
    return pd.DataFrame(
        {
            "Close": close,
            "High": close * (1.01 + 0.002 * np.sin(step / 7.0)),
            "Low": close * (0.99 - 0.002 * np.cos(step / 9.0)),
            "Volume": scale * (1_000_000 + 2_500 * step + 50_000 * np.cos(step / 13.0)),
        },
        index=idx,
    )


def test_metric_series_matches_documented_formulas():
    df = _frame()
    ret = df["Close"].pct_change()
    expected = {
        "mom_12_1": df["Close"].shift(21) / df["Close"].shift(252) - 1,
        "adr20": (df["High"] / df["Low"]).rolling(20).mean() - 1,
        "sigma_mad": ret.rolling(63).std() / ret.abs().rolling(63).mean(),
        "autocorr": ret.rolling(63).corr(ret.shift(1)),
        "dvol_roc": (df["Close"] * df["Volume"]).rolling(20).mean().pct_change(21),
        "rvol_roc": ret.rolling(20).std().pct_change(21),
    }

    for key, series in expected.items():
        pdt.assert_series_equal(_metric_series(df, key, 63), series)


def test_builder_ranks_raw_values_cross_sectionally():
    data = {"AAA": _frame(1.0), "BBB": _frame(1.7), "CCC": _frame(0.65)}
    specs = [
        {"metric": "mom_12_1", "window": 63},
        {"metric": "adr20", "window": 63},
        {"metric": "sigma_mad", "window": 42},
        {"metric": "autocorr", "window": 42},
        {"metric": "dvol_roc", "window": 63},
        {"metric": "rvol_roc", "window": 63},
    ]

    actual = build_xsec_metric_matrices(data, specs)
    assert set(actual) == {spec["metric"] for spec in specs}
    for spec in specs:
        key = spec["metric"]
        raw = pd.DataFrame({
            ticker: _metric_series(df, key, spec["window"])
            for ticker, df in data.items()
        })
        pdt.assert_frame_equal(actual[key], raw.rank(axis=1, pct=True) * 100.0)


def test_missing_required_columns_are_excluded_and_warmup_stays_nan():
    complete = _frame()
    no_volume = complete.drop(columns="Volume")
    close_only = complete[["Close"]]

    dvol = build_xsec_metric_matrices(
        {"GOOD": complete, "NO_VOLUME": no_volume},
        [{"metric": "dvol_roc", "window": 63}],
    )["dvol_roc"]
    assert list(dvol.columns) == ["GOOD"]
    assert dvol.iloc[:40, 0].isna().all()

    adr = build_xsec_metric_matrices(
        {"GOOD": complete, "CLOSE_ONLY": close_only},
        [{"metric": "adr20", "window": 63}],
    )["adr20"]
    assert list(adr.columns) == ["GOOD"]
    assert adr.iloc[:19, 0].isna().all()


def test_wide_open_band_excludes_only_metric_warmup_rows():
    ranks = build_xsec_metric_matrices(
        {"AAA": _frame(1.0), "BBB": _frame(1.4)},
        [{"metric": "rvol_roc", "window": 63}],
    )["rvol_roc"]
    wide_open = (ranks >= 0.0) & (ranks <= 100.0)
    pdt.assert_frame_equal(wide_open, ranks.notna())
