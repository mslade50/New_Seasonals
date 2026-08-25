import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import build_betas
from scripts.site_r2_pipeline import GENERATED_INPUTS
from scripts.validate_site_freshness import (
    PAGE_BUILD_PAYLOADS,
    REQUIRED_PORTFOLIO_PAYLOADS,
)
from strategy_config import ACCOUNT_VALUE


ROOT = Path(__file__).resolve().parents[1]


def _prices() -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-02", periods=270)
    spy_returns = 0.0004 + 0.008 * np.sin(np.arange(len(dates) - 1) / 5.0)
    noise = 0.0002 * np.cos(np.arange(len(dates) - 1) / 7.0)
    double_returns = 2.0 * spy_returns + noise

    spy_close = np.r_[100.0, 100.0 * np.cumprod(1.0 + spy_returns)]
    double_close = np.r_[40.0, 40.0 * np.cumprod(1.0 + double_returns)]
    short_dates = dates[-20:]
    short_close = np.linspace(25.0, 27.0, len(short_dates))

    return pd.concat(
        [
            pd.DataFrame({"ticker": "SPY", "date": dates, "Close": spy_close}),
            pd.DataFrame({"ticker": "DOUBLE", "date": dates, "Close": double_close}),
            pd.DataFrame(
                {"ticker": "SHORT", "date": short_dates, "Close": short_close}
            ),
        ],
        ignore_index=True,
    )


def test_known_beta_spy_identity_and_short_history_schema():
    payload = build_betas.build_beta_payload(
        _prices(),
        universe={"SPY", "DOUBLE", "SHORT", "MISSING"},
        generated_utc="2026-08-25T21:31:00Z",
    )

    assert set(payload) == {
        "asof",
        "generated_utc",
        "method",
        "spy_last",
        "account_value",
        "tickers",
    }
    assert payload["generated_utc"] == "2026-08-25T21:31:00Z"
    assert payload["method"] == build_betas.METHOD
    assert payload["account_value"] == ACCOUNT_VALUE
    assert payload["asof"] == "2026-01-14"

    records = payload["tickers"]
    assert set(records) == {"SPY", "DOUBLE", "SHORT", "MISSING"}
    assert records["DOUBLE"]["beta63"] == pytest.approx(2.0, abs=0.05)
    assert records["DOUBLE"]["beta252"] == pytest.approx(2.0, abs=0.05)
    assert records["SPY"]["beta63"] == 1.0
    assert records["SPY"]["beta252"] == 1.0
    assert records["SPY"]["idio_vol63"] == 0.0
    assert records["SPY"]["n63"] == 63
    assert records["SPY"]["n252"] == 252
    assert records["SHORT"] == {
        "beta63": None,
        "beta252": None,
        "idio_vol63": None,
        "n63": 19,
        "n252": 19,
    }
    assert records["MISSING"]["n63"] == 0
    assert records["MISSING"]["beta63"] is None


def test_write_payload_is_compact_valid_json(tmp_path):
    output = tmp_path / "data" / "betas.json"
    payload = build_betas.build_beta_payload(
        _prices(), universe={"SPY", "DOUBLE"}, generated_utc="2026-08-25T21:31:00Z"
    )

    build_betas.write_payload(payload, output)

    raw = output.read_text(encoding="utf-8")
    assert json.loads(raw) == payload
    assert '\n  "' not in raw


def test_price_frame_requires_spy_history():
    frame = _prices().query("ticker != 'SPY'")
    with pytest.raises(ValueError, match="SPY requires"):
        build_betas.build_beta_payload(frame, universe={"DOUBLE"})


def test_missing_ticker_session_does_not_create_a_multi_session_return():
    prices = _prices()
    dates = prices.loc[prices["ticker"].eq("SPY"), "date"].sort_values()
    missing_date = dates.iloc[-20]
    sparse = prices[
        ~(
            prices["ticker"].eq("DOUBLE")
            & prices["date"].eq(missing_date)
        )
    ]

    payload = build_betas.build_beta_payload(
        sparse,
        universe={"SPY", "DOUBLE"},
        generated_utc="2026-08-25T21:31:00Z",
    )

    # The missing close invalidates its own return and the following session's
    # return; neither may be replaced with a multi-session move.
    assert payload["tickers"]["DOUBLE"]["n63"] == 61
    assert payload["tickers"]["DOUBLE"]["beta63"] == pytest.approx(2.0, abs=0.05)


def test_private_site_wiring_keeps_betas_optional():
    item = {entry.name: entry for entry in GENERATED_INPUTS}["betas"]
    assert item.key == "betas.json"
    assert item.path == "data/betas.json"
    assert item.required is False
    assert "betas" not in REQUIRED_PORTFOLIO_PAYLOADS
    assert not any(flag == "betas" for _path, flag in PAGE_BUILD_PAYLOADS)

    workflow = (ROOT / ".github" / "workflows" / "deploy_site.yml").read_text(
        encoding="utf-8"
    )
    beta_step = workflow.index("run: python scripts/build_betas.py")
    assert workflow.index("run: python scripts/build_atr_downside_stats.py") < beta_step
    assert beta_step < workflow.index("run: python scripts/build_risk_json.py")
    assert "continue-on-error: true" in workflow[beta_step - 180 : beta_step]
