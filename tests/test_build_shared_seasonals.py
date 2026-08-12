from pathlib import Path

import pandas as pd
import pytest

from scripts.build_shared_seasonals import build_shared_site, validate_shared_output


def _prices() -> pd.DataFrame:
    rows = []
    for ticker, base in (("SPY", 100.0), ("QQQ", 200.0)):
        for index, date in enumerate(pd.bdate_range("2000-01-03", periods=20)):
            close = base + index
            rows.append({
                "ticker": ticker,
                "date": date,
                "Open": close - 0.25,
                "High": close + 1.0,
                "Low": close - 1.0,
                "Close": close,
                "Volume": 1_000_000,
            })
    return pd.DataFrame(rows)


def test_builder_emits_only_share_allow_list(tmp_path: Path):
    prices = tmp_path / "prices.parquet"
    output = tmp_path / "shared"
    _prices().to_parquet(prices, index=False)

    manifest = build_shared_site(prices, output)

    assert manifest["ticker_count"] == 2
    assert (output / "index.html").is_file()
    assert (output / "assets/seasonality.js").is_file()
    assert (output / "data/seasonality/manifest.json").is_file()
    html = (output / "index.html").read_text(encoding="utf-8")
    assert "Forward Distributions" in html
    assert "exclude the current" in html
    js = (output / "assets/seasonality.js").read_text(encoding="utf-8")
    assert "sl-ticker-options" in js
    assert "sl-quick-ticker" not in js
    assert not (output / "data/trades.json").exists()
    assert not (output / "execution.html").exists()
    validate_shared_output(output)


def test_validator_fails_closed_on_private_payload(tmp_path: Path):
    prices = tmp_path / "prices.parquet"
    output = tmp_path / "shared"
    _prices().to_parquet(prices, index=False)
    build_shared_site(prices, output)
    (output / "data/trades.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="non-shareable file"):
        validate_shared_output(output)


def test_builder_refuses_to_replace_existing_output(tmp_path: Path):
    prices = tmp_path / "prices.parquet"
    output = tmp_path / "shared"
    _prices().to_parquet(prices, index=False)
    output.mkdir()

    with pytest.raises(FileExistsError, match="refusing to replace"):
        build_shared_site(prices, output)
