import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import build_survivorship_prices as builder
from survivorship_contract import (
    SURVIVORSHIP_BASIS,
    SURVIVORSHIP_CONTRACT_VERSION,
    SURVIVORSHIP_SCOPE_START,
    sha256_file,
    validate_survivorship_artifact,
)


def _prices(ticker="OLD", rows=260):
    dates = pd.bdate_range("2020-01-02", periods=rows)
    close = np.linspace(20.0, 30.0, rows)
    return pd.DataFrame({
        "ticker": ticker,
        "date": dates,
        "Open": close,
        "High": close + 1.0,
        "Low": close - 1.0,
        "Close": close,
        "Volume": np.full(rows, 1_000_000.0),
    })


def _write_pair(tmp_path: Path, *, unresolved=None):
    artifact = tmp_path / "survivorship_prices.parquet"
    manifest = tmp_path / "survivorship_prices.meta.json"
    frame = _prices()
    frame.to_parquet(artifact, index=False)
    payload = {
        "contract_version": SURVIVORSHIP_CONTRACT_VERSION,
        "basis": SURVIVORSHIP_BASIS,
        "scope_start": SURVIVORSHIP_SCOPE_START,
        "required_tickers": ["OLD", "PRIMARY"],
        "covered_by_primary_sources": ["PRIMARY"],
        "artifact_tickers": ["OLD"],
        "artifact_rows": len(frame),
        "artifact_sha256": sha256_file(artifact),
        "unresolved_required": list(unresolved or []),
    }
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return artifact, manifest


def test_survivorship_pair_validates_complete_declared_scope(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "survivorship_contract.SURVIVORSHIP_REQUIRED_TICKERS_V1",
        frozenset({"OLD", "PRIMARY"}),
    )
    artifact, manifest = _write_pair(tmp_path)
    result = validate_survivorship_artifact(artifact, manifest)
    assert result["required_tickers"] == ["OLD", "PRIMARY"]


def test_survivorship_pair_fails_on_digest_drift_or_unresolved_name(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "survivorship_contract.SURVIVORSHIP_REQUIRED_TICKERS_V1",
        frozenset({"OLD", "PRIMARY"}),
    )
    artifact, manifest = _write_pair(tmp_path)
    frame = pd.read_parquet(artifact)
    frame.loc[0, "Close"] += 1.0
    frame.to_parquet(artifact, index=False)
    with pytest.raises(ValueError, match="SHA-256"):
        validate_survivorship_artifact(artifact, manifest)

    artifact, manifest = _write_pair(tmp_path, unresolved=["MISS"])
    with pytest.raises(ValueError, match="incomplete"):
        validate_survivorship_artifact(artifact, manifest)


def test_catalog_rejects_reused_ticker_identity_and_accepts_reviewed_rename():
    index_rows = [
        {
            "date": "2021-01-04",
            "removedTicker": "FRX",
            "removedSecurity": "Forest Laboratories Inc",
            "index": "SP500",
        },
        {
            "date": "2021-01-04",
            "removedTicker": "ADS",
            "removedSecurity": "Alliance Data Systems Corp",
            "index": "SP500",
        },
    ]
    delisted_rows = [
        {
            "symbol": "FRX",
            "companyName": "Forest Road Acquisition Corp",
            "exchange": "NYSE",
            "ipoDate": "2020-01-01",
            "delistedDate": "2021-06-01",
        },
        {
            "symbol": "ADS",
            "companyName": "Bread Financial Holdings, Inc.",
            "exchange": "NYSE",
            "ipoDate": "2000-01-01",
            "delistedDate": "2022-04-04",
        },
    ]
    catalog, rejected = builder.build_catalog(
        delisted_rows, index_rows, as_of="2026-08-20"
    )
    assert [row["ticker"] for row in catalog] == ["ADS"]
    assert catalog[0]["identity_alias"] is True
    assert [row["ticker"] for row in rejected] == ["FRX"]


def test_builder_writes_validated_pair_without_upload(tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "SURVIVORSHIP_REQUIRED_TICKERS_V1", frozenset({"OLD"}))
    monkeypatch.setattr(
        "survivorship_contract.SURVIVORSHIP_REQUIRED_TICKERS_V1",
        frozenset({"OLD"}),
    )
    index_rows = [{
        "date": "2020-02-03",
        "removedTicker": "OLD",
        "removedSecurity": "Old Company Inc",
        "index": "SP500",
    }]
    delisted_rows = [{
        "symbol": "OLD",
        "companyName": "Old Company, Inc.",
        "exchange": "NYSE",
        "ipoDate": "2019-01-01",
        "delistedDate": "2022-01-03",
    }]
    monkeypatch.setattr(
        builder,
        "fetch_price_history",
        lambda *_args, **_kwargs: _prices("OLD"),
    )
    output = tmp_path / "survivorship_prices.parquet"
    manifest = tmp_path / "survivorship_prices.meta.json"
    payload = builder.build(
        output=output,
        manifest_path=manifest,
        master_source=tmp_path / "missing-master.parquet",
        overflow_source=tmp_path / "missing-overflow.parquet",
        as_of="2026-08-20",
        api_key="test-key",
        delisted_rows=delisted_rows,
        index_rows=index_rows,
    )
    assert payload["required_tickers"] == ["OLD"]
    assert payload["unresolved_required"] == []
    validate_survivorship_artifact(output, manifest)


def test_builder_rejects_partial_provider_catalog(tmp_path, monkeypatch):
    monkeypatch.setattr(
        builder, "SURVIVORSHIP_REQUIRED_TICKERS_V1", frozenset({"OLD", "MISSING"})
    )
    index_rows = [{
        "date": "2020-02-03",
        "removedTicker": "OLD",
        "removedSecurity": "Old Company Inc",
        "index": "SP500",
    }]
    delisted_rows = [{
        "symbol": "OLD",
        "companyName": "Old Company, Inc.",
        "exchange": "NYSE",
        "ipoDate": "2019-01-01",
        "delistedDate": "2022-01-03",
    }]
    with pytest.raises(RuntimeError, match="does not match reviewed v1 scope"):
        builder.build(
            output=tmp_path / "prices.parquet",
            manifest_path=tmp_path / "prices.meta.json",
            master_source=tmp_path / "master.parquet",
            overflow_source=tmp_path / "overflow.parquet",
            as_of="2026-08-20",
            api_key="test-key",
            delisted_rows=delisted_rows,
            index_rows=index_rows,
        )


def test_bank_failure_terminal_mark_prevents_stale_last_close():
    frame = _prices("SIVB")
    frame = frame.loc[frame["date"] < pd.Timestamp("2023-03-10")]
    marked, policy = builder.apply_terminal_value_policy(frame, "SIVB")
    assert policy["date"] == "2023-03-10"
    final = marked.iloc[-1]
    assert final["date"] == pd.Timestamp("2023-03-10")
    assert final[["Open", "High", "Low", "Close"]].tolist() == [0.01] * 4
    assert final["Volume"] == 0.0


def test_contract_rejects_omitted_required_bank_terminal_mark(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "survivorship_contract.SURVIVORSHIP_REQUIRED_TICKERS_V1",
        frozenset({"SIVB"}),
    )
    artifact = tmp_path / "survivorship_prices.parquet"
    manifest = tmp_path / "survivorship_prices.meta.json"
    frame = _prices("SIVB")
    frame.to_parquet(artifact, index=False)
    manifest.write_text(json.dumps({
        "contract_version": SURVIVORSHIP_CONTRACT_VERSION,
        "basis": SURVIVORSHIP_BASIS,
        "scope_start": SURVIVORSHIP_SCOPE_START,
        "required_tickers": ["SIVB"],
        "covered_by_primary_sources": [],
        "artifact_tickers": ["SIVB"],
        "artifact_rows": len(frame),
        "artifact_sha256": sha256_file(artifact),
        "unresolved_required": [],
        "terminal_marks": [],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="terminal mark is missing for SIVB"):
        validate_survivorship_artifact(artifact, manifest)


def test_contract_rejects_silently_shrunken_reviewed_catalog(tmp_path):
    artifact, manifest = _write_pair(tmp_path)
    with pytest.raises(ValueError, match="does not match the reviewed v1 catalog"):
        validate_survivorship_artifact(artifact, manifest)
