import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import regenerate_atr_seasonal_ranks as regen
from scripts import site_r2_pipeline


def _prices(path: Path, ticker: str = "SPY") -> None:
    dates = pd.bdate_range("2000-01-03", "2026-09-04")
    close = 100 * np.exp(np.cumsum(np.random.default_rng(7).normal(0.0002, 0.008, len(dates))))
    frame = pd.DataFrame(
        {
            "ticker": ticker,
            "date": dates,
            "Open": close * 0.999,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": 1_000_000,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _old_ranks(path: Path, *, current: bool = False) -> None:
    frame = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-01-02"]),
            **{column: [50.0] for column in regen.RANK_COLUMNS},
            "ticker": ["SPY"],
        }
    )
    if current:
        frame.attrs["seasonal_rank_version"] = regen.SEASONAL_RANK_VERSION
    frame.to_parquet(path, index=False)


def test_regeneration_is_frozen_complete_and_idempotent(tmp_path, monkeypatch):
    prices = tmp_path / "data/master_prices.parquet"
    ranks = tmp_path / "atr_seasonal_ranks.parquet"
    receipt = tmp_path / "data/receipt.json"
    _prices(prices)
    _old_ranks(ranks)
    monkeypatch.setattr(regen, "CSV_UNIVERSE", ["SPY"])
    monkeypatch.setattr(regen, "LIQUID_PLUS_COMMODITIES", ["SPY"])
    clock = dt.datetime(2026, 9, 6, tzinfo=dt.timezone.utc)

    first = regen.regenerate_if_needed(
        prices_path=prices,
        ranks_path=ranks,
        receipt_path=receipt,
        now=clock,
    )
    assert first["status"] == "REGENERATED"
    rebuilt = pd.read_parquet(ranks)
    assert rebuilt.attrs["seasonal_rank_version"] == regen.SEASONAL_RANK_VERSION
    assert rebuilt["ticker"].unique().tolist() == ["SPY"]
    assert rebuilt["Date"].dt.year.max() == 2026
    assert np.isfinite(rebuilt[list(regen.RANK_COLUMNS)].to_numpy()).all()
    first_bytes = ranks.read_bytes()

    second = regen.regenerate_if_needed(
        prices_path=prices,
        ranks_path=ranks,
        receipt_path=receipt,
        now=clock,
    )
    assert second["status"] == "CURRENT"
    assert ranks.read_bytes() == first_bytes


def test_regeneration_refuses_missing_frozen_price_history(tmp_path, monkeypatch):
    prices = tmp_path / "data/master_prices.parquet"
    ranks = tmp_path / "atr_seasonal_ranks.parquet"
    _prices(prices)
    _old_ranks(ranks)
    monkeypatch.setattr(regen, "CSV_UNIVERSE", ["SPY", "QQQ"])
    monkeypatch.setattr(regen, "LIQUID_PLUS_COMMODITIES", ["SPY", "QQQ"])
    with pytest.raises(regen.RegenerationError, match="missing=.*QQQ"):
        regen.regenerate_if_needed(
            prices_path=prices,
            ranks_path=ranks,
            receipt_path=tmp_path / "receipt.json",
            now=dt.datetime(2026, 9, 6, tzinfo=dt.timezone.utc),
        )


def test_regeneration_drops_stale_predecessor_tickers(tmp_path, monkeypatch):
    prices = tmp_path / "data/master_prices.parquet"
    ranks = tmp_path / "atr_seasonal_ranks.parquet"
    receipt = tmp_path / "receipt.json"
    _prices(prices)
    predecessor = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-01-02", "2026-01-02"]),
            **{column: [50.0, 50.0] for column in regen.RANK_COLUMNS},
            "ticker": ["SPY", "DELISTED"],
        }
    )
    predecessor.to_parquet(ranks, index=False)
    monkeypatch.setattr(regen, "CSV_UNIVERSE", ["SPY"])
    monkeypatch.setattr(regen, "LIQUID_PLUS_COMMODITIES", ["SPY"])

    result = regen.regenerate_if_needed(
        prices_path=prices,
        ranks_path=ranks,
        receipt_path=receipt,
        now=dt.datetime(2026, 9, 6, tzinfo=dt.timezone.utc),
    )

    assert result["status"] == "REGENERATED"
    assert pd.read_parquet(ranks)["ticker"].unique().tolist() == ["SPY"]


def test_future_dated_predecessor_cannot_expand_rebuild_horizon(tmp_path, monkeypatch):
    prices = tmp_path / "data/master_prices.parquet"
    ranks = tmp_path / "atr_seasonal_ranks.parquet"
    receipt = tmp_path / "receipt.json"
    _prices(prices)
    frame = pd.DataFrame({
        "Date": pd.to_datetime(["2099-01-02"]),
        **{column: [50.0] for column in regen.RANK_COLUMNS},
        "ticker": ["SPY"],
    })
    frame.attrs["seasonal_rank_version"] = regen.SEASONAL_RANK_VERSION
    frame.to_parquet(ranks, index=False)
    monkeypatch.setattr(regen, "CSV_UNIVERSE", ["SPY"])
    monkeypatch.setattr(regen, "LIQUID_PLUS_COMMODITIES", ["SPY"])
    result = regen.regenerate_if_needed(
        prices_path=prices,
        ranks_path=ranks,
        receipt_path=receipt,
        now=dt.datetime(2026, 9, 6, tzinfo=dt.timezone.utc),
    )
    assert result["target_year"] == 2026
    assert pd.read_parquet(ranks)["Date"].dt.year.max() == 2026


def test_current_health_rejects_partial_target_year_and_extra_columns(monkeypatch):
    monkeypatch.setattr(regen, "generate_trading_dates", lambda year: pd.DataFrame({
        "Date": pd.to_datetime([f"{year}-01-02", f"{year}-01-05"]),
        "day_count": [1, 2],
    }))
    frame = pd.DataFrame({
        "Date": pd.to_datetime(["2026-01-02"]),
        **{column: [50.0] for column in regen.RANK_COLUMNS},
        "ticker": ["SPY"],
        "unexpected": [1],
    })
    metadata = {
        "seasonal_rank_version": regen.SEASONAL_RANK_VERSION,
        "regeneration_schema": regen.RECEIPT_SCHEMA,
        "ticker_count": 1,
        "ticker_digest": regen.sha256_json(["SPY"]),
        "year_start": regen.FULL_START_YEAR,
        "year_end": 2026,
    }
    healthy, problems = regen.current_health(
        frame,
        metadata=metadata,
        tickers=["SPY"],
        target_year=2026,
    )
    assert healthy is False
    assert "rank columns do not match the production contract" in problems


def test_current_health_requires_complete_target_year_for_every_ticker(monkeypatch):
    monkeypatch.setattr(regen, "generate_trading_dates", lambda year: pd.DataFrame({
        "Date": pd.to_datetime([f"{year}-01-02", f"{year}-01-05"]),
        "day_count": [1, 2],
    }))
    frame = pd.DataFrame({
        "Date": pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-02"]),
        **{column: [50.0, 50.0, 50.0] for column in regen.RANK_COLUMNS},
        "ticker": ["QQQ", "QQQ", "SPY"],
    })
    tickers = ["QQQ", "SPY"]
    metadata = {
        "seasonal_rank_version": regen.SEASONAL_RANK_VERSION,
        "regeneration_schema": regen.RECEIPT_SCHEMA,
        "ticker_count": 2,
        "ticker_digest": regen.sha256_json(tickers),
        "year_start": regen.FULL_START_YEAR,
        "year_end": 2026,
    }
    healthy, problems = regen.current_health(
        frame,
        metadata=metadata,
        tickers=tickers,
        target_year=2026,
    )
    assert healthy is False
    assert any("target-year trading-date coverage is incomplete" in problem for problem in problems)


def _stage(root: Path) -> None:
    root.mkdir(parents=True)
    (root / ".private-site-cloud-stage.json").write_text(
        json.dumps({"mode": "private-site-cloud-source", "source_sha": "a" * 40}),
        encoding="utf-8",
    )


def test_conditional_canonical_promotion_refreshes_generator_provenance(tmp_path, monkeypatch):
    root = tmp_path / "generator"
    _stage(root)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("PRIVATE_SITE_CLOUD_BUILD", "1")
    item = next(i for i in site_r2_pipeline.CANONICAL_INPUTS if i.name == "atr_seasonal_ranks")
    target = root / item.path
    target.write_bytes(b"new-corrected-ranks")
    old = b"old-ranks"
    old_sha = hashlib.sha256(old).hexdigest()
    new_sha = hashlib.sha256(target.read_bytes()).hexdigest()
    store = {item.key: old}
    etags = {item.key: '"old-etag"'}
    upload_order = []

    def head(key):
        value = store.get(key)
        return None if value is None else {
            "ETag": etags[key], "ContentLength": len(value), "LastModified": "now"
        }

    def conditional(local, key, *, create_only=False, expected_etag=None):
        upload_order.append(key)
        if create_only:
            if key in store:
                return "precondition_failed", None
        elif expected_etag != etags.get(key):
            return "precondition_failed", None
        store[key] = Path(local).read_bytes()
        etags[key] = '"new-etag"'
        return "uploaded", etags[key]

    def download(key, local):
        value = store.get(key)
        if value is None:
            return False
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        Path(local).write_bytes(value)
        return True

    monkeypatch.setattr(site_r2_pipeline.cache_io, "head", head)
    monkeypatch.setattr(site_r2_pipeline.cache_io, "conditional_upload_from_local", conditional)
    monkeypatch.setattr(site_r2_pipeline.cache_io, "download_to_local", download)
    provenance = {
        "mode": "r2-only",
        "phase": "generator",
        "run_id": None,
        "source_sha": "a" * 40,
        "materialized_at": "2026-09-06T00:00:00+00:00",
        "entries": [{
            "name": item.name,
            "key": item.key,
            "path": item.path,
            "required": True,
            "sha256": old_sha,
            "etag": "old-etag",
            "last_modified": "now",
            "size": len(old),
        }],
    }
    provenance_path = root / site_r2_pipeline.PROVENANCE_PATH
    provenance_path.parent.mkdir(parents=True)
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    receipt = root / "data/receipt.json"
    receipt.write_text(
        json.dumps({
            "schema_version": "atr-seasonal-regeneration.v1",
            "status": "REGENERATED",
            "checked_at": "2026-09-06T00:00:00+00:00",
            "rank_version": "annual-outcome-cutoff-v2",
            "before_sha256": old_sha,
            "after_sha256": new_sha,
            "prices_sha256": "b" * 64,
            "ticker_count": 1,
            "target_year": 2026,
            "reasons": ["old version"],
        }),
        encoding="utf-8",
    )

    result = site_r2_pipeline.promote_canonical(
        root,
        name=item.name,
        receipt_path=receipt,
        run_id="123-1",
    )
    assert result["status"] == "PROMOTED"
    assert store[item.key] == b"new-corrected-ranks"
    updated = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert updated["entries"][0]["sha256"] == new_sha
    assert store["migrations/atr_seasonal_ranks/123-1/predecessor.parquet"] == old
    assert "migrations/atr_seasonal_ranks/123-1/receipt.json" in store
    assert result["predecessor_key"].endswith("/predecessor.parquet")
    assert upload_order.index(result["predecessor_key"]) < upload_order.index(item.key)


def test_current_receipt_never_writes_r2(tmp_path, monkeypatch):
    root = tmp_path / "generator"
    _stage(root)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("PRIVATE_SITE_CLOUD_BUILD", "1")
    item = next(i for i in site_r2_pipeline.CANONICAL_INPUTS if i.name == "atr_seasonal_ranks")
    target = root / item.path
    target.write_bytes(b"current")
    sha = hashlib.sha256(b"current").hexdigest()
    provenance_path = root / site_r2_pipeline.PROVENANCE_PATH
    provenance_path.parent.mkdir(parents=True)
    provenance_path.write_text(json.dumps({
        "phase": "generator", "source_sha": "a" * 40,
        "entries": [{"name": item.name, "sha256": sha}],
    }), encoding="utf-8")
    receipt = root / "receipt.json"
    receipt.write_text(json.dumps({
        "schema_version": "atr-seasonal-regeneration.v1", "status": "CURRENT",
        "checked_at": "2026-09-06T00:00:00Z", "rank_version": "annual-outcome-cutoff-v2",
        "before_sha256": sha, "after_sha256": sha, "prices_sha256": "b" * 64,
        "ticker_count": 1, "target_year": 2026, "reasons": [],
    }), encoding="utf-8")
    monkeypatch.setattr(
        site_r2_pipeline.cache_io,
        "conditional_upload_from_local",
        lambda *args, **kwargs: pytest.fail("CURRENT receipt must not upload"),
    )
    assert site_r2_pipeline.promote_canonical(
        root, name=item.name, receipt_path=receipt, run_id="123-1"
    )["status"] == "CURRENT"


def test_cloud_workflow_promotes_corrected_ranks_before_rebuilding_ledger():
    root = Path(__file__).resolve().parents[1]
    workflow = (root / ".github" / "workflows" / "deploy_site.yml").read_text(encoding="utf-8")
    regenerate = workflow.index("regenerate_atr_seasonal_ranks.py")
    promote = workflow.index("promote-canonical")
    ledger = workflow.index("scripts/build_trade_ledger.py --upload")
    publish = workflow.index("publish-generated")
    assert regenerate < promote < ledger < publish
    assert "--receipt data/atr_seasonal_rank_regeneration.json" in workflow
