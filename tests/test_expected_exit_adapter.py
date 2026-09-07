import pytest
from scripts.run_expected_exit_monitor import observe
from scripts.build_algorithm_family_catalog import catalog_from_source
from tagged_inventory import TaggedInventory


def test_observer_uses_algo_catalog_and_keeps_read_failure_unknown(monkeypatch):
    catalog=catalog_from_source(as_of="2026-09-07T00:00:00Z")
    captured={}
    def inventory(**kwargs):
        captured.update(kwargs)
        return TaggedInventory(status="unknown",reasons=["reviewed seed required"])
    def broken(*args):
        raise RuntimeError("private source error")
    monkeypatch.setenv("STATUS_TOKEN","fixture")
    result,book,fills=observe("fixture-seed",catalog,inventory_loader=inventory,book_loader=broken,fills_loader=broken)
    assert result["status"]=="unknown" and book=={} and fills=={}
    assert captured["max_age_seconds"]==90
    assert "Monthly Trend" in captured["algo_strategies"]
    assert set(captured["algo_strategies"])=={r["name"] for r in catalog["records"]}


def test_observer_without_credentials_makes_no_broker_request(monkeypatch):
    monkeypatch.delenv("STATUS_TOKEN",raising=False)
    def forbidden(*args): pytest.fail("broker request without credentials")
    result,book,fills=observe("seed",catalog_from_source(as_of="2026-09-07T00:00:00Z"),
        inventory_loader=lambda **kwargs:TaggedInventory(),book_loader=forbidden,fills_loader=forbidden)
    assert result["status"]=="unknown" and book=={} and fills=={}
