import pytest
import io
import json
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


def test_status_publication_cannot_replace_newer_or_unreadable_report(tmp_path):
    from scripts.run_expected_exit_monitor import publish_status
    def report(at): return json.dumps({"schema_version":1,"account_key":"primary","generated_at":at}).encode()
    path=tmp_path/"status.json";path.write_bytes(report("2026-09-06T20:10:00Z"))
    class Client:
        body=report("2026-09-06T20:11:00Z")
        writes=[]
        def get_object(self,**kwargs): return {"Body":io.BytesIO(self.body),"ETag":"fixture-etag"}
        def put_object(self,**kwargs): self.writes.append(kwargs)
    client=Client()
    assert publish_status(path,client=client,bucket="fixture")=="newer_report_retained" and not client.writes
    client.body=report("2026-09-06T20:09:00Z")
    assert publish_status(path,client=client,bucket="fixture")=="published"
    assert client.writes[-1]["IfMatch"]=="fixture-etag"
    client.body=b"corrupt"
    with pytest.raises(ValueError): publish_status(path,client=client,bucket="fixture")


def test_observer_without_credentials_makes_no_broker_request(monkeypatch):
    monkeypatch.delenv("STATUS_TOKEN",raising=False)
    def forbidden(*args): pytest.fail("broker request without credentials")
    result,book,fills=observe("seed",catalog_from_source(as_of="2026-09-07T00:00:00Z"),
        inventory_loader=lambda **kwargs:TaggedInventory(),book_loader=forbidden,fills_loader=forbidden)
    assert result["status"]=="unknown" and book=={} and fills=={}


def test_current_status_publishes_even_if_email_is_uncertain(tmp_path,monkeypatch):
    from scripts import run_expected_exit_monitor as runner, monitor_expected_exits as monitor
    monkeypatch.setattr(runner, "load_read_environment", lambda *args: None)
    catalog=tmp_path/"catalog.json";catalog.write_text("{}")
    monkeypatch.setattr(runner,"observe",lambda *args:({},{},{}))
    def uncertain(argv):
        from pathlib import Path
        Path(argv[argv.index("--output")+1]).write_text(json.dumps({
            "schema_version":1,"account_key":"primary","generated_at":"2026-09-06T20:10:00Z"}))
        return 2
    monkeypatch.setattr(monitor,"main",uncertain)
    published=[]
    monkeypatch.setattr(runner,"publish_status",lambda path:published.append(path) or "published")
    code=runner.main(["--config-root",str(tmp_path),"--seed",str(tmp_path/"seed.json"),
        "--algorithm-catalog",str(catalog),"--state",str(tmp_path/"state.json"),
        "--artifacts",str(tmp_path/"runs"),"--upload"])
    assert code==2 and len(published)==1


def test_observer_reuses_coherent_fill_book_without_any_refresh(monkeypatch):
    monkeypatch.setenv("STATUS_TOKEN", "fixture")
    coherent = {"fills": [], "book": {"at": 42, "accounts": []}}
    def inventory(**kwargs):
        assert kwargs["fills_loader"]("unused", "unused") is coherent
        return TaggedInventory(status="unknown", reasons=["history gap"])
    def forbidden(*args):
        pytest.fail("coherent observation must not fetch another broker book")
    snapshot, book, fills = observe(None, catalog_from_source(as_of="2026-09-07T00:00:00Z"),
        inventory_loader=inventory, book_loader=forbidden, fills_loader=lambda *_: coherent)
    assert book is coherent["book"] and fills is coherent
    assert snapshot["reasons"] == ["history gap"]


def test_observer_failed_feed_disables_inventory_refresh_fallback(monkeypatch):
    monkeypatch.setenv("STATUS_TOKEN", "fixture")
    def unavailable(*args):
        raise ConnectionError("sensitive error must not be exposed")
    def inventory(**kwargs):
        assert kwargs["fills_loader"]("unused", "unused") == {}
        raise ValueError("sensitive inventory details")
    snapshot, book, fills = observe(None, catalog_from_source(as_of="2026-09-07T00:00:00Z"),
        inventory_loader=inventory, book_loader=unavailable, fills_loader=unavailable)
    assert fills == {} and book == {} and snapshot["status"] == "unknown"
    assert snapshot["reasons"] == ["reviewed inventory source failed (ValueError)"]


def test_read_environment_imports_only_read_broker_credentials(tmp_path, monkeypatch):
    from scripts.run_expected_exit_monitor import load_read_environment
    for name in ("STATUS_TOKEN", "EXEC_BROKER_URL", "EXEC_AGENT_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    env = tmp_path / "broker.env"
    (tmp_path / ".env").write_text("")
    env.write_text("STATUS_TOKEN=fixture-read\nEXEC_AGENT_TOKEN=fixture-write\nEXEC_BROKER_URL=https://fixture.invalid\n")
    load_read_environment(tmp_path, env)
    import os
    assert os.environ["STATUS_TOKEN"] == "fixture-read"
    assert os.environ["EXEC_BROKER_URL"] == "https://fixture.invalid"
    assert "EXEC_AGENT_TOKEN" not in os.environ


def test_producer_publishes_current_blocked_status_without_email(tmp_path, monkeypatch):
    from scripts import run_expected_exit_monitor as runner
    import daily_execution_report
    monkeypatch.setattr(runner, "load_read_environment", lambda *args: None)
    monkeypatch.setattr(runner, "observe", lambda *args: ({"status": "unknown",
        "reasons": ["verified history does not reach the reviewed seed"], "tranches": []}, {}, {}))
    monkeypatch.setattr(daily_execution_report, "send_email", lambda *args: pytest.fail("unexpected email"))
    published = []
    def publish(path):
        published.append(json.loads(path.read_text(encoding="utf-8")))
        return "published"
    monkeypatch.setattr(runner, "publish_status", publish)
    assert runner.main(["--config-root", str(tmp_path), "--state", str(tmp_path / "state.json"),
                        "--artifacts", str(tmp_path / "runs"), "--upload"]) == 0
    assert len(published) == 1 and published[0]["status"] == "degraded"
    assert published[0]["counts"]["unable_to_verify"] == 1
    assert "verified history does not reach" in published[0]["source_error"]
