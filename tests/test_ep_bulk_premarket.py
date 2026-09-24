import copy
import json
from datetime import datetime, timezone

import pytest

from episodic_pivot import bulk_premarket as bulk
from episodic_pivot import morning_completion as mc
from episodic_pivot.premarket import premarket_move_is_verified
from episodic_pivot.schema import PremarketSnapshot
from scripts.capture_ep_daily_yfinance import _load_discovery_inputs
from scripts import prepare_ep_morning as prep

NOW = datetime(2026, 9, 24, 12, 20, tzinfo=timezone.utc)
TARGET = "2026-09-24"


def evidence():
    value = {"record_type": bulk.RECORD_TYPE, "provider": "TRADINGVIEW", "session": "premarket",
             "target_session_date": TARGET, "started_at": NOW.isoformat(), "captured_at": NOW.isoformat(),
             "url": bulk.URL, "request": bulk.request_body(),
             "response": {"totalCount": 1, "data": [{"s": "NASDAQ:DEMO", "d": [
                 "DEMO", "Fictional test issuer", "NASDAQ", "stock", "common", 10,
                 11, 10, 1, 250000, NOW.replace(hour=8).timestamp()]}]}}
    value["snapshots"] = bulk.normalize(value)
    return value


def test_complete_public_response_replays_into_existing_atr_pipeline(tmp_path):
    raw = evidence()
    path = tmp_path / "capture.json"
    path.write_text(json.dumps(raw))
    snapshots, target, count, inputs, warnings = _load_discovery_inputs([path])
    assert target == TARGET and count == len(snapshots) == 1
    row = snapshots[0]
    assert row.source == bulk.SOURCE and not row.tradeable
    assert row.previous_close == 10 and row.discovery_gap_pct == 10
    assert premarket_move_is_verified(row, as_of=NOW, require_fresh_at_as_of=True)
    assert inputs[0]["path"] == str(path)
    assert "TRADINGVIEW_PREMARKET_NOT_INCLUDED" not in warnings


@pytest.mark.parametrize("mutation", ["count", "empty", "duplicate", "old_session", "future_bar", "bad_pct",
                                    "query", "field_count", "nan", "identity", "negative_volume", "slow", "after_open"])
def test_unverified_feed_never_becomes_discovery(mutation):
    raw = evidence()
    row = raw["response"]["data"][0]
    if mutation == "count": raw["response"]["totalCount"] = 2
    elif mutation == "empty": raw["response"] = {"totalCount": 0, "data": []}
    elif mutation == "duplicate": raw["response"]["data"].append(copy.deepcopy(row)); raw["response"]["totalCount"] = 2
    elif mutation == "old_session": row["d"][-1] -= 86400
    elif mutation == "future_bar": row["d"][-1] = NOW.replace(hour=13).timestamp()
    elif mutation == "bad_pct": row["d"][7] = 25
    elif mutation == "query": raw["request"]["range"][1] = 10
    elif mutation == "field_count": row["d"].pop()
    elif mutation == "nan": row["d"][6] = float("nan")
    elif mutation == "identity": row["s"] = "NYSE:OTHER"
    elif mutation == "negative_volume": row["d"][9] = -1
    elif mutation == "slow": raw["captured_at"] = NOW.replace(minute=22).isoformat()
    elif mutation == "after_open": raw["started_at"] = raw["captured_at"] = NOW.replace(hour=14).isoformat()
    with pytest.raises(ValueError): bulk.normalize(raw)


def test_edited_snapshot_is_rejected_by_daily_ingestion(tmp_path):
    raw = evidence()
    raw["snapshots"][0]["last"] = 999
    path = tmp_path / "capture.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="differ"):
        _load_discovery_inputs([path])


def test_drs_supported_without_preferred_stock_leakage():
    raw = evidence()
    raw["response"]["data"][0]["d"][3:5] = ["dr", ""]
    assert len(bulk.normalize(raw)) == 1
    raw["response"]["data"][0]["d"][3:5] = ["stock", "preferred"]
    with pytest.raises(ValueError, match="no verified equity"):
        bulk.normalize(raw)


def test_capture_records_real_http_response_and_request_clock():
    class Response:
        def raise_for_status(self): pass
        def json(self): return evidence()["response"]
    calls = []
    def post(url, **kwargs):
        calls.append((url, kwargs))
        return Response()
    raw = bulk.capture(now_fn=lambda: NOW, post=post)
    bulk.validate_capture(raw)
    assert calls == [(bulk.URL, {"json": bulk.request_body(), "timeout": 30})]


def test_retry_then_success_freezes_once_and_preserves_notes(tmp_path):
    root = tmp_path / "ep"
    def outage(): raise OSError("synthetic provider outage")
    failed = prep.prepare(root, now_fn=lambda: NOW, capture_fn=outage)
    assert failed["preparation"] == "RETRY_PENDING" and failed["status"] == "RESUME"
    assert not list(root.rglob("email_delivery.json"))
    calls = []
    def step(script, args, log):
        calls.append(script)
        key = "--output" if script.startswith("capture") else "--prepare-google-review"
        args[args.index(key) + 1].write_text(json.dumps({"coverage": {
            "broad_nomination_count": 1, "daily_metrics_verified_count": 1}}))
    ready = prep.prepare(root, now_fn=lambda: NOW, capture_fn=evidence, step_fn=step)
    assert ready["preparation"] == "QUEUE_FROZEN"
    notes = root / "notes.json"
    notes.write_text("[]")
    mc.checkpoint(root, TARGET, "RESEARCH", {"notes": notes}, now=NOW)
    again = prep.prepare(root, now_fn=lambda: NOW, capture_fn=outage)
    assert again["preparation"] == "QUEUE_ALREADY_FROZEN"
    assert len(calls) == 2 and "notes" in again["progress"]["artifacts"]


def test_preparation_never_runs_after_deadline_or_during_pause(tmp_path):
    def forbidden(): pytest.fail("must not capture")
    late = prep.prepare(tmp_path, now_fn=lambda: NOW.replace(hour=14), capture_fn=forbidden)
    assert late["status"] == "DEADLINE_MISSED"
    mc.checkpoint(tmp_path, TARGET, "PAUSED_BY_USER", {}, now=NOW)
    assert prep.prepare(tmp_path, now_fn=lambda: NOW, capture_fn=forbidden)["status"] == "PAUSED_BY_USER"


def test_total_daily_outage_keeps_retry_open(tmp_path):
    calls = []
    def step(script, args, log):
        calls.append(script)
        args[args.index("--output") + 1].write_text(json.dumps({"coverage": {
            "broad_nomination_count": 1, "daily_metrics_verified_count": 0}}))
    state = prep.prepare(tmp_path, now_fn=lambda: NOW, capture_fn=evidence, step_fn=step)
    assert state["preparation"] == "RETRY_PENDING"
    assert len(calls) == 1 and "queue" not in state["progress"]["artifacts"]


def test_public_capture_through_daily_atr_to_full_research_queue(tmp_path):
    import pandas as pd
    from episodic_pivot.config import DEFAULT_POLICY
    from episodic_pivot.daily_prices import enrich_snapshots_from_yfinance
    from episodic_pivot.pipeline import run_shadow_pipeline
    from episodic_pivot.reviewed_news import make_queue
    from trading_calendar import TRADING_DAY

    path = tmp_path / "capture.json"
    path.write_text(json.dumps(evidence()))
    rows, target, *_ = _load_discovery_inputs([path])
    frame = pd.DataFrame({"Open": 10.0, "High": 10.3, "Low": 9.7, "Close": 10.0,
                          "Volume": 1000000}, index=pd.date_range(end="2026-09-23", periods=150, freq=TRADING_DAY))
    frame.columns = pd.MultiIndex.from_tuples([(c, "DEMO") for c in frame.columns], names=["Price", "Ticker"])
    enriched = enrich_snapshots_from_yfinance(rows, session_date=NOW.date(),
        download=lambda *a, **kw: frame, fetched_at=NOW)
    assert enriched.verified_count == 1
    assert enriched.snapshots[0].prior_atr_pct > 4
    result = run_shadow_pipeline(list(enriched.snapshots), as_of=NOW,
                                target_session_date=target, policy=DEFAULT_POLICY)
    queue = make_queue(result.candidates, prepared_at=NOW, target_session_date=target, policy=DEFAULT_POLICY)
    assert [t["symbol"] for t in queue["targets"]] == ["DEMO"]
    assert not result.previews
