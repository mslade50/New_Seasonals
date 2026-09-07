"""Offline regressions for the September whole-repository audit."""
import copy
import json
from pathlib import Path

import pandas as pd
import pytest
from concurrent.futures import ThreadPoolExecutor

import pitch_journal
import posts_journal
from fundamental.research_controls import apply_research_controls
from fundamental.research_state import _material_changes
from fundamental.site_payload import build_fundamental_site_payload
from fundamental.underwrite import validate_underwrite_record
from scripts import build_fundamental_report
from scripts.configure_shared_access import configure_access, TARGET_DOMAIN, PREVIEW_DOMAIN
from fundamental.research_controls import completed_diligence_requests


@pytest.mark.parametrize("journal,kind", [(pitch_journal, "idea"), (posts_journal, "draft")])
def test_corrupt_journal_is_preserved_and_cannot_accept_append(tmp_path, journal, kind):
    path = tmp_path / "journal.jsonl"
    original = json.dumps({"kind": kind, "id": "old"}) + '\n{"kind":'
    path.write_text(original, encoding="utf-8")
    with pytest.raises(ValueError, match="corrupt|invalid|Malformed"):
        journal.append([{"kind": kind, "id": "new"}], path, push=False)
    assert path.read_text(encoding="utf-8") == original
    with pytest.raises(ValueError, match="corrupt|invalid|Malformed"):
        journal.load(path, pull=False)


@pytest.mark.parametrize("journal,kind", [(pitch_journal, "idea"), (posts_journal, "draft")])
def test_complete_unterminated_journal_line_is_separated(tmp_path, journal, kind):
    path = tmp_path / "journal.jsonl"
    path.write_text(json.dumps({"kind": kind, "id": "old"}), encoding="utf-8")
    journal.append([{"kind": kind, "id": "new"}], path, push=False)
    assert [r["id"] for r in journal.load(path, pull=False)] == ["old", "new"]


def test_concurrent_journal_batches_are_never_interleaved_or_lost(tmp_path):
    path = tmp_path / "journal.jsonl"
    def append(batch):
        pitch_journal.append([{"kind": "idea", "batch": batch, "sequence": i} for i in range(25)], path, push=False)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(append, range(8)))
    records = pitch_journal.load(path, pull=False)
    assert len(records) == 200
    assert len({(r["batch"], r["sequence"]) for r in records}) == 200
    assert all(len({r["batch"] for r in records[start:start + 25]}) == 1 for start in range(0, 200, 25))


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_nonfinite_jsonl_is_preserved_and_blocks_append(tmp_path, constant):
    from research_io import append_jsonl, read_jsonl
    path = tmp_path / "invalid.jsonl"
    original = ('{"value":' + constant + '}\n').encode()
    path.write_bytes(original)
    with pytest.raises(ValueError, match="corrupt"):
        read_jsonl(path)
    with pytest.raises(ValueError, match="corrupt"):
        append_jsonl(path, [{"value": 1}])
    assert path.read_bytes() == original


def test_nonobject_append_rejects_whole_batch_before_writing(tmp_path):
    from research_io import append_jsonl
    path = tmp_path / "journal.jsonl"
    with pytest.raises(ValueError, match="object"):
        append_jsonl(path, [{"good": 1}, ["bad"]])
    assert not path.exists()


@pytest.mark.parametrize("action", ["PASS", "WATCH"])
def test_suppressed_review_is_archived_not_in_current_inbox(tmp_path, action, v2_underwrite_factory):
    candidate = apply_research_controls(
        pd.DataFrame([{"ticker": "AAA", "research_queue_priority": 1}]),
        {"AAA": {"action": action, "updated_at": "2026-08-05T12:00:00Z"}},
    ).iloc[0].to_dict()
    path = tmp_path / "daily.json"
    path.write_text(json.dumps({"health": {"as_of": "2026-08-05"},
                               "candidates": [candidate],
                               "underwrite_decisions": [v2_underwrite_factory()]}, default=str))
    payload = build_fundamental_site_payload(path, tmp_path / "missing.json")
    assert payload["reviews"] == []
    assert payload["active_research"] == []
    assert payload["archived_research"][0]["ticker"] == "AAA"


def test_no_current_candidate_cannot_reappear_as_review(tmp_path, v2_underwrite_factory):
    path = tmp_path / "daily.json"
    path.write_text(json.dumps({"health": {"as_of": "2026-08-05"}, "candidates": [],
                               "underwrite_decisions": [v2_underwrite_factory()]}))
    assert build_fundamental_site_payload(path, tmp_path / "missing.json")["reviews"] == []


def test_old_material_evidence_flag_cannot_bypass_consumption():
    payload = {"schema_version": "fundamental-evidence.v1", "updated_at": "2026-08-05T12:00:00Z",
               "evidence": [{"evidence_id": "e-old", "claim_id": "claim-old", "ticker": "AAA",
                             "claim": "A material research fact changed.", "source_id": "filing",
                             "direction": "CONFIRM", "materiality": "THESIS_CHANGING",
                             "observed_at": "2026-08-03T10:00:00Z", "new_since_last_run": True}]}
    result = _material_changes(payload, as_of="2026-08-05", since="2026-08-04T20:00:00Z")
    assert result["changed_tickers"] == []


@pytest.mark.parametrize("defect", ["currency", "ev", "price_source"])
def test_inconsistent_underwrite_cannot_promote(defect, v2_underwrite_factory):
    record = v2_underwrite_factory()
    if defect == "currency":
        record["valuation"]["currency"] = "JPY"
    elif defect == "ev":
        record["price_snapshot"]["enterprise_value"] = -1_000_000
    else:
        record["sources"][0]["as_of"] = "2020-01-01"
    assert not validate_underwrite_record(record)["valid_for_quick_review"]


def test_historical_snapshot_restores_issuer_missing_from_current(tmp_path, monkeypatch):
    current = pd.DataFrame([{"ticker": "AAA", "snapshot_as_of": "2026-08-06", "value": 2},
                            {"ticker": "BBB", "snapshot_as_of": "2026-08-04", "value": 3}])
    current.to_parquet(tmp_path / "fmp_latest.parquet", index=False)
    archived = pd.DataFrame([{"ticker": "AAA", "snapshot_as_of": "2026-08-04", "value": 1},
                             {"ticker": "BBB", "snapshot_as_of": "2026-08-04", "value": 3}])
    monkeypatch.setattr(build_fundamental_report, "CURRENT_ROOT", tmp_path)
    monkeypatch.setattr(build_fundamental_report, "load_latest_snapshot_parts", lambda *a: archived)
    result = build_fundamental_report._load_current_or_snapshots("fmp", "2026-08-05", None)
    assert set(result["ticker"]) == {"AAA", "BBB"}
    assert result.set_index("ticker").loc["AAA", "value"] == 1


@pytest.mark.parametrize("policies", [
    [{"name": "Denali Team", "decision": "allow", "include": [{"everyone": {}}]}],
    [{"name": "Denali Team", "decision": "allow", "include": [{"email": {"email": "one@example.com"}}]},
     {"name": "Public", "decision": "bypass", "include": [{"everyone": {}}]}],
])
def test_shared_access_rejects_public_effective_policy(policies):
    class Client:
        def list_apps(self):
            return [{"id": "prod", "domain": TARGET_DOMAIN}, {"id": "preview", "domain": PREVIEW_DOMAIN}]
        def list_policies(self, _app_id):
            return copy.deepcopy(policies)
    with pytest.raises(RuntimeError):
        configure_access(Client())


@pytest.mark.parametrize("kind,observed,suppressed", [
    ("REOPEN", "2026-08-04T12:00:00Z", True),
    ("KILL", "2026-08-06T12:00:00Z", True),
    ("WARNING", "2026-08-06T12:00:00Z", True),
    ("PROOF", "2026-08-06T12:00:00Z", False),
    ("REOPEN", "2026-08-06T12:00:00Z", False),
])
def test_only_new_constructive_trigger_reopens_watch(kind, observed, suppressed):
    result = apply_research_controls(pd.DataFrame([{"ticker": "AAA", "research_queue_priority": 1}]),
        {"AAA": {"action": "WATCH", "updated_at": "2026-08-05T12:00:00Z"}},
        trigger_events=[{"ticker": "AAA", "evaluation": "FIRED", "kind": kind, "observed_at": observed}])
    assert bool(result.iloc[0]["research_suppressed"]) is suppressed


def test_deepen_is_consumed_only_by_matching_completed_diligence(v2_underwrite_factory):
    controls = {"AAA": {"action": "DEEPEN", "updated_at": "2026-08-05T12:00:00Z"}}
    record = v2_underwrite_factory()
    assert completed_diligence_requests([record], controls) == {}
    record.update(research_control_updated_at=controls["AAA"]["updated_at"], completed_at="2026-08-05T15:00:00Z")
    consumed = completed_diligence_requests([record], controls)
    pending = apply_research_controls(pd.DataFrame([{"ticker": "AAA", "research_queue_priority": 5}]), controls)
    assert pending.iloc[0]["research_queue_priority"] == 10_000
    result = apply_research_controls(pending,
                                     controls, completed_control_requests=consumed)
    assert result.iloc[0]["research_queue_priority"] == 5
    assert result.iloc[0]["control_disposition"] == "COMPLETED_BOUNDED_DILIGENCE_PASS"
    controls["AAA"]["updated_at"] = "2026-08-06T12:00:00Z"
    assert completed_diligence_requests([record], controls) == {}


def test_report_suppression_matches_site_and_keeps_original_decision(tmp_path, v2_underwrite_factory):
    from fundamental.report import render_candidate_report
    record = v2_underwrite_factory()
    candidate = apply_research_controls(pd.DataFrame([{
        "ticker": "AAA", "research_priority": "A", "source_posture": "SEC package", "research_queue_priority": 1,
    }]), {"AAA": {"action": "PASS", "updated_at": "2026-08-05T12:00:00Z"}})
    output = tmp_path / "report.html"
    render_candidate_report(candidate, {"as_of": "2026-08-05"}, output, underwrite_decisions=[record])
    assert record["verdict"] not in output.read_text(encoding="utf-8")
    assert record["decision"] == "QUICK_REVIEW"  # archival evidence is untouched


def test_capital_structure_units_and_sourced_ev_adjustments(v2_underwrite_factory):
    record = v2_underwrite_factory()
    record["price_snapshot"].update(diluted_shares=100, share_unit_multiplier=1e6,
        money_unit_multiplier=1e6, net_debt=-1200, enterprise_value=-150,
        ev_adjustments=[{"label": "Minority interest", "amount": 50, "source_ids": ["filing"]}])
    # Legitimately negative EV: 10*100 - 1200 + 50 = -150 (all in millions).
    assert validate_underwrite_record(record)["valid_for_quick_review"]
    record["price_snapshot"]["ev_adjustments"][0]["source_ids"] = ["unknown"]
    assert not validate_underwrite_record(record)["valid_for_quick_review"]


def test_mixed_historical_dataset_parts_keep_statement_periods(tmp_path, monkeypatch):
    current = pd.DataFrame([
        {"ticker": "AAA", "endpoint": "income", "snapshot_as_of": "2026-08-06", "period": 2025},
        {"ticker": "AAA", "endpoint": "balance", "snapshot_as_of": "2026-08-04", "period": 2025},
    ])
    current.to_parquet(tmp_path / "fmp_latest.parquet", index=False)
    archive = pd.DataFrame([
        {"ticker": "AAA", "endpoint": "income", "snapshot_as_of": "2026-08-03", "period": period}
        for period in (2024, 2025)
    ])
    monkeypatch.setattr(build_fundamental_report, "CURRENT_ROOT", tmp_path)
    monkeypatch.setattr(build_fundamental_report, "load_latest_snapshot_parts", lambda *a: archive)
    result = build_fundamental_report._load_current_or_snapshots("fmp", "2026-08-05", ["AAA"])
    assert len(result) == 3
    assert result.groupby("endpoint").size().to_dict() == {"balance": 1, "income": 2}


def test_access_verification_reads_all_policy_pages(monkeypatch):
    from scripts.configure_shared_access import CloudflareAccessClient
    client = CloudflareAccessClient("synthetic", "not-a-token")
    calls = []
    def request(method, path, **kwargs):
        calls.append(path)
        return {"result": [{"page": len(calls)}], "result_info": {"total_pages": 2}}
    monkeypatch.setattr(client, "request", request)
    assert client.list_policies("app") == [{"page": 1}, {"page": 2}]
    assert calls[-1].endswith("page=2")



def test_raw_replay_loader_never_uses_adjusted_bars():
    from research_price_history import load_raw_prices
    from scripts.grade_pitch_journal import replay_leg
    def download(tickers, **kwargs):
        assert kwargs["auto_adjust"] is False and kwargs["back_adjust"] is False
        assert kwargs["start"] == "2026-08-06" and kwargs["end"] == "2026-08-08"
        assert tickers == ["AAA"]
        values = pd.DataFrame([[100, 102, 99, 101], [102, 104, 100, 103]],
                              index=pd.to_datetime(["2026-08-06", "2026-08-07"]), columns=["Open", "High", "Low", "Close"])
        values.columns = pd.MultiIndex.from_product([values.columns, ["AAA"]], names=["Price", "Ticker"])
        return values
    prices = load_raw_prices(["AAA"], start="2026-08-06", end="2026-08-08", download=download)
    row = {"Ticker": "AAA", "Execute_On": "2026-08-06", "Action": "BUY", "ATR": 1,
           "Quantity": 100, "Entry_Type": "LIMIT", "Limit_Price": 95, "Entry_Expire_Date": "2026-08-07"}
    assert replay_leg(prices.set_index("date"), row)["status"] == "no_fill"
    assert prices.attrs["price_basis"] == "RAW"
    with pytest.raises(ValueError, match="unavailable"):
        load_raw_prices(["AAA"], start="2026-08-06", end="2026-08-08", download=lambda *a, **k: pd.DataFrame())


def test_ep_ambiguous_data_delivery_blocks_retry(tmp_path, monkeypatch):
    from episodic_pivot import email_delivery as delivery
    class SMTP:
        accepted = 0
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def ehlo(self): pass
        def starttls(self, **kwargs): pass
        def login(self, *args): pass
        def send_message(self, message):
            type(self).accepted += 1
            raise OSError("disconnect while awaiting DATA acknowledgement")
    monkeypatch.setattr(delivery.smtplib, "SMTP", SMTP)
    payload = delivery.test_payload(output_root=tmp_path)
    settings = delivery.EmailSettings("sender@example.com", "fake", ("user@example.com",))
    with pytest.raises(delivery.EmailDeliveryError, match="ambiguous"):
        delivery.deliver_email(payload, settings, send=True)
    for resend in (False, True):
        with pytest.raises(delivery.EmailDeliveryError, match="ambiguous"):
            delivery.deliver_email(payload, settings, send=True, resend=resend)
    assert SMTP.accepted == 1
    assert json.loads(payload.receipt_path.read_text())["status"] == "AMBIGUOUS"


def test_ep_quit_failure_after_acceptance_does_not_resend(tmp_path, monkeypatch):
    from episodic_pivot import email_delivery as delivery
    class SMTP:
        accepted = 0
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): raise OSError("QUIT disconnected after acceptance")
        def ehlo(self): pass
        def starttls(self, **kwargs): pass
        def login(self, *args): pass
        def send_message(self, message): type(self).accepted += 1
    monkeypatch.setattr(delivery.smtplib, "SMTP", SMTP)
    payload = delivery.test_payload(output_root=tmp_path)
    settings = delivery.EmailSettings("sender@example.com", "fake", ("user@example.com",))
    assert delivery.deliver_email(payload, settings, send=True) == "SENT"
    assert delivery.deliver_email(payload, settings, send=True) == "ALREADY_SENT"
    assert SMTP.accepted == 1
