"""Offline regressions for the September whole-repository audit."""
import copy
import json
from pathlib import Path

import pandas as pd
import pytest
from concurrent.futures import ThreadPoolExecutor

import pitch_journal
import posts_journal
from scripts.configure_shared_access import configure_access, TARGET_DOMAIN, PREVIEW_DOMAIN


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
