import hashlib
import json
import pandas as pd
import pytest
from alpha_calendar_snapshot import SnapshotError, daily_alpha
from scripts.compare_earnings_shadow import parse_alpha_csv

NOW = pd.Timestamp("2026-09-24T11:00:00Z")
RAW = "symbol,name,reportDate,fiscalDateEnding,estimate,currency\nAAA,A,2026-09-29,2026-08-31,1,USD\n"


class Store:
    def __init__(self):
        self.value = None
        self.etag = None
    def read(self, key):
        return self.value, self.etag
    def write(self, key, value, etag=None):
        if (etag is None and self.value is not None) or (etag is not None and etag != self.etag):
            raise SnapshotError("conflict")
        self.value = value.copy(); self.etag = str(int(self.etag or 0)+1)
        return self.etag


def call(tmp_path, store, fetch):
    return daily_alpha(config_root=tmp_path, store=store, fetch=fetch, parse=parse_alpha_csv, key="fake", now=NOW)


def test_repeated_producer_observer_calls_fetch_once(tmp_path):
    store = Store(); calls = []
    def fetch(key):
        calls.append(key); return RAW, parse_alpha_csv(RAW)
    a = call(tmp_path, store, fetch)
    b = call(tmp_path, store, fetch)
    assert len(calls) == 1 and a[0] == b[0] == RAW


def test_failed_attempt_does_not_allow_second_request(tmp_path):
    store = Store(); calls = []
    def fetch(key):
        calls.append(key); raise ValueError("quota")
    with pytest.raises(ValueError): call(tmp_path, store, fetch)
    with pytest.raises(SnapshotError, match="already attempted"): call(tmp_path, store, fetch)
    assert len(calls) == 1


def test_existing_authenticated_today_capture_is_seeded_without_request(tmp_path):
    p = tmp_path / "artifacts/earnings_shadow/authenticated/morning"; p.mkdir(parents=True)
    (p / "summary.json").write_text(json.dumps(dict(mode="authenticated", captured_at_utc=NOW.isoformat())))
    (p / "alpha_raw.csv").write_text(RAW)
    raw, _, meta = call(tmp_path, Store(), lambda key: pytest.fail("duplicate Alpha request"))
    assert raw == RAW and meta["origin"] == "existing_authenticated_observer"


def test_wrong_date_and_tampered_payload_rejected(tmp_path):
    store = Store()
    store.value = dict(state="ready", mode="authenticated", captured_at_utc="2026-09-23T11:00:00Z",
                       raw=RAW, sha256=hashlib.sha256(RAW.encode()).hexdigest())
    with pytest.raises(SnapshotError, match="wrong capture date"):
        call(tmp_path, store, lambda key: pytest.fail("invalid snapshot caused request"))
    store.value["captured_at_utc"] = NOW.isoformat(); store.value["raw"] += "tamper"
    with pytest.raises(SnapshotError, match="digest"):
        call(tmp_path, store, lambda key: pytest.fail("invalid snapshot caused request"))


def test_lost_claim_never_calls_provider(tmp_path):
    class Race(Store):
        def write(self, *args): raise SnapshotError("conflict")
    with pytest.raises(SnapshotError):
        call(tmp_path, Race(), lambda key: pytest.fail("lost claim called provider"))


def test_future_same_day_capture_is_rejected(tmp_path):
    store = Store()
    store.value = dict(state="ready", mode="authenticated", captured_at_utc="2026-09-24T23:00:00Z",
                       raw=RAW, sha256=hashlib.sha256(RAW.encode()).hexdigest())
    with pytest.raises(SnapshotError, match="future"):
        call(tmp_path, store, lambda key: pytest.fail("future capture caused a request"))
