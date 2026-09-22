from copy import deepcopy
from datetime import datetime, timezone

import pytest

from episodic_pivot import short_discovery as sd

SETTINGS = {"min_price": 10, "min_vol": 100000, "vol_thresh": 2}
TARGET = "2026-09-22"
PRIOR = "2026-09-21"
NOW = datetime(2026, 9, 22, 12, 30, tzinfo=timezone.utc)
BAR_TIME = datetime(2026, 9, 21, 13, 30, tzinfo=timezone.utc).timestamp()


@pytest.fixture(autouse=True)
def small_fixture(monkeypatch):
    monkeypatch.setattr(sd, "MIN_ROWS", 1)


def row(symbol="TEST", exchange="NASDAQ", *, volume=300000, avg60=100000, price=30, sma50=20, bar_time=BAR_TIME):
    return {"s": f"{exchange}:{symbol}", "d": [symbol, "Company ADR", exchange, price, volume, avg60, sma50, bar_time]}


def response(*rows):
    return {"totalCount": len(rows), "data": list(rows)}


def listings(*symbols):
    return {symbol: {"exchange": "NASDAQ"} for symbol in symbols}


def evidence():
    return {"schema": sd.SCHEMA, "session_date": TARGET, "url": sd.URL,
            "captured_at": "2026-09-22T12:15:00Z", "request": sd.request_for(SETTINGS),
            "response": response(row())}


def test_query_keeps_adrs_and_has_no_change_rank_or_static_tickers():
    query = sd.request_for(SETTINGS)
    assert query["symbols"] == {"query": {"types": []}, "tickers": []}
    assert query["filter"] == [
        {"left": "close", "operation": "egreater", "right": 9.5},
        {"left": "volume", "operation": "egreater", "right": 180000}]
    assert query["range"] == [0, sd.MAX_ROWS]


@pytest.mark.parametrize("including_signal", [True, False])
@pytest.mark.parametrize("signal_volume", [210000, 300000, 2000000])
@pytest.mark.parametrize("old_volume", [0, 100000, 2000000])
def test_loose_volume_bound_preserves_exact_qualifiers(including_signal, signal_volume, old_volume):
    history = [old_volume] * 3 + [100000] * 59 + [signal_volume]
    avg63 = sum(history) / 63
    avg60 = sum(history[-60:] if including_signal else history[-61:-1]) / 60
    targets, _ = sd.select_targets(response(row(volume=signal_volume, avg60=avg60)), listings("TEST"), SETTINGS, PRIOR)
    if avg63 >= SETTINGS["min_vol"] and signal_volume > SETTINGS["vol_thresh"] * avg63:
        assert targets == ["TEST"]


def test_loose_volume_boundary_and_sma_buffer():
    boundary = 0.9 * 2 * (60 / 63) * 100000
    source = response(row("BOUNDARY", volume=boundary), row("ABOVE", volume=boundary + 1),
                      row("LOW", price=19.59), row("BUFFER", price=19.6))
    targets, stats = sd.select_targets(source, listings("BOUNDARY", "ABOVE", "LOW", "BUFFER"), SETTINGS, PRIOR)
    assert targets == ["ABOVE", "BUFFER"]
    assert stats["volume_filtered"] == stats["below_sma50_filtered"] == 1


@pytest.mark.parametrize("metric", ["volume", "avg60", "price", "sma50"])
def test_unknown_metrics_reach_exact_check(metric):
    targets, stats = sd.select_targets(response(row(**{metric: None})), listings("TEST"), SETTINGS, PRIOR)
    assert targets == ["TEST"] and stats["unknown_metrics_retained"] == 1


def test_exchange_identity_excludes_etfs_unlisted_and_wrong_venue():
    source = response(row(), row("ETF"), row("TEST", exchange="OTC"))
    targets, stats = sd.select_targets(source, listings("TEST"), SETTINGS, PRIOR)
    assert targets == ["TEST"] and stats["outside_listing_scope"] == 2


@pytest.mark.parametrize("kind", ["truncated", "duplicate", "empty", "tiny", "oversized", "identity", "columns", "row_type", "response_type"])
def test_untrustworthy_response_fails(kind, monkeypatch):
    source = response(row())
    if kind == "truncated": source["totalCount"] = 2
    elif kind == "duplicate": source = response(row(), row())
    elif kind == "empty": source = response()
    elif kind == "tiny": monkeypatch.setattr(sd, "MIN_ROWS", 2)
    elif kind == "oversized": monkeypatch.setattr(sd, "MAX_ROWS", 0)
    elif kind == "identity": source["data"][0]["s"] = "NYSE:TEST"
    elif kind == "columns": source["data"][0]["d"].pop()
    elif kind == "row_type": source["data"][0] = None
    else: source = []
    with pytest.raises(ValueError):
        sd.select_targets(source, listings("TEST"), SETTINGS, PRIOR)


def test_stale_feed_unavailable_but_isolated_unknown_date_retained():
    with pytest.raises(ValueError, match="prior NYSE"):
        sd.select_targets(response(row(bar_time=None)), listings("TEST"), SETTINGS, PRIOR)
    names = [f"T{i}" for i in range(10)]
    source = response(*[row(symbol) for symbol in names[:9]], row(names[-1], bar_time=None, volume=1))
    targets, stats = sd.select_targets(source, listings(*names), SETTINGS, PRIOR)
    assert targets == sorted(names) and stats["unverified_dates_retained"] == 1


def test_capacity_failure_never_truncates(monkeypatch):
    monkeypatch.setattr(sd, "MAX_HISTORY_TARGETS", 1)
    with pytest.raises(ValueError, match="2 targets, exceeding capacity 1"):
        sd.select_targets(response(row("AAA"), row("BBB")), listings("AAA", "BBB"), SETTINGS, PRIOR)


def test_valid_capture_and_empty_shortlist_are_distinct_from_missing_feed():
    assert sd.validate_discovery(evidence(), TARGET, listings("TEST"), SETTINGS, now=NOW)[0] == ["TEST"]
    packet = evidence()
    packet["response"]["data"][0] = row(volume=200000, avg60=1000000)
    assert sd.validate_discovery(packet, TARGET, listings("TEST"), SETTINGS, now=NOW)[0] == []


@pytest.mark.parametrize("kind", ["future", "previous", "after_open", "before_premarket", "query", "source", "wrong_bar_date"])
def test_capture_provenance_gates(kind):
    packet = deepcopy(evidence())
    if kind == "future": packet["captured_at"] = "2026-09-22T12:31:00Z"
    elif kind == "previous": packet["captured_at"] = "2026-09-21T12:15:00Z"
    elif kind == "after_open": packet["captured_at"] = "2026-09-22T13:30:00Z"
    elif kind == "before_premarket": packet["captured_at"] = "2026-09-22T07:59:00Z"
    elif kind == "query": packet["request"]["filter"][0]["right"] = 20
    elif kind == "source": packet["url"] = "https://example.com"
    else: packet["response"]["data"][0]["d"][-1] += 86400
    with pytest.raises(ValueError):
        sd.validate_discovery(packet, TARGET, listings("TEST"), SETTINGS, now=NOW)


def test_afternoon_capture_stops_before_network(monkeypatch):
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 9, 22, 20, 30, tzinfo=timezone.utc)

    monkeypatch.setattr(sd, "datetime", Clock)
    monkeypatch.setattr(sd.requests, "post", lambda *_args, **_kwargs: pytest.fail("Network called after hours"))
    with pytest.raises(ValueError, match="premarket-only"):
        sd.capture_discovery(TARGET, SETTINGS)
