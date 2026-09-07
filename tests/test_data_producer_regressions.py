"""Producer failure paths are exercised with isolated files and no external I/O."""
import importlib.util
import json
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tests.test_data_audit_regressions import ROOT, functions
from tests.test_sheets_io import Worksheet


def module(path):
    spec = importlib.util.spec_from_file_location("probe_" + path.replace("/", "_").replace(".", "_"), ROOT / path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.mark.parametrize("repull", ["short", "empty", "cliff", "complete"])
def test_basis_repair_never_splices_a_rejected_overlap(tmp_path, monkeypatch, repull):
    mod = module("scripts/update_master_prices.py")
    dates = pd.bdate_range(end="2026-09-04", periods=100)
    old = pd.DataFrame({"ticker": "TEST", "date": dates, "Open": 100., "High": 101., "Low": 99., "Close": 100., "Volume": 1000.})
    path = tmp_path / "master.parquet"
    old.to_parquet(path, index=False)
    all_changed = old.set_index("date").drop(columns="ticker")
    all_changed[["Open", "High", "Low", "Close"]] *= 2
    overlap = all_changed.tail(20)
    if repull == "short":
        repaired = {"TEST": overlap}
    elif repull == "empty":
        repaired = {}
    elif repull == "cliff":
        damaged = all_changed.copy()
        damaged.iloc[30:50, damaged.columns.get_indexer(["Open", "High", "Low", "Close"])] *= 3
        repaired = {"TEST": damaged}
    else:
        repaired = {"TEST": all_changed}
    responses = iter([{"TEST": overlap}, repaired])
    monkeypatch.setattr(mod, "PATH", str(path))
    monkeypatch.setattr(mod, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(mod, "_today", lambda: pd.Timestamp("2026-09-06"), raising=False)
    monkeypatch.setattr(mod, "download_chunk", lambda *args: next(responses))
    monkeypatch.setattr(mod.time, "sleep", lambda *_: None)
    monkeypatch.setattr(sys, "argv", ["update_master_prices.py", "--no-upload"])
    assert mod.main() == 0
    out = pd.read_parquet(path)
    assert out.Close.unique().tolist() == ([200.] if repull == "complete" else [100.])
    if repull != "complete":
        receipt = json.loads((tmp_path / "master.parquet.status.json").read_text())
        assert receipt["status"] == "degraded"
        assert receipt["unresolved_basis"] == ["TEST"]


def earnings_row():
    return {"date": "2026-09-01", "epsActual": 1., "epsEstimated": 1.,
            "revenueActual": 100., "revenueEstimated": 100., "lastUpdated": "2026-09-02"}


@pytest.mark.parametrize("baseline", ["absent", "corrupt"])
def test_earnings_no_trusted_baseline_never_publishes(tmp_path, monkeypatch, baseline):
    mod = module("scripts/build_earnings_calendar.py")
    path = tmp_path / "earnings.parquet"
    if baseline == "corrupt":
        path.write_bytes(b"not parquet")
    before = path.read_bytes() if path.exists() else None
    monkeypatch.setitem(sys.modules, "cache_io", SimpleNamespace(download_to_local=lambda *_: False))
    monkeypatch.setattr(mod, "fetch_ticker", lambda ticker, _: [earnings_row()] if ticker == "GOOD" else None)
    monkeypatch.setattr(mod, "SLEEP_BETWEEN_CALLS", 0)
    uploads = []
    monkeypatch.setattr(mod, "upload_to_r2", lambda *a, **k: uploads.append(a) or True)
    with pytest.raises(SystemExit, match="baseline"):
        mod.build_calendar(["GOOD", "FAILED"], "not-a-key", str(path))
    assert uploads == []
    assert (path.read_bytes() if path.exists() else None) == before


def test_earnings_failed_ticker_keeps_prior_events(tmp_path, monkeypatch):
    mod = module("scripts/build_earnings_calendar.py")
    path = tmp_path / "earnings.parquet"
    pd.DataFrame({"ticker": ["GOOD", "FAILED"], "date": pd.to_datetime(["2026-09-01", "2026-09-02"])}).to_parquet(path, index=False)
    monkeypatch.setattr(mod, "fetch_ticker", lambda ticker, _: [earnings_row()] if ticker == "GOOD" else None)
    monkeypatch.setattr(mod, "SLEEP_BETWEEN_CALLS", 0)
    monkeypatch.setattr(mod, "upload_to_r2", lambda *a, **k: True)
    mod.build_calendar(["GOOD", "FAILED"], "not-a-key", str(path))
    assert set(pd.read_parquet(path).ticker) == {"GOOD", "FAILED"}
    assert json.loads((tmp_path / "earnings.parquet.status.json").read_text())["status"] == "degraded"


def test_indicator_identity_tracks_interior_price_and_dependency_content(tmp_path):
    from cache_fingerprint import content_fingerprint
    fn = functions("pages/strat_backtester.py", ["_indicator_cache_path"], parent_dir=str(tmp_path), INDICATOR_CACHE_VERSION="test")["_indicator_cache_path"]
    prices = pd.DataFrame({"Close": [1.,2.,3.]}, index=pd.bdate_range("2026-01-01", periods=3))
    revised = prices.copy()
    revised.iloc[1, 0] += 1
    assert fn("A", prices, ()) != fn("A", revised, ())
    one = content_fingerprint({"SPY": prices})
    assert one != content_fingerprint({"QQQ": prices})
    assert one != content_fingerprint({"SPY": revised})
    assert fn("A", prices, (one,)) != fn("A", prices, (content_fingerprint({"SPY": revised}),))


def test_scan_coverage_uses_requested_universe_and_unknown_is_not_zero():
    from producer_status import scan_coverage
    receipt = scan_coverage(["A", "B", "C"], {"A": None, "B": None}, {"A": None}, {"B": "2026-09-01"}, [{"id": "S", "universe_tickers": ["A", "B", "C"]}])
    assert receipt["status"] == "degraded"
    assert receipt["missing"] == ["C"]
    assert receipt["available"] == 1
    assert receipt["strategies"][0]["unavailable"] == ["B", "C"]


@pytest.mark.parametrize("available", [False, True])
def test_fill_source_failure_preserves_state_and_processes_other_rows(tmp_path, monkeypatch, available):
    mod = module("verify_fills.py")
    date = pd.Timestamp.today().date().isoformat()
    columns = ["Date", "Ticker", "Time Exit", "Fill_Status", "Fill_Date", "Fill_Price", "Entry", "ATR", "Action"]
    rows = [[date, "MISSING", date, "PENDING", "", "", "100", "1", "BUY"],
            [date, "GOOD", date, "PENDING", "", "", "100", "1", "BUY"]]
    ws = Worksheet([columns] + rows)
    monkeypatch.setattr(mod, "__file__", str(tmp_path / "verify_fills.py"))
    monkeypatch.setattr(mod, "get_google_client", lambda: SimpleNamespace(open=lambda _: SimpleNamespace(sheet1=ws)))
    monkeypatch.setattr(mod, "build_strategy_map", lambda: {})
    monkeypatch.setattr(mod, "classify_order", lambda *a: ("TEST", 0, "DAY"))
    monkeypatch.setattr(mod, "fetch_price_data", lambda *a: {"GOOD": pd.DataFrame({"Close": [100.]})} if available else {})
    monkeypatch.setattr(mod, "check_fill", lambda **kw: ("FILLED", date, 100.))
    monkeypatch.setattr(mod.time, "sleep", lambda *_: None)
    if available:
        mod.run_fill_verification()
        assert ws.values[1] == rows[0]
        assert ws.values[2][3] == "FILLED"
    else:
        with pytest.raises(RuntimeError, match="existing fill states preserved"):
            mod.run_fill_verification()
        assert ws.requests == []
    assert json.loads((tmp_path / "data/fill_verification_status.json").read_text())["status"] == "degraded"


def test_unavailable_staging_preserves_original_obligation_and_dates(monkeypatch):
    import daily_scan
    ws = Worksheet([["Symbol", "Scan_Date", "Quantity"], ["MISSING", "2026-09-04", "12"], ["EVALUATED", "2026-09-04", "4"]])
    monkeypatch.setattr(daily_scan, "get_google_client", lambda: SimpleNamespace(open=lambda _: SimpleNamespace(worksheet=lambda _: ws)))
    daily_scan.save_staging_orders([], [], preserve_unavailable=["MISSING"])
    assert ws.values == [["Symbol", "Scan_Date", "Quantity"], ["MISSING", "2026-09-04", "12"]]


def test_signal_rescan_keeps_finalized_fill_and_frozen_prices(monkeypatch):
    import daily_scan
    headers = ['Ticker', 'Date', 'Strategy_ID', 'Entry', 'ATR', 'Fill_Status', 'Fill_Price', 'Fill_Date']
    original = ['SPY', '2026-09-01', 'S', '100', '2', 'FILLED', '99.5', '2026-09-02']
    ws = Worksheet([headers, original])
    monkeypatch.setattr(daily_scan, 'get_google_client', lambda: SimpleNamespace(open=lambda _: SimpleNamespace(sheet1=ws)))
    daily_scan.save_signals_to_gsheet(pd.DataFrame([{'Ticker':'SPY', 'Date':'2026-09-01', 'Strategy_ID':'S', 'Entry':90., 'ATR':1.}]))
    row = dict(zip(ws.values[0], ws.values[1]))
    assert row['Entry'] == '100'
    assert row['Fill_Status'] == 'FILLED'
    assert row['Fill_Price'] == '99.5'


def test_indicator_cache_recomputes_when_same_date_dependencies_change(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    engine = functions('pages/strat_backtester.py', [
        '_indicator_cache_has_required_schema', '_indicator_cache_path', 'precompute_all_indicators'],
        parent_dir=str(tmp_path), INDICATOR_CACHE_VERSION='test',
        _INDICATOR_CACHE_REQUIRED_COLUMNS={'pivot'}, ATR_SZNL_COLS=[],
        ThreadPoolExecutor=ThreadPoolExecutor, as_completed=as_completed)
    monkeypatch.setitem(sys.modules, 'cache_io', SimpleNamespace(is_configured=lambda: False, download_to_local=lambda *a: False))
    calls = []

    def calculate(frame, seasonal, ticker, market, vix, **kwargs):
        calls.append(ticker)
        output = frame.copy()
        for col in engine['_INDICATOR_CACHE_REQUIRED_COLUMNS']:
            output[col] = 1.
        output['dependency_result'] = seasonal['value'] + vix.iloc[-1]
        return output

    engine['calculate_indicators'] = calculate
    index = pd.bdate_range('2025-01-01', periods=260)
    frame = pd.DataFrame({'Close': np.arange(260)+100.}, index=index)
    vix = pd.Series(20., index=index)
    book = [{'id':'S','name':'S','universe_tickers':['A'],'settings':{}}]
    compute = engine['precompute_all_indicators']
    one = compute({'A':frame}, book, {'value':10.}, vix)
    again = compute({'A':frame}, book, {'value':10.}, vix)
    two = compute({'A':frame}, book, {'value':30.}, vix)
    vix.iloc[-1] = 40.
    three = compute({'A':frame}, book, {'value':30.}, vix)
    assert calls == ['A','A','A']
    assert [result['A'].dependency_result.iloc[-1] for result in [one,again,two,three]] == [30.,30.,50.,70.]


def test_cross_sectional_cache_matches_cold_result_after_membership_change(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    engine = functions('pages/strat_backtester.py', [
        '_indicator_cache_has_required_schema', '_indicator_cache_path', 'precompute_all_indicators'],
        parent_dir=str(tmp_path), INDICATOR_CACHE_VERSION='test',
        _INDICATOR_CACHE_REQUIRED_COLUMNS={'Close'}, ATR_SZNL_COLS=[],
        ThreadPoolExecutor=ThreadPoolExecutor, as_completed=as_completed,
        st=SimpleNamespace(empty=lambda: SimpleNamespace(text=lambda *_: None, empty=lambda: None)),
        calculate_indicators=lambda frame, *a, **kw: frame.copy())
    monkeypatch.setitem(sys.modules, 'cache_io', SimpleNamespace(is_configured=lambda: False, download_to_local=lambda *a: False))
    index = pd.bdate_range('2024-01-01', periods=400)
    rng = np.random.default_rng(123)
    frames = [pd.DataFrame({'Close':100*np.exp(np.cumsum(rng.normal(0,.01,len(index))))},index=index) for _ in range(3)]
    book = [{'id':'S','name':'S','universe_tickers':['A'], 'settings':{'use_xsec_filter':True, 'xsec_filters':[{'window':5}]}}]
    run = lambda inputs: engine['precompute_all_indicators'](inputs, book, {}, None)['A']['xsec_rank_ret_5d']
    first = run({'A':frames[0], 'B':frames[1]})
    changed = run({'A':frames[0], 'C':frames[2]})
    engine['parent_dir'] = str(tmp_path/'cold')
    cold = run({'A':frames[0], 'C':frames[2]})
    assert not first.equals(cold)
    pd.testing.assert_series_equal(changed, cold, check_freq=False)
