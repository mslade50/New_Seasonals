"""Guard: PM book side (pm_agent_book) and the daily check-in (scripts/pm_daily_check.py)."""
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pm_agent_book as B  # noqa: E402
import pm_agent_data as pad  # noqa: E402
import pm_agent_grammar as G  # noqa: E402
import pm_agent_universe as U  # noqa: E402


def _snap(session, nlv, positions=(), orders=()):
    return {"session": session, "observed_at": f"{session}T20:05:00Z",
            "book": {"accounts": [{"key": "primary", "nlv": nlv, "positions": list(positions),
                                   "orders": list(orders)},
                                  {"key": "pa", "nlv": 1.0, "positions": []}]}}


def _pos(sym, qty, px, sec="STK", mult=None):
    p = {"symbol": sym, "sec_type": sec, "position": qty, "market_price": px,
         "market_value": qty * px, "unrealized_pnl": 0.0}
    if mult:
        p["multiplier"] = mult
        p["market_value"] = 0.0
    return p


def _fill(sym, side, qty, ref_action, strategy, session="2026-10-09", acct="primary"):
    return {"exec_id": f"{sym}-{side}-{qty}-{session}", "session_date": session, "account_key": acct,
            "symbol": sym, "side": side, "qty": float(qty), "price": 10.0, "ref_action": ref_action,
            "strategy": strategy, "realized_pnl": 0.0, "commission": 1.0}


# ---------------------------------------------------------------------------
# live
# ---------------------------------------------------------------------------
def test_nav_and_vol_exclude_suspected_flows():
    s = [_snap(f"2026-09-{d:02d}", v) for d, v in zip(range(14, 30), np.linspace(600000, 606000, 16))]
    s[8]["book"]["accounts"][0]["nlv"] = s[7]["book"]["accounts"][0]["nlv"] * 1.10   # a deposit
    for x in s[9:]:
        x["book"]["accounts"][0]["nlv"] *= 1.10
    nav = B.nav_series(s)
    v = B.live_vol(nav)
    assert len(v["suspected_flows"]) == 1 and v["suspected_flows"][0]["chg_pct"] > 9
    assert v["ann_vol_all_pct"] < 2          # the flow is not counted as volatility
    assert v["n_returns"] == 14


def test_live_exposure_signs_futures_and_orders():
    snap = _snap("2026-10-09", 500000.0,
                 [_pos("AAA", 1000, 50.0), _pos("BBB", -500, 40.0), _pos("MNQ", -2, 20000.0, "FUT", 2.0)],
                 [{"action": "SELL", "remaining": 100, "lmt": 50.0}, {"action": "BUY", "remaining": 0, "lmt": 9}])
    e = B.live_exposure(snap)
    assert e["long_pct"] == 10.0 and e["short_pct"] == pytest.approx(4 + 16)
    assert e["net_pct"] == pytest.approx(10 - 20)
    assert e["working_orders"] == 1 and e["working_sell_pct"] == 1.0
    assert e["top"][0]["symbol"] == "MNQ"


def test_fills_window_groups_by_strategy_and_untagged():
    f = pd.DataFrame([_fill("A", "BOT", 10, "BUY", "OLV"), _fill("B", "SLD", 5, None, None),
                      _fill("C", "BOT", 1, "BUY", "OLV", session="2026-09-01")])
    w = B.fills_window(f, "2026-10-05", "2026-10-09")
    assert w["n"] == 2 and w["untagged_pct"] == 50.0
    assert {r["strategy"] for r in w["by_strategy"]} == {"OLV", "untagged"}


# ---------------------------------------------------------------------------
# ledger
# ---------------------------------------------------------------------------
def _ledger():
    rows = [("S1", "Liquid", "Long", "2026-09-01", "2026-09-10", 1000.0, 500.0, 1.0),
            ("S1", "Liquid", "Long", "2026-09-20", "2026-10-09", 1000.0, 500.0, 1.0),     # held through asof
            ("S2", "Overflow", "Short", "2026-09-05", "2026-09-15", -200.0, 500.0, -0.4)]
    return pd.DataFrame([{"Strategy": s, "Tier": t, "Direction": d, "Entry Date": e, "Exit Date": x,
                          "PnL_flat_750k": p, "Risk_flat_750k": r, "R_Multiple": rm, "Shares_flat": 1000,
                          "Entry Price": 75.0} for s, t, d, e, x, p, r, rm in rows])


def test_ledger_exposure_counts_positions_on_the_last_day():
    e = B.ledger_exposure(_ledger(), asof="2026-10-09", days=40)
    assert e["gross_pct_last"] == 10.0 and e["net_pct_last"] == 10.0


def test_capital_efficiency_ratio():
    ce = B.capital_efficiency(_ledger(), asof="2026-10-09")
    rows = {r["strategy"]: r for r in ce["full"]["rows"]}
    # S1: 2000 of 1800 total P&L over 2/3 of risk -> CER (2000/1800)/(2/3)
    assert rows["S1"]["cer"] == pytest.approx(round((2000 / 1800) / (2 / 3), 2))
    assert rows["S2"]["cer"] < 0


def test_ledger_vol_reports_reference_not_target():
    d = pd.DataFrame({"date": pd.bdate_range(end="2026-10-09", periods=900),
                      "pnl_flat": np.random.default_rng(1).normal(300, 7500, 900)})
    v = B.ledger_vol(d)
    v.pop("_daily_ret")
    assert 10 < v["ann_vol_252d_pct"] < 20 and "ref_63d_vol_3y_median_pct" in v
    assert not any("target" in k for k in v)


def test_dynamic_keys_pick_newest_complete_ledger_run_and_windows():
    listing = {"site/builds/100-1/backtest_trades_full.parquet", "site/builds/100-1/backtest_daily_pnl.parquet",
               "site/builds/999-1/backtest_trades_full.parquet",                       # incomplete
               "site/builds/200-1/backtest_trades_full.parquet", "site/builds/200-1/backtest_daily_pnl.parquet",
               "ops/olv_capacity/2026-10-09.json", "ops/olv_capacity/2026-01-02.json",
               "automation/receipts/v1/2026-10-08/scan_pm/latest.json",
               "automation/receipts/v1/2026-09-01/scan_pm/latest.json",
               "pitch_delivery_receipts/2026-10-09.json"}
    k = B.dynamic_keys(dt.date(2026, 10, 9), listing)
    assert k["ledger"] == ["site/builds/200-1/backtest_trades_full.parquet",
                           "site/builds/200-1/backtest_daily_pnl.parquet"]
    assert k["snapshots"] == ["ops/olv_capacity/2026-10-09.json"]
    assert k["receipts"] == ["automation/receipts/v1/2026-10-08/scan_pm/latest.json"]
    assert all(U.r2_key_allowed(x) for v in k.values() for x in v)


def test_runtime_issues_respect_enable_flags():
    rt = {"event_enabled": True, "trend_moo_enabled": False, "legend_enabled": False,
          "tasks": {"event": {"state": "Ready", "last_result": 1}, "trend": {"state": "Missing"},
                    "legend": {"state": "Ready", "last_result": 2}, "chain": {"state": "Ready", "last_result": 0}}}
    assert [t["task"] for t in B.runtime_issues(rt)] == ["event"]


def test_book_notes_must_name_their_basis_and_no_vol_target():
    errs = []
    G.lint_text("We should set a vol target of 12%.", "x", errs)
    assert any("vol target" in e for e in errs)


# ---------------------------------------------------------------------------
# daily check-in
# ---------------------------------------------------------------------------
@pytest.fixture()
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("PM_AGENT_HOME", str(tmp_path / "pm"))
    c = U.cache_dir()
    c.mkdir(parents=True)

    def put(key, obj):
        p = pad.local_path(key, c)
        if isinstance(obj, pd.DataFrame):
            obj.to_parquet(p)
        else:
            p.write_text(json.dumps(obj), encoding="utf-8")

    snaps = {"2026-10-08": _snap("2026-10-08", 600000.0),
             "2026-10-09": _snap("2026-10-09", 630000.0, [_pos("BNS", -700, 88.0), _pos("EEM", -300, 66.0),
                                                          _pos("RHI", 1134, 34.0)])}
    for d, s in snaps.items():
        put(f"ops/olv_capacity/{d}.json", s)
    put("ops/olv_capacity/latest.json", snaps["2026-10-09"])
    put("live_fills.parquet", pd.DataFrame([
        _fill("BNS", "BOT", 2272, "BUY", "LT Trend ST OS", "2026-10-08"),
        _fill("BNS", "SLD", 700, "5b93628d-uuid", "unified-close"),
        _fill("BNS", "SLD", 2272, "BUY", "LT Trend ST OS"),
        _fill("EEM", "SLD", 300, None, None),
        _fill("RHI", "BOT", 1134, "BUY", "Oversold Low Volume", "2026-10-05")]))
    put("live_fills_status.json", {"last_session": "2026-10-09", "gap": {"gap": False},
                                   "completeness": {"accounts": {"primary": {"complete": True}}}})
    put("ops/sleeve_runtime_status.json", {"event_enabled": True, "tasks": {"event": {"state": "Ready", "last_result": 0}}})
    put("ops/expected_exit_status.json", {"status": "ok", "counts": {"missed": 0}})
    rec = {"automation/receipts/v1/2026-10-09/macro_releases/latest.json":
           {"job_id": "macro_releases", "run_date_et": "2026-10-09", "status": "failure"},
           "automation/receipts/v1/2026-10-09/scan_pm/latest.json":
           {"job_id": "scan_pm", "run_date_et": "2026-10-09", "status": "success", "health_status": "degraded"}}
    for k, v in rec.items():
        put(k, v)
    deliveries = {"pitch_delivery_receipts/2026-10-09.json": {"status": "sent"},
                  "seasonal_agent_delivery_receipts/2026-10-09.json": {"status": "sent"}}
    for k, v in deliveries.items():
        put(k, v)
    keys = {"snapshots": [f"ops/olv_capacity/{d}.json" for d in snaps], "receipts": list(rec),
            "deliveries": list(deliveries), "ledger": []}
    B._save_index(keys, c)
    return c


def test_checkin_catalogue_on_a_bad_day(cache):
    import pm_daily_check as C
    exc, facts = C.exceptions(dt.date(2026, 10, 9))
    kinds = {(e["kind"], e["key"]) for e in exc}
    assert ("job_failed", "job:macro_releases") in kinds
    assert ("position_flip", "flip:BNS") in kinds
    assert not any(k == "flip:EEM" for _, k in kinds)          # untagged short: not a flip
    assert ("delivery_missing", "delivery:Risk Agent") in kinds
    assert ("nlv_move", "nlv") in kinds                         # +5% day
    assert not any(e["key"] == "job:scan_pm" for e in exc)      # degraded is not a failure
    assert facts["prev_session"] == "2026-10-08"


def test_checkin_journals_quiet_days_and_counts_streaks(cache, monkeypatch):
    import pm_daily_check as C
    import pm_agent_journal as J
    monkeypatch.setattr(J, "sync_up", lambda path: True)
    prev = [{"kind": "check_in", "date": "2026-10-08",
             "exceptions": [{"key": "job:macro_releases"}]}]
    J.append(prev)
    rc = C.main(["--today", "2026-10-09", "--no-sync", "--no-send", "--no-r2"])
    assert rc == 0
    rec = [r for r in J.load() if r.get("kind") == "check_in"][-1]
    mr = next(e for e in rec["exceptions"] if e["key"] == "job:macro_releases")
    assert mr["days_running"] == 2
    assert C.main(["--today", "2026-10-09", "--no-sync", "--no-send", "--no-r2"]) == 0   # idempotent
    assert len([r for r in J.load() if r.get("kind") == "check_in"]) == 2
    assert C.main(["--today", "2026-10-10", "--no-sync", "--no-send", "--no-r2"]) == 0   # Saturday: skipped
