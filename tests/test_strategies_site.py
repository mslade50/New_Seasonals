"""Guards for the Strategies site tab (2026-09-26).

The tab is a catalog of every strategy the book trades, merged at build time
with ledger-replay cadence/outcome stats and live-fill attribution. Pinned
here: the committed catalog's schema, the build_site registration, the merge
behavior on a synthetic ledger + fills store, the degrade-to-null path when
neither exists, and the nav/page/JS wiring.
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"
CATALOG = SITE / "research" / "strategy_catalog.json"
sys.path.insert(0, str(ROOT / "scripts"))

from scripts import build_site  # noqa: E402

ENTRY_KEYS = {
    "id", "name", "family", "status", "live_since", "direction", "instruments",
    "universe_size", "captures", "entry", "exit", "sizing", "risk_controls",
    "order_ref_tags", "doc", "stats_source", "frozen_stats", "notes",
}
OPTIONAL_KEYS = {"components"}
FAMILIES = {"Systematic equity book", "Futures", "Intraday ETF", "Sleeves",
            "Agent products"}
STATUSES = {"live", "pilot", "manual", "paper", "retired"}
STATS_SOURCES = {"ledger_replay", "research_backtest", "frozen_evidence",
                 "live_journal", "none"}
DIRECTIONS = {"long", "short", "long_short", "long_flat", "mixed"}
FROZEN_KEYS = {"span", "n_trades", "trades_per_year", "trades_per_month",
               "win_rate", "avg_r", "profit_factor", "sharpe", "median_hold",
               "notes"}
FRONTEND_PENDING = ("Strategies frontend not landed yet: {what}. The data side "
                    "is done; this assertion waits on strategies.html / "
                    "assets/strategies.js / the common.js PAGES entry.")


def _catalog() -> dict:
    return json.loads(CATALOG.read_text(encoding="utf-8"))


# ------------------------------------------------------------------ catalog
def test_catalog_top_level():
    cat = _catalog()
    assert cat["schema"] == 1
    assert pd.Timestamp(cat["updated"])
    assert isinstance(cat["strategies"], list) and cat["strategies"]


def test_catalog_entries_match_schema():
    for e in _catalog()["strategies"]:
        keys = set(e)
        assert ENTRY_KEYS <= keys, f"{e.get('id')}: missing {ENTRY_KEYS - keys}"
        assert keys <= ENTRY_KEYS | OPTIONAL_KEYS, f"{e['id']}: extra {keys - ENTRY_KEYS - OPTIONAL_KEYS}"
        assert e["family"] in FAMILIES, e["id"]
        assert e["status"] in STATUSES, e["id"]
        assert e["stats_source"] in STATS_SOURCES, e["id"]
        assert e["direction"] in DIRECTIONS, e["id"]
        assert isinstance(e["order_ref_tags"], list), e["id"]
        if e["live_since"] is not None:
            pd.Timestamp(e["live_since"])
        for k in ("captures", "entry", "exit", "sizing", "risk_controls"):
            assert isinstance(e[k], str) and len(e[k]) > 20, f"{e['id']}.{k}"
        fs = e["frozen_stats"]
        if fs is not None:
            assert FROZEN_KEYS <= set(fs), f"{e['id']}: frozen_stats missing {FROZEN_KEYS - set(fs)}"
        if e["stats_source"] == "ledger_replay":
            assert e["family"] == "Systematic equity book", e["id"]
        if e["stats_source"] in ("research_backtest", "frozen_evidence"):
            assert fs is not None, f"{e['id']}: {e['stats_source']} needs frozen_stats"
        if e["doc"]:
            assert (ROOT / e["doc"]).exists(), f"{e['id']}: doc {e['doc']} missing"


def test_catalog_ids_unique_and_live_entries_tagged():
    entries = _catalog()["strategies"]
    ids = [e["id"] for e in entries]
    assert len(ids) == len(set(ids))
    for e in entries:
        if e["status"] in ("live", "pilot"):
            assert e["order_ref_tags"], f"{e['id']} is {e['status']} with no orderRef tag"
            assert all(isinstance(t, str) and t.strip() for t in e["order_ref_tags"])


def test_catalog_covers_the_whole_book():
    from strategy_config import STRATEGY_BOOK
    book = {s["name"] for s in STRATEGY_BOOK}
    sys_entries = [e for e in _catalog()["strategies"]
                   if e["family"] == "Systematic equity book"]
    assert {e["name"] for e in sys_entries} == book
    for e in sys_entries:
        assert e["stats_source"] == "ledger_replay"
        assert e["order_ref_tags"] == [e["name"]]


def test_catalog_event_components_are_frozen_evidence():
    import event_sleeve as es
    ev = next(e for e in _catalog()["strategies"] if e["id"] == "event_sleeve")
    comps = {c["id"]: c for c in ev["components"]}
    assert set(comps) == set(es.EVENT_SLEEVE) == set(ev["order_ref_tags"])
    for tid, frozen in es.BACKTEST_EVIDENCE.items():
        for k in ("n", "avg_bps", "t", "hit"):
            assert comps[tid][k] == frozen[k], f"{tid}.{k} drifted from BACKTEST_EVIDENCE"
        assert comps[tid]["nav_frac"] == es.EVENT_SLEEVE[tid]["nav_frac"]


def test_catalog_prose_has_no_em_dashes():
    text = CATALOG.read_text(encoding="utf-8")
    assert "—" not in text and "–" not in text


# ------------------------------------------------------------------ build_site
def test_build_site_registers_payload():
    src = (ROOT / "scripts" / "build_site.py").read_text(encoding="utf-8")
    assert "def build_strategies" in src
    assert '"strategies": False' in src
    assert 'best_effort("strategies", build_strategies)' in src
    val = (ROOT / "scripts" / "validate_site_freshness.py").read_text(encoding="utf-8")
    assert '("strategies.json", "strategies")' in val


def _synthetic_ledger(path: Path) -> None:
    rows = []
    # Strategy A: 4 closed trades over exactly two years (2 per year).
    for i, (sd, r, pnl) in enumerate([("2020-01-02", 1.0, 1000.0), ("2020-07-01", -1.0, -500.0),
                                      ("2021-01-04", 2.0, 2000.0), ("2022-01-03", 0.5, 250.0)]):
        rows.append({"trade_id": i, "Strategy": "Oversold Low Volume", "Ticker": f"T{i}",
                     "Tranche": "", "Signal Date": sd, "Entry Date": sd, "Exit Date": sd,
                     "R_Multiple": r, "PnL_flat_750k": pnl, "Risk_flat_750k": 1000.0,
                     "Shares_flat": 100.0})
    # OVS: one position booked as two tranche rows (near/far) counts once.
    for j, (tr, r, sh, pnl) in enumerate([("near", 1.0, 40.0, 400.0), ("far", 2.0, 60.0, 1200.0)]):
        rows.append({"trade_id": 10 + j, "Strategy": "Overbot Vol Spike", "Ticker": "XYZ",
                     "Tranche": tr, "Signal Date": "2024-03-01", "Entry Date": "2024-03-04",
                     "Exit Date": "2024-03-05", "R_Multiple": r, "PnL_flat_750k": pnl,
                     "Risk_flat_750k": 1000.0, "Shares_flat": sh})
    df = pd.DataFrame(rows)
    for c in ("Signal Date", "Entry Date", "Exit Date"):
        df[c] = pd.to_datetime(df[c])
    df.to_parquet(path)


def _synthetic_fills(path: Path) -> None:
    pd.DataFrame([
        {"exec_id": "a", "session_date": "2026-09-10", "account_key": "primary",
         "symbol": "HXL", "side": "BOT", "qty": 10.0,
         "order_ref": "HXL|BUY|Oversold Low Volume|2026-09-09",
         "strategy": "Oversold Low Volume", "realized_pnl": 0.0, "commission": 1.0},
        {"exec_id": "b", "session_date": "2026-09-24", "account_key": "pa",
         "symbol": "HXL", "side": "SLD", "qty": 10.0,
         "order_ref": "HXL|BUY|Oversold Low Volume|2026-09-09",
         "strategy": "", "realized_pnl": -150.5, "commission": 1.0},
        {"exec_id": "c", "session_date": "2026-09-22", "account_key": "primary",
         "symbol": "SPY", "side": "BOT", "qty": 5.0,
         "order_ref": "SPY|BUY|Pitch-2026-09-22-1|2026-09-22",
         "strategy": "Pitch-2026-09-22-1", "realized_pnl": 0.0, "commission": 1.0},
        {"exec_id": "d", "session_date": "2026-09-23", "account_key": "primary",
         "symbol": "SPY", "side": "BOT", "qty": 1.0,
         "order_ref": "SPY|BUY|Legend_EMA_TEST|2026-09-23",
         "strategy": "Legend_EMA_TEST", "realized_pnl": 0.0, "commission": 1.0},
    ]).to_parquet(path)


def _by_id(payload: dict) -> dict:
    return {s["id"]: s for s in payload["strategies"]}


def test_build_strategies_merges_ledger_and_fills(tmp_path):
    ledger, fills = tmp_path / "ledger.parquet", tmp_path / "fills.parquet"
    _synthetic_ledger(ledger)
    _synthetic_fills(fills)
    payload = build_site.build_strategies(str(CATALOG), str(ledger), str(fills))
    rows = _by_id(payload)
    assert len(rows) == len(_catalog()["strategies"])

    olv = rows["olv"]["ledger_stats"]
    assert olv["n_trades"] == 4
    assert olv["trades_per_year"] == pytest.approx(4 / ((pd.Timestamp("2022-01-03") - pd.Timestamp("2020-01-02")).days / 365.25), abs=0.01)
    assert olv["trades_per_month"] == pytest.approx(olv["trades_per_year"] / 12, rel=1e-2)
    assert olv["first_date"] == "2020-01-02" and olv["last_date"] == "2022-01-03"
    assert olv["win_rate"] == pytest.approx(0.75)
    assert olv["profit_factor"] == pytest.approx(3250 / 500, rel=1e-3)
    assert olv["total_pnl_flat"] == pytest.approx(2750)

    ovs = rows["ovs"]["ledger_stats"]
    assert ovs["n_trades"] == 1 and ovs["n_rows"] == 2
    assert ovs["avg_r"] == pytest.approx((1.0 * 40 + 2.0 * 60) / 100)

    # A ledger_replay entry with no ledger rows gets a null block.
    assert rows["monthly_weak_close"]["ledger_stats"] is None

    live = rows["olv"]["live"]
    assert live["n_fills"] == 2  # blank strategy field falls back to order_ref
    assert live["first_fill"] == "2026-09-10" and live["last_fill"] == "2026-09-24"
    assert live["by_account"]["pa"]["n_fills"] == 1
    assert live["realized_pnl"] == pytest.approx(-150.5)
    assert live["symbols"] == ["HXL"]
    assert rows["daily_pitch"]["live"]["n_fills"] == 1  # Pitch-* prefix tag
    assert rows["legend_ema"]["live"]["n_fills"] == 0   # _TEST refs excluded
    assert rows["event_sleeve"]["live"]["n_fills"] == 0
    assert payload["sources"]["fills"]["first_session"] == "2026-09-10"
    assert payload["sources"]["fills"]["untagged_rows"] == 0

    out = tmp_path / "dist" / "strategies.json"
    build_site.write_json(payload, str(out))
    assert json.loads(out.read_text(encoding="utf-8"))["strategies"]


def test_build_strategies_never_keys_on_raw_account_ids(tmp_path):
    fills = tmp_path / "fills.parquet"
    pd.DataFrame([
        {"exec_id": "a", "session_date": "2026-09-10", "account": "U16584234",
         "symbol": "HXL", "side": "BOT", "qty": 10.0,
         "order_ref": "HXL|BUY|Oversold Low Volume|2026-09-09",
         "strategy": "Oversold Low Volume", "realized_pnl": 0.0, "commission": 1.0},
    ]).to_parquet(fills)
    payload = build_site.build_strategies(str(CATALOG), str(tmp_path / "none.parquet"), str(fills))
    assert _by_id(payload)["olv"]["live"]["n_fills"] == 1
    assert "U16584234" not in json.dumps(payload)


def test_build_strategies_without_ledger_or_fills(tmp_path):
    payload = build_site.build_strategies(
        str(CATALOG), str(tmp_path / "no_ledger.parquet"), str(tmp_path / "no_fills.parquet"))
    assert payload["sources"]["ledger"]["available"] is False
    assert payload["sources"]["fills"]["available"] is False
    for row in payload["strategies"]:
        assert row["live"] is None
        if row["stats_source"] == "ledger_replay":
            assert row["ledger_stats"] is None
    rows = _by_id(payload)
    # 2737 = NQ+ES base with the shipped 1.25 prior-range skip (was 3374 unfiltered).
    assert rows["open_breakout"]["frozen_stats"]["n_trades"] == 2737
    out = tmp_path / "strategies.json"
    build_site.write_json(payload, str(out))
    assert out.exists()


def test_strategy_stats_cadence_keys_are_additive(tmp_path):
    ledger = tmp_path / "ledger.parquet"
    _synthetic_ledger(ledger)
    df = build_site.load_ledger(str(ledger))
    for c, v in (("Direction", "Long"), ("Entry Price", 100.0), ("Exit Price", 101.0),
                 ("ATR", 2.0), ("Exit Type", "Time")):
        df[c] = v
    stats = build_site.build_strategy_stats(df)
    olv = stats["Oversold Low Volume"]
    for k in ("n", "win_rate", "avg_r", "median_hold", "terminal_move", "loser_mix",
              "outcome_hist", "trades_per_year", "trades_per_month", "first_date", "last_date"):
        assert k in olv
    assert olv["first_date"] == "2020-01-02"
    assert stats["Overbot Vol Spike"]["n"] == 2  # existing consumers still see rows


# ------------------------------------------------------------------ frontend wiring
def test_nav_has_strategies_tab():
    common = (SITE / "assets" / "common.js").read_text(encoding="utf-8")
    assert 'href: "strategies.html"' in common and 'label: "Strategies"' in common, \
        FRONTEND_PENDING.format(what="PAGES entry labelled Strategies in common.js")


def test_strategies_page_wiring():
    html_path = SITE / "strategies.html"
    js_path = SITE / "assets" / "strategies.js"
    assert html_path.exists(), FRONTEND_PENDING.format(what="site/strategies.html")
    assert js_path.exists(), FRONTEND_PENDING.format(what="site/assets/strategies.js")
    html = html_path.read_text(encoding="utf-8")
    assert "assets/strategies.js" in html and "assets/common.js" in html, \
        FRONTEND_PENDING.format(what="script tags in strategies.html")
    js = js_path.read_text(encoding="utf-8")
    assert "data/strategies.json" in js, FRONTEND_PENDING.format(what="payload fetch in strategies.js")
    assert 'renderNav("strategies.html")' in js, FRONTEND_PENDING.format(what="renderNav call")
