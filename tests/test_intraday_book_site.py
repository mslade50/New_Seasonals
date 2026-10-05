"""Guards for the Portfolio page's intraday book (2026-09-28).

build_site.build_intraday_book turns the committed research replay
(site/research/intraday_replay.json, written by scripts/build_intraday_replay.py)
into dist/data/intraday_daily.json in the strategy_daily shape, plus the
combined swing + intraday series. Pinned here: payload shape from a fixture,
the absent-file path (no payload, swing payloads untouched), the combined
arithmetic over the union of dates, book tags, and the build registration.
"""
import copy
import json
import os
import re
import sys
from pathlib import Path

import pytest

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
ROOT = Path(__file__).resolve().parent.parent
REAL_REPLAY = ROOT / "site" / "research" / "intraday_replay.json"
sys.path.insert(0, str(ROOT / "scripts"))

from scripts import build_site  # noqa: E402


def _fixture() -> dict:
    return {
        "schema": 1, "generated": "2026-09-28T12:00:00Z", "basis_usd": 750000,
        "strategies": [
            {"id": "open_breakout", "name": "Open Breakout", "book": "intraday",
             "instruments": "MNQ/MES", "span": ["2018-01-02", "2018-01-05"],
             "sizing": "15 bps NQ / 10 bps ES", "costs": "base",
             "daily": [["2018-01-02", 1000], ["2018-01-03", -500],
                       ["2018-01-04", 0], ["2018-01-05", 250]],
             "by_market": {"NQ": [["2018-01-02", 600], ["2018-01-03", -500]],
                           "ES": [["2018-01-02", 400], ["2018-01-05", 250]]},
             "stats": {"trades_per_year": 300, "sum_usd": 750, "sharpe_daily": 1.2,
                       "max_dd_usd": -500, "years": 0.02},
             "notes": "fixture"},
            {"id": "legend_ema", "name": "Legend EMA", "book": "intraday",
             "instruments": "SPY/QQQ", "span": ["2018-01-04", "2018-01-08"],
             "sizing": "40 percent SPY", "costs": "2 bps",
             "daily": [["2018-01-04", 300], ["2018-01-05", 0], ["2018-01-08", -100]],
             "by_market": {"SPY": [["2018-01-04", 300], ["2018-01-08", -100]]},
             "stats": {"trades_per_year": 7}, "notes": "fixture"},
            {"id": "empty_one", "name": "Empty One", "book": "intraday",
             "span": None, "daily": [], "by_market": {}, "stats": {},
             "notes": "no replay yet"},
        ],
    }


def _write(tmp_path, obj) -> Path:
    p = tmp_path / "intraday_replay.json"
    p.write_text(json.dumps(obj), encoding="utf-8")
    return p


def _swing() -> dict:
    return {
        "dates": ["2003-01-02", "2018-01-02", "2018-01-03", "2018-01-04"],
        "series": {"Alpha||Liquid": [10.0, 20.0, 30.0, 40.0]},
        "total_flat": [10.0, 20.0, 30.0, 40.0],
        "total_compounded": [10.0, 20.0, 30.0, 40.0],
        "equity_compounded": [750010.0, 750030.0, 750060.0, 750100.0],
        "start_equity": 750000.0,
    }


def test_payload_matches_strategy_daily_shape(tmp_path):
    out = build_site.build_intraday_book(_write(tmp_path, _fixture()))
    assert out["book"] == "intraday"
    assert out["label"] == "research replay at live sizing"
    assert out["start_equity"] == float(build_site.ACCOUNT_VALUE)
    assert out["generator"] == "scripts/build_intraday_replay.py"
    assert out["dates"] == ["2018-01-02", "2018-01-03", "2018-01-04",
                            "2018-01-05", "2018-01-08"]
    assert set(out["series"]) == {"Open Breakout||Intraday", "Legend EMA||Intraday"}
    for arr in out["series"].values():
        assert len(arr) == len(out["dates"])
    assert out["series"]["Open Breakout||Intraday"] == [1000, -500, 0, 250, 0]
    assert out["series"]["Legend EMA||Intraday"] == [0, 0, 300, 0, -100]
    assert out["total_flat"] == [1000, -500, 300, 250, -100]
    assert "combined" not in out  # no swing series passed


def test_roster_book_tags_and_empty_strategy(tmp_path):
    out = build_site.build_intraday_book(_write(tmp_path, _fixture()))
    roster = {s["id"]: s for s in out["strategies"]}
    assert set(roster) == {"open_breakout", "legend_ema", "empty_one"}
    for s in roster.values():
        assert s["book"] == "intraday" and s["Tier"] == "Intraday"
        assert s["key"] == f"{s['name']}||Intraday" and s["Strategy"] == s["name"]
    ob = roster["open_breakout"]
    assert ob["has_daily"] and ob["n_days"] == 4 and ob["active_days"] == 3
    assert ob["by_market_usd"] == {"NQ": 100.0, "ES": 650.0}
    assert ob["span"] == ["2018-01-02", "2018-01-05"]
    assert roster["empty_one"]["has_daily"] is False
    assert "Empty One||Intraday" not in out["series"]
    books = build_site.intraday_books_meta(out)
    assert books["intraday"]["label"] == "research replay at live sizing"
    assert books["intraday"]["span"] == ["2018-01-02", "2018-01-08"]
    assert [s["id"] for s in books["intraday"]["strategies"]] == [
        "open_breakout", "legend_ema", "empty_one"]


def test_combined_is_day_by_day_sum_over_union(tmp_path):
    swing = _swing()
    before = copy.deepcopy(swing)
    out = build_site.build_intraday_book(_write(tmp_path, _fixture()), swing_daily=swing)
    assert swing == before, "swing payload must not be mutated"
    comb = out["combined"]
    assert comb["dates"] == ["2003-01-02", "2018-01-02", "2018-01-03", "2018-01-04",
                             "2018-01-05", "2018-01-08"]
    assert comb["swing_first"] == "2003-01-02"
    assert comb["intraday_first"] == "2018-01-02"
    assert comb["swing_flat"] == [10, 20, 30, 40, 0, 0]
    assert comb["intraday_flat"] == [0, 1000, -500, 300, 250, -100]
    assert comb["total_flat"] == [s + i for s, i in zip(comb["swing_flat"], comb["intraday_flat"])]
    assert sum(comb["total_flat"]) == sum(swing["total_flat"]) + sum(out["total_flat"])


def test_absent_file_builds_nothing_and_leaves_swing_alone(tmp_path):
    swing = _swing()
    before = copy.deepcopy(swing)
    assert build_site.build_intraday_book(tmp_path / "missing.json", swing_daily=swing) is None
    assert swing == before


def test_basis_and_schema_mismatch_fail_loudly(tmp_path):
    bad = _fixture()
    bad["basis_usd"] = 1_000_000
    with pytest.raises(ValueError, match="basis"):
        build_site.build_intraday_book(_write(tmp_path, bad))
    bad = _fixture()
    bad["schema"] = 2
    with pytest.raises(ValueError, match="schema"):
        build_site.build_intraday_book(_write(tmp_path, bad))


def test_build_registration_and_swing_book_tag():
    src = (ROOT / "scripts" / "build_site.py").read_text(encoding="utf-8")
    assert re.search(r'best_effort\("intraday_daily", build_intraday_book', src)
    assert '"intraday_daily": False' in src
    assert 'rec["book"] = "swing"' in src
    assert 'meta["books"] = intraday_books_meta(intraday)' in src
    # meta.books only when the payload exists, so a build without the replay
    # ships the same Portfolio roster as before
    assert re.search(r"if intraday:\s+meta\[\"books\"\]", src)


def test_frontend_wiring():
    html = (ROOT / "site" / "index.html").read_text(encoding="utf-8")
    for el in ('id="bookScope"', 'id="bookScopeNote"', 'id="bookCorrNote"'):
        assert el in html
    js = (ROOT / "site" / "assets" / "portfolio.js").read_text(encoding="utf-8")
    assert 'fetchSitePayload(meta, "data/intraday_daily.json")' in js
    assert "flags.intraday_daily" in js


def test_trade_rows_reconcile_and_preserve_no_stop_r(tmp_path):
    replay = _fixture()
    for s in replay["strategies"]:
        s["trades"] = [
            {"trade_id": f"intraday:{s['id']}:{d}", "Strategy": s["name"], "Tier": "Intraday",
             "book": "intraday", "Exit_Date": d, "Entry_Date": d, "PnL_flat": v,
             "R": None if s["id"] == "legend_ema" else v / 100}
            for d, v in s["daily"] if v
        ]
    out = build_site.build_intraday_book(_write(tmp_path, replay))
    assert out["has_trades"]
    assert len(out["trades"]) == 5
    assert sum(t["PnL_flat"] for t in out["trades"]) == sum(out["total_flat"])
    assert all(t["R"] is None for t in out["trades"] if t["Strategy"] == "Legend EMA")
    assert out["strategies"][0]["coverage_through"] == "2018-01-05"
    replay["strategies"][0]["trades"][0]["PnL_flat"] += 10
    with pytest.raises(ValueError, match="reconcile"):
        build_site.build_intraday_book(_write(tmp_path, replay))


def test_committed_trades_match_daily_and_coverage():
    out = build_site.build_intraday_book(REAL_REPLAY)
    assert out["has_trades"]
    assert len(out["trades"]) == 3238 + 73
    assert sum(t["PnL_flat"] for t in out["trades"]) == pytest.approx(sum(out["total_flat"]))
    for t in out["trades"]:
        if t["R"] is not None:
            assert t["PnL_flat"] == pytest.approx(t["R"] * t["Risk_flat"], abs=.011)
    assert {s["id"]: s["coverage_through"] for s in out["strategies"]} == {
        "open_breakout": "2026-08-28", "legend_ema": "2026-08-05"}


@pytest.mark.skipif(not REAL_REPLAY.exists(), reason="intraday replay not committed yet")
def test_committed_replay_builds():
    out = build_site.build_intraday_book(REAL_REPLAY)
    assert out["dates"] == sorted(out["dates"])
    assert out["series"], "committed replay has no daily series"
    n = len(out["dates"])
    assert all(len(v) == n for v in out["series"].values())
    for s in out["strategies"]:
        assert s["book"] == "intraday"
