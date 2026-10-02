"""The --product switch resolves every Daily Seasonal path, subject and tab,
and the default pitch product resolves the historical ones byte for byte.

Live rule: docs/claude_ref/daily_seasonal.md. The pitch runs live at 5:10 AM,
so the pitch half of every test here is the one that matters most.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import daily_pitch as dp  # noqa: E402
import pitch_delivery  # noqa: E402
import pitch_grammar as pg  # noqa: E402
import pitch_journal  # noqa: E402
import pitch_lab  # noqa: E402
import pitch_products as pp  # noqa: E402
import check_pitch_delivered as cpd  # noqa: E402

ASOF = pd.Timestamp("2026-09-30")
DATA = ROOT / "data"


def test_pitch_product_is_the_historical_paths():
    p = pp.get_product()
    assert p is pp.PITCH and pp.get_product("PITCH") is pp.PITCH
    assert p.journal_path == DATA / "pitch_journal.jsonl" == pitch_journal.JOURNAL_PATH
    assert p.journal_r2_key == "pitch_journal.jsonl" == pitch_journal.JOURNAL_R2_KEY
    assert p.scoreboard_path == DATA / "pitch_scoreboard.json" == dp.SCOREBOARD_PATH
    assert p.state_path == DATA / "pitch_state.json" == dp.STATE_PATH
    assert p.default_ideas == DATA / "pitch_ideas.json" == dp.DEFAULT_IDEAS
    assert p.watchlist_path == DATA / "pitch_watchlist.json" == pitch_lab.WATCHLIST_PATH
    assert p.negative_registry_path == DATA / "pitch_negative_registry.md"
    assert p.checks_root == ROOT / "scratch" / "pitch_checks" == pg.CHECKS_ROOT
    assert p.receipt_dir == pitch_delivery.RECEIPT_DIR == DATA / "pitch_delivery_receipts"
    assert p.receipt_r2_prefix == pitch_delivery.R2_RECEIPT_PREFIX == "pitch_delivery_receipts"
    assert p.receipt_download_dir == pitch_delivery.R2_DOWNLOAD_DIR
    assert p.tab_name == dp.TAB_NAME == "Pitch"
    assert p.label == "Daily Pitch"


def test_seasonal_product_paths():
    s = pp.get_product("seasonal")
    assert s.journal_path == DATA / "seasonal_agent_journal.jsonl"
    assert s.journal_r2_key == "seasonal_agent_journal.jsonl"
    assert s.scoreboard_path == DATA / "seasonal_agent_scoreboard.json"
    assert s.watchlist_path == DATA / "seasonal_agent_watchlist.json"
    assert s.negative_registry_path == DATA / "seasonal_agent_negative_registry.md"
    assert s.state_path == DATA / "seasonal_state.json"
    assert s.default_ideas == DATA / "seasonal_agent_ideas.json"
    assert s.checks_root == ROOT / "scratch" / "seasonal_checks" == pg.SEASONAL_CHECKS_ROOT
    assert s.tab_name == "Seasonal Agent" and s.label == "Daily Seasonal"
    with pytest.raises(ValueError):
        pp.get_product("posts")


def test_seeded_files_load_as_empty(tmp_path):
    # Production research records are no longer empty seeds. Test the initial
    # file formats without reading or changing the live tracked journals.
    s = SimpleNamespace(
        journal_path=tmp_path / "seasonal_agent_journal.jsonl",
        watchlist_path=tmp_path / "seasonal_agent_watchlist.json",
        scoreboard_path=tmp_path / "seasonal_agent_scoreboard.json",
        negative_registry_path=tmp_path / "seasonal_agent_negative_registry.md",
    )
    s.journal_path.write_text("", encoding="utf-8")
    s.watchlist_path.write_text('{"entries": []}', encoding="utf-8")
    s.scoreboard_path.write_text("{}", encoding="utf-8")
    s.negative_registry_path.write_text(
        "# Negative registry\n" + "".join(f"\n## Section {i}\n" for i in range(5)),
        encoding="utf-8",
    )
    assert pitch_journal.load(s.journal_path, pull=False) == []
    assert pitch_lab.load_watchlist(s.watchlist_path) == {"entries": []}
    assert json.loads(s.scoreboard_path.read_text(encoding="utf-8")) == {}
    text = s.negative_registry_path.read_text(encoding="utf-8")
    assert text.count("\n## ") == 5


def test_receipts_are_namespaced_per_product():
    assert pitch_delivery.default_receipt_path("2026-09-30") == \
        DATA / "pitch_delivery_receipts" / "2026-09-30.json"
    assert pitch_delivery.r2_key("2026-09-30") == "pitch_delivery_receipts/2026-09-30.json"
    assert pitch_delivery.default_receipt_path("2026-09-30", "seasonal") == \
        DATA / "seasonal_agent_delivery_receipts" / "2026-09-30.json"
    assert pitch_delivery.r2_key("2026-09-30", "seasonal") == \
        "seasonal_agent_delivery_receipts/2026-09-30.json"
    assert pitch_delivery.r2_key("2026-09-30", "seasonal") != pitch_delivery.r2_key("2026-09-30")


def test_journal_r2_keys_route_by_production_path(tmp_path):
    assert pitch_journal.r2_key_for(pp.PITCH.journal_path) == "pitch_journal.jsonl"
    assert pitch_journal.r2_key_for(pp.SEASONAL.journal_path) == "seasonal_agent_journal.jsonl"
    assert pitch_journal.r2_key_for(tmp_path / "dev.jsonl") is None


def test_redirected_pitch_journal_never_syncs_the_real_path(tmp_path, monkeypatch):
    monkeypatch.setattr(pitch_journal, "JOURNAL_PATH", tmp_path / "j.jsonl")
    assert pitch_journal.r2_key_for(tmp_path / "j.jsonl") == "pitch_journal.jsonl"
    assert pitch_journal.r2_key_for(pp.PITCH.journal_path) is None


def test_subjects():
    assert dp.email_subject("pitch", ASOF, n_ideas=3) == "Daily Pitch - 2026-09-30 - 3 ideas"
    assert dp.email_subject("pitch", ASOF, n_ideas=1) == "Daily Pitch - 2026-09-30 - 1 idea"
    assert dp.email_subject("pitch", ASOF, stand_down_killed=7) == \
        "Daily Pitch - 2026-09-30 - NO TRADES (7 killed)"
    assert dp.email_subject("seasonal", ASOF, n_ideas=2) == "Daily Seasonal - 2026-09-30 - 2 ideas"
    assert dp.email_subject("seasonal", ASOF, stand_down_killed=6) == \
        "Daily Seasonal - 2026-09-30 - NO TRADES (6 killed)"


def test_tabs():
    assert dp.product_tab() == ("Pitch", dp.TAB_COLUMNS)
    name, cols = dp.product_tab("seasonal")
    assert name == "Seasonal Agent"
    assert cols[:len(dp.TAB_COLUMNS)] == dp.TAB_COLUMNS
    assert cols[len(dp.TAB_COLUMNS):] == ["Trail_Arm_ATR", "Trail_ATR"]


class FakeWorksheet:
    def __init__(self):
        self.written = None

    def clear(self):
        pass

    def update(self, values):
        self.written = values

    def get_all_records(self):
        return []


class FakeSheet:
    def __init__(self):
        self.asked, self.ws = [], FakeWorksheet()

    def worksheet(self, name):
        self.asked.append(name)
        return self.ws


def test_write_tab_targets_the_products_tab():
    base = {c: "" for c in dp.TAB_COLUMNS}
    rows = [{**base, "Idea_Id": "2026-09-30-S1", "Trail_Arm_ATR": 1.5, "Trail_ATR": 1.0},
            {**base, "Idea_Id": "2026-09-30-S2"}]
    sheet = FakeSheet()
    dp.write_tab(sheet, rows, "seasonal")
    assert sheet.asked == ["Seasonal Agent"]
    header, first, second = sheet.ws.written
    assert header[-2:] == ["Trail_Arm_ATR", "Trail_ATR"]
    assert first[-2:] == ["1.5", "1.0"] and second[-2:] == ["", ""]
    pitch = FakeSheet()
    dp.write_tab(pitch, [base])
    assert pitch.asked == ["Pitch"] and pitch.ws.written[0] == dp.TAB_COLUMNS
    assert dp.capture_approvals(FakeSheet(), ASOF, "Seasonal Agent") == []


def test_delivery_receipt_settings_per_product(tmp_path):
    args = SimpleNamespace(delivery_receipt=None, product="seasonal")
    path, r2 = dp.delivery_receipt_settings(args, pp.SEASONAL.journal_path, ASOF)
    assert (path, r2) == (DATA / "seasonal_agent_delivery_receipts" / "2026-09-30.json", True)
    # the pitch journal under the seasonal product is NOT production
    path, r2 = dp.delivery_receipt_settings(args, pitch_journal.JOURNAL_PATH, ASOF)
    assert r2 is False
    pitch_args = SimpleNamespace(delivery_receipt=None)
    path, r2 = dp.delivery_receipt_settings(pitch_args, pitch_journal.JOURNAL_PATH, ASOF)
    assert (path, r2) == (DATA / "pitch_delivery_receipts" / "2026-09-30.json", True)


def test_delivery_check_receipt_path_per_product():
    args = SimpleNamespace(delivery_receipt=None, product="seasonal", asof="2026-09-30",
                           journal=str(pp.SEASONAL.journal_path))
    assert cpd._receipt_path(args) == DATA / "seasonal_agent_delivery_receipts" / "2026-09-30.json"
    args = SimpleNamespace(delivery_receipt=None, product="pitch", asof="2026-09-30",
                           journal=str(pitch_journal.JOURNAL_PATH))
    assert cpd._receipt_path(args) == DATA / "pitch_delivery_receipts" / "2026-09-30.json"


def test_seasonal_never_touches_the_site_pitch_tab(tmp_path, monkeypatch):
    import cache_io
    monkeypatch.setattr(cache_io, "upload_from_local",
                        lambda *a, **k: pytest.fail("seasonal must not upload"))
    monkeypatch.setattr(dp, "SITE_PAYLOAD_PATH", tmp_path / "pitch_today.json")
    args = SimpleNamespace(delivery_receipt=None, product="seasonal")
    assert dp.publish_site_payload(dp.site_payload(ASOF, []), pp.SEASONAL.journal_path,
                                   args, ASOF)
    assert not (tmp_path / "pitch_today.json").exists()


def test_email_heading_and_tab_by_product():
    html = dp.render_email({"ideas": []}, [], ASOF, {}, None, "seasonal")
    assert "Daily Seasonal &mdash;" in html and "Seasonal Agent tab" in html
    assert "Daily Pitch" not in html
    pitch = dp.render_email({"ideas": []}, [], ASOF, {}, None)
    assert "Daily Pitch &mdash;" in pitch and "the Pitch tab" in pitch
    sd = dp.render_stand_down({"stand_down": {}}, ASOF, {}, None, "seasonal")
    assert "Daily Seasonal &mdash;" in sd and "Seasonal Agent tab is empty" in sd


def test_seasonal_idea_ids_never_collide_with_pitch_ids(survey, monkeypatch):
    fixture = json.loads((ROOT / "tests" / "fixtures" / "pitch_ideas_fixture.json")
                         .read_text(encoding="utf-8"))
    idea = fixture["ideas"][0]
    idea["sizing"] = {"risk_bps": 30, "stop_atr_for_sizing": 3.0}
    payload = {"asof": "2026-09-30", "ideas": [idea, dict(fixture["ideas"][1], sizing={
        "risk_bps": 30, "stop_atr_for_sizing": 3.0}, novelty_axis="calendar_cell")],
        "killed": fixture.get("killed", []),
        "short_slate": {"reason": "x" * 130, "candidates_considered": 8,
                        "axes": ["a", "b", "c", "d"], "asset_classes": ["1", "2", "3", "4"],
                        "closest": [{"title": "t", "decisive": "d", "why_died": "w"}]}}
    day = survey(payload)
    monkeypatch.setattr(pg, "SEASONAL_CHECKS_ROOT", day.parent)
    tickers = sorted({l["ticker"] for i in payload["ideas"] for l in i["legs"]})
    idx = pd.bdate_range(end="2026-09-29", periods=60)
    prices = pd.concat([pd.DataFrame({"ticker": t, "date": idx, "Open": 50.0,
                                      "High": 50.5, "Low": 49.5, "Close": 50.0,
                                      "Volume": 1e6}) for t in tickers])
    monkeypatch.setattr(pitch_journal, "load", lambda *a, **k: [])
    ideas, rows = dp.prepare(payload, ASOF, prices, [], product="seasonal")
    assert [i["idea_id"] for i in ideas] == ["2026-09-30-S1", "2026-09-30-S2"]
    assert all(r["Scan_Source"] == "Seasonal_Agent" for r in rows)


def test_seasonal_prepare_blocks_a_recent_pitch_fingerprint(survey, monkeypatch, capsys):
    fixture = json.loads((ROOT / "tests" / "fixtures" / "pitch_ideas_fixture.json")
                         .read_text(encoding="utf-8"))
    idea = dict(fixture["ideas"][0], sizing={"risk_bps": 30, "stop_atr_for_sizing": 3.0})
    payload = {"asof": "2026-09-30", "ideas": [idea], "killed": [], "short_slate": {}}
    day = survey(payload)
    monkeypatch.setattr(pg, "SEASONAL_CHECKS_ROOT", day.parent)
    pitch_records = [{"kind": "idea", "date": "2026-09-28",
                      "fingerprint": pg.fingerprint(idea)}]
    monkeypatch.setattr(pitch_journal, "load", lambda *a, **k: pitch_records)
    with pytest.raises(SystemExit):
        dp.prepare(payload, ASOF, pd.DataFrame(), [], product="seasonal")
    assert "was pitched on 2026-09-28" in capsys.readouterr().out
    # control: the same payload with no pitch record is not blocked for repetition
    monkeypatch.setattr(pitch_journal, "load", lambda *a, **k: [])
    with pytest.raises(SystemExit):
        dp.prepare(payload, ASOF, pd.DataFrame(), [], product="seasonal")
    assert "was pitched on" not in capsys.readouterr().out
