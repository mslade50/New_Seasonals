"""Pitch transport: daily_pitch.py -> R2 pitch_today.json -> Pages Function -> Pitch tab.

Same theme as the radar chain: the publisher copies a whitelist of fields
verbatim, the Function streams the bytes, and the tab formats them. These tests
pin the whitelist, the R2 key that ties publisher and Function together, the
nav registration, the stand-down payload, and that a dev run never uploads.
"""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import daily_pitch as dp  # noqa: E402
import pitch_journal as pj  # noqa: E402

FUNCTION = ROOT / "functions" / "pitch-today.js"
PITCH_JS = ROOT / "site" / "assets" / "pitch.js"
PITCH_HTML = ROOT / "site" / "pitch.html"
COMMON_JS = ROOT / "site" / "assets" / "common.js"
ASOF = pd.Timestamp("2026-09-23")


def _row(**kw):
    row = {c: "" for c in dp.TAB_COLUMNS}
    row.update({"Idea_Id": "2026-09-23-1", "Leg": 1, "Ticker": "XLE", "Sec_Type": "STK",
                "Action": "BUY", "Entry_Type": "LIMIT", "Entry_Anchor": "CLOSE",
                "Entry_Offset_ATR": -0.5, "Order_Type": "LMT", "TIF": "GTD",
                "Limit_Price": 98.75, "Quantity": 250, "Stop_Price": 95.75,
                "Target_Price": 104.75, "Time_Exit_Date": "2026-09-30",
                "Time_Exit_Order": "MOC", "Entry_Expire_Date": "2026-09-24",
                "Ref_Close": 100.0, "ATR": 2.5, "Multiplier": 1.0, "Place_Pass": "auction",
                "Execute_On": "2026-09-23", "Scan_Source": "Pitch", "Approve": ""})
    row.update(kw)
    return row


def _idea(**kw):
    idea = {"idea_id": "2026-09-23-1", "rank": 1, "title": "Energy dip", "grade": "B",
            "horizon_td": 5, "thesis": "th", "survived": "sv", "what_kills_it": "wk",
            "fingerprint": "fp-internal", "novelty_axis": "event_fingerprint",
            "overlap": "none", "legs": [{"ticker": "XLE"}],
            "evidence": {"summary": "N=31 sign test", "n": 31, "script": "scratch/x.py",
                         "dev_script": "scratch/y.py", "table": [[1, 2]]},
            "orders": [_row(), _row(Leg=2, Ticker="XOP", Internal_Debug="drop me")]}
    idea.update(kw)
    return idea


class Args:
    def __init__(self, **kw):
        self.delivery_receipt = kw.get("delivery_receipt")


# --- the publisher --------------------------------------------------------------

def test_site_payload_copies_whitelisted_fields_verbatim():
    p = dp.site_payload(ASOF, [_idea()])
    assert p["date"] == "2026-09-23" and p["stand_down"] is False
    assert p["account_value"] == dp.ACCOUNT_VALUE
    idea = p["ideas"][0]
    assert set(idea) == set(dp.SITE_IDEA_FIELDS) | {"evidence", "orders"}
    assert idea["evidence"] == {"summary": "N=31 sign test", "n": 31}
    assert idea["place_pass"] == "auction", "falls back to the first order's Place_Pass"
    leg = idea["orders"][0]
    assert leg["Limit_Price"] == 98.75 and leg["Quantity"] == 250
    assert leg["Stop_Price"] == 95.75 and leg["Multiplier"] == 1.0
    assert leg["Time_Exit_Order"] == "MOC"


def test_site_payload_drops_unwhitelisted_fields():
    p = dp.site_payload(ASOF, [_idea()])
    idea = p["ideas"][0]
    for k in ("fingerprint", "novelty_axis", "legs", "overlap"):
        assert k not in idea
    assert "script" not in idea["evidence"] and "table" not in idea["evidence"]
    assert all("Approve" not in leg for leg in idea["orders"])
    assert "Internal_Debug" not in idea["orders"][1]
    assert set(idea["orders"][0]) == set(dp.SITE_ORDER_FIELDS)


def test_order_whitelist_is_the_tab_schema_minus_approve_plus_multiplier():
    assert "Approve" not in dp.SITE_ORDER_FIELDS
    assert set(dp.SITE_ORDER_FIELDS) == (set(dp.TAB_COLUMNS) - {"Approve"}) | {"Multiplier"}


def test_stand_down_payload():
    p = dp.site_payload(ASOF, [], stand_down={"reason": "nothing survived the battery"})
    assert p["stand_down"] is True
    assert p["stand_down_reason"] == "nothing survived the battery"
    assert p["ideas"] == []
    assert json.loads(json.dumps(p))["stand_down"] is True


def test_dev_run_writes_locally_and_never_uploads(tmp_path, monkeypatch):
    import cache_io
    monkeypatch.setattr(cache_io, "upload_from_local",
                        lambda *a, **k: pytest.fail("a dev run must never touch R2"))
    monkeypatch.setattr(cache_io, "is_configured", lambda: True)
    monkeypatch.setattr(dp, "SITE_PAYLOAD_PATH", tmp_path / "prod_pitch_today.json")
    journal = tmp_path / "j.jsonl"
    assert dp.publish_site_payload(dp.site_payload(ASOF, [_idea()]), journal, Args(), ASOF)
    out = tmp_path / "j.pitch_today.json"
    assert json.loads(out.read_text(encoding="utf-8"))["ideas"][0]["idea_id"] == "2026-09-23-1"
    assert not (tmp_path / "prod_pitch_today.json").exists()
    # An explicit receipt is a dev run even against the production journal path.
    assert dp.publish_site_payload(dp.site_payload(ASOF, []), pj.JOURNAL_PATH.with_name(
        "never_written.jsonl"), Args(delivery_receipt=str(tmp_path / "r.json")), ASOF)


def test_production_run_uploads_to_the_function_key(tmp_path, monkeypatch):
    import cache_io
    calls = []
    monkeypatch.setattr(cache_io, "upload_from_local", lambda src, key: calls.append(key) or True)
    monkeypatch.setattr(cache_io, "is_configured", lambda: True)
    monkeypatch.setattr(dp, "SITE_PAYLOAD_PATH", tmp_path / "pitch_today.json")
    assert dp.publish_site_payload(dp.site_payload(ASOF, []), pj.JOURNAL_PATH, Args(), ASOF)
    assert calls == [dp.SITE_R2_KEY]


def test_upload_failure_is_loud_but_never_raises(tmp_path, monkeypatch, capsys):
    import cache_io
    monkeypatch.setattr(cache_io, "upload_from_local", lambda *a: False)
    monkeypatch.setattr(cache_io, "is_configured", lambda: True)
    monkeypatch.setattr(dp, "SITE_PAYLOAD_PATH", tmp_path / "pitch_today.json")
    assert dp.publish_site_payload(dp.site_payload(ASOF, []), pj.JOURNAL_PATH, Args(), ASOF) is False
    assert "FAILED" in capsys.readouterr().out
    monkeypatch.setattr(dp, "SITE_PAYLOAD_PATH", tmp_path / "missing" / "\0bad")
    assert dp.publish_site_payload(dp.site_payload(ASOF, []), pj.JOURNAL_PATH, Args(), ASOF) is False


# --- the wiring -----------------------------------------------------------------

def test_publisher_and_function_agree_on_the_r2_key():
    key = re.search(r'CHARTS\.get\("([^"]+)"\)', FUNCTION.read_text(encoding="utf-8"))
    assert key, "the Function lost its R2 get"
    assert key.group(1) == dp.SITE_R2_KEY == "pitch_today.json"


def test_function_is_read_only_and_no_store():
    src = FUNCTION.read_text(encoding="utf-8")
    assert "onRequestGet" in src and "onRequestPost" not in src
    assert '"Cache-Control": "no-store"' in src
    assert ".put(" not in src and ".delete(" not in src


def test_tab_fetches_the_function_route():
    assert 'PITCH_ENDPOINT = "/pitch-today"' in PITCH_JS.read_text(encoding="utf-8")
    assert FUNCTION.name == "pitch-today.js"


def test_pitch_page_is_registered_in_the_nav_right_after_radar():
    src = COMMON_JS.read_text(encoding="utf-8")
    assert ('{ href: "radar.html",    label: "Radar" },\n'
            '  { href: "pitch.html",    label: "Pitch" },') in src
    html = PITCH_HTML.read_text(encoding="utf-8")
    assert 'assets/pitch.js' in html and 'data-page="pitch"' in html


def test_site_tag_matches_pitch_moo_strategy():
    assert "strat: `Pitch-${idea.idea_id}`" in PITCH_JS.read_text(encoding="utf-8")
    runner = Path("~").expanduser() / "OneDrive" / "trading_ibkr" / "pitch_moo.py"
    if not runner.exists():
        pytest.skip(f"live execution dir not present: {runner.parent}")
    assert "f\"Pitch-{row['Idea_Id']}\"" in runner.read_text(encoding="utf-8")


def test_tab_never_substitutes_or_derives_off_reference_close():
    src = PITCH_JS.read_text(encoding="utf-8")
    assert "refCloseLevels" not in src and "substitute: market order" not in src
    assert 'const PITCH_OPG_CUTOFF = "09:25"' in src
    assert 'PITCH_PASS_AFTER = { auction: "09:05", open: "09:32" }' in src
    exec_js = (ROOT / "site" / "assets" / "execution.js").read_text(encoding="utf-8")
    assert 'const PITCH_STAGE_TYPES = ["LMT", "MOO", "MOC"]' in exec_js
    assert 'const PITCH_OPG_CUTOFF = "09:25"' in exec_js


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_pitch_tab_javascript_contract():
    result = subprocess.run([shutil.which("node"), str(ROOT / "tests" / "js" / "test_pitch_tab.js")],
                            cwd=ROOT, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
