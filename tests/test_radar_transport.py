"""Radar recs transport: local publisher -> R2 -> Pages Function -> Radar tab.

The theme is that nothing in this chain may invent a number. The radar's book
engine mints the plans; the publisher copies a whitelist of fields, the Function
streams the bytes, and the tab formats them. These tests pin the whitelist, the
R2 key that ties publisher and Function together, and the strategy tag that ties
a staged order to `radar_trail_sync.py`.
"""
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
UPLOADER = ROOT / "scripts" / "upload_radar_recs.py"
FUNCTION = ROOT / "functions" / "radar-recs.js"
RADAR_JS = ROOT / "site" / "assets" / "radar.js"
RADAR_HTML = ROOT / "site" / "radar.html"
COMMON_JS = ROOT / "site" / "assets" / "common.js"
IBKR_DIR = Path("~").expanduser() / "OneDrive" / "trading_ibkr"


@pytest.fixture(scope="module")
def uploader():
    spec = importlib.util.spec_from_file_location("upload_radar_recs", UPLOADER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["upload_radar_recs"] = mod
    spec.loader.exec_module(mod)
    return mod


def _recs():
    return {
        "date": "2026-08-16", "account_value": 250000, "generated_at": "ignored",
        "regime": {"active_rule": "full"}, "budget": {"prorata_scale": 0.75},
        "staleness": {"momentum_json_td": 1}, "mint_blocked": None,
        "scoreboard": {"n_closed": 0},
        "new_recs": [{
            "plan_id": "AMG-2026-08-16", "ticker": "AMG", "setup_grade": "A",
            "plan_type": "breakout_stop", "status": "RECOMMENDED",
            "entry": {"order": "BUY_STOP_LIMIT", "trigger": 383.12, "limit_cap": 400.17},
            "stop": {"price": 355.84, "atr_mult": 2.0},
            "targets": {"t1": 437.68, "t1_frac": 0.3333},
            "time": {"valid_through": "2026-08-28", "time_exit_date": "2026-11-13"},
            "sizing": {"shares": 33, "risk_bps": 36.01},
            "ticket": "BUY STOP 383.12 cap 400.17 | ...",
            "covariates": {"rs_rank": 87.1},          # not whitelisted
            "adv_dollars": 99171830.9664917,          # not whitelisted
        }],
        "open_positions": [{
            "plan_id": "X-1", "ticker": "X", "status": "OPEN", "current_stop": 12.5,
            "stop_kind": "trail", "shares_remaining": 100, "next_action": "RAISE STOP TO 12.50",
            "secret_internal": "drop me",
        }],
        "plan_only": [1, 2], "watch_only": [1], "budget_cut": [], "zeroed": [],
        "closed_this_week": [], "expired": [], "rebased": [],
    }


# --- the publisher --------------------------------------------------------------

def test_payload_copies_plan_numbers_verbatim(uploader):
    p = uploader.build_payload(_recs(), Path("a/b/data/recs/2026-08-16.json"), "pulled")
    rec = p["new_recs"][0]
    assert rec["entry"] == {"order": "BUY_STOP_LIMIT", "trigger": 383.12, "limit_cap": 400.17}
    assert rec["stop"]["price"] == 355.84
    assert rec["sizing"]["shares"] == 33
    assert rec["ticket"].startswith("BUY STOP 383.12")


def test_payload_drops_unwhitelisted_fields(uploader):
    """A new engine field must not reach the browser unreviewed."""
    p = uploader.build_payload(_recs(), Path("a/b/data/recs/2026-08-16.json"), "pulled")
    assert "covariates" not in p["new_recs"][0]
    assert "adv_dollars" not in p["new_recs"][0]
    assert "secret_internal" not in p["open_positions"][0]
    assert p["open_positions"][0]["current_stop"] == 12.5


def test_payload_carries_provenance_and_age(uploader):
    p = uploader.build_payload(_recs(), Path("a/b/data/recs/2026-08-16.json"), "pulled")
    assert p["date"] == "2026-08-16"
    assert isinstance(p["age_days"], int) and p["age_days"] >= 0
    assert p["pull"] == "pulled"
    assert p["generated_at"].endswith("UTC")
    assert p["counts"]["new_recs"] == 1 and p["counts"]["plan_only"] == 2


def test_payload_survives_an_unparseable_date(uploader):
    bad = _recs() | {"date": "not-a-date"}
    assert uploader.build_payload(bad, Path("a/b/c/d.json"), "pulled")["age_days"] is None


def test_payload_is_json_serialisable(uploader):
    p = uploader.build_payload(_recs(), Path("a/b/data/recs/2026-08-16.json"), "pulled")
    assert json.loads(json.dumps(p))["new_recs"][0]["ticker"] == "AMG"


# --- the wiring -----------------------------------------------------------------

def test_publisher_and_function_agree_on_the_r2_key(uploader):
    key = re.search(r'CHARTS\.get\("([^"]+)"\)', FUNCTION.read_text(encoding="utf-8"))
    assert key, "the Function lost its R2 get"
    assert key.group(1) == uploader.R2_KEY == "radar_recs.json"


def test_function_is_read_only_and_no_store():
    src = FUNCTION.read_text(encoding="utf-8")
    assert "onRequestGet" in src and "onRequestPost" not in src
    assert '"Cache-Control": "no-store"' in src
    assert ".put(" not in src and ".delete(" not in src


def test_tab_fetches_the_function_route():
    assert 'RADAR_ENDPOINT = "/radar-recs"' in RADAR_JS.read_text(encoding="utf-8")
    assert FUNCTION.name == "radar-recs.js"


def test_radar_page_is_registered_in_the_nav_and_loads_its_asset():
    assert '{ href: "radar.html",    label: "Radar" },' in COMMON_JS.read_text(encoding="utf-8")
    html = RADAR_HTML.read_text(encoding="utf-8")
    assert 'assets/radar.js' in html and 'data-page="radar"' in html


def test_site_tag_matches_what_the_trail_job_looks_for():
    """A staged order the trail job cannot find is a stop that never moves."""
    site_tag = re.search(r'RADAR_STRATEGY = "([^"]+)"', RADAR_JS.read_text(encoding="utf-8"))
    assert site_tag, "radar.js lost its strategy tag"
    sync = IBKR_DIR / "radar_trail_sync.py"
    if not sync.exists():
        pytest.skip(f"live execution dir not present: {IBKR_DIR}")
    job_tag = re.search(r'RADAR_STRATEGY = "([^"]+)"', sync.read_text(encoding="utf-8"))
    assert job_tag and job_tag.group(1) == site_tag.group(1) == "Momentum_Radar"
