"""Guards for the Events site tab wiring (2026-08-21).

The tab exists so an event-sleeve order is never a surprise: nav entry,
page/JS wiring, the payload registration in build_site, and the sizing-basis
disclosure (fixed ACCOUNT_VALUE, not live NLV) all pinned here.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"


def test_nav_has_events_tab():
    common = (SITE / "assets" / "common.js").read_text(encoding="utf-8")
    assert "events.html" in common


def test_events_page_wiring():
    html = (SITE / "events.html").read_text(encoding="utf-8")
    assert "assets/events.js" in html and "assets/common.js" in html
    # The sizing-basis disclosure lives on the page itself.
    assert "not live" in html and "$750k" in html
    js = (SITE / "assets" / "events.js").read_text(encoding="utf-8")
    assert "data/event_sleeve.json" in js
    assert 'renderNav("events.html")' in js


def test_build_site_registers_payload():
    src = (ROOT / "scripts" / "build_site.py").read_text(encoding="utf-8")
    assert "def build_event_sleeve" in src
    assert '"event_sleeve": False' in src
    assert 'best_effort("event_sleeve", build_event_sleeve)' in src


def test_scan_email_subject_flags_staged_event():
    src = (ROOT / "daily_scan.py").read_text(encoding="utf-8")
    assert "_staged_event" in src
    assert "_event_suffix" in src
    # Both subject branches (signals and no-signals) carry the flag.
    assert src.count("{_event_suffix}") == 2
