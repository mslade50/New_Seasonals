"""Stop-limit (STP LMT) entry on the site ticket -- the repo half (2026-08-18).

The executor and agent gates live in OneDrive and are covered by
`test_stop_limit_entry.py` there; this file guards the two things CI can see:
the browser ticket carries the new type and its cap field, and every layer's
entry-type vocabulary still agrees. The invariant behind all of it: a stop-limit
fills anywhere between trigger and cap, so risk / notional / R:R / the target
comparison read the CAP, never the trigger.
"""
import os
import re
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXEC_JS = ROOT / "site" / "assets" / "execution.js"
SCHEMA_DOC = ROOT / "docs" / "site_execution_schema.md"
IBKR_DIR = Path(os.path.expanduser("~")) / "OneDrive" / "trading_ibkr"

ENTRY_TYPES = {"LMT", "STP_LMT", "MKT", "MOO", "MOC"}


@pytest.fixture(scope="module")
def js():
    return EXEC_JS.read_text(encoding="utf-8")


def test_ticket_offers_stop_limit_with_a_cap_field(js):
    assert '<option value="STP_LMT"' in js
    assert 'inp("f_entry_cap"' in js
    assert '"f_entry_cap"' in js.split("const TICKET_FIELDS")[1].split("]")[0], \
        "f_entry_cap must be in TICKET_FIELDS or a Type toggle wipes it"
    # The cap input and the entry-expiry field are both shown for STP_LMT.
    sync = js.split("function syncEntryTypeFields()")[1].split("\n}")[0]
    assert 'typ === "STP_LMT" ? "contents"' in sync
    assert '(typ === "LMT" || typ === "STP_LMT")' in sync


def test_payload_emits_entry_cap_only_for_stop_limit(js):
    assert 'entry_cap: entry_type === "STP_LMT" ? numOrNull("f_entry_cap") : null' in js
    # Entry expiry rides along with the stop-limit parent (the radar's fill window).
    assert 'expiry: (entry_type === "LMT" || entry_type === "STP_LMT")' in js


def test_client_gates_and_readout_use_the_worst_fill(js):
    warns = js.split("function bracketWarnings()")[1].split("\n}")[0]
    assert "const worst = cap != null ? cap : entry;" in warns
    assert 'warns.push("BUY STP_LMT needs trigger < cap")' in warns
    assert 'warns.push("SELL STP_LMT needs cap < trigger")' in warns
    assert "worst < target" in warns and "target < worst" in warns
    # Risk, R:R and notional in the readout all price the worst fill, so the
    # operator reads the same numbers the agent enforces.
    assert "const dist = Math.abs(worst - stop);" in js
    assert "Math.abs(target - worst) / dist" in js
    assert "qty * worst * mult" in js


def test_stop_limit_is_stock_only_on_every_layer(js):
    assert 'warns.push("STP_LMT entry supports stocks only")' in js


def test_schema_doc_documents_the_cap_and_the_gap_behaviour():
    doc = SCHEMA_DOC.read_text(encoding="utf-8")
    assert "LMT|STP_LMT|MKT|MOO|MOC" in doc
    assert "entry_cap" in doc
    # The one operational surprise worth keeping written down.
    assert "does **not** die" in doc


def test_entry_type_vocabulary_matches_across_layers(js):
    """A type added on one side only is the drift this catches."""
    js_list = re.search(r'if \(!\[([^\]]+)\]\.includes\(orderType\)\)', js).group(1)
    js_types = set(re.findall(r'"([A-Z_]+)"', js_list))
    assert js_types == ENTRY_TYPES, js_types

    if not IBKR_DIR.is_dir():
        pytest.skip(f"live execution dir not present: {IBKR_DIR}")
    sys.path.insert(0, str(IBKR_DIR))
    for module_name in ("execute_order", "exec_agent"):
        try:
            src = (IBKR_DIR / f"{module_name}.py").read_text(encoding="utf-8")
        except OSError as exc:
            pytest.skip(f"{module_name} unreadable here ({exc})")
        found = re.search(r'entry_type not in \(([^)]+)\)', src)
        assert found, f"{module_name} lost its entry_type whitelist"
        assert set(re.findall(r'"([A-Z_]+)"', found.group(1))) == ENTRY_TYPES, module_name
