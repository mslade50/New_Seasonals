"""Skip markers for the RETIRED one-shot broker preparers (2026-09-21).

`broker_runtime/prepare_*.py` modules are one-shot installers: each builds a
candidate copy of the live trading runtime from pinned source fragments, the
owner installs it by hand, and after that the fragment's anchors are gone from
live by construction. A preparer whose patch is installed is SPENT.

The tests below were written to exercise the *patched* runtime by re-applying
the preparer to the live source. Once the patch ships, that rebuild cannot
succeed, and re-applying it is not merely useless -- `prepare_execution_repairs`
would double-inject the capped-limit option block and regress cancel/modify
from `manual_order_actions` back to `order_mutations`.

So these tests are retired with an explicit reason rather than left red. The
coverage they provided lives on in the guard suites that read the INSTALLED
runtime directly (`tests/test_execution_entry_controls.py`,
`tests/test_exec_reporting_fixes.py`, and the OneDrive suite).

Inventory and evidence:
`artifacts/recon_2026-09-17/reviewed_runtime_drift_review.md`
("Preparer fragments -- reviewed, NOT refreshed" + "Test impact").
"""
from __future__ import annotations

import pytest

from broker_runtime.prepare_entry_controls import SPENT as ENTRY_CONTROLS_SPENT
from broker_runtime.prepare_execution_repairs import SPENT as EXECUTION_REPAIRS_SPENT

RECONCILIATION_SPENT = (
    "prepare_broker_reconciliation is a retired one-shot installer: the reconciliation "
    "modules were installed 2026-09-17 12:34 and live is byte-identical to the reviewed "
    "candidate, so prepare() now refuses with 'A reconciliation module already exists'. "
    "See artifacts/recon_2026-09-17/reviewed_runtime_drift_review.md items 3-5.")

MANUAL_ORDER_ACTIONS_SPENT = (
    "prepare_manual_order_actions is a retired one-shot installer: live routes cancel/modify "
    "through manual_order_actions already (installed 2026-09-17 12:34) and delegates the "
    "handlers to execution_lifecycle, so the rewrite it expects to make is a no-op on every "
    "file but main. See artifacts/recon_2026-09-17/reviewed_runtime_drift_review.md.")

PREPARE_OLV_SPENT = (
    "broker_runtime.prepare.patch_olv is a retired one-shot installer: the 2026-09-09 primary "
    "OLV cutover replaced the olv_exit_moo.py monolith with a 15-line dispatcher, so the "
    "load_exit_rows anchor it rewrites no longer exists there. The live implementation is "
    "olv_exit_primary.py / olv_exit_pa_legacy.py. See "
    "artifacts/recon_2026-09-17/reviewed_runtime_drift_review.md item 9.")

PREPARE_AUCTION_SPENT = (
    "broker_runtime.prepare.patch_auction is a retired one-shot installer: the durable "
    "auction-identity change it applies was installed into event_moo.py on 2026-09-17 12:09 "
    "(commit 076a113c), so its anchors are consumed. See "
    "artifacts/recon_2026-09-17/reviewed_runtime_drift_review.md item 10.")

retired_execution_repairs = pytest.mark.skip(reason=EXECUTION_REPAIRS_SPENT)
retired_entry_controls = pytest.mark.skip(reason=ENTRY_CONTROLS_SPENT)
retired_broker_reconciliation = pytest.mark.skip(reason=RECONCILIATION_SPENT)
retired_manual_order_actions = pytest.mark.skip(reason=MANUAL_ORDER_ACTIONS_SPENT)
retired_prepare_olv = pytest.mark.skip(reason=PREPARE_OLV_SPENT)
retired_prepare_auction = pytest.mark.skip(reason=PREPARE_AUCTION_SPENT)


def skip_execution_repairs():
    """For a fixture, where a skip mark cannot be applied."""
    pytest.skip(EXECUTION_REPAIRS_SPENT)


def skip_prepare_olv():
    pytest.skip(PREPARE_OLV_SPENT)


def skip_prepare_auction():
    pytest.skip(PREPARE_AUCTION_SPENT)
