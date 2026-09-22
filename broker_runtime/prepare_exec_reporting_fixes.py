"""Prepare the execution-reporting fixes; never install, arm, or connect.

Built 2026-09-21 from `docs/exec_reporting_fix_brief_2026-09-21.md`, agent-side
items 1-6. Same shape as ``prepare_entry_controls.py``: pinned source fragments
applied with ``replace_once`` into a candidate directory that also keeps a
``.original`` copy of every file and a ``manifest.json`` of source/candidate
sha256s. Nothing here writes to the live runtime.

Hunks, by brief item:

1.  ``test_olv_exits.py`` -- the OneDrive guard suite has not collected since
    the 2026-09-09 primary OLV cutover turned ``olv_exit_moo`` into a 15-line
    dispatcher. The file's 16 ``runner.*`` attributes all live on
    ``olv_exit_pa_legacy`` (the monolith's descendant: same ``OLV_Exits`` tab,
    same ``olv_exit_placed.json``), so the binding moves there and the shim
    keeps a dispatcher assertion of its own.
2.  ``position_actions.PreTransmitRefusal`` + ``blocking_lock`` /
    ``blocking_summary`` + ``created`` and ``command_type`` stamps, a matching
    catch at the command boundary in ``execute_order.main``, and forwarding of
    the structured field through ``exec_agent``. A refusal decided before any
    broker mutation now reports ``state="rejected"`` with a ``lock`` object
    {symbol, action_type, action_id, created_at, discrepancy} that the site's
    ``lockRejection()`` parses into a Clear-lock button; ``unknown`` is
    reserved for exceptions after something was transmitted.
3.  ``execute_order`` IBKR notice classification: 399 and the 2100-2199
    warning block are recorded as notices, never as placement errors, and are
    appended to a successful result's detail.
4.  Diagnostic only (no behaviour change) on the cancel/resize loop in
    ``position_actions.adjust_exits``: name the leg and the already-cancelled
    rung count, because the live failures were an OCA sibling going terminal
    on its own. See the design question in the install runbook.
5.  ``position_action_resolve``: a new read-only command that records the live
    book against one unresolved action and marks it done. Handler in
    ``position_actions.resolve``, dispatch + ``SUPPORTED`` in
    ``execute_order``, validate/preview/describe + the ``resolved`` rename in
    ``exec_agent``. Payload {account, symbol, action_id, operator_note};
    response {state: resolved|rejected, reason, snapshot:{positions,
    open_orders}} per ``docs/site_execution_schema.md`` (site PR #71).
6.  No change: the 2026-09-14 ``futures_front.py`` install already supplies the
    validated exchange, front expiry and conId the two mapping failures wanted.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

# --------------------------------------------------------------------------
# Item 1 -- test_olv_exits.py binds to the module that owns the implementation
# --------------------------------------------------------------------------
OLV_TEST_IMPORT_OLD = """import olv_exit_moo as runner
import order_staging as staging
"""
OLV_TEST_IMPORT_NEW = '''# 2026-09-09 primary OLV cutover: olv_exit_moo.py is now a 15-line dispatcher
# that calls olv_exit_primary.main() and olv_exit_pa_legacy.main() in turn. The
# monolith these fixtures were written against survives as olv_exit_pa_legacy
# (same OLV_Exits tab, same olv_exit_placed.json), so the runner binding moves
# there. Binding it to the shim raised AttributeError at import and aborted
# collection of the WHOLE 18-file suite, which is why no execution guard ran
# between 2026-09-09 and 2026-09-21.
import olv_exit_moo as dispatcher
import olv_exit_pa_legacy as runner
import order_staging as staging
'''

OLV_TEST_DISPATCH_ANCHOR = """def _base_sig(symbol='A', texit='2026-09-10'):
    return runner.signal_ref(symbol, 'SELL', 'Oversold Low Volume', TODAY, texit)
"""
OLV_TEST_DISPATCH_NEW = """def test_scheduled_entry_point_dispatches_to_both_runners():
    \"\"\"The Task Scheduler entry point is olv_exit_moo.main; it must keep calling
    BOTH runners and return the worst exit code. This is the assertion that was
    lost when the module became a shim and the fixtures stopped collecting.\"\"\"
    source = Path(dispatcher.__file__).resolve().read_text(encoding='utf-8')
    assert 'olv_exit_primary' in source and 'olv_exit_pa_legacy' in source
    assert callable(dispatcher.main)


def _base_sig(symbol='A', texit='2026-09-10'):
    return runner.signal_ref(symbol, 'SELL', 'Oversold Low Volume', TODAY, texit)
"""

# ROOT must name the RUNTIME tree, not the tree the test file happens to sit in:
# the executor source-tree hash and the runner-source assertion below are about
# the installed runtime, and a candidate overlay puts the test file elsewhere.
OLV_TEST_ROOT_OLD = """ROOT = Path(__file__).resolve().parent
"""
OLV_TEST_ROOT_NEW = """ROOT = Path(runner.__file__).resolve().parent
"""

# The shim carries no placement code, so the OCA assertion has to read the
# runner that actually places the standalone sell.
OLV_TEST_SOURCE_OLD = """    source = (ROOT / 'olv_exit_moo.py').read_text(encoding='utf-8')
"""
OLV_TEST_SOURCE_NEW = """    source = Path(runner.__file__).resolve().read_text(encoding='utf-8')
"""

# --------------------------------------------------------------------------
# Item 3 -- IBKR notice classification (execute_order.py)
# --------------------------------------------------------------------------
NOTICE_OLD = '''# Capture real IBKR errors (e.g. a cancel refusal) so the result says WHY.
_ERRORS = []
_BENIGN = {1102, 2103, 2104, 2105, 2106, 2107, 2108, 2110, 2119, 2150, 2158}


def _on_err(reqId, code, msg, *a):
    if code not in _BENIGN:
        _ERRORS.append(f"{code}:{msg}")


def _out(ok, state, detail, fill=None):
    print(json.dumps({"ok": ok, "state": state, "detail": detail, "fill": fill}))
    return 0
'''
NOTICE_NEW = '''# Capture real IBKR errors (e.g. a cancel refusal) so the result says WHY.
_ERRORS = []
# Informational IBKR order/system messages. These arrive ON ORDERS THAT ARE
# LIVE, so counting them as placement failures told the operator to verify a
# working order by hand and not to retry (2026-09-16: a repriced RTX entry, two
# outside-RTH exit_attach warnings). They are recorded separately and surfaced
# on the successful result instead.
#   399         "Order Message:" -- e.g. repriced so as not to cross a resting order
#   2100-2199   the TWS API warning block (2109 = outside-RTH attribute ignored)
_NOTICES = []
_NOTICE_CODES = {399}
_NOTICE_RANGE = (2100, 2199)
_BENIGN = {1102, 2103, 2104, 2105, 2106, 2107, 2108, 2110, 2119, 2150, 2158}


def _is_notice(code):
    """True for an informational IBKR message that does not deny placement."""
    try:
        code = int(code)
    except (TypeError, ValueError):
        return False
    return code in _NOTICE_CODES or _NOTICE_RANGE[0] <= code <= _NOTICE_RANGE[1]


def _on_err(reqId, code, msg, *a):
    if code in _BENIGN:
        return
    if _is_notice(code):
        _NOTICES.append(f"{code}:{msg}")
        return
    _ERRORS.append(f"{code}:{msg}")


def _out(ok, state, detail, fill=None, **extra):
    # A notice never changes ok/state; it is appended so the operator still
    # reads what the broker said about an order that went through.
    if ok and _NOTICES:
        detail = f"{detail} [IBKR notice: {'; '.join(_NOTICES[-4:])}]"
    result = {"ok": ok, "state": state, "detail": detail, "fill": fill}
    # Structured side channel for the Execution tab: `lock` (which unresolved
    # action refused this command) and `snapshot` (what position_action_resolve
    # recorded). exec_agent forwards them verbatim; the ok/state/detail contract
    # its trustworthy-result gate checks is unchanged, and a None is dropped so
    # every existing caller still emits exactly the four original keys.
    result.update({k: v for k, v in extra.items() if v is not None})
    print(json.dumps(result))
    return 0
'''

# --------------------------------------------------------------------------
# Items 2 + 5 -- execute_order: SUPPORTED, boundary catch, resolve dispatch
# --------------------------------------------------------------------------
# The command TYPE is journalled on a new position-action record so a later
# lock refusal can say WHICH action blocks (the site renders "Blocked by
# unresolved <action_type> on <symbol>"). It travels through globals(), not the
# payload: position_actions compares a reloaded record's payload byte for byte
# against the incoming one, so a new payload key would make every pre-install
# unresolved action unreconcilable -- exactly the records this fix exists for.
COMMAND_TYPE_OLD = '''    p["_command_id"] = str(cmd.get("id") or "").strip()
'''
COMMAND_TYPE_NEW = '''    p["_command_id"] = str(cmd.get("id") or "").strip()
    globals()["_COMMAND_TYPE"] = str(t or "")
'''

SUPPORTED_OLD = 'SUPPORTED.add("reconcile_exits")\n'
SUPPORTED_NEW = ('SUPPORTED.add("reconcile_exits")\n'
                 '# Read-only operator clearance of ONE unresolved position action. Places no\n'
                 '# order; it records the live book against the action and marks it done.\n'
                 'SUPPORTED.add("position_action_resolve")\n')

DISPATCH_OLD = '''        if t == "reconcile_exits":
            import reconcile_position_exits
            return reconcile_position_exits.run(globals(), ib, p, host, port, cid)
'''
DISPATCH_NEW = '''        if t == "reconcile_exits":
            import reconcile_position_exits
            return reconcile_position_exits.run(globals(), ib, p, host, port, cid)
        if t == "position_action_resolve":
            return _do_position_action_resolve(ib, p, acct)
'''

BOUNDARY_OLD = '''    except Exception as e:  # noqa: BLE001
        return _out(
            False,
            "unknown",
            f"execute became indeterminate ({type(e).__name__}); "
            "VERIFY IN TWS AND DO NOT RETRY",
        )
'''
BOUNDARY_NEW = '''    except Exception as e:  # noqa: BLE001
        # A refusal decided BEFORE any broker mutation is a clean no-op, not an
        # indeterminate outcome. Reporting it as "unknown / VERIFY IN TWS AND DO
        # NOT RETRY" was the single worst reporting defect in the 2026-09 audit:
        # it presented a working reconciliation lock as a possibly half-placed
        # order. "unknown" is now reserved for exceptions after transmission.
        if _pre_transmit_refusal(e):
            return _out(False, "rejected", f"Nothing sent: {e}", lock=_refusal_lock(e))
        return _out(
            False,
            "unknown",
            f"execute became indeterminate ({type(e).__name__}); "
            "VERIFY IN TWS AND DO NOT RETRY",
        )
'''

RESOLVE_HANDLER_OLD = '''def _do_cancel(ib, p, host, port, main_cid):
'''
RESOLVE_HANDLER_NEW = '''def _pre_transmit_refusal(exc):
    """True when `exc` was raised before any broker mutation was attempted."""
    try:
        import position_actions
    except Exception:  # noqa: BLE001
        return False
    return isinstance(exc, position_actions.PreTransmitRefusal)


def _refusal_lock(exc):
    """The structured lock a refusal carries, or None.

    The Execution tab's `lockRejection()` needs symbol + action_type +
    action_id together before it will offer a Clear-lock button; anything less
    renders as plain detail text, so a partial dict is dropped here rather than
    shipped as a half-identity the site would have to guess at.
    """
    lock = getattr(exc, "lock", None)
    if not isinstance(lock, dict):
        return None
    if not all(str(lock.get(k) or "") for k in ("symbol", "action_type", "action_id")):
        return None
    return lock


def _do_position_action_resolve(ib, p, acct):
    import position_actions
    return position_actions.resolve(globals(), ib, p, acct)


def _do_cancel(ib, p, host, port, main_cid):
'''

# --------------------------------------------------------------------------
# Items 2/4/5 -- position_actions
# --------------------------------------------------------------------------
PA_REFUSAL_OLD = '''TERMINAL = {"Filled", "Cancelled", "ApiCancelled", "Inactive"}
ACKNOWLEDGED = {"Submitted", "PreSubmitted"}
'''
PA_REFUSAL_NEW = '''TERMINAL = {"Filled", "Cancelled", "ApiCancelled", "Inactive"}
ACKNOWLEDGED = {"Submitted", "PreSubmitted"}
OPERATOR_NOTE_MIN = 8


class PreTransmitRefusal(ValueError):
    """A refusal decided BEFORE any broker mutation was attempted.

    Nothing was sent, so the correct report is ``state="rejected"``. The command
    boundary maps this class specifically; a bare ValueError from further in
    still degrades to ``unknown`` because it cannot prove non-delivery.

    ``lock`` carries the structured identity of the blocking action when there
    is one, so the Execution tab can offer to clear exactly that action instead
    of asking the operator to find it.
    """

    def __init__(self, message, lock=None):
        super().__init__(message)
        self.lock = lock if isinstance(lock, dict) else None


def blocking_lock(record, kind):
    """The structured identity of the action that blocks this command.

    Field names are the site contract (`docs/site_execution_schema.md`,
    `lockRejection()`): symbol, action_type, action_id, created_at, discrepancy.
    ``action_type`` is the journalled command type where the record has one;
    records written before this change carry none, so it degrades to the generic
    kind rather than guessing a type from the record's shape.
    """
    payload = record.get("payload") or {}
    con_id = payload.get("con_id") or (record.get("identity") or [None, None])[1]
    return {"symbol": str(payload.get("symbol") or "").upper() or f"conId {con_id}",
            "action_type": str(record.get("command_type") or "").strip() or kind,
            "action_id": str(record.get("id") or ""),
            "created_at": str(record.get("created") or record.get("session") or ""),
            "discrepancy": str((record.get("observation") or {}).get("detail")
                               or record.get("error")
                               or ((record.get("result") or {}).get("detail") if record.get("result") else "")
                               or f"phase {record.get('phase')}"),
            "account": str(record.get("account_key") or "")}


def blocking_summary(lock):
    """One line naming WHICH earlier action blocks this one, and since when.

    The audit's complaint about 'An earlier position action is unresolved' was
    that it never said which one, on what, from when, or what the broker
    actually disagreed about. Kept as prose as well as structure because a
    relay that drops the structured field must still print something useful.
    """
    return (f"blocked by {lock['action_type']} {lock['action_id'] or 'unknown'} "
            f"on {lock['symbol']} (opened {lock['created_at'] or 'unknown time'}): "
            f"{lock['discrepancy']}")
'''

PA_LOCK_OLD = '''            for previous in records(root):
                if previous["phase"] != "done" and previous["payload"]["_broker_account"] == payload["_broker_account"] and previous["payload"]["con_id"] == payload["con_id"]:
                    raise ValueError("An earlier position action is unresolved; reconcile it before another")
            for previous in records(root / "order_edits"):
                if (previous["phase"] != "done" and previous["identity"][:2]
                        == [payload["_broker_account"], int(payload["con_id"])]):
                    raise ValueError("An earlier order edit is unresolved; reconcile it before a position action")
'''
PA_LOCK_NEW = '''            for previous in records(root):
                if previous["phase"] != "done" and previous["payload"]["_broker_account"] == payload["_broker_account"] and previous["payload"]["con_id"] == payload["con_id"]:
                    lock = blocking_lock(previous, "position action")
                    raise PreTransmitRefusal(
                        "An earlier position action is unresolved; reconcile it before another -- "
                        + blocking_summary(lock), lock)
            for previous in records(root / "order_edits"):
                if (previous["phase"] != "done" and previous["identity"][:2]
                        == [payload["_broker_account"], int(payload["con_id"])]):
                    lock = blocking_lock(previous, "order edit")
                    raise PreTransmitRefusal(
                        "An earlier order edit is unresolved; reconcile it before a position action -- "
                        + blocking_summary(lock), lock)
'''

PA_CREATED_OLD = '''                      "session": datetime.now(ZoneInfo("America/New_York")).date().isoformat(),
'''
PA_CREATED_NEW = '''                      "session": datetime.now(ZoneInfo("America/New_York")).date().isoformat(),
                      "created": datetime.now(ZoneInfo("America/New_York")).isoformat(timespec="seconds"),
                      "command_type": str(ns.get("_COMMAND_TYPE") or ""),
'''

PA_EXCEPT_OLD = '''    except Exception as exc:
        attempted = record is not None and any(record.get(key) for key in ("mutation", "wire", "addition"))
'''
PA_EXCEPT_NEW = '''    except Exception as exc:
        # A PreTransmitRefusal is decided before this command touches the broker,
        # so an EARLIER command's mutation marker on a reloaded record must not
        # promote it to "unknown".
        attempted = (not isinstance(exc, PreTransmitRefusal)
                     and record is not None
                     and any(record.get(key) for key in ("mutation", "wire", "addition")))
'''

PA_DIAGNOSTIC_OLD = '''        for (connection, original), (_, target, leg) in zip(resolved, plans):
            trade = life.fresh_for_edit(ns, connection, tuple(leg["identity"]))
'''
PA_DIAGNOSTIC_NEW = '''        for (connection, original), (_, target, leg) in zip(resolved, plans):
            try:
                trade = life.fresh_for_edit(ns, connection, tuple(leg["identity"]))
            except ValueError as exc:
                # Diagnostic only -- the abort is DELIBERATELY unchanged; see the
                # design question in the install runbook. All six live cases
                # (MCHP/HXL/SNA/RTX 2026-09-16, ENTG 09-17, JHX 09-21, all PA,
                # all full closes) are the same shape: every rung scales to 0,
                # rung 1 of an OCA pair is cancelled, IBKR takes its sibling
                # terminal, and find_exact -- which skips terminal rows -- then
                # matches ZERO orders. The message said "cannot resolve exactly
                # one" and named nothing, so it read like an ambiguity.
                raise ValueError(
                    f"{exc} [leg {leg['source_key']} -> {leg['scaled_qty']}, "
                    f"OCA {leg.get('oca_group') or 'none'}; "
                    f"{len(record['removed'])} of {len(plans)} rung(s) already cancelled "
                    "this run -- a cancelled OCA sibling goes terminal on its own]") from exc
'''

PA_RESOLVE_OLD = '''def run(ns, ib, payload, account_key, host, port, cid, *, adding=False):
'''
PA_RESOLVE_NEW = '''def open_orders_for(ns, ib, broker_account, con_id):
    """Every order on this exact account/contract, as plain rows.

    Reads through the executor's ``_fresh_open_trades`` when it is available:
    ``ib.openTrades()`` alone returns only THIS client's orders, and the whole
    point of the snapshot is to show the operator what every client left behind
    on the contract. Terminal rows are kept -- a stale working leg that has
    since gone Cancelled is exactly the evidence that clears a lock.
    """
    reader = ns.get("_fresh_open_trades") if hasattr(ns, "get") else None
    if callable(reader):
        trades = list(reader(ib))
    else:
        ib.reqAllOpenOrders()
        trades = list(ib.openTrades())
    rows = []
    for trade in trades:
        order = trade.order
        if (str(getattr(order, "account", "")) != broker_account
                or int(getattr(trade.contract, "conId", 0) or 0) != int(con_id)):
            continue
        rows.append({"client_id": int(getattr(order, "clientId", -1)),
                     "order_id": int(getattr(order, "orderId", 0) or 0),
                     "perm_id": int(getattr(order, "permId", 0) or 0),
                     "action": str(getattr(order, "action", "")),
                     "order_type": str(getattr(order, "orderType", "")),
                     "quantity": float(getattr(order, "totalQuantity", 0) or 0),
                     "filled": float(getattr(trade.orderStatus, "filled", 0) or 0),
                     "status": str(getattr(trade.orderStatus, "status", "")),
                     "order_ref": str(getattr(order, "orderRef", "") or "")})
    return rows


def positions_for(ib, broker_account, con_id):
    """Held rows on this exact account/contract."""
    ib.reqPositions()
    return [{"account": str(row.account), "con_id": int(row.contract.conId),
             "symbol": str(getattr(row.contract, "symbol", "")),
             "position": float(row.position), "avg_cost": float(getattr(row, "avgCost", 0) or 0)}
            for row in ib.positions()
            if str(row.account) == broker_account and int(row.contract.conId) == int(con_id)]


def resolve(ns, ib, payload, account_key):
    """Operator clearance of ONE unresolved position action. Sends NO order.

    Re-reads the live book for the action's exact account/contract, records the
    positions and working orders it saw into the action's state record, and
    marks the record done with ``resolved_by = "operator"``. This is the site's
    equivalent of the hand-written ``resolution`` blocks that until now could
    only be produced by a one-off script on the trading box, which is why an
    unresolved action latched a lock the Execution tab could create but never
    clear.

    Refuses (pre-transmit, so ``rejected``) when the action is unknown, already
    resolved, on another account, on another symbol, or when the operator note
    is missing.
    """
    root = journal_root(ns)
    try:
        if account_key not in {"primary", "pa"}:
            raise PreTransmitRefusal("unknown execution account")
        action_id = str(payload.get("action_id") or "").strip()
        if not action_id:
            raise PreTransmitRefusal("action_id is required")
        note = str(payload.get("operator_note") or "").strip()
        if len(note) < OPERATOR_NOTE_MIN:
            raise PreTransmitRefusal(
                f"operator_note of at least {OPERATOR_NOTE_MIN} characters is required; "
                "clearing a lock is a judgement and the journal records who made it")
        symbol = str(payload.get("symbol") or "").strip().upper()
        with operation_lock(root):
            target, scope = None, "position action"
            for candidate_root, label in ((root, "position action"),
                                          (root / "order_edits", "order edit")):
                path = record_path(candidate_root, action_id)
                if path.exists():
                    target, scope = path, label
                    break
            if target is None:
                raise PreTransmitRefusal(f"no position action or order edit with id {action_id}")
            record = json.loads(target.read_text(encoding="utf-8"))
            if record.get("phase") == "done":
                raise PreTransmitRefusal(
                    f"{scope} {action_id} is already resolved; nothing to clear")
            record_payload = record.get("payload") or {}
            if str(record.get("account_key") or account_key) != account_key:
                raise PreTransmitRefusal(
                    f"{scope} {action_id} belongs to account {record.get('account_key')}")
            recorded_symbol = str(record_payload.get("symbol") or "").strip().upper()
            if symbol and recorded_symbol and symbol != recorded_symbol:
                raise PreTransmitRefusal(
                    f"{scope} {action_id} is on {recorded_symbol}, not {symbol}")
            broker_account = str(record_payload.get("_broker_account") or "")
            con_id = int(record_payload.get("con_id") or 0)
            if not broker_account or con_id <= 0:
                raise PreTransmitRefusal(
                    f"{scope} {action_id} has no exact account/contract to re-check")
            positions = positions_for(ib, broker_account, con_id)
            orders = open_orders_for(ns, ib, broker_account, con_id)
            held = sum(row["position"] for row in positions)
            working = sum(1 for row in orders if row["status"] not in TERMINAL)
            detail = (f"{scope} {action_id} on {recorded_symbol or con_id} cleared by operator; "
                      f"live book now holds {held:g} unit(s) with {working} working order(s). "
                      f"Note: {note}")
            snapshot = {"positions": positions, "open_orders": orders}
            record["resolution"] = {
                "at": datetime.now(ZoneInfo("America/New_York")).isoformat(timespec="seconds"),
                "resolved_by": "operator", "note": note,
                "positions": positions, "open_orders": orders,
                "previous_phase": record.get("phase"), "previous_error": record.get("error"),
            }
            record["phase"] = "done"
            record["result"] = dict(ok=True, state="executed", detail=detail, fill=None)
            save(target.parent, record)
            # state stays "executed" on the wire: exec_agent only trusts a child
            # result in {executed, rejected, unknown}, and weakening that gate to
            # teach it one read-only word would weaken it for every transmit.
            # The agent renames this one to the site's "resolved" and forwards
            # the snapshot (docs/site_execution_schema.md).
            return ns["_out"](ok=True, state="executed", detail=detail, fill=None,
                              snapshot=snapshot)
    except PreTransmitRefusal as exc:
        return ns["_out"](ok=False, state="rejected", detail=f"Nothing sent: {exc}", fill=None)
    except Exception as exc:  # noqa: BLE001
        # This path never places an order, so a read failure is still a clean
        # refusal: the record is only written after both reads succeed.
        return ns["_out"](ok=False, state="rejected",
                          detail=f"Nothing sent: resolve could not read the live book ({exc})",
                          fill=None)


def run(ns, ib, payload, account_key, host, port, cid, *, adding=False):
'''

# --------------------------------------------------------------------------
# Item 5 -- exec_agent wiring (validate / preview / describe)
# --------------------------------------------------------------------------
AGENT_DESCRIBE_OLD = '''    if t == "reconcile_exits":
        return f"reconcile existing exits for {p.get('symbol', '?')} ({acct}) proportionally to live holdings"
'''
AGENT_DESCRIBE_NEW = '''    if t == "reconcile_exits":
        return f"reconcile existing exits for {p.get('symbol', '?')} ({acct}) proportionally to live holdings"
    if t == "position_action_resolve":
        return (f"clear unresolved position action {p.get('action_id', '?')} on "
                f"{p.get('symbol', '?')} ({acct}) -- records the live book, places NO order")
'''

AGENT_VALIDATE_OLD = '''    if t == "reconcile_exits":
        import reconcile_position_exits
        return reconcile_position_exits.validate(cmd)
    import position_action_agent
'''
AGENT_VALIDATE_NEW = '''    if t == "reconcile_exits":
        import reconcile_position_exits
        return reconcile_position_exits.validate(cmd)
    if t == "position_action_resolve":
        import position_actions
        problems = []
        if acct not in ("primary", "pa"):
            problems.append(f"unknown account {acct!r}")
        if not str(p.get("action_id") or "").strip():
            problems.append("action_id is required")
        if len(str(p.get("operator_note") or "").strip()) < position_actions.OPERATOR_NOTE_MIN:
            problems.append(
                f"operator_note of at least {position_actions.OPERATOR_NOTE_MIN} characters is required")
        return (not problems), problems
    import position_action_agent
'''

# Items 2 + 5 -- the structured keys have to survive the agent hop. `reply` is
# what the broker stores as the command's `result`, and it is built field by
# field, so a `lock` or `snapshot` on the child's result would otherwise be
# dropped between execute_order and the Execution tab.
AGENT_FORWARD_OLD = '''                res = await _execute_live(cmd)          # gated transmit (armed only)
                reply.update(state=res["state"], ok=res["ok"],
                             detail=res["detail"], validation={"ok": True, "reasons": []},
                             preview=preview, fill=res.get("fill"))
'''
AGENT_FORWARD_NEW = '''                res = await _execute_live(cmd)          # gated transmit (armed only)
                reply.update(state=res["state"], ok=res["ok"],
                             detail=res["detail"], validation={"ok": True, "reasons": []},
                             preview=preview, fill=res.get("fill"))
                # Structured reporting channel (docs/site_execution_schema.md).
                # `lock` identifies the unresolved action that refused this
                # command so the site can offer to clear exactly that one;
                # `snapshot` is what position_action_resolve recorded.
                if isinstance(res.get("lock"), dict):
                    reply["lock"] = res["lock"]
                if isinstance(res.get("snapshot"), dict):
                    reply["snapshot"] = res["snapshot"]
                # execute_order's terminal vocabulary is {executed, rejected,
                # unknown} and _execute_live_unlocked refuses anything else, so
                # the read-only resolve reports "executed" on the wire and is
                # renamed here to the word the Execution tab documents.
                if cmd.get("type") == "position_action_resolve" and res["state"] == "executed":
                    reply.update(state="resolved", reason=res["detail"])
                elif res["state"] == "rejected":
                    reply["reason"] = res["detail"]
'''

AGENT_PREVIEW_OLD = '''    if t == "reconcile_exits":
        return {"summary": "Reconcile existing exits to live position",
                "legs": ["Proportional exit quantities; prices and schedules preserved. No new close or re-add."]}
'''
AGENT_PREVIEW_NEW = '''    if t == "reconcile_exits":
        return {"summary": "Reconcile existing exits to live position",
                "legs": ["Proportional exit quantities; prices and schedules preserved. No new close or re-add."]}
    if t == "position_action_resolve":
        return {"summary": f"Clear unresolved position action {p.get('action_id', '?')} on {p.get('symbol', '?')}",
                "legs": ["READ-ONLY: re-reads positions and working orders for the action's exact contract.",
                         "Records what it saw into the action's journal record and marks it done.",
                         "Places, cancels and modifies NOTHING. The operator note is journalled."]}
'''


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Expected exactly one occurrence: {old[:90]!r}")
    return source.replace(old, new, 1)


def patch_olv_test(source):
    """Item 1."""
    source = replace_once(source, OLV_TEST_IMPORT_OLD, OLV_TEST_IMPORT_NEW)
    source = replace_once(source, OLV_TEST_ROOT_OLD, OLV_TEST_ROOT_NEW)
    source = replace_once(source, OLV_TEST_SOURCE_OLD, OLV_TEST_SOURCE_NEW)
    source = replace_once(source, OLV_TEST_DISPATCH_ANCHOR, OLV_TEST_DISPATCH_NEW)
    ast.parse(source)
    return source


def patch_execute_order(source):
    """Items 2, 3, 5."""
    source = replace_once(source, NOTICE_OLD, NOTICE_NEW)                    # item 3
    source = replace_once(source, COMMAND_TYPE_OLD, COMMAND_TYPE_NEW)        # item 2
    source = replace_once(source, SUPPORTED_OLD, SUPPORTED_NEW)              # item 5
    source = replace_once(source, RESOLVE_HANDLER_OLD, RESOLVE_HANDLER_NEW)  # items 2, 5
    source = replace_once(source, DISPATCH_OLD, DISPATCH_NEW)                # item 5
    source = replace_once(source, BOUNDARY_OLD, BOUNDARY_NEW)                # item 2
    ast.parse(source)
    return source


def patch_position_actions(source):
    """Items 2, 4, 5."""
    source = replace_once(source, PA_REFUSAL_OLD, PA_REFUSAL_NEW)          # item 2
    source = replace_once(source, PA_RESOLVE_OLD, PA_RESOLVE_NEW)          # item 5
    source = replace_once(source, PA_LOCK_OLD, PA_LOCK_NEW)                # item 2
    source = replace_once(source, PA_CREATED_OLD, PA_CREATED_NEW)          # item 2
    source = replace_once(source, PA_DIAGNOSTIC_OLD, PA_DIAGNOSTIC_NEW)    # item 4
    source = replace_once(source, PA_EXCEPT_OLD, PA_EXCEPT_NEW)            # item 2
    ast.parse(source)
    return source


def patch_exec_agent(source):
    """Items 2 + 5."""
    source = replace_once(source, AGENT_DESCRIBE_OLD, AGENT_DESCRIBE_NEW)    # item 5
    source = replace_once(source, AGENT_VALIDATE_OLD, AGENT_VALIDATE_NEW)    # item 5
    source = replace_once(source, AGENT_PREVIEW_OLD, AGENT_PREVIEW_NEW)      # item 5
    source = replace_once(source, AGENT_FORWARD_OLD, AGENT_FORWARD_NEW)      # items 2, 5
    ast.parse(source)
    return source


PATCHES = {
    "test_olv_exits.py": patch_olv_test,
    "execute_order.py": patch_execute_order,
    "position_actions.py": patch_position_actions,
    "exec_agent.py": patch_exec_agent,
}

# The new command type the owner must add to LIVE_TYPES before the site control
# can use it. Nothing here edits exec_agent.env.
PROPOSED_ENVIRONMENT = {"LIVE_TYPES": "+position_action_resolve"}


def prepare(source_dir, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "prepared_only", "brief": "docs/exec_reporting_fix_brief_2026-09-21.md",
                "items": ["1 test-suite import", "2 structured lock rejections",
                          "3 IBKR notice classification", "4 diagnostic (design question in runbook)",
                          "5 position_action_resolve", "6 no change (futures_front covers it)"],
                "files": {}, "proposed_environment": PROPOSED_ENVIRONMENT}
    for name, patch in PATCHES.items():
        original = (Path(source_dir) / name).read_bytes()
        candidate = patch(original.decode("utf-8-sig").replace("\r\n", "\n")).encode("utf-8")
        (output / name).write_bytes(candidate)
        (output / (name + ".original")).write_bytes(original)
        manifest["files"][name] = {
            "source_sha256": hashlib.sha256(original).hexdigest(),
            "candidate_sha256": hashlib.sha256(candidate).hexdigest(),
        }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output), indent=2))
