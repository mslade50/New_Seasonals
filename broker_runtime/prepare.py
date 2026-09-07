"""Prepare a NEW broker candidate checkout; never modifies the source directory.

The source hashes pin the reviewed deployed code. All transformations are
reviewable here. Candidate outputs contain the operator's original configuration
and therefore belong only under ignored artifacts until approved promotion.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError("reviewed source fragment changed or is ambiguous")
    return source.replace(old, new, 1)


def change_function(source, name, transform):
    tree = ast.parse(source)
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name]
    if len(nodes) != 1:
        raise ValueError(f"missing unique function: {name}")
    node = nodes[0]
    lines = source.splitlines(keepends=True)
    original = "".join(lines[node.lineno - 1:node.end_lineno])
    replacement = transform(original).rstrip() + "\n"
    return "".join(lines[:node.lineno - 1]) + replacement + "".join(lines[node.end_lineno:])


def wrapper(name, signature, call):
    return f"def {name}({signature}):\n    import execution_lifecycle as lifecycle\n    return lifecycle.{call}\n"


def patch_execute(source):
    for name, signature, call in [
        ("_restore_leg_quantities", "ib, p, contract, originals, host, port, main_cid", "restore(globals(), ib, p, contract, originals, host, port, main_cid)"),
        ("_resize_legs_via_owners", "ib, host, port, main_cid, plan, signal_id", "resize(globals(), ib, host, port, main_cid, plan, signal_id)"),
        ("_cancel_via_owners", "ib, host, port, main_cid, trades", "cancel_many(globals(), ib, host, port, main_cid, trades)"),
        ("_do_cancel", "ib, p, host, port, main_cid", "mutate_one(globals(), ib, p, host, port, main_cid)"),
        ("_do_modify", "ib, p, host, port, main_cid, acct", "mutate_one(globals(), ib, p, host, port, main_cid, modify=True, account_key=acct)"),
        ("_do_add_to_position", "ib, p, acct, host, port, main_cid", "add(globals(), ib, p, acct, host, port, main_cid)"),
        ("_do_trim_readd", "ib, p, acct, host, port, main_cid", "trim_readd(globals(), ib, p, acct, host, port, main_cid)"),
    ]:
        source = change_function(source, name, lambda _old, n=name, sig=signature, c=call: wrapper(n, sig, c))

    def flatten(text):
        start = text.index("    matches = [x for x in ib.positions()")
        end = text.index("    ref = pos.contract", start)
        text = text[:start] + '    pos, err = _exact_position(ib, p)\n    if err:\n        return _out(False, "rejected", err)\n' + text[end:]
        text = text.replace("if x.position and _contract_matches(x.contract, ref)",
                            'if x.position and str(x.account) == p["_broker_account"]\n                    and int(x.contract.conId) == int(p["con_id"])')
        text = text.replace('signal_id=_command_signal(p, "flatten:close"),',
                            'signal_id=_command_signal(p, "flatten:close"),\n        account=p["_broker_account"],')
        text = text.replace('return _out(False, "rejected",\n                    f"live gate: cancelled {len(targets)} order(s) but CLOSE did NOT fill',
                            'return _out(False, "unknown" if st not in TERMINAL or float(t.orderStatus.filled or 0) > 0 else "rejected",\n                    f"DO NOT RETRY before reconciliation: cancelled {len(targets)} order(s) but CLOSE did NOT fill')
        return text
    source = change_function(source, "_do_flatten", flatten)
    def close_resize(text):
        text = text.replace('state = "rejected" if dead else "unknown"',
                            'state = "rejected" if dead and not float(trade.orderStatus.filled or 0) else "unknown"')
        text = replace_once(text, '    held = int(round(abs(float(pos.position))))',
                             '    held = int(round(abs(float(pos.position))))\n    p["_recovery_held"] = held')
        text = replace_once(text, '    status = trade.orderStatus.status',
                             '    p["_recovery_close"] = trade\n    status = trade.orderStatus.status')
        return text
    source = change_function(source, "_do_close_resize", close_resize)
    # Cap validation must also allow the independently attached ADD quantity,
    # not merely the old bracket plus the addition.
    source = change_function(source, "_prepare_fast_position", lambda text: text.replace(
        'totals = [held + qty]', 'totals = [qty]'))
    # New lifecycle implementations replace the generic hard-disable. Explicit
    # LIVE_TYPES/account arming and scheduled-option gate remain intact.
    source = change_function(source, "main", lambda text: text.replace(
        'if t in DISABLED_UNSAFE_MUTATIONS:', 'if t in DISABLED_UNSAFE_MUTATIONS - {"cancel", "modify", "trim_readd", "add_to_position"}:'))
    return source


def patch_agent(source):
    for name in ("_validate", "_live_eligible"):
        source = change_function(source, name, lambda text: text.replace(
            'in DISABLED_UNSAFE_MUTATIONS:', 'in DISABLED_UNSAFE_MUTATIONS - {"cancel", "modify", "trim_readd", "add_to_position"}:'))
    # Mutating commands are resolved against the current broker by the executor.
    # The cached read-only book remains useful for previews, never a veto.
    source = change_function(source, "_validate", lambda text: replace_once(
        text, '    reasons = []', '''    reasons = []
    if t in {"cancel", "modify", "trim_readd", "add_to_position", "close_resize", "close_only", "flatten"} and not cmd.get("dry_run"):
        if acct not in ("primary", "pa"):
            return False, ["unknown account"]
        if not p.get("con_id"):
            return False, ["exact contract id is required"]
        if t in {"cancel", "modify"} and any(p.get(key) is None for key in ("client_id", "order_id", "perm_id")):
            return False, ["exact owner/order identity is required"]
        return True, []'''))
    for name in ("_fetch_option", "_fetch_workbench", "_fetch_futures_front", "_fetch_book"):
        def helper(text):
            text = text.replace('    try:\n        proc = await asyncio.create_subprocess_exec(', '    proc = None\n    try:\n        proc = await asyncio.create_subprocess_exec(', 1)
            # A finally block handles both cancellation and timeout, and reaps
            # the child even after JSON decoding or connection errors.
            return text.rstrip() + '''
    finally:
        if proc is not None:
            if proc.returncode is None:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
            await proc.communicate()
'''
        source = change_function(source, name, helper)
    return source


def patch_snapshot(source):
    def snapshot(text):
        text = replace_once(text, '"nlv": None, "positions": [], "orders": [], "fills": []}',
                             '"nlv": None, "positions": [], "orders": [], "fills": [],\n           "fills_complete": False, "fills_source_at": None, "fills_source_session": None}')
        text = replace_once(text, '            ib.reqExecutions()', '            executions = ib.reqExecutions()\n            if executions is None:\n                raise RuntimeError("execution request did not complete")')
        text = replace_once(text, '        except Exception as e:  # noqa: BLE001\n            out["fills_error"]', '''            from datetime import datetime, timezone
            from zoneinfo import ZoneInfo
            stamp = datetime.now(timezone.utc)
            out["fills_complete"] = True
            out["fills_source_at"] = int(stamp.timestamp() * 1000)
            out["fills_source_session"] = stamp.astimezone(ZoneInfo("America/New_York")).date().isoformat()
        except Exception as e:  # noqa: BLE001
            out["fills_error"]''')
        return text
    return change_function(source, "snap_account", snapshot)


def patch_entry(source):
    def run(text):
        # All early exits in the input/connect preflight are failures, except an
        # explicitly empty and valid input basket (the normal no-action case).
        marker = text.index('    order_summary = []')
        head, tail = text[:marker], text[marker:]
        head = head.replace('        return\n', '        return 1\n')
        head = replace_once(head, '        print("[WARN] File is empty.")\n        return 1\n', '        print("[OK] Valid input contains no orders.")\n        return 0\n')
        # Missing/unreadable/stale input and connection failures are errors;
        # a successfully parsed empty basket is an intentional no-action run.
        tail += '\n    return 1 if any(o.get("Status") not in {"SENT", "SKIPPED_DUP", "SKIPPED_FLAT"} for o in order_summary) or stale_count > 0 else 0\n'
        return head + tail
    source = change_function(source, "run_execution", run)
    source = replace_once(source, '    run_execution()', '    raise SystemExit(run_execution())')
    return source


def patch_batch(source):
    source = replace_once(source, 'setlocal', 'setlocal\nset "FINAL_RC=0"')
    source = replace_once(source, 'set STAGE_RC=%ERRORLEVEL%', 'set STAGE_RC=%ERRORLEVEL%\nif not %STAGE_RC%==0 set "FINAL_RC=%STAGE_RC%"')
    for name in ("eq_order_entry.py", "pa_order_entry.py", "morning_order_summary.py"):
        old = f'echo [{name} exit code: %ERRORLEVEL%]'
        source = replace_once(source, old, f'set "STEP_RC=%ERRORLEVEL%"\nif not %STEP_RC%==0 set "FINAL_RC=%STEP_RC%"\necho [{name} exit code: %STEP_RC%]')
    return replace_once(source, 'endlocal', 'endlocal & exit /b %FINAL_RC%')


def patch_auction(source, *, event):
    loader = "load_event_rows" if event else "load_trend_rows"
    def load(text):
        due = 'raw[(execute_on == today_ts) | ((execute_on < today_ts) & raw["Action"].isin({"SELL", "BUY_TO_COVER"}))].copy()' if event else 'raw[execute_on <= today_ts].copy()'
        text = replace_once(text, 'raw[execute_on == today_ts].copy()', due)
        text = replace_once(text, '    if (due["Quantity"] <= 0).any():', '    if (due["Quantity"] % 1 != 0).any():\n        errors.append("Quantity must be whole shares")\n    if (due["Quantity"] <= 0).any():')
        return text
    source = change_function(source, loader, load)
    def process(text):
        text = replace_once(text, '        ib.sleep(1)', '        ib.sleep(1)\n        from auction_lifecycle import resolve_primary, claim\n        from pathlib import Path\n        account = resolve_primary(ib)\n        ib.reqAllOpenOrders()\n        ib.reqExecutions()')
        text = text.replace('if str(getattr(trade.order, "orderRef", "") or "").strip()', 'if str(getattr(trade.order, "account", "") or "") == account\n            and str(getattr(trade.order, "orderRef", "") or "").strip()')
        text = replace_once(text, '                if ref:', '                if ref and str(getattr(fill.execution, "acctNumber", "") or "") == account:')
        text = replace_once(text, '            print(f"[WARN] could not read session fills for dedup: {exc}")', '            raise RuntimeError("session execution reconciliation failed") from exc')
        text = replace_once(text, '        for pos in ib.positions():', '        for pos in ib.positions():\n            if str(getattr(pos, "account", "")) != account or pos.contract.secType != "STK" or pos.contract.currency != "USD":\n                continue')
        if event:
            text = replace_once(text, 'sig = signal_ref(symbol, staged_action, row["Trade"], today_str)', 'sig = signal_ref(symbol, staged_action, row["Trade"], str(row.get("Entry_Date") or row["Execute_On"]))')
        else:
            text = replace_once(text, 'sig = signal_ref(symbol, action, STRATEGY_REF, today_str)', 'sig = signal_ref(symbol, action, STRATEGY_REF, str(row["Asof"]))')
        text = replace_once(text, '            order.orderRef = sig', '            order.orderRef = sig\n            order.account = account')
        marker = '                trade = guarded_place_order('
        exception = '                    summary.append(missed(row, "PENDING_RECONCILIATION"))' if event else '                    summary.append({"Symbol": symbol, "Action": action, "Quantity": qty, "Status": "PENDING_RECONCILIATION"})'
        text = replace_once(text, marker, '                if not claim(Path(__file__).resolve().parent / "auction_intents", account, contract.conId, sig, qty, order.orderType, order.tif):\n' + exception + '\n                    continue\n' + marker)
        text = replace_once(text, '                    signal_id=sig,', '                    signal_id=sig,\n                    account=account,')
        text = text.replace('if status in TERMINAL_REJECT_STATUSES:', 'if status in TERMINAL_REJECT_STATUSES or status in {"UNKNOWN", "PendingSubmit", "ApiPending"}:')
        return text
    return change_function(source, "process_orders", process)


PATCHERS = {"execute_order.py": patch_execute, "exec_agent.py": patch_agent,
            "book_snapshot.py": patch_snapshot, "eq_order_entry.py": patch_entry,
            "run_order_staging.bat": patch_batch,
            "event_moo.py": lambda s: patch_auction(s, event=True),
            "trend_moo.py": lambda s: patch_auction(s, event=False)}


def prepare(source_dir: Path, output_dir: Path):
    source_dir, output_dir = source_dir.resolve(), output_dir.resolve()
    if output_dir == source_dir or source_dir in output_dir.parents or output_dir in source_dir.parents:
        raise ValueError("candidate output must be separate from the deployed source")
    if output_dir.exists():
        raise ValueError("candidate output must be new; existing files are never overwritten")
    manifest = json.loads((HERE / "source_hashes.json").read_text())
    candidates = {}
    for filename, patch in PATCHERS.items():
        body = (source_dir / filename).read_bytes()
        if hashlib.sha256(body).hexdigest() != manifest[filename]:
            raise ValueError(f"reviewed source hash changed: {filename}")
        candidate = patch(body.decode("utf-8-sig").replace("\r\n", "\n"))
        if filename.endswith(".py"):
            compile(candidate, filename, "exec")
        candidates[filename] = candidate
    output_dir.mkdir(parents=True)
    for filename, candidate in candidates.items():
        (output_dir / filename).write_text(candidate, encoding="utf-8")
    for helper in ("execution_lifecycle.py", "auction_lifecycle.py"):
        (output_dir / helper).write_bytes((HERE / helper).read_bytes())
    return sorted(candidates)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print("Prepared candidate files: " + ", ".join(prepare(args.source, args.output)))
