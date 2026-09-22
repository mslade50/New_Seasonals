"""Prepare exact-source execution repairs in a new directory; never install/run.

RETIRED 2026-09-21 -- SPENT ONE-SHOT INSTALLER, kept as the record of what was
installed. Every transform below is already in the live runtime (installs of
2026-09-14 18:32, 2026-09-16 13:28 and 2026-09-17 12:34), so the anchors no
longer match and `prepare()` could not build a candidate even if it were run.
Re-applying it would be worse than useless:

* `patch_options` looks for `_do_dynamic_option_market`, which became
  `_do_dynamic_option_limit` when scheduled options moved to capped-limit
  pricing; the transform also injects the `option_limit_pricing` import and a
  live-quote block that live already carries, so it would DOUBLE-INJECT.
* `patch_executor` rewrites `_do_cancel` / `_do_modify` onto `order_mutations`.
  Live routes both to `manual_order_actions` (2026-09-17), which is stricter --
  applying this would REGRESS the live cancel/modify path.
* `patch_agent`'s first fragment and the `SUPPORTED` / `DISABLED_UNSAFE_MUTATIONS`
  swaps are present in live verbatim.
* `patch_front` is the `select_front_details` + `con_id` change that shipped
  with the 2026-09-14 `futures_front.py` install.

Evidence: `artifacts/recon_2026-09-17/reviewed_runtime_drift_review.md`
("Preparer fragments -- reviewed, NOT refreshed") and
`site_execution_audit.md` section 3.2. The house convention for a fragment that
reached live by another route is to retire it (see `prepare_entry_controls`'s
own docstring); this is that follow-up.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from broker_runtime.prepare import change_function, replace_once

HERE = Path(__file__).resolve().parent

SPENT = ("prepare_execution_repairs is a retired one-shot installer: its patches are "
         "already in the live runtime (2026-09-14/16/17). Re-applying would double-inject "
         "the capped-limit option block and regress cancel/modify from manual_order_actions "
         "back to order_mutations. See artifacts/recon_2026-09-17/reviewed_runtime_drift_review.md.")


def patch_executor(source):
    source = patch_options(source)
    source = patch_remaining_execution(source)
    for name, adding in (("_do_close_resize", False), ("_do_add_to_position", True)):
        source = change_function(source, name, lambda _, name=name, adding=adding:
            f"def {name}(ib, p, acct, host, port, main_cid):\n"
            f"    import position_actions\n"
            f"    return position_actions.run(globals(), ib, p, acct, host, port, main_cid, adding={adding})\n")
    for name, modify, tail in (("_do_cancel", False, ""), ("_do_modify", True, ", acct")):
        source = change_function(source, name, lambda _, name=name, modify=modify, tail=tail:
            f"def {name}(ib, p, host, port, main_cid{tail}):\n"
            f"    import order_mutations\n"
            f"    return order_mutations.run(globals(), ib, p, host, port, main_cid, modify={modify}, "
            f"account_key={'acct' if modify else 'None'})\n")
    source = replace_once(source,
        'SUPPORTED = {"entry_bracket", "close_only", "close_resize", "flatten",\n             "option_spread", "exit_attach"}',
        'SUPPORTED = {"entry_bracket", "close_only", "close_resize", "flatten",\n'
        '             "option_spread", "exit_attach", "add_to_position", "cancel", "modify"}')
    source = replace_once(source,
        'if t in DISABLED_UNSAFE_MUTATIONS and not (acct == "primary" and t == "add_to_position"):',
        'if t in DISABLED_UNSAFE_MUTATIONS - {"add_to_position", "cancel", "modify"}:')
    # Qualify the exact held contract before checking its exchange metadata.
    source = change_function(source, "_do_close_only", lambda text: replace_once(text,
        "        _cluster_symbol(pos.contract)   # validates mapped equity-index futures",
        "        from execution_contracts import qualify_position\n"
        "        pos = qualify_position(ib, pos)\n"
        "        _cluster_symbol(pos.contract)   # still validates exact exchange and expiry"))
    return source


def patch_remaining_execution(source):
    """Repair the remaining site handlers, preserving Flatten's cancel-first behavior."""
    def flatten(text):
        start = text.index("    matches = [x for x in ib.positions()")
        end = text.index("    ref = pos.contract", start)
        text = text[:start] + '''    ib.reqPositions()
    pos, err = _exact_position(ib, p)
    if err:
        return _out(False, "rejected", err)
    from execution_contracts import qualify_position
    try:
        pos = qualify_position(ib, pos)
    except (ValueError, BrokerMutationBlocked) as exc:
        return _out(False, "rejected", str(exc))
''' + text[end:]
        text = text.replace("if x.position and _contract_matches(x.contract, ref)",
                            'if x.position and str(x.account) == p["_broker_account"]\n'
                            '                    and int(x.contract.conId) == int(p["con_id"])')
        text = text.replace('    pos = next((x for x in ib.positions()',
                            '    ib.reqPositions()\n    pos = next((x for x in ib.positions()')
        text = replace_once(text, '    live_held = int(abs(pos.position))',
                            '    live_held = int(abs(pos.position))\n'
                            '    if partial and (pos.position > 0) != (close_action == "SELL"):\n'
                            '        return _out(False, "unknown", "Position direction changed during cancellation; reconcile before retry")')
        text = replace_once(text, '    if n <= 0:',
                            '    if not partial:\n        n = live_held  # full Flatten includes fills racing entry cancellation\n'
                            '    if n <= 0:')
        start = text.index('    c = pos.contract')
        end = text.index('    action = "SELL"', start)
        text = text[:start] + '    c = ref  # qualified exact-account contract from before cancellation\n' + text[end:]
        text = replace_once(text, 'signal_id=_command_signal(p, "flatten:close"),',
                            'signal_id=_command_signal(p, "flatten:close"),\n        account=p["_broker_account"],')
        text = text.replace('return _out(False, "rejected", f"live gate: {err}; close ABORTED -- manage in TWS")',
                            'return _out(False, "unknown", f"live gate: {err}; cancellation needs reconciliation; DO NOT RETRY")')
        text = text.replace('return _out(False, "rejected",\n                        f"live gate: could not cancel',
                            'return _out(False, "unknown",\n                        f"live gate: could not cancel')
        text = replace_once(text, '        live_qty = int(round(abs(rem.position))) if rem is not None else 0',
                            '        if rem is not None and (rem.position > 0) != (close_action == "SELL"):\n'
                            '            return _out(False, "unknown", "Position direction changed before exit restoration; DO NOT RETRY", fill=fill)\n'
                            '        live_qty = int(round(abs(rem.position))) if rem is not None else 0\n'
                            '        live_qty = min(live_qty, max(0, live_held - int(t.orderStatus.filled or 0)))')
        text = replace_once(text, '        rem = next((x for x in ib.positions()',
                            '        ib.reqPositions()\n        rem = next((x for x in ib.positions()')
        text = replace_once(text, '                fill["reattached"] = len(placed)',
                            '                fill["reattach_orders"] = _placed_ids(placed)\n'
                            '                problem = _placement_problem(placed, ib=ib)\n'
                            '                if problem or len(placed) != len(legs):\n'
                            '                    raise ValueError(problem or "exit restoration count mismatch")\n'
                            '                fill["reattached"] = len(placed)')
        text = text.replace('return _out(False, "rejected",\n                    f"live gate: cancelled {len(targets)} order(s) but CLOSE did NOT fill',
                            'return _out(False, "unknown" if st not in TERMINAL or float(t.orderStatus.filled or 0) > 0 else "rejected",\n'
                            '                    f"DO NOT RETRY before reconciliation: cancelled {len(targets)} order(s) but CLOSE did NOT fill')
        text = text.replace('return _out(False, "rejected",\n                    f"live gate: trim {action}',
                            'return _out(False, "unknown",\n                    f"live gate: trim {action}')
        text = text.replace('return _out(False, "rejected",\n                    f"live gate: cancelled {len(targets)} order(s) but the LMT close was',
                            'return _out(False, "unknown" if float(t.orderStatus.filled or 0) > 0 or reattach_failed else "rejected",\n'
                            '                    f"live gate: cancelled {len(targets)} order(s) but the LMT close was')
        anchor = '    naked_note = ""'
        text = replace_once(text, anchor, '''    if st not in {"Submitted", "PreSubmitted", "Filled"}:
        return _out(False, "unknown", f"Flatten close lacks broker acknowledgement ({st}); DO NOT RETRY", fill=fill)
''' + anchor)
        text = text.replace('REMAINDER ~{remaining_qty} HAS NO EXITS', 'REMAINDER ~{remaining_qty} HAS UNVERIFIED EXIT COVERAGE')
        text = text.replace('but the LMT close was ', 'but the LMT close ended ')
        text = text.replace('f"REJECTED ({action}', 'f"({action}')
        text = text.replace('POSITION STILL OPEN ~{abs(pos.position):g}', 'confirmed filled {t.orderStatus.filled:g}/{n}')
        text = text.replace('f"~{abs(pos.position):g}{reattach_note}', 'f"confirmed filled {t.orderStatus.filled:g}/{n}{reattach_note}')
        text = text.replace('-> {st}); POSITION STILL OPEN "', '-> {st}); "')
        return text
    source = change_function(source, "_do_flatten", flatten)
    source = change_function(source, "_cancel_via_owners", lambda _: '''def _cancel_via_owners(ib, host, port, main_cid, targets):
    import execution_lifecycle
    return execution_lifecycle.cancel_many(globals(), ib, host, port, main_cid, targets)
''')
    def placement(text):
        text = replace_once(text, 'def _placement_problem(trades):', '''def _placement_problem(trades, ib=None):
    if ib is not None:
        for _ in range(24):
            if all(t.orderStatus.status in {"Submitted", "PreSubmitted", "Filled"} for t in trades):
                break
            if any(t.orderStatus.status in TERMINAL - {"Filled"} for t in trades):
                break
            ib.sleep(0.25)''')
        return text.replace('{"PENDINGSUBMIT", "PRESUBMITTED", "SUBMITTED", "FILLED"}',
                            '{"PRESUBMITTED", "SUBMITTED", "FILLED"}')
    source = change_function(source, "_placement_problem", placement)
    for name in ("_do_entry_bracket", "_do_exit_attach", "_do_option_spread"):
        source = change_function(source, name, lambda text: text.replace(
            'problem = _placement_problem(trades)', 'problem = _placement_problem(trades, ib=ib)').replace(
            'problem = _placement_problem(placed)', 'problem = _placement_problem(placed, ib=ib)').replace(
            'problem = _placement_problem([t])', 'problem = _placement_problem([t], ib=ib)'))
    def close(text):
        text = text.replace('except BrokerMutationBlocked as exc:', 'except (ValueError, BrokerMutationBlocked) as exc:')
        text = text.replace('state = "rejected" if terminal else "unknown"',
                            'state = "rejected" if terminal and not float(trade.orderStatus.filled or 0) else "unknown"')
        text = text.replace('return _out(False, "rejected",\n                    f"live gate: CLOSE ONLY was rejected',
                            'return _out(False, "unknown" if float(trade.orderStatus.filled or 0) > 0 else "rejected",\n'
                            '                    f"live gate: CLOSE ONLY was rejected')
        anchor = '    resting = " (RESTING, not yet filled)"'
        return replace_once(text, anchor, '''    if status not in {"Submitted", "PreSubmitted", "Filled"}:
        return _out(False, "unknown", f"Close lacks broker acknowledgement ({status}); DO NOT RETRY", fill=fill)
''' + anchor)
    source = change_function(source, "_do_close_only", close)
    def attach(text):
        text = replace_once(text, '    ib.qualifyContracts(ref)', '''    from execution_contracts import qualify_position
    try:
        pos = qualify_position(ib, pos)
    except (ValueError, BrokerMutationBlocked) as exc:
        return _out(False, "rejected", str(exc))
    ref = pos.contract''')
        text = text.replace('if (stop is not None and stop <= 0) or (target is not None and target <= 0):',
                            'if any(not math.isfinite(v) or v <= 0 for v in (stop, target) if v is not None):')
        return text
    source = change_function(source, "_do_exit_attach", attach)
    for name in ("_do_entry_bracket", "_do_exit_attach"):
        def dates(text):
            start = text.index('        d = str(time_stop)')
            end = text.index('\n', text.index('        time_gat = ', start))
            return text[:start] + '''        try:
            time_gat = _execution_deadline(time_stop, "15:59:00")
        except ValueError as exc:
            return _out(False, "rejected", f"live gate: time_stop must be a future valid date ({exc})")''' + text[end:]
        source = change_function(source, name, dates)
    def entry(text):
        text = text.replace('if qty_raw <= 0 or qty_raw != int(qty_raw):',
                            'if not math.isfinite(qty_raw) or qty_raw <= 0 or qty_raw != int(qty_raw):')
        text = replace_once(text, '''    if min(entry, entry_cap if entry_cap is not None else entry,
           stop if stop is not None else entry,
           target if target is not None else entry) <= 0:''',
                            '''    if any(not math.isfinite(v) or v <= 0 for v in
           (entry, entry_cap, stop, target) if v is not None):''')
        start = text.index('        de = str(expiry)')
        end = text.index('\n', text.index('        parent_gtd = ', start))
        return text[:start] + '''        try:
            parent_gtd = _execution_deadline(expiry, "16:00:00")
        except ValueError as exc:
            return _out(False, "rejected", f"live gate: expiry must be a future valid date ({exc})")''' + text[end:]
    source = change_function(source, "_do_entry_bracket", entry)
    def option(text):
        text = text.replace('qty = int(p.get("quantity"))', 'qty = float(p.get("quantity"))')
        text = text.replace('if qty <= 0:', 'if not math.isfinite(qty) or qty <= 0 or qty != int(qty):')
        text = text.replace('if limit <= 0 or claimed_risk <= 0:',
                            'if not math.isfinite(limit) or not math.isfinite(claimed_risk) or limit <= 0 or claimed_risk <= 0:')
        return replace_once(text, '    legs, err = _parse_spread_legs(p)', '    qty = int(qty)\n    legs, err = _parse_spread_legs(p)')
    source = change_function(source, "_do_option_spread", option)
    source = change_function(source, "_parse_spread_legs", lambda text: text.replace(
        'if ratio < 1:', 'if ratio < 1 or float(l.get("ratio", 1) or 1) != ratio:').replace(
        'if strike <= 0:', 'if not math.isfinite(strike) or strike <= 0:'))
    deadline = '''def _execution_deadline(value, clock):
    from zoneinfo import ZoneInfo
    raw = str(value).strip()
    if not re.fullmatch(r"\\d{4}-\\d{2}-\\d{2}|\\d{8}", raw):
        raise ValueError("use YYYY-MM-DD")
    date = datetime.datetime.strptime(raw.replace("-", ""), "%Y%m%d")
    when = datetime.datetime.combine(date.date(), datetime.time.fromisoformat(clock), ZoneInfo("America/New_York"))
    if when <= datetime.datetime.now(ZoneInfo("America/New_York")):
        raise ValueError("deadline has already passed")
    return when.strftime("%Y%m%d %H:%M:%S") + " US/Eastern"


'''
    source = change_function(source, "_do_exit_attach", lambda text: deadline + text)
    return source


def patch_agent(source):
    source = replace_once(source,
        'if armed_type in DISABLED_UNSAFE_MUTATIONS and not (cmd.get("account") == "primary" and armed_type == "add_to_position"):',
        'if armed_type in DISABLED_UNSAFE_MUTATIONS - {"add_to_position", "cancel", "modify"}:')
    source = change_function(source, "_validate", lambda text: replace_once(text,
        "    if t in DISABLED_UNSAFE_MUTATIONS:",
        '    if t in DISABLED_UNSAFE_MUTATIONS - {"cancel", "modify"}:'))
    def validate(text):
        text = replace_once(text,
            '        reasons.append(\n            "scheduled_option is disabled until execution has a broker-enforced "\n            "maximum-debit price"\n        )',
            '        if p.get("pricing_policy") != "capped_limit_v1":\n'
            '            reasons.append("scheduled option requires capped-limit pricing")')
        return text
    source = change_function(source, "_validate", validate)
    source = change_function(source, "_validate_scheduled_option", lambda text: text.replace(
        '!= "MKT"', '!= "LMT"').replace("order_type must be MKT", "order_type must be LMT")
        .replace('if budget <= 0:', 'if not math.isfinite(budget) or budget <= 0:'))
    def eligible(text):
        start = text.index('    if (\n        cmd.get("type") == "scheduled_option"')
        end = text.index('    if armed_type in ', start)
        return text[:start] + '''    p = cmd.get("payload") or {}
    if cmd.get("type") == "scheduled_option" or p.get("dynamic_selection"):
        if p.get("order_type") != "LMT" or p.get("pricing_policy") != "capped_limit_v1":
            return False, "scheduled option requires capped-limit pricing"
    armed_type = "option_spread" if cmd.get("type") in {"scheduled_option", "scheduled_option_cancel"} else cmd.get("type")
''' + text[end:]
    source = change_function(source, "_live_eligible", eligible)
    source = change_function(source, "_preview", lambda text: text.replace(
        "ask-based size toward", "premium capped at").replace("SMART MKT DAY", "SMART LMT DAY")
        .replace("MKT fill may exceed target; expires after 5 minutes",
                 "premium capped excluding commissions; LMT may remain unfilled; trigger expires after 5 minutes")
        .replace("{expiry}  MKT DAY", "{expiry}  capped LMT DAY"))
    for name in ("_fetch_option", "_fetch_workbench", "_fetch_futures_front", "_fetch_book"):
        def reap(text):
            text = replace_once(text, '    try:\n        proc = await asyncio.create_subprocess_exec(',
                                '    proc = None\n    try:\n        proc = await asyncio.create_subprocess_exec(')
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
        source = change_function(source, name, reap)
    return source


def patch_options(source):
    def dynamic(text):
        # Replace the deliberately disabled legacy entry point with a different
        # contract. Never accept or reinterpret a saved MKT intent.
        start = text.index('    sym = str(p.get("symbol")')
        text = 'def _do_dynamic_option_limit(ib, p, acct):\n' + text[start:]
        text = replace_once(text, '    sym = str(p.get("symbol")', '''    from option_limit_pricing import capped_market_size, market_increments, fresh_live_ask, validate_intent
    try:
        validate_intent(p)
    except (ValueError, KeyError, TypeError) as exc:
        return _out(False, "rejected", str(exc))
    attempted = False
    sym = str(p.get("symbol")''')
        text = text.replace('!= "MKT"', '!= "LMT"').replace("must be MKT DAY", "must be LMT DAY")
        text = text.replace('budget <= 0:', '(not math.isfinite(budget) or budget <= 0):')
        start = text.index('        ask = float(row["ask"])')
        end = text.index('        contract = Option(', start)
        text = text[:start] + text[end:]
        anchor = '        trusted, topology_error = _trusted_option_topology('
        text = replace_once(text, anchor, '''        details = ib.reqContractDetails(contract)
        if len(details) != 1 or int(details[0].contract.conId) != cid:
            raise ValueError("exact option contract details required")
        bands = market_increments(ib, contract, details[0])
        quote_requested_at = datetime.datetime.now(datetime.timezone.utc)
        quotes = ib.reqTickers(contract)
        if len(quotes) != 1:
            raise ValueError("exact option quote required")
        quote = quotes[0]
        ask = fresh_live_ask(quote, cid, quote_requested_at)
        qty, limit = capped_market_size(budget, ask, bands, contract.multiplier)
        if not _uncapped_options(acct) and qty > LIVE_MAX_OPT_CONTRACTS:
            raise ValueError("option quantity exceeds the existing account cap")
        validate_intent(p)  # slow chain lookup must not cross the deadline
''' + anchor)
        text = text.replace('            ask,\n            qty,', '            limit,\n            qty,')
        text = text.replace('order = MarketOrder("BUY", qty)', 'order = LimitOrder("BUY", qty, limit)')
        text = text.replace('"dynamic-option-market"', '"dynamic-option-limit"')
        text = text.replace('estimated_premium = ask * 100.0 * qty', 'estimated_premium = limit * 100.0 * qty')
        text = text.replace('        trade = guarded_place_order(', '        attempted = True\n        trade = guarded_place_order(')
        text = text.replace('        attempted = True\n        trade = guarded_place_order(',
                            '        validate_intent(p)\n        fresh_live_ask(quote, cid, quote_requested_at)\n'
                            '        attempted = True\n        trade = guarded_place_order(')
        text = text.replace('Dynamic MKT sizing requires genuinely live marks.', 'Scheduled limit sizing requires live marks.')
        text = text.replace('ask-based option risk', 'limit-based option risk')
        text = text.replace('dynamic option MKT rejected by IBKR', 'option limit broker response requires reconciliation')
        text = text.replace('if _ERRORS:\n            return _out(False, "rejected",',
                            'if _ERRORS:\n            return _out(False, "unknown",')
        text = text.replace('        return _out(True, "executed",', '''        if status not in {"Submitted", "PreSubmitted", "Filled"}:
            return _out(False, "unknown", f"Option limit outcome {status}; reconcile before retry",
                        fill={"order_id": trade.order.orderId, "status": status})
        return _out(True, "executed",''')
        text = text.replace('MKT DAY; delta', 'LMT {limit:g} DAY; delta').replace("ask-size est", "maximum premium")
        text = text.replace('"ask_at_submit": ask,', '"ask_at_submit": ask, "limit": limit,')
        text = text.replace('return _out(False, "rejected", f"live gate: {e}")',
                            'return _out(False, "unknown" if attempted else "rejected", f"scheduled option: {e}")')
        text = text.replace('return _out(False, "error", f"dynamic option execution error:',
                            'return _out(False, "unknown" if attempted else "rejected", f"dynamic option execution error:')
        return text
    source = change_function(source, "_do_dynamic_option_market", dynamic)
    def route(text):
        start = text.index('    if p.get("dynamic_selection"):')
        end = text.index('    broker_account =', start)
        return text[:start] + '''    if p.get("dynamic_selection"):
        return _do_dynamic_option_limit(ib, p, acct)
''' + text[end:]
    return change_function(source, "_do_option_spread", route)


def prepare(source_root, output):
    # Fail on the reason, not 25 frames deep on a vanished anchor.
    raise ValueError(SPENT)


def _prepare_retired(source_root, output):
    if output.exists():
        raise ValueError("candidate directory must be new")
    expected = json.loads((HERE / "execution_repair_source_hashes.json").read_text())
    for name, digest in expected.items():
        if hashlib.sha256((source_root / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"reviewed source changed: {name}")
    rendered = {}
    for name, patch in (("execute_order.py", patch_executor), ("exec_agent.py", patch_agent),
                        ("futures_front.py", patch_front)):
        raw = (source_root / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected[name]:
            raise ValueError(f"reviewed source changed: {name}")
        rendered[name] = patch(raw.decode("utf-8-sig").replace("\r\n", "\n"))
    for name in ("position_actions.py", "position_action_agent.py", "execution_lifecycle.py",
                 "order_mutations.py", "execution_contracts.py", "option_limit_pricing.py", "order_edit_context.py"):
        rendered[name] = (HERE / name).read_text(encoding="utf-8")
    for name, source in rendered.items():
        compile(source, name, "exec")
    output.mkdir(parents=True)
    for name, source in rendered.items():
        (output / name).write_text(source, encoding="utf-8")
    manifest = {"source": expected, "candidate": {
        name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in rendered}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def patch_front(source):
    def resolve(text):
        start = text.index('    front, upcoming, last_trade = pick_front(')
        end = text.index('    contract = front_cd.contract', start)
        text = text[:start] + '''    from execution_contracts import select_front_details
    try:
        front_cd, front, upcoming, last_trade = select_front_details(
            cds, today, _roll_buffer_days(asset_class))
    except ValueError as exc:
        return {"error": str(exc), "symbol": symbol}
''' + text[end:]
        return text.replace('"exchange": exch, "expiry": front, "last_trade": last_trade,',
                            '"exchange": exch, "expiry": front, "last_trade": last_trade, "con_id": int(contract.conId),')
    return change_function(source, "resolve", resolve)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output), indent=2))
