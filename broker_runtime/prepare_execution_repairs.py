"""Prepare exact-source execution repairs in a new directory; never install/run."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from broker_runtime.prepare import change_function, replace_once

HERE = Path(__file__).resolve().parent


def patch_executor(source):
    source = patch_options(source)
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
        "        from execution_contracts import qualify_held\n"
        "        pos.contract = qualify_held(ib, pos.contract)\n"
        "        _cluster_symbol(pos.contract)   # still validates exact exchange and expiry"))
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
    return source


def patch_options(source):
    def dynamic(text):
        # Replace the deliberately disabled legacy entry point with a different
        # contract. Never accept or reinterpret a saved MKT intent.
        start = text.index('    sym = str(p.get("symbol")')
        text = 'def _do_dynamic_option_limit(ib, p, acct):\n' + text[start:]
        text = replace_once(text, '    sym = str(p.get("symbol")', '''    from option_limit_pricing import capped_size, validate_intent
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
        text = replace_once(text, anchor, '''        quotes = ib.reqTickers(contract)
        details = ib.reqContractDetails(contract)
        if len(quotes) != 1 or len(details) != 1 or int(details[0].contract.conId) != cid:
            raise ValueError("exact option quote and contract details required")
        quote = quotes[0]
        if int(getattr(quote, "marketDataType", 0) or 0) != 1:
            raise ValueError("scheduled option requires a fresh live quote")
        if int(getattr(quote.contract, "conId", 0) or 0) != cid:
            raise ValueError("option quote identity changed")
        ask = float(quote.ask)
        qty, limit = capped_size(budget, ask, details[0].minTick, contract.multiplier)
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
                            '        validate_intent(p)\n        attempted = True\n        trade = guarded_place_order(')
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
