"""Reviewed CLI boundary fixture; no broker/configuration imports or entrypoint.

Extracted from the installed executor on 2026-09-14. Tests inject only fake
brokers; test_reviewed_cli_fixture_matches_installed_boundaries guards drift.
"""

SUPPORTED = {"entry_bracket", "close_only", "close_resize", "flatten",
             "option_spread", "exit_attach", "add_to_position", "cancel", "modify"}

DISABLED_UNSAFE_MUTATIONS = {
    "cancel", "modify", "trim_readd", "add_to_position",
}

def _out(ok, state, detail, fill=None):
    print(json.dumps({"ok": ok, "state": state, "detail": detail, "fill": fill}))
    return 0

def _do_add_to_position(ib, p, acct, host, port, main_cid):
    import position_actions
    return position_actions.run(globals(), ib, p, acct, host, port, main_cid, adding=True)

def _do_close_resize(ib, p, acct, host, port, main_cid):
    import position_actions
    return position_actions.run(globals(), ib, p, acct, host, port, main_cid, adding=False)

def _do_cancel(ib, p, host, port, main_cid):
    import order_mutations
    return order_mutations.run(globals(), ib, p, host, port, main_cid, modify=False, account_key=None)

def _do_modify(ib, p, host, port, main_cid, acct):
    import order_mutations
    return order_mutations.run(globals(), ib, p, host, port, main_cid, modify=True, account_key=acct)

def main():
    try:
        cmd = json.loads(sys.argv[1])
    except Exception:
        return _out(False, "error", "bad command json")
    t = cmd.get("type")
    p = dict(cmd.get("payload") or {})
    p["_command_id"] = str(cmd.get("id") or "").strip()
    acct = cmd.get("account")

    if not LIVE_ENABLED:
        return _out(False, "rejected", "live gate: AGENT_LIVE_ENABLED not set")
    if not p["_command_id"]:
        return _out(False, "rejected", "live gate: durable command id is required")
    if acct not in LIVE_ACCOUNTS:
        return _out(False, "rejected", f"live gate: account {acct} not armed")
    if t not in LIVE_TYPES:
        return _out(False, "rejected", f"live gate: type {t} not armed")
    if t in DISABLED_UNSAFE_MUTATIONS - {"add_to_position", "cancel", "modify"}:
        return _out(
            False,
            "rejected",
            f"live gate: {t} is disabled until it has an atomic, exact-identity "
            "broker lifecycle; to reduce a position use close_only (bare), "
            "close_resize (partial, shrinks the exits first) or flatten",
        )
    if t not in SUPPORTED:
        return _out(False, "rejected", f"live gate: {t} not supported live")

    host, port, cid = PORTS[acct]
    ib = IB()
    try:
        ib.connect(host, port, clientId=cid, timeout=8)        # NOT readonly
    except Exception as e:  # noqa: BLE001
        return _out(False, "rejected", f"live gate: connect failed ({type(e).__name__})")
    ib.errorEvent += _on_err
    try:
        try:
            p["_broker_account"] = _resolve_broker_account(ib, acct)
        except Exception as exc:
            return _out(False, "rejected", f"live gate: broker account resolution failed ({exc})")
        if t == "entry_bracket":
            return _do_entry_bracket(ib, p, acct)
        if t == "exit_attach":
            return _do_exit_attach(ib, p, acct)
        if t == "close_only":
            return _do_close_only(ib, p)
        if t == "close_resize":
            return _do_close_resize(ib, p, acct, host, port, cid)
        if t == "flatten":
            return _do_flatten(ib, p, host, port, cid)
        if t == "trim_readd":
            return _do_trim_readd(ib, p, acct, host, port, cid)
        if t == "add_to_position":
            return _do_add_to_position(ib, p, acct, host, port, cid)
        if t == "cancel":
            return _do_cancel(ib, p, host, port, cid)
        if t == "modify":
            return _do_modify(ib, p, host, port, cid, acct)
        if t == "option_spread":
            return _do_option_spread(ib, p, acct)
        return _out(False, "rejected", f"live gate: {t} unhandled")
    except Exception as e:  # noqa: BLE001
        return _out(
            False,
            "unknown",
            f"execute became indeterminate ({type(e).__name__}); "
            "VERIFY IN TWS AND DO NOT RETRY",
        )
    finally:
        try:
            ib.disconnect()
        except Exception:  # noqa: BLE001
            pass
