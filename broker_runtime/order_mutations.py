"""Durable exact-order edits. Uses existing broker guards and account limits."""
import json
try:
    from . import position_actions as actions
    from . import execution_lifecycle as life
except ImportError:
    import position_actions as actions
    import execution_lifecycle as life


def _result(ok, state, detail, fill=None):
    return dict(ok=ok, state=state, detail=detail, fill=fill)


def run(ns, ib, payload, host, port, cid, *, modify=False, account_key=None):
    root = actions.journal_root(ns) / "order_edits"
    record = None
    try:
        wanted = life.payload_identity(payload)
        with actions.operation_lock(actions.journal_root(ns)):
            path = actions.record_path(root, payload.get("_command_id"))
            if path.exists():
                saved = json.loads(path.read_text(encoding="utf-8"))
                if saved["payload"] != payload or saved["modify"] != modify:
                    raise ValueError("command id reused with a different edit")
                if saved["phase"] == "done":
                    return ns["_out"](**saved["result"])
                raise ValueError("earlier edit needs broker reconciliation; do not retry")
            for saved in actions.records(root):
                if saved["phase"] != "done" and saved["identity"] == list(wanted):
                    raise ValueError("earlier edit on this order remains unresolved")
            for saved in actions.records(actions.journal_root(ns)):
                p = saved["payload"]
                if (saved["phase"] != "done" and p["_broker_account"] == wanted[0]
                        and int(p["con_id"]) == wanted[1]):
                    raise ValueError("earlier position action needs reconciliation before editing its orders")
            # Write before mutation; crash/restart cannot send the edit twice.
            record = dict(version=1, id=payload["_command_id"], payload=dict(payload),
                          modify=modify, identity=list(wanted), phase="mutating")
            actions.save(root, record)
            # The executable's _out prints JSON and returns exit code 0. Collect
            # the inner result silently, persist it, then emit exactly once.
            result = life.mutate_one(dict(ns, _out=_result), ib, payload, host, port, cid,
                                     modify=modify, account_key=account_key)
            record["result"] = result
            record["phase"] = "attention" if result["state"] == "unknown" else "done"
            actions.save(root, record)
            return ns["_out"](**result)
    except Exception as exc:
        return ns["_out"](False, "unknown" if record else "rejected",
                          f"Order edit requires review: {exc}")
