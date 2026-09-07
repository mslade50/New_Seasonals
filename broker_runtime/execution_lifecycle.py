"""Exact-identity lifecycle helpers vendored into the broker checkout on promotion.

No connection or order is initiated at import. ``ns`` is the executor's existing
broker adapter; tests supply a fake adapter. All mutation attempts are recorded
before calling the broker, because an exception is not proof of non-delivery.
"""
from __future__ import annotations

import copy
import math
from contextlib import contextmanager

TERMINAL = {"Filled", "Cancelled", "ApiCancelled", "Inactive"}


def identity(trade):
    order = trade.order
    values = (str(getattr(order, "account", "") or "").strip(),
              int(getattr(trade.contract, "conId", 0) or 0),
              int(getattr(order, "clientId", -1)), int(getattr(order, "orderId", 0) or 0),
              int(getattr(order, "permId", 0) or 0))
    if not values[0] or values[1] <= 0 or values[2] < 0 or values[3] <= 0 or values[4] <= 0:
        raise ValueError("exact account/conId/client/order/perm identity is required")
    return values


def payload_identity(payload):
    values = (str(payload.get("_broker_account") or "").strip(),
              int(payload.get("con_id") or 0), int(payload.get("client_id", -1)),
              int(payload.get("order_id") or 0), int(payload.get("perm_id") or 0))
    if not values[0] or values[1] <= 0 or values[2] < 0 or values[3] <= 0 or values[4] <= 0:
        raise ValueError("exact account/conId/client/order/perm identity is required")
    return values


def find_exact(connection, wanted):
    connection.reqAllOpenOrders()
    connection.sleep(0.25)
    matches = []
    for trade in connection.openTrades():
        if trade.orderStatus.status in TERMINAL:
            continue
        try:
            if identity(trade) == wanted:
                matches.append(trade)
        except (ValueError, TypeError):
            continue
    if len(matches) != 1:
        raise ValueError("owning client cannot resolve exactly one matching open order")
    return matches[0]


@contextmanager
def owners(ns, ib, host, port, main_cid, trades):
    """Connect and resolve EVERY owner before the first mutation."""
    connections, opened = {int(main_cid): ib}, []
    identities = [identity(trade) for trade in trades]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate order in mutation plan")
    try:
        for owner in sorted({key[2] for key in identities}):
            if owner not in connections:
                connection = ns["IB"]()
                opened.append(connection)
                connection.connect(host, port, clientId=owner, timeout=8)
                connections[owner] = connection
        resolved = [(connections[key[2]], find_exact(connections[key[2]], key))
                    for key in identities]
        yield resolved
    finally:
        for connection in reversed(opened):
            try:
                connection.disconnect()
            except Exception:
                pass


def resize(ns, ib, host, port, main_cid, plan, signal_id):
    done = []
    try:
        with owners(ns, ib, host, port, main_cid, [t for t, _ in plan]) as resolved:
            for (connection, original), (_, quantity) in zip(resolved, plan):
                trade = find_exact(connection, identity(original))
                quantity = int(quantity)
                old = int(trade.order.totalQuantity)
                filled = int(trade.orderStatus.filled or 0)
                if quantity <= filled:
                    raise ValueError("resize must retain quantity above already filled shares")
                if old == quantity:
                    continue
                order = copy.deepcopy(trade.order)
                order.totalQuantity, order.transmit = quantity, True
                done.append((int(order.permId), old, quantity))
                ns["guarded_place_order"](
                    connection, trade.contract, order,
                    mutation_kind="modify" if int(getattr(order, "parentId", 0) or 0) else "exit",
                    account=order.account, signal_id=f"{signal_id}|{order.permId}")
        return done, None
    except Exception as exc:
        return done, f"resize requires reconciliation ({type(exc).__name__})"


def cancel_many(ns, ib, host, port, main_cid, trades):
    attempted = 0
    try:
        with owners(ns, ib, host, port, main_cid, trades) as resolved:
            for connection, original in resolved:
                trade = find_exact(connection, identity(original))
                attempted += 1
                ns["guarded_cancel_order"](connection, trade.order)
                connection.sleep(0.5)
        return attempted, None
    except Exception as exc:
        return attempted, f"cancel requires reconciliation ({type(exc).__name__})"


def restore(ns, ib, payload, contract, originals, host, port, main_cid):
    """Restore only capacity left after actual fills and other working closes."""
    if not originals:
        return ""
    try:
        position, error = ns["_exact_position"](ib, payload)
        if error:
            raise ValueError(error)
        closing = "SELL" if position.position > 0 else "BUY"
        trades = ns["_orders_for_contract"](ib, contract, payload["_broker_account"])
        protective, _ = ns["_split_pending_entry_legs"](trades)
        own, other = [], []
        for trade in protective:
            if trade.order.action != closing:
                continue
            (own if int(trade.order.permId or 0) in originals else other).append(trade)
        # OCA siblings reserve their maximum remaining quantity once; independent
        # exits each consume their own capacity. No symbol-only position lookup.
        groups = {}
        for trade in other:
            key = getattr(trade.order, "ocaGroup", "") or identity(trade)
            remaining = max(0, int(trade.order.totalQuantity) - int(trade.orderStatus.filled or 0))
            groups[key] = max(groups.get(key, 0), remaining)
        held = int(abs(position.position))
        close = payload.get("_recovery_close")
        if close is not None:
            # Execution callbacks can precede position callbacks. Never let a
            # lagging position cache restore shares already confirmed filled.
            held = min(held, max(0, int(payload["_recovery_held"]) - int(close.orderStatus.filled or 0)))
            if close.orderStatus.status not in TERMINAL:
                groups[identity(close)] = max(groups.get(identity(close), 0),
                    int(close.order.totalQuantity) - int(close.orderStatus.filled or 0))
        capacity = max(0, held - sum(groups.values()))
        legs, error = ns["_capture_exit_legs"](own, closing)
        if error or not legs:
            raise ValueError(error or "resized protection is no longer present")
        if capacity == 0:
            count, error = cancel_many(ns, ib, host, port, main_cid, own)
            if error or ns["_confirm_cancelled"](ib, own):
                raise ValueError(error or "zero-capacity exits did not cancel")
            return "; no uncommitted inventory remains; resized exits cancelled"
        scaled, error = ns["_scaled_exit_legs"](legs, capacity)
        if error:
            raise ValueError(error)
        by_key = {leg["source_key"]: int(leg["scaled_qty"]) for leg in scaled}
        plan = [(trade, by_key[ns["_order_source_key"](trade.order)]
                 + int(trade.orderStatus.filled or 0)) for trade in own]
        _, error = resize(ns, ib, host, port, main_cid, plan,
                          ns["_command_signal"](payload, "close-resize:recover"))
        if error or ns["_confirm_leg_quantities"](ib, {int(t.order.permId): q for t, q in plan}):
            raise ValueError(error or "recovery acknowledgement missing")
        return f"; exits reconciled to {capacity} uncommitted shares"
    except Exception as exc:
        return f"; exit recovery uncertain ({type(exc).__name__}); VERIFY IN TWS; DO NOT RETRY"


def mutate_one(ns, ib, payload, host, port, main_cid, *, modify=False, account_key=None):
    attempted = False
    try:
        wanted = payload_identity(payload)
        trade = find_exact(ib, wanted)
        with owners(ns, ib, host, port, main_cid, [trade]) as resolved:
            connection, original = resolved[0]
            trade = find_exact(connection, identity(original))
            if not modify:
                attempted = True
                ns["guarded_cancel_order"](connection, trade.order)
                for _ in range(12):
                    connection.sleep(0.25)
                    if trade.orderStatus.status in TERMINAL:
                        break
                status = trade.orderStatus.status
                if status not in {"Cancelled", "ApiCancelled"}:
                    return ns["_out"](False, "unknown", "Cancel outcome requires reconciliation; DO NOT RETRY",
                                      fill={"status": status, "filled": trade.orderStatus.filled})
                return ns["_out"](True, "executed", "Cancellation confirmed",
                                  fill={"status": status, "filled": trade.orderStatus.filled})
            order = copy.deepcopy(trade.order)
            changed = {}
            for key, field, allowed in (("new_qty", "totalQuantity", True),
                                        ("new_limit", "lmtPrice", "LMT" in order.orderType),
                                        ("new_stop", "auxPrice", order.orderType.startswith("STP"))):
                if payload.get(key) in (None, ""):
                    continue
                value = float(payload[key])
                if not allowed or not math.isfinite(value) or value <= 0:
                    raise ValueError("requested modification is incompatible with the order")
                if key == "new_qty" and (value != int(value) or value <= float(trade.orderStatus.filled or 0)):
                    raise ValueError("quantity must be whole and greater than already filled quantity")
                setattr(order, field, value)
                changed[field] = value
            if not changed or not str(order.tif or "").strip():
                raise ValueError("a supported change and explicit existing TIF are required")
            if order.totalQuantity > trade.order.totalQuantity:
                if getattr(trade.contract, "secType", "") == "FUT":
                    if not ns["_uncapped_futures"](account_key) and order.totalQuantity > ns["LIVE_MAX_FUT_CONTRACTS"]:
                        raise ValueError("modified size exceeds account contract cap")
                elif ns["LIVE_MAX_QTY"] > 0 and order.totalQuantity > ns["LIVE_MAX_QTY"]:
                    raise ValueError("modified size exceeds account quantity cap")
                price = ns["_px"](order.lmtPrice) or ns["_px"](order.auxPrice)
                if not price or price * order.totalQuantity > ns["_max_notional"](account_key):
                    raise ValueError("increased quantity needs a price within the account notional cap")
            kind = str(payload.get("mutation_kind") or "").lower()
            if int(getattr(order, "parentId", 0) or 0):
                kind = "modify"
            if kind not in {"entry", "exit", "modify"}:
                raise ValueError("explicit entry/exit semantics are required")
            risk_usd, risk_bps = payload.get("risk_usd"), payload.get("risk_bps")
            if kind == "entry" and (risk_usd in (None, "")) == (risk_bps in (None, "")):
                raise ValueError("entry modification requires exactly one complete risk value")
            order.transmit = True
            attempted = True
            result = ns["guarded_place_order"](
                connection, trade.contract, order, mutation_kind=kind, account=wanted[0],
                portfolio_direction=payload.get("portfolio_direction") if kind == "entry" else None,
                risk_usd=risk_usd if kind == "entry" else None,
                risk_bps=risk_bps if kind == "entry" else None,
                signal_id=ns["_command_signal"](payload, "modify"))
            connection.sleep(1.0)
            current = find_exact(connection, wanted)
            if result.orderStatus.status in TERMINAL or any(getattr(current.order, field) != value for field, value in changed.items()):
                raise RuntimeError("modification acknowledgement differs from requested fields")
            return ns["_out"](True, "executed", "Exact order modification confirmed",
                              fill={"status": current.orderStatus.status, "filled": current.orderStatus.filled})
    except Exception as exc:
        return ns["_out"](False, "unknown" if attempted else "rejected",
                          f"{'Reconcile; DO NOT RETRY' if attempted else 'Nothing changed'} ({type(exc).__name__})")


def stage_attached(ns, ib, context, qty, signal_id, *, market=False):
    """Stage the parent without transmission, then release attached protection."""
    parent = (ns["MarketOrder"](context["entry_action"], qty) if market else
              ns["LimitOrder"](context["entry_action"], qty, context["avg_cost"]))
    parent.orderId, parent.tif, parent.transmit = ib.client.getReqId(), "DAY", False
    parent.account, parent.orderRef = context["account"], signal_id
    placed = []
    try:
        parent_trade = ns["guarded_place_order"](
            ib, context["ref"], parent, account=context["account"], mutation_kind="entry",
            portfolio_direction="long" if context["entry_action"] == "BUY" else "short",
            risk_usd=ns["_fast_guard_risk_usd"](context, qty), signal_id=signal_id)
        placed.append(parent_trade)
        children, _ = ns["_place_exit_legs"](
            ib, context["ref"], context["legs"], context["close_action"], qty,
            context["account"], f"{signal_id}|children", parent_id=parent.orderId, transmit_chain=True)
        placed.extend(children)
        problem = ns["_placement_problem"](placed)
        filled = float(parent_trade.orderStatus.filled or 0)
        if 0 < filled < qty:
            problem = "partial entry fill; attached exit activation requires reconciliation; DO NOT RETRY"
        if str(parent_trade.orderStatus.status or "") in {"", "PendingSubmit", "ApiPending"}:
            problem = "entry acknowledgement is pending; reconcile before retry"
        result = {"parent": ns["_placed_ids"]([parent_trade]), "children": ns["_placed_ids"](children)}
        return result, problem
    except Exception as exc:
        # Parent was never intentionally released without its children. The
        # release itself may be uncertain, so don't cancel a possibly filled
        # parent's protection or pretend this is a definitive rejection.
        return {"parent": ns["_placed_ids"](placed), "children": []}, f"attached staging uncertain ({type(exc).__name__}); reconcile before retry"


def add(ns, ib, payload, account_key, host, port, main_cid):
    if str(payload.get("order_type") or "MKT").upper() != "MKT":
        return ns["_out"](False, "rejected", "Add requires a market entry with attached protection")
    context, error = ns["_prepare_fast_position"](ib, payload, account_key, partial=False)
    if error:
        return ns["_out"](False, "rejected", error)
    # The existing bracket is untouched. New shares have their own attached
    # bracket from submission, including when the entry fills only partially.
    result, error = stage_attached(ns, ib, context, context["qty"],
                                   ns["_command_signal"](payload, "add-attached"), market=True)
    return ns["_out"](not error, "unknown" if error else "executed",
                       error or "Add and attached protection submitted; broker fills remain authoritative", fill={"add": result})


def trim_readd(ns, ib, payload, account_key, host, port, main_cid):
    if str(payload.get("close_order_type") or "MKT").upper() != "MKT" or str(payload.get("readd_tif") or "DAY").upper() != "DAY":
        return ns["_out"](False, "rejected", "Trim requires MKT and a DAY re-add")
    if not isinstance(payload.get("readd", True), bool):
        return ns["_out"](False, "rejected", "readd must be boolean")
    context, error = ns["_prepare_fast_position"](ib, payload, account_key, partial=True)
    if error:
        return ns["_out"](False, "rejected", error)
    close = dict(payload, qty=context["qty"], action=context["close_action"], order_type="MKT")
    original_out = ns["_out"]
    ns["_out"] = lambda ok, state, detail, fill=None: {"ok": ok, "state": state, "detail": detail, "fill": fill}
    try:
        outcome = ns["_do_close_resize"](ib, close, account_key, host, port, main_cid)
    finally:
        ns["_out"] = original_out
    fill = outcome.get("fill") or {}
    if fill.get("status") != "Filled" or int(fill.get("filled") or 0) != context["qty"]:
        return ns["_out"](**outcome)
    position, error = ns["_exact_position"](ib, payload)
    expected = context["held"] - context["qty"]
    if error or abs(position.position) != expected or (position.position > 0) != (context["pos"].position > 0):
        return ns["_out"](False, "unknown", "Trim filled; remaining inventory changed; no re-add staged", fill=fill)
    if not payload.get("readd", True):
        return ns["_out"](**outcome)
    result, error = stage_attached(ns, ib, context, context["qty"], ns["_command_signal"](payload, "trim-readd"))
    return ns["_out"](not error, "unknown" if error else "executed",
                       error or "Trim filled, remainder protected, DAY re-add with attached exits submitted",
                       fill={"trim": fill, "readd": result})
