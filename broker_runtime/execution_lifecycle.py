"""Exact-identity lifecycle helpers vendored into the broker checkout on promotion.

No connection or order is initiated at import. ``ns`` is the executor's existing
broker adapter; tests supply a fake adapter. All mutation attempts are recorded
before calling the broker, because an exception is not proof of non-delivery.
"""
from __future__ import annotations

import copy
import math
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal

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
    wanted = tuple(wanted)
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


def wait_modified(connection, wanted, changes, filled_before, attempts=24, ns=None):
    """Read fresh broker snapshots; placeOrder's returned Trade may stay pending.

    Never retransmit while waiting. A fill, rejection or lost identity stops the
    operation rather than allowing the next leg or close to proceed.
    """
    for _ in range(attempts):
        current = fresh_for_edit(ns, connection, wanted) if ns else find_exact(connection, wanted)
        status = current.orderStatus.status
        filled = getattr(current.orderStatus, "filled", None)
        if filled is None or float(filled) != float(filled_before):
            raise ValueError("fill quantity changed while waiting for modification")
        if status in {"Submitted", "PreSubmitted"} and all(
                getattr(current.order, field) == value for field, value in changes.items()):
            return current
        connection.sleep(0.25)
    raise ValueError("modification lacks broker acknowledgement; reconcile before retry")


def fresh_for_edit(ns, connection, wanted):
    native = find_exact(connection, wanted)
    reader = ns.get("_orders_for_contract")
    if reader is None:
        return native
    rows = reader(connection, native.contract, wanted[0])
    exact = [t for t in rows if identity(t) == tuple(wanted)]
    if len(exact) != 1 or not hasattr(exact[0].orderStatus, "filled"):
        raise ValueError("fresh complete order and fill counters required before an edit")
    return exact[0]


def wait_cancelled(connection, native, attempts=24):
    # Raw open-order snapshots are detached objects. Observe the native status
    # callbacks after cancelling, including a fill racing the cancellation.
    for _ in range(attempts):
        if native.orderStatus.status in TERMINAL:
            break
        connection.sleep(0.25)
    return native.orderStatus


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


def market_edit_price(connection, contract):
    """Read an exact, newly received live ask for an increased market order."""
    requested = datetime.now(timezone.utc)
    quotes = connection.reqTickers(contract)
    if len(quotes) != 1:
        raise ValueError("market quantity increase requires one fresh exact quote")
    quote = quotes[0]
    received = getattr(quote, "time", None)
    if (int(getattr(getattr(quote, "contract", None), "conId", 0) or 0) != int(contract.conId)
            or int(getattr(quote, "marketDataType", 0) or 0) != 1
            or not isinstance(received, datetime) or received.tzinfo is None
            or received < requested
            or not 0 <= (datetime.now(timezone.utc) - received).total_seconds() <= 30):
        raise ValueError("market quantity increase requires a fresh exact live quote")
    price = float(quote.ask)
    if not math.isfinite(price) or not 0 < price < 1e20:
        raise ValueError("market quantity increase requires a positive finite live ask")
    return price


def option_edit_context(ns, connection, trade, edited, account_key, changed):
    """Derive option edit risk from exact broker contracts, never ticket claims.

    The same standard-option/vertical topology and account caps used by new
    option tickets apply. Reverse BAG orientations retain their signed broker
    price while risk is normalized to the executor's canonical orientation.
    """
    if edited.orderType != "LMT" or int(getattr(edited, "parentId", 0) or 0):
        raise ValueError("option edits require an independent limit order with provable risk")
    if account_key not in ns["OPTION_ACCOUNTS"]:
        raise ValueError("options are not enabled for this account")
    if getattr(edited, "orderComboLegs", None) or any(
            str(getattr(tag, "tag", "")) == "NonGuaranteed" and str(getattr(tag, "value", "0")) != "0"
            for tag in getattr(edited, "smartComboRoutingParams", ()) or ()):
        raise ValueError("option edits require a guaranteed net-price order")
    contract = trade.contract
    is_combo = contract.secType == "BAG"
    raw_legs = list(getattr(contract, "comboLegs", ()) or ()) if is_combo else []
    if is_combo and (len(raw_legs) != 2 or any(float(leg.ratio) != 1 for leg in raw_legs)):
        raise ValueError("option edits support only exact 1:1 verticals")
    details, legs, contracts = [], [], []
    for raw in raw_legs if is_combo else [None]:
        requested = (ns["Contract"](conId=int(raw.conId), secType="OPT", exchange=raw.exchange or "SMART")
                     if raw is not None else copy.deepcopy(contract))
        rows = connection.reqContractDetails(requested)
        if len(rows) != 1 or int(rows[0].contract.conId) != int(requested.conId):
            raise ValueError("exact option contract details unavailable")
        detail, qualified = rows[0], rows[0].contract
        right = str(qualified.right).upper()
        expiry = str(qualified.lastTradeDateOrContractMonth)[:8]
        strike = float(qualified.strike)
        if right not in {"C", "P"} or not math.isfinite(strike) or strike <= 0 or len(expiry) != 8 or not expiry.isdigit():
            raise ValueError("exact option series is invalid")
        details.append(detail)
        contracts.append(qualified)
        legs.append(dict(side=str(raw.action).upper() if raw is not None else "BUY",
                         ratio=1, right=right, expiry=expiry, strike=strike))
    if is_combo:
        if {leg["side"] for leg in legs} != {"BUY", "SELL"}:
            raise ValueError("vertical requires one BUY and one SELL leg")
        bought, sold = (next(leg for leg in legs if leg["side"] == side) for side in ("BUY", "SELL"))
        reverse = bought["strike"] > sold["strike"] if bought["right"] == "C" else bought["strike"] < sold["strike"]
        if reverse:
            legs = [dict(leg, side="SELL" if leg["side"] == "BUY" else "BUY") for leg in legs]
    else:
        reverse = False
    closing_single = not is_combo and edited.action == "SELL"
    if edited.action not in {"BUY", "SELL"}:
        raise ValueError("option action must be BUY or SELL")

    def trusted(order):
        price, quantity = float(order.lmtPrice), float(order.totalQuantity)
        if (not math.isfinite(price) or price == 0 or not math.isfinite(quantity)
                or quantity <= 0 or quantity != int(quantity)):
            raise ValueError("option limit and whole quantity must be finite and nonzero")
        action = "BUY" if closing_single else order.action
        if reverse:
            action, price = ("SELL" if action == "BUY" else "BUY"), -price
        result, error = ns["_trusted_option_topology"](action, price, int(quantity), legs, contracts)
        if error:
            raise ValueError(error)
        risk = float(result["risk_usd"])
        if not math.isfinite(risk) or risk <= 0 or result["portfolio_direction"] not in {"long", "short"}:
            raise ValueError("option risk and direction are unavailable")
        return result

    context = trusted(edited)
    previous = trusted(trade.order)
    if "lmtPrice" in changed:
        value = Decimal(str(abs(float(edited.lmtPrice))))
        if is_combo:
            low, tick = Decimal(0), Decimal(str(ns["OPT_COMBO_TICK"]))
        else:
            try:
                from .option_limit_pricing import market_increments
            except ImportError:
                from option_limit_pricing import market_increments
            routed = copy.deepcopy(contracts[0])
            routed.exchange = contract.exchange
            bands = market_increments(connection, routed, details[0])
            low, tick = next((low, tick) for low, tick in reversed(bands) if value >= low)
        if not tick.is_finite() or tick <= 0 or (value - low) % tick:
            raise ValueError("modified option limit is not on the applicable price increment")
    increasing_size = float(edited.totalQuantity) > float(trade.order.totalQuantity)
    increasing_risk = context["risk_usd"] > previous["risk_usd"] + 1e-9
    if not ns["_uncapped_options"](account_key):
        if increasing_size and edited.totalQuantity > ns["LIVE_MAX_OPT_CONTRACTS"]:
            raise ValueError("modified size exceeds account option contract cap")
        if not closing_single and increasing_risk and context["risk_usd"] > ns["_max_opt_risk"](account_key):
            raise ValueError("modified option risk exceeds account option risk cap")
    elif not closing_single and increasing_risk:
        nlv = ns["_nlv"](connection, edited.account)
        if not nlv or not math.isfinite(float(nlv)) or context["risk_usd"] > .05 * float(nlv):
            raise ValueError("option risk increase exceeds the existing 5% NLV review threshold or NLV is unavailable")
    # A single SELL is supported only with exact long inventory, checked below
    # on every edit. A vertical is independently defined risk on either side.
    context["mutation_kind"] = "exit" if closing_single else "entry"
    context["risk_bps"] = None
    return context


def mutate_one(ns, ib, payload, host, port, main_cid, *, modify=False, account_key=None):
    attempted = False
    try:
        wanted = payload_identity(payload)
        trade = find_exact(ib, wanted)
        with owners(ns, ib, host, port, main_cid, [trade]) as resolved:
            connection, original = resolved[0]
            trade = fresh_for_edit(ns, connection, identity(original))
            if not modify:
                attempted = True
                ns["guarded_cancel_order"](connection, trade.order)
                observed = wait_cancelled(connection, original)
                status = observed.status
                if status not in {"Cancelled", "ApiCancelled"}:
                    return ns["_out"](False, "unknown", "Cancel outcome requires reconciliation; DO NOT RETRY",
                                      fill={"status": status, "filled": observed.filled})
                return ns["_out"](True, "executed", "Cancellation confirmed",
                                  fill={"status": status, "filled": observed.filled})
            order = copy.deepcopy(trade.order)
            sec_type = str(getattr(trade.contract, "secType", ""))
            option = sec_type in {"OPT", "BAG"}
            changed = {}
            for key, field, allowed in (("new_qty", "totalQuantity", True),
                                        ("new_limit", "lmtPrice", "LMT" in order.orderType),
                                        ("new_stop", "auxPrice", order.orderType.startswith("STP"))):
                if payload.get(key) in (None, ""):
                    continue
                value = float(payload[key])
                signed_limit = sec_type == "BAG" and key == "new_limit"
                if not allowed or not math.isfinite(value) or value == 0 or (value < 0 and not signed_limit):
                    raise ValueError("requested modification is incompatible with the order")
                if key == "new_qty" and (value != int(value) or value <= float(trade.orderStatus.filled or 0)):
                    raise ValueError("quantity must be whole and greater than already filled quantity")
                setattr(order, field, value)
                changed[field] = value
            if not changed or not str(order.tif or "").strip():
                raise ValueError("a supported change and explicit existing TIF are required")
            if not option and order.totalQuantity > trade.order.totalQuantity:
                uncapped_future = sec_type == "FUT" and ns["_uncapped_futures"](account_key)
                if sec_type == "FUT":
                    if not ns["_uncapped_futures"](account_key) and order.totalQuantity > ns["LIVE_MAX_FUT_CONTRACTS"]:
                        raise ValueError("modified size exceeds account contract cap")
                elif ns["LIVE_MAX_QTY"] > 0 and order.totalQuantity > ns["LIVE_MAX_QTY"]:
                    raise ValueError("modified size exceeds account quantity cap")
                if not uncapped_future:
                    base_usd = sec_type == "CASH" and str(getattr(trade.contract, "symbol", "")).upper() == "USD"
                    if sec_type == "CASH" and not base_usd and str(getattr(trade.contract, "currency", "")).upper() != "USD":
                        raise ValueError("CASH edits support USD pairs only; exact USD conversion unavailable")
                    price = (1.0 if base_usd else market_edit_price(connection, trade.contract)
                             if order.orderType == "MKT" else ns["_px"](order.lmtPrice) or ns["_px"](order.auxPrice))
                    multiplier = 1.0 if sec_type == "CASH" else float(getattr(trade.contract, "multiplier", "") or 1)
                    if (not math.isfinite(multiplier) or multiplier <= 0 or not price
                            or price * multiplier * order.totalQuantity > ns["_max_notional"](account_key)):
                        raise ValueError("increased quantity needs a price within the account notional cap")
            if option:
                payload = dict(payload, **option_edit_context(ns, connection, trade, order, account_key, changed))
            elif not payload.get("mutation_kind"):
                try:
                    from .order_edit_context import infer
                except ImportError:
                    from order_edit_context import infer
                payload = dict(payload, **infer(ns, connection, trade, order))
            kind = str(payload.get("mutation_kind") or "").lower()
            if int(getattr(order, "parentId", 0) or 0):
                kind = "modify"
            if kind not in {"entry", "exit", "modify"}:
                raise ValueError("explicit entry/exit semantics are required")
            risk_usd, risk_bps = payload.get("risk_usd"), payload.get("risk_bps")
            if kind == "entry" and (risk_usd in (None, "")) == (risk_bps in (None, "")):
                raise ValueError("entry modification requires exactly one complete risk value")
            if kind == "entry":
                risk = float(risk_usd if risk_usd not in (None, "") else risk_bps)
                if not math.isfinite(risk) or risk <= 0:
                    raise ValueError("entry risk must be positive and finite")
                direction = "long" if order.action == "BUY" else "short"
                if not option and payload.get("portfolio_direction") != direction:
                    raise ValueError("entry direction does not match the actual order")
            if kind in {"exit", "modify"} and (option or float(order.totalQuantity) > float(trade.order.totalQuantity)):
                connection.reqPositions()
                holdings = [p for p in connection.positions()
                            if str(p.account) == wanted[0] and int(p.contract.conId) == wanted[1]]
                if len(holdings) != 1 or not holdings[0].position:
                    raise ValueError("exit increase requires an exact held position")
                if order.action != ("SELL" if holdings[0].position > 0 else "BUY"):
                    raise ValueError("exit direction would increase the held position")
                capacity = {}
                reader = ns.get("_orders_for_contract")
                orders = reader(connection, trade.contract, wanted[0]) if reader else connection.openTrades()
                for other in orders:
                    if (str(other.order.account) != wanted[0] or int(other.contract.conId) != wanted[1]
                            or other.order.action != order.action or other.orderStatus.status in TERMINAL):
                        continue
                    quantity = order.totalQuantity if identity(other) == wanted else other.order.totalQuantity
                    remaining = float(quantity) - float(other.orderStatus.filled)
                    if not math.isfinite(remaining) or remaining < 0:
                        raise ValueError("fresh remaining exit quantities required")
                    group = (getattr(other.order, "ocaGroup", "")
                             if getattr(other.order, "ocaType", 0) in {1, 2} else "") or identity(other)
                    capacity[group] = max(capacity.get(group, 0), remaining)
                if sum(capacity.values()) > abs(holdings[0].position):
                    raise ValueError("increased exit quantity exceeds uncommitted held inventory")
            order.transmit = True
            attempted = True
            filled_before = float(trade.orderStatus.filled or 0)
            ns["guarded_place_order"](
                connection, trade.contract, order, mutation_kind=kind, account=wanted[0],
                portfolio_direction=payload.get("portfolio_direction"),
                risk_usd=risk_usd,
                risk_bps=risk_bps,
                signal_id=ns["_command_signal"](payload, "modify"))
            current = wait_modified(connection, wanted, changed, filled_before, ns=ns)
            return ns["_out"](True, "executed", "Exact order modification confirmed",
                              fill={"status": current.orderStatus.status, "filled": current.orderStatus.filled})
    except Exception as exc:
        return ns["_out"](False, "unknown" if attempted else "rejected",
                          f"{'Reconcile; DO NOT RETRY' if attempted else 'Nothing changed'}: {exc}")


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
        acknowledged = {"Submitted", "PreSubmitted", "Filled"}
        for _ in range(24):
            if all(str(t.orderStatus.status or "") in acknowledged for t in placed):
                break
            if any(str(t.orderStatus.status or "") in TERMINAL - {"Filled"} for t in placed):
                break
            ib.sleep(0.25)
        problem = ns["_placement_problem"](placed)
        filled = float(parent_trade.orderStatus.filled or 0)
        if 0 < filled < qty:
            problem = "partial entry fill; attached exit activation requires reconciliation; DO NOT RETRY"
        if str(parent_trade.orderStatus.status or "") in {"", "PendingSubmit", "ApiPending"}:
            problem = "entry acknowledgement is pending; reconcile before retry"
        if len(children) != len(context["legs"]) or any(
                str(t.orderStatus.status or "") not in acknowledged for t in children):
            problem = "attached exit acknowledgement is incomplete; reconcile before retry"
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
