"""Derive edit metadata from broker evidence instead of asking the operator."""
import math


def price(order):
    typ = str(order.orderType)
    field = "lmtPrice" if "LMT" in typ else "auxPrice" if typ.startswith("STP") else None
    value = float(getattr(order, field, 0) or 0) if field else 0
    return value if math.isfinite(value) and 0 < value < 1e20 else None


def infer(ns, ib, trade, edited):
    account, contract = str(edited.account), trade.contract
    ib.reqPositions()
    positions = [p for p in ib.positions() if str(p.account) == account
                 and int(p.contract.conId) == int(contract.conId) and p.position]
    if len(positions) > 1:
        raise ValueError("multiple exact positions returned for this order")
    position = positions[0] if positions else None
    reader = ns.get("_orders_for_contract")
    rows = reader(ib, contract, account) if reader else ib.openTrades()
    rows = [t for t in rows if str(t.order.account) == account
            and int(t.contract.conId) == int(contract.conId)]
    child = int(getattr(edited, "parentId", 0) or 0) > 0
    children = [t for t in rows if int(getattr(t.order, "parentId", 0) or 0) == edited.orderId
                and t.order.clientId == edited.clientId]
    semantic, guard, registry = None, None, None
    if ns.get("_cluster_symbol") and ns["_cluster_symbol"](contract) is not None:
        import legend_reservation_guard as guard
        config = guard._load_config()
        if config:
            _, registry = guard._intent_registry(config["reservation_dir"], account, guard._cluster_symbol(contract))
            semantic = guard._guarded_order_semantic(registry, trade, account)
    reducing = position is not None and edited.action == ("SELL" if position.position > 0 else "BUY")
    # Durable provenance wins over net holdings (another sleeve can hold the
    # opposite direction). A broker-linked parent is an entry even while held.
    kind = "modify" if child else semantic or ("entry" if children or not reducing else "exit")
    result = {"mutation_kind": kind}
    needs_risk = kind == "entry"
    if kind != "entry" and guard and registry is not None:
        needs_risk, _, _ = guard._exit_modify_requires_capacity(ib, contract, edited, account, registry)
    if not needs_risk:
        return result
    multiplier = float(getattr(contract, "multiplier", "") or 1)
    if not math.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("contract multiplier unavailable")
    direction = ("long" if edited.action == "BUY" else "short") if kind == "entry" else (
        "long" if edited.action == "SELL" else "short")
    if kind == "entry":
        reference, quantity = price(edited), float(edited.totalQuantity)
        exits = children
    else:
        parents = [t for t in rows if t.order.orderId == edited.parentId
                   and t.order.clientId == edited.clientId] if child else []
        reference = float(position.avgCost) / multiplier if position else (price(parents[0].order) if len(parents) == 1 else None)
        quantity = abs(float(position.position)) if position else (float(parents[0].order.totalQuantity) if len(parents) == 1 else 0)
        exits = [t for t in rows if t.order.action == edited.action]
    if not reference:
        quotes = ib.reqTickers(contract)
        reference = float(quotes[0].ask if direction == "long" else quotes[0].bid) if len(quotes) == 1 else 0
    if not math.isfinite(reference) or reference <= 0 or quantity <= 0:
        raise ValueError("current broker price/quantity unavailable for automatic edit risk")
    groups, distances = {}, []
    for t in exits:
        o = edited if (t.order.clientId, t.order.orderId, t.order.permId) == (edited.clientId, edited.orderId, edited.permId) else t.order
        if o.orderType != "STP" or o.action != ("SELL" if direction == "long" else "BUY") or getattr(o, "goodAfterTime", ""):
            continue
        stop = price(o)
        if not stop:
            continue
        filled = float(t.orderStatus.filled)
        remaining = float(o.totalQuantity) - filled
        if not math.isfinite(remaining) or remaining < 0:
            raise ValueError("fresh fill counters required for automatic edit risk")
        group = getattr(o, "ocaGroup", "") or (o.clientId, o.orderId)
        groups[group] = max(groups.get(group, 0), remaining)
        distances.append(abs(reference - stop))
    protected = min(quantity, sum(groups.values()))
    # Price stops are trigger levels, not guaranteed fills. This is the same
    # stop-distance reservation convention as entry staging; uncovered units
    # reserve full notional instead of inventing a smaller loss estimate.
    risk = ((protected * max(distances, default=reference)) + (quantity - protected) * reference) * multiplier
    result.update(portfolio_direction=direction, risk_usd=max(risk, .01))
    return result
