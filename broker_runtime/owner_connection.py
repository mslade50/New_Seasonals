"""Authenticated local access to an already-connected order owner.

Only existing exact orders can be edited/cancelled. New orders still go through
the execution agent and its existing gates. Imports never connect or transmit.
"""
from __future__ import annotations

import asyncio
import copy
from dataclasses import asdict
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import socket
from types import SimpleNamespace

LIMIT = 1024 * 1024
TERMINAL = {"Filled", "Cancelled", "ApiCancelled", "Inactive"}


def registry():
    return Path(os.environ.get("EXECUTION_OWNER_REGISTRY") or
                Path(os.environ.get("LOCALAPPDATA") or Path.home() / ".local") /
                "New_Seasonals" / "execution_owners")


def address(host, port, client_id):
    key = hashlib.sha256(f"{host}:{port}:{client_id}".encode()).hexdigest()
    return registry() / (key + ".json")


def identity(trade):
    o = trade.order
    return [o.account, trade.contract.conId, o.clientId, o.orderId, o.permId]


def pack(trade):
    return {key: asdict(getattr(trade, key))
            for key in ("contract", "order", "orderStatus")}


def unpack(row):
    # Futures orders have no combo/tag objects. Preserve IB's native dataclasses
    # so the existing reservation guard can inspect the same broker fields.
    from ib_insync import Contract, Order, OrderStatus, SoftDollarTier, TagValue
    fields = dict(row["order"])
    if fields.get("conditions"):
        raise ValueError("conditional owner orders require their native controller")
    fields["softDollarTier"] = SoftDollarTier(**fields["softDollarTier"])
    for key in ("algoParams", "smartComboRoutingParams", "orderMiscOptions"):
        fields[key] = [TagValue(**v) for v in fields.get(key) or []]
    return SimpleNamespace(contract=Contract(**row["contract"]),
                           order=Order(**fields),
                           orderStatus=OrderStatus(**row["orderStatus"]))


class OwnerConnection:
    def __init__(self, descriptor, main, read_all=None):
        self.descriptor, self.main, self.trades = descriptor, main, {}
        self.read_all = read_all
        self.client = SimpleNamespace(clientId=descriptor["client_id"])

    def __getattr__(self, name):
        # Account/position/execution reads retain the executor's own connection.
        return getattr(self.main, name)

    def request(self, action, **payload):
        message = dict(token=self.descriptor["token"], action=action, **payload)
        with socket.create_connection(("127.0.0.1", self.descriptor["port"]), timeout=12) as stream:
            stream.sendall(json.dumps(message).encode() + b"\n")
            with stream.makefile("rb") as reader:
                raw = reader.readline(LIMIT + 1)
        if not raw.endswith(b"\n") or len(raw) > LIMIT:
            raise ValueError("order owner returned an incomplete response")
        result = json.loads(raw)
        if not result.get("ok"):
            raise ValueError(result.get("error") or "order owner refused request")
        seen = set()
        for row in result["trades"]:
            fresh = unpack(row)
            key = tuple(identity(fresh))
            seen.add(key)
            if key in self.trades:
                old = self.trades[key]
                for field in ("contract", "order", "orderStatus"):
                    vars(getattr(old, field)).update(vars(getattr(fresh, field)))
            else:
                self.trades[key] = fresh
        for missing in set(self.trades) - seen:
            del self.trades[missing]
        return list(self.trades.values())

    def reqAllOpenOrders(self):
        self.request("snapshot")
        return self.openTrades()

    def reqAllOpenOrdersRaw(self):
        owned = self.reqAllOpenOrders()
        # Reservation/capacity checks still need every other client's orders.
        # Only the owning client's rows are supplied by the handoff.
        if self.read_all is None:
            return owned
        others = [t for t in self.read_all(self.main)
                  if t.order.clientId != self.descriptor["client_id"]
                  or t.order.account != self.descriptor["account"]]
        return owned + others

    def openTrades(self):
        return [t for t in self.trades.values() if t.orderStatus.status not in TERMINAL]

    def sleep(self, seconds):
        self.main.sleep(seconds)
        self.request("snapshot")

    def placeOrder(self, contract, order):
        selected = SimpleNamespace(contract=contract, order=order)
        wanted = identity(selected)
        current = self.trades.get(tuple(wanted))
        if current is None:
            raise ValueError("owner connection cannot submit a new order")
        changed = {f: getattr(order, f) for f in ("totalQuantity", "lmtPrice", "auxPrice")
                   if getattr(order, f) != getattr(current.order, f)}
        self.request("modify", identity=wanted, changes=changed,
                     filled_before=current.orderStatus.filled)
        return self.trades[tuple(wanted)]

    def cancelOrder(self, order):
        matches = [t for t in self.trades.values() if t.order.orderId == order.orderId
                   and t.order.permId == order.permId and t.order.clientId == order.clientId]
        if len(matches) != 1:
            raise ValueError("owner cancellation needs one exact order")
        self.request("cancel", identity=identity(matches[0]),
                     filled_before=matches[0].orderStatus.filled)

    def disconnect(self):
        pass  # Never disconnect the live strategy's broker session.


def connect_existing(host, port, client_id, main, read_all=None):
    path = address(host, port, client_id)
    if not path.exists():
        return None
    descriptor = json.loads(path.read_text(encoding="utf-8"))
    if (descriptor.get("host") != host or descriptor.get("broker_port") != port
            or descriptor.get("client_id") != client_id):
        raise ValueError("registered order owner identity differs")
    connection = OwnerConnection(descriptor, main, read_all)
    # A stale registration fails closed; never try a duplicate owner connection.
    connection.reqAllOpenOrders()
    return connection


class OwnerServer:
    def __init__(self, broker, before_mutation):
        self.broker, self.before_mutation = broker, before_mutation
        self.token = secrets.token_hex(32)
        self.lock = asyncio.Lock()
        self.handlers = set()
        self.server = None

    async def snapshot(self):
        ib = self.broker.ib
        async with self.broker.snapshot_lock:
            # Capture complete echoes before ib_insync merges six editable fields
            # into its cache. Keep native status callbacks for fill/cancel races.
            echoes = {}
            prior = ib.wrapper.openOrder
            had_override = "openOrder" in vars(ib.wrapper)
            def capture(oid, contract, order, state):
                raw = copy.deepcopy(order)
                raw.orderId = oid
                echoes[(raw.clientId, oid, raw.permId)] = (copy.deepcopy(contract), raw)
                return prior(oid, contract, order, state)
            ib.wrapper.openOrder = capture
            try:
                await asyncio.wait_for(ib.reqOpenOrdersAsync(), 5)
            finally:
                if had_override:
                    ib.wrapper.openOrder = prior
                else:
                    del ib.wrapper.openOrder
        rows = []
        for native in ib.trades():
            if (native.order.account != self.broker.config.account
                    or native.order.clientId != self.broker.config.client_id
                    or native.contract.secType != "FUT"):
                continue
            row = SimpleNamespace(contract=copy.deepcopy(native.contract),
                                  order=copy.deepcopy(native.order),
                                  orderStatus=copy.deepcopy(native.orderStatus))
            echo = echoes.get((native.order.clientId, native.order.orderId, native.order.permId))
            if echo:
                row.contract, row.order = echo
            elif native.orderStatus.status not in TERMINAL:
                continue  # A cached working order is not a fresh broker echo.
            rows.append(row)
        return rows

    async def dispatch(self, message):
        if not hmac.compare_digest(str(message.get("token") or ""), self.token):
            raise PermissionError("order owner authentication failed")
        async with self.lock:
            rows = await self.snapshot()
            if message["action"] == "snapshot":
                return rows
            if message["action"] not in {"modify", "cancel"}:
                raise ValueError("owner supports only existing-order edits")
            matches = [t for t in rows if identity(t) == message.get("identity")
                       and t.orderStatus.status not in TERMINAL]
            if len(matches) != 1:
                raise ValueError("selected order is no longer uniquely working at its owner")
            current = matches[0]
            if current.order.conditions:
                raise ValueError("conditional owner orders require their native controller")
            if current.orderStatus.filled != message.get("filled_before"):
                raise ValueError("selected order filled before owner handoff; refresh before editing")
            self.broker.config.authorize(self.broker.session)
            if self.broker.config.mode not in {"paper", "live"}:
                raise PermissionError("read-only owner cannot edit orders")
            native = next(t for t in self.broker.ib.trades() if identity(t) == identity(current))
            if message["action"] == "modify":
                changes = message.get("changes") or {}
                if not changes or set(changes) - {"totalQuantity", "lmtPrice", "auxPrice"}:
                    raise ValueError("unsupported owner edit fields")
                import math
                if any(not math.isfinite(float(v)) for v in changes.values()):
                    raise ValueError("owner edit needs finite quantities/prices")
                order = copy.deepcopy(current.order)
                for field, value in changes.items():
                    setattr(order, field, value)
                self.before_mutation(current)
                self.broker.ib.placeOrder(current.contract, order)
            else:
                self.before_mutation(current)
                self.broker.ib.cancelOrder(native.order)
            await asyncio.sleep(.05)
            return await self.snapshot()

    async def handle(self, reader, writer):
        task = asyncio.current_task()
        self.handlers.add(task)
        try:
            try:
                raw = await asyncio.wait_for(reader.readline(), 5)
                if len(raw) > LIMIT or not raw.endswith(b"\n"):
                    raise ValueError("incomplete owner request")
                rows = await self.dispatch(json.loads(raw))
                result = dict(ok=True, trades=[pack(t) for t in rows])
            except Exception as exc:
                result = dict(ok=False, error=f"{type(exc).__name__}: {exc}")
            writer.write(json.dumps(result).encode() + b"\n")
            await writer.drain()
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            finally:
                self.handlers.discard(task)

    async def start(self):
        self.server = await asyncio.start_server(self.handle, "127.0.0.1", 0, limit=LIMIT)
        config = self.broker.config
        self.path = address(config.host, config.port, config.client_id)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = dict(host=config.host, broker_port=config.port, client_id=config.client_id,
                          account=config.account, port=self.server.sockets[0].getsockname()[1], token=self.token)
        temporary = self.path.with_suffix(".pending")
        temporary.write_text(json.dumps(descriptor), encoding="utf-8")
        temporary.chmod(0o600)
        os.replace(temporary, self.path)

    async def close(self):
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            if self.handlers:
                await asyncio.gather(*tuple(self.handlers), return_exceptions=True)
            if self.path.exists() and json.loads(self.path.read_text())["token"] == self.token:
                self.path.unlink()
