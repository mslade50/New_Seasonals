"""Offline candidate: durable, opposite-only Legend/OpenBreakout coordination.

No broker imports. The close controller consumes an explicit broker adapter;
the copied production transport is deliberately not wired to that adapter.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, time as wall_time
import json
import os
from pathlib import Path
import sqlite3
import time
from typing import Callable
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")
FAMILY = {"ES": "ES", "MES": "ES", "NQ": "NQ", "MNQ": "NQ"}
TERMINAL = {"Filled", "Cancelled", "ApiCancelled", "Rejected"}
NORMALIZED_STATUSES = TERMINAL | {"ApiPending", "PendingSubmit", "PreSubmitted", "Submitted", "PendingCancel"}
CANCELLED = {"Cancelled", "ApiCancelled"}


class CoordinationError(RuntimeError):
    pass


class EntryBlocked(CoordinationError):
    pass


def stamp(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise CoordinationError("An aware timestamp is required")
    return value.astimezone(NY)


def direction(side: int) -> int:
    if type(side) is not int or side not in (-1, 1):
        raise CoordinationError("Direction must be exactly -1 or +1")
    return side


@dataclass(frozen=True)
class Scope:
    day: str
    account: str
    family: str

    def __post_init__(self):
        datetime.strptime(self.day, "%Y-%m-%d")
        if not self.account or self.account.strip() != self.account or self.family not in {"ES", "NQ"}:
            raise CoordinationError("Exact account and supported index family are required")

    @classmethod
    def for_symbol(cls, day, account, symbol):
        if symbol not in FAMILY:
            raise CoordinationError("Unsupported symbol")
        return cls(day, account, FAMILY[symbol])

    @property
    def key(self):
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class LegendSignal:
    scope: Scope
    side: int
    signal_id: str
    decided_at: datetime
    enabled: bool
    finalized: bool
    actionable: bool
    quantity: int
    normal_session: bool = True

    def validate(self, now):
        direction(self.side)
        at, now = stamp(self.decided_at), stamp(now)
        if not all(flag is True for flag in (self.enabled, self.finalized, self.actionable, self.normal_session)):
            raise CoordinationError("Only finalized enabled actionable production signals may publish")
        if type(self.quantity) is not int or self.quantity < 1 or not self.signal_id:
            raise CoordinationError("Positive whole quantity and stable signal identity are required")
        if at.date().isoformat() != self.scope.day or now.date().isoformat() != self.scope.day:
            raise CoordinationError("Signal belongs to another trading day")
        if not wall_time(9, 31) <= at.time().replace(tzinfo=None) <= wall_time(9, 31, 20):
            raise CoordinationError("Signal was not finalized in Legend's normal decision window")
        if not 0 <= (now - at).total_seconds() <= 20 or now.time().replace(tzinfo=None) > wall_time(9, 31, 20):
            raise CoordinationError("Stale, future, or late signal")


class Ledger:
    """One local ledger shared by both workers; commit intents before any send.

    A separate OS file lock remains held across the intent commit and synchronous
    transport invocation. A process death leaves a durable uncertain intent,
    never permission to resend. There is no lease expiry that recreates entries.
    """
    def __init__(self, path, *, enabled=False, lock_timeout=2.0):
        self.enabled = enabled is True
        self.path = Path(path)
        self.lock_timeout = lock_timeout
        if self.enabled:
            if not self.path.is_absolute() or any(p.lower().startswith("onedrive") for p in self.path.parts):
                raise CoordinationError("Use an absolute local ledger path outside OneDrive")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.locked():
                with self.db() as db:
                    db.execute("CREATE TABLE IF NOT EXISTS records (scope TEXT PRIMARY KEY, body TEXT NOT NULL)")

    @contextmanager
    def db(self):
        db = sqlite3.connect(self.path, timeout=self.lock_timeout)
        db.execute("PRAGMA synchronous=FULL")
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    @contextmanager
    def locked(self):
        if not self.enabled:
            yield
            return
        handle = open(str(self.path) + ".lock", "a+b")
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        deadline = time.monotonic() + self.lock_timeout
        acquired = False
        try:
            while not acquired:
                try:
                    handle.seek(0)
                    if os.name == "nt":
                        import msvcrt
                        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                except OSError:
                    if time.monotonic() >= deadline:
                        raise CoordinationError("Coordination lock unavailable; do not submit")
                    time.sleep(0.01)
            yield
        finally:
            if acquired:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle, fcntl.LOCK_UN)
            handle.close()

    def _read(self, scope):
        with self.db() as db:
            row = db.execute("SELECT body FROM records WHERE scope=?", (scope.key,)).fetchone()
        return json.loads(row[0]) if row else {"signal": None, "entries": {}, "operation": None}

    def _save(self, scope, body):
        with self.db() as db:
            db.execute("INSERT INTO records VALUES (?,?) ON CONFLICT(scope) DO UPDATE SET body=excluded.body",
                       (scope.key, json.dumps(body, sort_keys=True, allow_nan=False)))

    def read(self, scope):
        if not self.enabled:
            return {"signal": None, "entries": {}, "operation": None}
        with self.locked():
            return self._read(scope)

    def publish(self, signal, now):
        if not self.enabled:
            return False
        # Exact retries are idempotent, even on restart after the entry window.
        with self.locked():
            body = self._read(signal.scope)
            existing = body["signal"]
            identity = (signal.signal_id, signal.side, signal.decided_at.isoformat())
            if existing:
                if (existing["id"], existing["side"], existing["at"]) == identity:
                    return False
                raise CoordinationError("A different finalized signal already owns this scope; reconcile")
            signal.validate(now)
            body["signal"] = {"id": signal.signal_id, "side": signal.side,
                              "at": signal.decided_at.isoformat(), "status": "eligible"}
            body["operation"] = {"id": signal.signal_id + ":breakout-close", "phase": "cancel_entries",
                                 "cancelled": {}, "close": None, "reason": "", "exit_plan": []}
            self._save(signal.scope, body)
        return True

    def signal_status(self, scope, signal_id, status):
        """No automatic release on revocation/rejection: that policy is unresolved.

        A previously accepted opposite-side latch stays in force for this day;
        an invalid/revoked candidate that was never accepted cannot create one.
        Same-direction admission remains allowed. No further close is attempted.
        """
        if not self.enabled:
            return False
        if status not in {"revoked", "broker_rejected", "delivery_unknown"}:
            raise CoordinationError("Unsupported signal lifecycle update")
        with self.locked():
            body = self._read(scope)
            if not body["signal"] or body["signal"]["id"] != signal_id:
                raise CoordinationError("No accepted signal to update")
            body["signal"]["status"] = status
            if body["operation"]["phase"] != "done":
                body["operation"]["phase"] = "reconcile"
                body["operation"]["reason"] = "Signal changed; operator decision required; do not restore entries"
            self._save(scope, body)

    def allowed(self, scope, side):
        direction(side)
        body = self.read(scope)
        return not body["signal"] or body["signal"]["side"] == side

    def submit_entry(self, scope, side, token, order_identity, submit: Callable):
        """Entry-only fence. Protection/closing orders must never use this gate."""
        direction(side)
        if not self.enabled:
            return submit()
        if not token or order_identity.get("strategy") != "OpenBreakout" or order_identity.get("role") != "ENTRY":
            raise CoordinationError("Exact OpenBreakout entry provenance required")
        with self.locked():
            body = self._read(scope)
            if body["signal"] and body["signal"]["side"] != side:
                raise EntryBlocked("Opposite finalized Legend signal owns this trading day")
            if body['operation'] and body['operation']['phase'] in {
                    'cancel_exits', 'submit_close', 'close_working', 'restore_protection', 'restore_pending',
                    'protected_submit', 'protected_working', 'protected_restore', 'protected_restore_pending'}:
                raise EntryBlocked('Owned close handoff in progress; defer same-direction entry, do not suppress its signal')
            if token in body["entries"]:
                raise EntryBlocked("Durable entry intent already exists; reconcile instead of resending")
            body["entries"][token] = {"side": side, "identity": order_identity, "status": "intent"}
            self._save(scope, body)
            try:
                result = submit()
            except BaseException:
                body["entries"][token]["status"] = "unknown"
                self._save(scope, body)
                raise
            body["entries"][token]["status"] = "submitted"
            self._save(scope, body)
            return result

    def bind_entry_identity(self, scope, token, confirmed_identity):
        """A qualified adapter may bind an initially unknown permId once.

        Client/order/account/contract/provenance cannot change. This is a broker
        acknowledgement update, never permission to resubmit a durable intent.
        """
        if not self.enabled:
            return False
        with self.locked():
            body = self._read(scope)
            row = body['entries'].get(token)
            if not row:
                raise CoordinationError('No durable entry identity to bind')
            before = row['identity']
            if (set(before) != set(confirmed_identity)
                    or any(before[k] != confirmed_identity[k] for k in before if k != 'perm_id')
                    or type(confirmed_identity['perm_id']) is not int or confirmed_identity['perm_id'] <= 0
                    or before['perm_id'] not in {0, confirmed_identity['perm_id']}):
                raise CoordinationError('Acknowledgement differs from durable entry identity')
            row['identity'] = confirmed_identity
            self._save(scope, body)
            return True

    def submit_legend(self, scope, signal_id, now, submit: Callable):
        """Serialize the actual Legend parent send with breakout admission.

        Only the synchronous parent transport call belongs here; fill waits and
        exit placement happen outside the lock. A lost send is never replayed.
        """
        if not self.enabled:
            return submit()
        with self.locked():
            body = self._read(scope)
            signal, op = body['signal'], body['operation']
            settled = (op or {}).get('settled_at')
            if (not signal or signal['id'] != signal_id or signal['status'] != 'eligible'
                    or not op or op['phase'] != 'done' or not settled
                    or stamp(now).date().isoformat() != scope.day
                    or stamp(now).time().replace(tzinfo=None) > wall_time(9, 31, 20)
                    or not 0 <= (stamp(now)-stamp(datetime.fromisoformat(settled))).total_seconds() <= 3
                    or body.get('legend_entry')):
                raise EntryBlocked('Legend parent is unsettled, late, or already journalled; reconcile')
            body['legend_entry'] = {'signal_id': signal_id, 'status': 'intent', 'at': stamp(now).isoformat()}
            self._save(scope, body)
            try:
                result = submit()
            except BaseException:
                body['legend_entry']['status'] = 'unknown'
                self._save(scope, body)
                raise
            body['legend_entry']['status'] = 'submitted'
            self._save(scope, body)
            return result

    def legend_ready(self, scope, now):
        if not self.enabled:
            return True
        body = self.read(scope)
        settled = (body['operation'] or {}).get('settled_at')
        return bool(body["signal"] and body["signal"]["status"] == "eligible"
                    and body["operation"]["phase"] == "done"
                    and settled and 0 <= (stamp(now)-stamp(datetime.fromisoformat(settled))).total_seconds() <= 3
                    and stamp(now).date().isoformat() == scope.day
                    and stamp(now).time().replace(tzinfo=None) <= wall_time(9, 31, 20))


@dataclass(frozen=True)
class OrderView:
    token: str
    account: str
    day: str
    symbol: str
    con_id: int
    client_id: int
    order_id: int
    perm_id: int
    strategy: str
    role: str
    owner_side: int
    status: str
    remaining: int
    filled: int = 0
    execution_filled: int = 0
    oca: str = ""
    terms: dict = field(default_factory=dict)

    @property
    def active(self):
        return self.status not in TERMINAL

    @property
    def identity(self):
        return {"client_id": self.client_id, "order_id": self.order_id, "perm_id": self.perm_id,
                "con_id": self.con_id, "account": self.account, "strategy": self.strategy, "role": self.role}


@dataclass(frozen=True)
class Snapshot:
    scope: Scope
    at: datetime
    con_id: int
    owned_qty: int
    account_qty: int
    orders: tuple[OrderView, ...]
    complete: bool = False
    ownership_known: bool = False
    manual_control: bool = False
    close_capable: bool = False


class LegacyCancelThenCloseModel:
    """REFERENCE MODEL ONLY: cancel-then-close includes a protection gap.

    Never wired to the native candidate. Its tests explain the risks and durable
    invariants of the previously inspected supported manual close flow. Native
    coordination uses AsyncOwnerCoordinator and never cancels protection to
    prepare a new market close.

    One bounded step at a time; every mutation has a durable stable ID.

    Adapter protocol (all mocked in this candidate): snapshot(scope),
    cancel_exact(order, operation_id), prepare_owned_close(snapshot, qty, id),
    submit_owned_close(snapshot, qty, id), find_close(id), restore_protection
    (scope, qty, stored_exit_plan, id). These are not aggregate flatten calls.
    prepare must return the reserved exact close order identity, including its
    client/order/account/contract IDs (permId may initially be zero), while the
    protective stop is still live. It must never place the close at preparation.
    A close cannot be submitted while any owned opposing entry/exit is working.
    """
    def __init__(self, ledger, broker, *, max_snapshot_age=3):
        self.ledger, self.broker = ledger, broker
        self.max_snapshot_age = max_snapshot_age

    @staticmethod
    def _close_receipt_valid(close, scope, snap, signal, op):
        expected = (op['close'] or {}).get('identity')
        if (not expected or not close or close.token != op['id'] or close.role != 'COORD_CLOSE'
                or close.account != scope.account or close.day != scope.day
                or FAMILY.get(close.symbol) != scope.family or close.con_id != snap.con_id
                or close.strategy != 'OpenBreakout' or close.owner_side != -signal['side']
                or close.status not in NORMALIZED_STATUSES
                or any(type(n) is not int or n < 0 for n in (close.remaining, close.filled, close.execution_filled))
                or close.filled != close.execution_filled or close.filled > op['close']['qty']
                or close.remaining + close.filled > op['close']['qty']
                or type(close.perm_id) is not int or close.perm_id <= 0
                or any(expected[k] != close.identity[k] for k in expected if k != 'perm_id')
                or expected['perm_id'] not in {0, close.perm_id}):
            return False
        op['close']['identity'] = close.identity
        return True

    def recover_close(self, scope, now):
        """Read-only broker recovery. Never resubmit an uncertain close.

        A late terminal receipt can advance to cleanup/restoration only with
        exact identity and fresh complete fills. Pending/absent receipts stay
        blocked. An operator must review other reconciliation causes.
        """
        if not self.ledger.enabled:
            return 'disabled'
        with self.ledger.locked():
            body = self.ledger._read(scope)
            op = body['operation']
            if not op or op['phase'] != 'reconcile' or not (op['close'] or {}).get('submitted'):
                return op['phase'] if op else 'no_signal'
            try:
                snap = self.broker.snapshot(scope)
                close = self.broker.find_close(op['id'])
            except Exception:
                return 'reconcile'
            if (snap.scope != scope or not snap.complete or not snap.ownership_known or snap.manual_control
                    or stamp(now).date().isoformat() != scope.day
                    or not 0 <= (stamp(now)-stamp(snap.at)).total_seconds() <= self.max_snapshot_age
                    or not self._close_receipt_valid(close, scope, snap, body['signal'], op) or close.active):
                return 'reconcile'
            qty = abs(snap.owned_qty) if snap.owned_qty * -body['signal']['side'] > 0 else 0
            if qty and snap.account_qty * -body['signal']['side'] < qty:
                return 'reconcile'
            op['phase'] = 'restore_protection' if qty else 'cleanup'
            op['reason'] = 'Late terminal close receipt reconciled; no close resubmission'
            self.ledger._save(scope, body)
            return op['phase']

    def step(self, scope, now):
        if not self.ledger.enabled:
            return "disabled"
        with self.ledger.locked():
            body = self.ledger._read(scope)
            op, signal = body["operation"], body["signal"]
            if not op or op["phase"] == "reconcile":
                return op["phase"] if op else "no_signal"
            def save(phase=None, reason=None):
                if phase:
                    op["phase"] = phase
                if op['phase'] == 'done':
                    op['settled_at'] = stamp(now).isoformat()
                if reason:
                    op["reason"] = reason
                self.ledger._save(scope, body)
                return op["phase"]
            def fail(reason):
                return save("reconcile", reason)
            if signal["status"] != "eligible":
                return fail("Legend signal is no longer eligible")
            if stamp(now).date().isoformat() != scope.day:
                return fail("Wrong trading day; no mutation")
            try:
                snap = self.broker.snapshot(scope)
            except Exception:
                return fail("Broker snapshot unavailable; no quantity/order inference")
            if (snap.scope != scope or not snap.complete or not snap.ownership_known
                    or not 0 <= (stamp(now) - stamp(snap.at)).total_seconds() <= self.max_snapshot_age):
                return fail("Unknown/stale broker ownership or order state")
            if snap.manual_control:
                return fail("Market handed to operator; no automatic close or cancellation")
            if (type(snap.owned_qty) is not int or type(snap.account_qty) is not int
                    or type(snap.con_id) is not int or snap.con_id <= 0):
                return fail("Non-whole broker quantity")
            if any(o.status not in NORMALIZED_STATUSES or type(o.owner_side) is not int or o.owner_side not in {-1,1}
                   for o in snap.orders):
                return fail('Broker state is not normalized and confirmed; Inactive alone is not cancellation proof')
            opposite = -signal["side"]
            own = [o for o in snap.orders if o.strategy == "OpenBreakout" and o.account == scope.account
                   and o.day == scope.day and FAMILY.get(o.symbol) == scope.family and o.owner_side == opposite]
            if any(o.con_id != snap.con_id or o.filled != o.execution_filled for o in own):
                return fail("Contract mismatch or an unreported fill; retain protection")
            if any(o.role not in {'ENTRY', 'STOP', 'TIME', 'COORD_CLOSE'}
                   or any(type(n) is not int or n < 0 for n in (o.remaining, o.filled, o.execution_filled, o.client_id))
                   or type(o.order_id) is not int or o.order_id <= 0
                   or type(o.perm_id) is not int or o.perm_id <= 0 for o in own):
                return fail('Unknown owned role or non-whole order quantity')
            # Missing durable intents are not proof of an unsent order.
            for token, row in body["entries"].items():
                if row["side"] == opposite and not any(o.token == token for o in own):
                    return fail("Entry intent absent from complete broker history; delivery unknown")
            if op['phase'] == 'done':
                if snap.owned_qty * opposite > 0 or any(o.active for o in own):
                    return fail('Late opposite fill/order after settlement; freeze new work and reconcile')
                return save('done')
            entries = [o for o in own if o.role == "ENTRY" and o.active]
            for order in entries:
                if order.token not in body["entries"] or body["entries"][order.token]["identity"] != order.identity:
                    return fail("Entry is not bound to the exact durable owner identity")
                if order.token not in op["cancelled"]:
                    op["cancelled"][order.token] = "requested"
                    save()
                    try:
                        self.broker.cancel_exact(order, ":".join((op["id"], "cancel", order.token)))
                    except Exception:
                        return fail("Entry cancellation acknowledgement/delivery unknown")
            if entries:
                return save("cancel_entries")
            if op["phase"] == "cancel_entries":
                return save("prepare_close" if snap.owned_qty * opposite > 0 else "cleanup")
            qty = abs(snap.owned_qty) if snap.owned_qty * opposite > 0 else 0
            exits = [o for o in own if o.role in {"STOP", "TIME", "COORD_CLOSE"} and o.active]
            if op["phase"] == "prepare_close":
                if not qty:
                    return save("cleanup")
                # Owned fill attribution is primary; net position is a ceiling,
                # never a sizing source. Opposite manual/Legend netting is unsafe.
                if snap.account_qty * opposite < qty:
                    return fail("Account net does not support owned opposing quantity; possible netting/manual close")
                if any(o.active and o.con_id == snap.con_id and o not in own for o in snap.orders):
                    return fail("Competing same-contract order is outside this owned close; do not mutate it")
                if not snap.close_capable:
                    return fail("Supported owned-close adapter has not been qualified; keep protection")
                if not any(o.role == "STOP" and o.remaining == qty for o in exits):
                    return fail("Existing protective stop does not cover the owned quantity")
                op["exit_plan"] = [asdict(o) for o in exits]
                op["close"] = {"id": op["id"], "qty_ceiling": qty, "status": "prepared", "submitted": False}
                save()
                try:
                    prepared = self.broker.prepare_owned_close(snap, qty, op["id"])
                    required = {'account', 'con_id', 'client_id', 'order_id', 'perm_id', 'strategy', 'role'}
                    owner_clients = {o.client_id for o in own if o.role == 'ENTRY'}
                    if (not isinstance(prepared, dict) or set(prepared) != required
                            or prepared['account'] != scope.account or prepared['con_id'] != snap.con_id
                            or prepared['strategy'] != 'OpenBreakout' or prepared['role'] != 'COORD_CLOSE'
                            or prepared['client_id'] not in owner_clients
                            or type(prepared['order_id']) is not int or prepared['order_id'] <= 0
                            or type(prepared['perm_id']) is not int or prepared['perm_id'] < 0):
                        return fail("Owned close preparation refused; original protection retained")
                    op['close']['identity'] = prepared
                except Exception:
                    return fail("Owned close preparation failed; original protection retained")
                return save("cancel_exits")
            if op["phase"] == "cancel_exits":
                # Prepared close only. Match every original exact identity before
                # removal; fills after preparation reduce the ultimate close size.
                planned = {o["token"]: o for o in op["exit_plan"]}
                identity_keys = ('client_id', 'order_id', 'perm_id', 'con_id', 'account', 'strategy', 'role')
                if any(not any(o.token == token and all(asdict(o)[key] == plan[key] for key in identity_keys)
                               for o in own) for token, plan in planned.items()):
                    return fail('Prepared exit absent from complete broker history; no close')
                for order in exits:
                    if order.token not in planned or any(asdict(order)[key] != planned[order.token][key] for key in identity_keys):
                        return fail("Exit identity changed after preparation; no close")
                    if order.token not in op["cancelled"]:
                        op["cancelled"][order.token] = "requested"
                        save()
                        try:
                            self.broker.cancel_exact(order, ":".join((op["id"], "cancel", order.token)))
                        except Exception:
                            return fail("Exit cancellation uncertain; do not submit a competing close")
                if exits:
                    return save()
                return save("submit_close" if qty else "cleanup")
            if op["phase"] == "submit_close":
                if exits or entries:
                    return fail("An exit/entry remains working; no competing close")
                if not qty:
                    return save("cleanup")
                if (snap.account_qty * opposite < qty or qty > op["close"]["qty_ceiling"]
                        or any(o.active and o.con_id == snap.con_id and o not in own for o in snap.orders)):
                    return fail("Owned quantity/order state changed before close; do not submit")
                op["close"]["status"] = "intent"
                op["close"]["submitted"] = True
                op["close"]["qty"] = qty
                save("close_working")  # crash after here is NEVER permission to retry
                try:
                    self.broker.submit_owned_close(snap, qty, op["id"])
                except Exception:
                    return fail("Close submission delivery unknown; do not resend or restore competing exits")
                return op["phase"]
            if op["phase"] == "close_working":
                try:
                    close = self.broker.find_close(op["id"])
                except Exception:
                    return fail("Close absent/unknown; never submit a replacement")
                if not self._close_receipt_valid(close, scope, snap, signal, op):
                    return fail("Close acknowledgement lacks exact provenance")
                if close.filled != close.execution_filled:
                    return fail("Close fill report incomplete; no quantity inference")
                if qty != op['close']['qty'] - close.execution_filled:
                    return fail('Attributed residual differs from confirmed close executions; no inferred restoration')
                if not close.active:
                    if qty:
                        # Terminal close with proven executions. Protection is
                        # restored only for the verified remaining owned quantity.
                        op["close"]["status"] = "terminal_residual"
                        save("restore_protection")
                        return op["phase"]
                    return save("cleanup")
                return save()
            if op["phase"] == "restore_protection":
                if not qty:
                    return save("cleanup")
                if snap.account_qty * opposite < qty or any(o.active and o.con_id == snap.con_id for o in snap.orders):
                    return fail("Residual restoration state is ambiguous")
                # At-most-once restoration, with its own stable operation ID.
                save("restore_pending")
                try:
                    self.broker.restore_protection(scope, qty, op["exit_plan"], op["id"] + ":restore")
                except Exception:
                    return fail("Protection restoration delivery unknown; inspect immediately")
                return op["phase"]
            if op["phase"] == "restore_pending":
                if not qty:
                    return save("cleanup")
                if not any(o.role == "STOP" and o.remaining == qty for o in exits):
                    return fail("Restored protection not acknowledged; manual reconciliation needed")
                return fail("Close rejected/cancelled with residual; protection restored, no automatic close retry")
            if op["phase"] == "cleanup":
                if qty:
                    return fail("Owned exposure reappeared during cleanup (late fill)")
                for order in exits:
                    if order.token not in op["cancelled"]:
                        op["cancelled"][order.token] = "requested"
                        save()
                        try:
                            self.broker.cancel_exact(order, ":".join((op["id"], "cleanup", order.token)))
                        except Exception:
                            return fail("Orphaned-exit cleanup uncertain; Legend remains blocked")
                if exits:
                    return save()
                return save("done")
            return fail("Unsupported reconciliation phase")


def from_environment():
    """Absent flag is inert: no file creation, broker calls, or timing change."""
    value = os.environ.get("INTRADAY_COORDINATION_ENABLED", "0")
    if value == "0":
        return None
    if value != "1":
        raise CoordinationError("Coordination enable flag must be exactly 0 or 1")
    path = os.environ.get("INTRADAY_COORDINATION_DB", "")
    if not path:
        raise CoordinationError("Explicit shared local coordination DB path required")
    return Ledger(path, enabled=True)
