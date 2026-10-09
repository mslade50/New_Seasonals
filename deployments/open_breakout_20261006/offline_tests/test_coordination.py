import multiprocessing as mp
import os
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
import time

import pytest

from intraday_coordination import (LegacyCancelThenCloseModel as CloseController, CoordinationError, EntryBlocked, Ledger,
                                  LegendSignal, NY, OrderView, Scope, Snapshot, from_environment)

DAY = '2026-10-06'


def at(text='09:31:01'):
    return datetime.fromisoformat(DAY + 'T' + text).replace(tzinfo=NY)


def scope(account='PRIMARY_TEST', symbol='MES'):
    return Scope.for_symbol(DAY, account, symbol)


def signal(s=None, side=1):
    return LegendSignal(s or scope(), side, 'final-legend-' + str(side), at(), True, True, True, 2)


def order(token, role='ENTRY', side=-1, qty=5, status='Submitted', *, s=None, filled=0):
    s = s or scope()
    oid = sum(map(ord, token))
    return OrderView(token, s.account, s.day, 'MES' if s.family == 'ES' else 'MNQ', 101,
                     91, oid, oid + 10000, 'OpenBreakout', role, side, status, qty,
                     filled, filled, 'group-1', {'kind': 'STP' if role == 'STOP' else 'MKT', 'stop': 99})


class Broker:
    """Inert broker. Signed owned quantity comes from simulated attributed fills."""
    def __init__(self, s=None):
        self.scope = s or scope()
        self.orders = {}
        self.owned = 0
        self.foreign = 0
        self.calls = []
        self.complete = True
        self.known = True
        self.manual = False
        self.capable = True
        self.snapshot_at = at()
        self.cancel_fill = None
        self.cancel_timeout = False
        self.close_timeout = False
        self.close_partial = None
        self.reject = False
        self.prepare_ok = True

    def snapshot(self, s):
        return Snapshot(s, self.snapshot_at, 101, self.owned, self.owned + self.foreign,
                        tuple(self.orders.values()), self.complete, self.known, self.manual, self.capable)

    def add_entry(self, ledger, token='entry', side=-1, qty=5, filled=0):
        row = order(token, side=side, qty=qty-filled, filled=filled,
                    status='Filled' if qty == filled else 'Submitted', s=self.scope)
        ledger.submit_entry(self.scope, side, token, row.identity, lambda: self.orders.setdefault(token, row))
        self.owned += side * filled
        if filled:
            self.protect(side, filled)
        return row

    def protect(self, side, qty, suffix=''):
        self.orders['stop'+suffix] = order('stop'+suffix, 'STOP', side, qty, s=self.scope)
        self.orders['time'+suffix] = order('time'+suffix, 'TIME', side, qty, s=self.scope)

    def fill(self, token, qty):
        row = self.orders[token]
        qty = min(qty, row.remaining)
        change = row.owner_side if row.role == 'ENTRY' else -row.owner_side
        self.owned += change * qty
        self.orders[token] = replace(row, remaining=row.remaining-qty, filled=row.filled+qty,
                                     execution_filled=row.execution_filled+qty,
                                     status='Filled' if row.remaining == qty else 'Submitted')
        if row.role == 'ENTRY':
            self.protect(row.owner_side, abs(self.owned))
        elif row.role == 'STOP':
            for key, sibling in list(self.orders.items()):
                if key != token and sibling.role in {'STOP', 'TIME'} and sibling.oca == row.oca:
                    self.orders[key] = replace(sibling, remaining=max(0, sibling.remaining-qty),
                                               status='Cancelled' if sibling.remaining <= qty else sibling.status)

    def cancel_exact(self, row, operation_id):
        self.calls.append(('cancel', row.token, operation_id))
        assert self.orders[row.token].identity == row.identity
        if self.cancel_fill:
            token, qty = self.cancel_fill
            self.cancel_fill = None
            self.fill(token, qty)
        if self.cancel_timeout:
            raise TimeoutError('delivery unknown')
        current = self.orders[row.token]
        if current.status != 'Filled':
            self.orders[row.token] = replace(current, status='Cancelled', remaining=0)

    def prepare_owned_close(self, snap, qty, operation_id):
        self.calls.append(('prepare', qty, operation_id))
        # Protection is still live at preparation, with an exact quantity.
        assert any(o.role == 'STOP' and o.active and o.remaining == qty for o in snap.orders)
        return order(operation_id, 'COORD_CLOSE', -1 if self.owned < 0 else 1, qty,
                     s=self.scope).identity if self.prepare_ok else False

    def submit_owned_close(self, snap, qty, operation_id):
        self.calls.append(('close', qty, operation_id))
        assert not any(o.active and o.owner_side == -1 and o.role in {'ENTRY', 'STOP', 'TIME'}
                       for o in self.orders.values())
        assert qty == abs(self.owned) and qty <= abs(snap.owned_qty)
        row = order(operation_id, 'COORD_CLOSE', -1 if self.owned < 0 else 1, qty, s=self.scope)
        self.orders[operation_id] = row
        if self.close_timeout:
            raise TimeoutError('ack lost after broker acceptance')
        if self.reject:
            self.orders[operation_id] = replace(row, status='Rejected')
        else:
            self.fill(operation_id, qty if self.close_partial is None else self.close_partial)

    def find_close(self, operation_id):
        return self.orders.get(operation_id)

    def restore_protection(self, s, qty, original, operation_id):
        self.calls.append(('restore', qty, operation_id))
        assert qty == abs(self.owned)
        self.protect(1 if self.owned > 0 else -1, qty, '-restored')


def advance(controller, s=None, until='done', count=12):
    s = s or scope()
    phases = []
    for _ in range(count):
        phases.append(controller.step(s, at()))
        if phases[-1] in {until, 'reconcile'}:
            break
    return phases


@pytest.fixture
def ledger(tmp_path):
    return Ledger(tmp_path / 'arbitration.sqlite', enabled=True)


def test_disabled_is_inert_even_for_invalid_signal(tmp_path):
    target = tmp_path/'never-created'/'db.sqlite'
    ledger = Ledger(target)  # default off
    assert not ledger.publish(replace(signal(), enabled=False), at())
    sent = []
    ledger.submit_entry(scope(), -1, '', {}, lambda: sent.append('baseline'))
    assert sent == ['baseline'] and not target.parent.exists()
    assert CloseController(ledger, object()).step(scope(), at()) == 'disabled'


def test_environment_requires_explicit_opt_in(monkeypatch, tmp_path):
    monkeypatch.delenv('INTRADAY_COORDINATION_ENABLED', raising=False)
    monkeypatch.setenv('INTRADAY_COORDINATION_DB', str(tmp_path/'unused.db'))
    assert from_environment() is None and not (tmp_path/'unused.db').exists()
    monkeypatch.setenv('INTRADAY_COORDINATION_ENABLED', 'yes')
    with pytest.raises(CoordinationError):
        from_environment()


@pytest.mark.parametrize('symbol', ['ES', 'MES', 'NQ', 'MNQ'])
@pytest.mark.parametrize('account', ['PRIMARY_TEST', 'PA_TEST'])
@pytest.mark.parametrize('side', [-1, 1])
def test_opposite_only_and_exact_account_family(ledger, symbol, account, side):
    s = scope(account, symbol)
    ledger.publish(signal(s, side), at())
    assert ledger.allowed(s, side) and not ledger.allowed(s, -side)
    assert ledger.allowed(scope(account+'OTHER', symbol), -side)
    assert ledger.allowed(scope(account, 'NQ' if s.family == 'ES' else 'ES'), -side)
    assert ledger.allowed(Scope('2026-10-07', account, s.family), -side)


@pytest.mark.parametrize('changes', [
    {'enabled':False}, {'finalized':False}, {'actionable':False}, {'normal_session':False},
    {'quantity':0}, {'quantity':True}, {'side':0}, {'decided_at':at('09:30:59')},
    {'decided_at':at('09:31:21')}, {'decided_at':at()+timedelta(days=-1)},
])
def test_invalid_candidate_never_creates_suppression(ledger, changes):
    with pytest.raises(CoordinationError):
        ledger.publish(replace(signal(), **changes), at())
    assert ledger.allowed(scope(), -1)


def test_stale_and_future_signal_never_publish(ledger):
    for when in (at('09:31:22'), at('09:30:59')):
        with pytest.raises(CoordinationError):
            ledger.publish(signal(), when)
    assert ledger.read(scope())['signal'] is None


def test_no_signal_does_not_wait_or_suppress(ledger):
    sent = []
    row = order('open-at-0930')
    ledger.submit_entry(scope(), -1, row.token, row.identity, lambda: sent.append('0930'))
    assert sent == ['0930']
    assert CloseController(ledger, object()).step(scope(), at()) == 'no_signal'


def test_durable_signal_retry_and_revocation_no_accidental_rearm(ledger):
    ledger.publish(signal(), at())
    restarted = Ledger(ledger.path, enabled=True)
    assert restarted.publish(signal(), at('11:00:00')) is False
    restarted.signal_status(scope(), signal().signal_id, 'revoked')
    assert not restarted.allowed(scope(), -1) and restarted.allowed(scope(), 1)
    assert CloseController(restarted, object()).step(scope(), at()) == 'reconcile'
    with pytest.raises(CoordinationError):
        restarted.publish(signal(side=-1), at())


def test_broker_rejected_legend_retains_opposite_latch_pending_decision(ledger):
    ledger.publish(signal(), at())
    ledger.signal_status(scope(), signal().signal_id, 'broker_rejected')
    assert ledger.allowed(scope(), 1) and not ledger.allowed(scope(), -1)
    assert not ledger.legend_ready(scope(), at())


def test_duplicate_or_uncertain_breakout_intent_never_resends(ledger):
    row = order('unique')
    def uncertain():
        raise TimeoutError()
    with pytest.raises(TimeoutError):
        ledger.submit_entry(scope(), -1, row.token, row.identity, uncertain)
    restarted = Ledger(ledger.path, enabled=True)
    with pytest.raises(EntryBlocked):
        restarted.submit_entry(scope(), -1, row.token, row.identity, lambda: pytest.fail('duplicate'))
    assert restarted.read(scope())['entries'][row.token]['status'] == 'unknown'


def test_unfilled_opposite_cancel_preserves_same_side_and_manual(ledger):
    b = Broker()
    b.add_entry(ledger, 'opposite')
    b.add_entry(ledger, 'same', 1)
    b.orders['manual'] = replace(order('manual', side=1), strategy='Manual', con_id=777)
    ledger.publish(signal(), at())
    assert advance(CloseController(ledger,b))[-1] == 'done'
    assert [c[1] for c in b.calls if c[0] == 'cancel'] == ['opposite']
    assert b.orders['same'].active and b.orders['manual'].active
    assert not any(c[0] == 'close' for c in b.calls)
    assert ledger.legend_ready(scope(), at())


@pytest.mark.parametrize('legend_side', [-1, 1])
def test_close_only_owned_opposing_filled_quantity(ledger, legend_side):
    b = Broker()
    b.add_entry(ledger, side=-legend_side, qty=4, filled=4)
    ledger.publish(signal(side=legend_side), at())
    assert advance(CloseController(ledger,b))[-1] == 'done'
    assert b.owned == 0
    assert [c[1] for c in b.calls if c[0] == 'close'] == [4]
    assert not any(o.active and o.strategy == 'OpenBreakout' for o in b.orders.values())


def test_cancel_fill_race_and_partial_entry_size_from_actual_fills(ledger):
    b = Broker()
    b.add_entry(ledger, qty=5, filled=1)
    b.cancel_fill = ('entry', 2)
    ledger.publish(signal(), at())
    assert advance(CloseController(ledger,b))[-1] == 'done'
    assert [c[1] for c in b.calls if c[0] == 'close'] == [3]


def test_stop_fill_during_exit_cancellation_reduces_close_quantity(ledger):
    b = Broker()
    b.add_entry(ledger, qty=5, filled=5)
    ledger.publish(signal(), at())
    controller = CloseController(ledger,b)
    assert controller.step(scope(), at()) == 'prepare_close'
    assert controller.step(scope(), at()) == 'cancel_exits'
    b.cancel_fill = ('stop', 2)
    assert advance(controller)[-1] == 'done'
    assert [c[1] for c in b.calls if c[0] == 'close'] == [3]


def test_full_stop_fill_during_close_preparation_never_sends_market_close(ledger):
    b = Broker()
    b.add_entry(ledger, qty=3, filled=3)
    ledger.publish(signal(), at())
    controller = CloseController(ledger,b)
    controller.step(scope(), at()); controller.step(scope(), at())
    b.fill('stop', 3)
    assert advance(controller)[-1] == 'done'
    assert not any(c[0] == 'close' for c in b.calls)


@pytest.mark.parametrize('condition', ['unknown', 'stale', 'manual', 'netted', 'unreported', 'unsupported'])
def test_unsafe_existing_position_retains_protection_and_flags_reconciliation(ledger, condition):
    b = Broker()
    b.add_entry(ledger, qty=3, filled=3)
    if condition == 'unknown': b.known = False
    if condition == 'stale': b.snapshot_at = at() - timedelta(seconds=4)
    if condition == 'manual': b.manual = True
    if condition == 'netted': b.foreign = 3
    if condition == 'unsupported': b.capable = False
    if condition == 'unreported': b.orders['entry'] = replace(b.orders['entry'], execution_filled=2)
    ledger.publish(signal(), at())
    assert advance(CloseController(ledger,b))[-1] == 'reconcile'
    assert b.orders['stop'].active and b.orders['time'].active
    assert not any(c[0] == 'close' for c in b.calls)


def test_competing_manual_order_same_contract_is_untouched(ledger):
    b = Broker()
    b.add_entry(ledger, qty=2, filled=2)
    b.orders['manual'] = replace(order('manual'), strategy='Manual')
    ledger.publish(signal(), at())
    assert advance(CloseController(ledger,b))[-1] == 'reconcile'
    assert b.orders['manual'].active and b.orders['stop'].active


def test_simultaneous_legend_and_breakout_fills_net_ambiguity_fails_closed(ledger):
    b = Broker()
    b.add_entry(ledger, qty=2, filled=2)
    b.foreign = 2  # A Legend fill raced ahead of the new handshake.
    ledger.publish(signal(), at())
    assert advance(CloseController(ledger,b))[-1] == 'reconcile'
    assert not ledger.legend_ready(scope(), at())
    assert b.owned == -2 and not any(c[0] == 'close' for c in b.calls)


def test_cancel_timeout_never_closes_or_retries_cancellation(ledger):
    b = Broker(); b.add_entry(ledger, filled=1); b.cancel_timeout = True
    ledger.publish(signal(), at()); controller = CloseController(ledger,b)
    assert advance(controller)[-1] == 'reconcile'
    assert controller.step(scope(), at()) == 'reconcile'
    assert len(b.calls) == 1 and b.orders['stop'].active


def test_partial_close_restart_waits_without_resending(ledger):
    b = Broker(); b.add_entry(ledger, qty=5, filled=5); b.close_partial = 2
    ledger.publish(signal(), at()); controller = CloseController(ledger,b)
    assert advance(controller, until='close_working')[-1] == 'close_working'
    restarted = CloseController(Ledger(ledger.path, enabled=True),b)
    assert advance(restarted, until='close_working')[-1] == 'close_working'
    op_id = ledger.read(scope())['operation']['id']
    b.fill(op_id, 3)
    assert advance(restarted)[-1] == 'done'
    assert len([c for c in b.calls if c[0] == 'close']) == 1


def test_lost_close_ack_late_receipt_reconciles_without_resubmit(ledger):
    b = Broker(); b.add_entry(ledger, qty=3, filled=3); b.close_timeout = True
    ledger.publish(signal(), at()); controller = CloseController(ledger,b)
    assert advance(controller)[-1] == 'reconcile'
    assert controller.recover_close(scope(), at()) == 'reconcile'  # still working
    op_id = ledger.read(scope())['operation']['id']; b.fill(op_id, 3)
    assert controller.recover_close(scope(), at()) == 'cleanup'
    assert advance(controller)[-1] == 'done'
    assert len([c for c in b.calls if c[0] == 'close']) == 1


def test_rejected_close_restores_known_residual_then_stops(ledger):
    b = Broker(); b.add_entry(ledger, qty=3, filled=3); b.reject = True
    ledger.publish(signal(), at()); controller = CloseController(ledger,b)
    assert advance(controller)[-1] == 'reconcile'
    assert b.orders['stop-restored'].active and b.orders['stop-restored'].remaining == 3
    assert [c[0] for c in b.calls].count('restore') == 1
    assert not ledger.legend_ready(scope(), at())


def test_same_direction_position_and_exits_remain_untouched(ledger):
    b = Broker(); b.add_entry(ledger, side=1, qty=3, filled=3)
    ledger.publish(signal(), at())
    assert advance(CloseController(ledger,b))[-1] == 'done'
    assert b.owned == 3 and b.orders['stop'].active and b.calls == []


def test_unknown_permanent_identity_can_bind_once_without_replacing_owner(ledger):
    row=order('entry'); before={**row.identity, 'perm_id':0}
    ledger.submit_entry(scope(),-1,row.token,before,lambda:None)
    assert ledger.bind_entry_identity(scope(),row.token,row.identity)
    assert ledger.bind_entry_identity(scope(),row.token,row.identity)
    for wrong in ({**row.identity,'order_id':999}, {**row.identity,'perm_id':999}):
        with pytest.raises(CoordinationError):
            ledger.bind_entry_identity(scope(),row.token,wrong)
    assert ledger.read(scope())['entries'][row.token]['identity']==row.identity


def test_unverified_snapshot_defaults_fail_closed(ledger):
    b=Broker(); b.add_entry(ledger,filled=1); ledger.publish(signal(),at())
    real=b.snapshot
    b.snapshot=lambda s:replace(real(s),complete=False,ownership_known=False)
    assert advance(CloseController(ledger,b))[-1]=='reconcile'
    assert b.orders['stop'].active and not b.calls


def test_inactive_alone_never_proves_cancellation(ledger):
    b=Broker(); row=b.add_entry(ledger,filled=1)
    b.orders[row.token]=replace(b.orders[row.token],status='Inactive')
    ledger.publish(signal(),at())
    assert advance(CloseController(ledger,b))[-1]=='reconcile'
    assert b.orders['stop'].active and not b.calls


def test_missing_prepared_exit_never_authorizes_market_close(ledger):
    b=Broker(); b.add_entry(ledger,qty=2,filled=2); ledger.publish(signal(),at())
    controller=CloseController(ledger,b)
    controller.step(scope(),at()); controller.step(scope(),at())
    del b.orders['time']
    assert controller.step(scope(),at())=='reconcile'
    assert b.orders['stop'].active and not any(c[0]=='close' for c in b.calls)


@pytest.mark.parametrize('changes', [
    {'account':'OTHER'}, {'day':'2026-10-05'}, {'symbol':'MNQ'}, {'con_id':999},
    {'client_id':999}, {'order_id':999}, {'perm_id':0}, {'strategy':'Manual'},
    {'role':'TIME'}, {'owner_side':1}, {'remaining':999}, {'execution_filled':1},
])
def test_close_receipt_requires_reserved_exact_identity_and_fills(ledger,changes):
    b=Broker(); b.add_entry(ledger,qty=3,filled=3); b.close_partial=0
    ledger.publish(signal(),at()); controller=CloseController(ledger,b)
    advance(controller,until='close_working')
    token=ledger.read(scope())['operation']['id']
    b.orders[token]=replace(b.orders[token],**changes)
    assert controller.step(scope(),at())=='reconcile'
    assert not ledger.legend_ready(scope(),at())
    assert len([c for c in b.calls if c[0]=='close'])==1
    assert not any(c[0]=='restore' for c in b.calls)


def test_legend_parent_send_requires_fresh_settlement_and_never_duplicates(ledger):
    b=Broker(); ledger.publish(signal(),at()); advance(CloseController(ledger,b))
    calls=[]
    assert ledger.submit_legend(scope(),signal().signal_id,at(),lambda:calls.append('send')) is None
    with pytest.raises(EntryBlocked):
        Ledger(ledger.path,enabled=True).submit_legend(scope(),signal().signal_id,at(),lambda:calls.append('duplicate'))
    assert calls==['send']


def test_legend_parent_unknown_delivery_never_resends(ledger):
    b=Broker(); ledger.publish(signal(),at()); advance(CloseController(ledger,b))
    def unknown():raise TimeoutError('send delivery unknown')
    with pytest.raises(TimeoutError):ledger.submit_legend(scope(),signal().signal_id,at(),unknown)
    with pytest.raises(EntryBlocked):ledger.submit_legend(scope(),signal().signal_id,at(),lambda:pytest.fail('duplicate'))
    assert ledger.read(scope())['legend_entry']['status']=='unknown'


def test_legend_parent_rechecks_settlement_age_at_send_boundary(ledger):
    b=Broker(); ledger.publish(signal(),at()); advance(CloseController(ledger,b))
    with pytest.raises(EntryBlocked):
        ledger.submit_legend(scope(),signal().signal_id,at()+timedelta(seconds=4),lambda:pytest.fail('late'))


def _race_worker(path, queue, kind):
    ledger = Ledger(Path(path), enabled=True, lock_timeout=5)
    try:
        if kind == 'entry':
            row = order('race-entry')
            def send():
                queue.put(('entered_transport', kind))
                time.sleep(.15)
            ledger.submit_entry(scope(), -1, row.token, row.identity, send)
            queue.put(('sent', kind))
        else:
            ledger.publish(signal(), at())
            queue.put(('published', kind))
    except EntryBlocked:
        queue.put(('blocked', kind))


def test_interprocess_signal_and_entry_send_are_serialized(tmp_path):
    path = tmp_path/'shared.db'; Ledger(path, enabled=True)
    ctx = mp.get_context('spawn'); queue = ctx.Queue()
    entry = ctx.Process(target=_race_worker, args=(str(path), queue, 'entry'))
    entry.start()
    assert queue.get(timeout=10)[0] == 'entered_transport'
    publisher = ctx.Process(target=_race_worker, args=(str(path), queue, 'signal'))
    publisher.start(); entry.join(10); publisher.join(10)
    assert entry.exitcode == publisher.exitcode == 0
    events = [queue.get(timeout=5), queue.get(timeout=5)]
    assert {e[0] for e in events} == {'sent', 'published'}
    ledger = Ledger(path, enabled=True)
    assert not ledger.allowed(scope(), -1)
    assert ledger.read(scope())['entries']['race-entry']['status'] == 'submitted'
    with pytest.raises(EntryBlocked):
        row = order('after'); ledger.submit_entry(scope(), -1, row.token, row.identity, lambda: pytest.fail('late send'))
