"""Offline connection and request-lifecycle regressions; no live runtime import."""
import ast
import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace as N

import pytest

from broker_runtime import execution_connection as connection
from broker_runtime.prepare_execution_connection import patch, prepare
from tests.execution_harness import load_agent

FIXTURE = Path(__file__).parent/'fixtures/execution_runtime/exec_agent_connection.py'


def test_stable_acknowledged_connection_resets_and_short_flaps_back_off():
    async def exercise():
        clock, waits, states = [0], [], []
        count = [0]
        async def once(state):
            monitor = connection.Session(state, lambda _: None, clock=lambda: clock[0])
            if count[0] == 4:
                clock[0] += 10
                monitor.observe({'type': 'ack', 'of': 'heartbeat'})
                clock[0] += 21
                monitor.observe({'type': 'ack', 'of': 'heartbeat'})
            else:
                clock[0] += 1
            states.append(state.stable)
            count[0] += 1
            raise ConnectionError('fixture transport loss')
        async def sleep(delay):
            waits.append(delay)
        ns = dict(WS_URL='fixture', TOKEN='fixture', _load_seen=lambda: None, _load_schedules=lambda: None,
                  in_window=lambda: count[0] < 8, log=lambda _: None, BACKOFF_MAX_S=30)
        await connection.run(ns, once, clock=lambda: clock[0], sleep=sleep, jitter=lambda *_: 1)
        assert waits == [1, 2, 4, 8, 1, 2, 4]
        assert states == [False]*4 + [True] + [False]*3
    asyncio.run(exercise())


def test_missing_ack_does_not_reset_or_change_trading_state():
    clock, logs = [0], []
    attempt = connection.Attempt(1)
    monitor = connection.Session(attempt, logs.append, clock=lambda: clock[0])
    for _ in range(7):
        clock[0] += 10
        monitor.heartbeat()
        monitor.tick()
    assert monitor.summary()['ack_missing']
    assert not attempt.stable
    assert any('heartbeat_health' in line for line in logs)
    assert all('command' not in line for line in logs)


def test_correlated_ack_rejects_wrong_session_unknown_sequence_and_old_reply():
    clock = [0]
    monitor = connection.Session(connection.Attempt(1), lambda _: None, clock=lambda: clock[0])
    first = monitor.heartbeat()
    monitor.observe(dict(type='ack', of='heartbeat', seq=first['seq'], session='other'))
    monitor.observe(dict(type='ack', of='heartbeat', seq=999, session=monitor.id))
    assert monitor.acks == 0
    clock[0] = 40
    monitor.observe(dict(type='ack', of='heartbeat', seq=first['seq'], session=monitor.id))
    assert monitor.acks == 0
    for advance in (1, 1):
        clock[0] += advance
        heartbeat = monitor.heartbeat()
        monitor.observe(dict(type='ack', of='heartbeat', seq=heartbeat['seq'], session=heartbeat['session']))
    assert monitor.attempt.stable and monitor.ack_mode == 'correlated'
    assert monitor.summary()['last_rtt_s'] == 0


@pytest.mark.parametrize('sample', [.8, 1, 1.2])
def test_flapping_retries_are_bounded_nonzero_with_jitter(sample):
    async def exercise():
        calls, delays = [0], []
        async def once(_):
            calls[0] += 1
            raise OSError('fixture')
        async def sleep(delay):
            delays.append(delay)
        ns = dict(WS_URL='fixture', TOKEN='fixture', _load_seen=lambda: None, _load_schedules=lambda: None,
                  in_window=lambda: calls[0] < 10, log=lambda _: None, BACKOFF_MAX_S=30)
        await connection.run(ns, once, sleep=sleep, jitter=lambda *_: sample)
        assert all(.8 <= delay <= 30 for delay in delays)
        assert delays[-1] >= 24 and len(delays) == 9
    asyncio.run(exercise())


def test_monotonic_lag_and_busy_consumer_are_reported():
    clock = [0]
    monitor = connection.Session(connection.Attempt(1), lambda _: None, clock=lambda: clock[0])
    clock[0] = 70
    monitor.loop_wake(10)
    monitor.busy = True
    assert monitor.summary()['max_loop_lag_s'] == 60
    assert monitor.summary()['consumer_busy'] and monitor.summary()['ack_missing']


def test_heartbeat_failure_finishes_inflight_request_before_reconnect():
    async def exercise():
        events, release = [], asyncio.Event()
        async def receive():
            events.append('request-start')
            await release.wait()
            events.append('durable-terminal-receipt')
        async def heartbeat():
            await asyncio.sleep(0)
            raise OSError('heartbeat transport failed')
        async def close():
            events.append('close-transport')
            release.set()
        receiver, beat = asyncio.create_task(receive()), asyncio.create_task(heartbeat())
        with pytest.raises(OSError, match='heartbeat transport failed'):
            await connection.supervise(receiver, beat, close)
        assert events == ['request-start', 'close-transport', 'durable-terminal-receipt']
        assert receiver.done() and not receiver.cancelled()
    asyncio.run(exercise())


def test_graceful_window_shutdown_drains_receiver_without_retry():
    async def exercise():
        done = asyncio.Event()
        async def receive():
            await done.wait()
        async def heartbeat():
            done.set()
        await connection.supervise(asyncio.create_task(receive()), asyncio.create_task(heartbeat()), lambda: None)
        count, delays = [0], []
        async def once(_):
            count[0] += 1
        async def sleep(delay):
            delays.append(delay)
        ns = dict(WS_URL='fixture', TOKEN='fixture', _load_seen=lambda: None, _load_schedules=lambda: None,
                  in_window=lambda: count[0] == 0, log=lambda _: None)
        await connection.run(ns, once, sleep=sleep)
        assert not delays
    asyncio.run(exercise())


def test_cancelled_retry_loop_propagates_without_replay():
    async def exercise():
        delays = []
        async def once(_):
            raise asyncio.CancelledError()
        async def sleep(delay):
            delays.append(delay)
        ns = dict(WS_URL='fixture', TOKEN='fixture', _load_seen=lambda: None, _load_schedules=lambda: None,
                  in_window=lambda: True, log=lambda _: None)
        with pytest.raises(asyncio.CancelledError):
            await connection.run(ns, once, sleep=sleep)
        assert not delays
    asyncio.run(exercise())


def test_exception_close_and_underlying_os_fields_redact_credentials():
    secret = 'fixture-secret-123'
    cause = OSError(104, 'Bearer ' + secret)
    exc = ConnectionError('wss://user:password@fixture.invalid/agent ' + secret)
    exc.__cause__ = cause
    exc.rcvd = N(code=1006, reason='Authorization: '+secret)
    fields = connection.exception_fields(exc, [secret])
    encoded = json.dumps(fields)
    assert secret not in encoded and 'password' not in encoded
    assert fields['rcvd']['code'] == 1006 and fields['cause']['errno'] == 104


def test_patcher_changes_transport_only_and_preserves_command_handlers():
    source = FIXTURE.read_text()
    updated = patch(source)
    compile(updated, '<inert-candidate>', 'exec')
    assert 'execution_connection.supervise(receiver, hb, ws.close)' in updated
    assert 'ping_interval=None' in updated
    assert updated.count('await _handle_command(') == source.count('await _handle_command(') == 1
    assert updated.count('await _fetch_workbench(') == source.count('await _fetch_workbench(') == 1
    with pytest.raises(ValueError):
        patch(updated)


def test_prepare_rejects_runtime_drift_and_inplace_output(tmp_path):
    runtime = tmp_path/'runtime'
    runtime.mkdir()
    (runtime/'exec_agent.py').write_text(FIXTURE.read_text())
    with pytest.raises(ValueError, match='separate'):
        prepare(runtime, runtime/'candidate')
    with pytest.raises(ValueError, match='changed'):
        prepare(runtime, tmp_path/'candidate')
    assert not (tmp_path/'candidate').exists()


@pytest.mark.parametrize('receipt_is_claimed', [True, False])
def test_duplicate_pending_receipt_never_calls_executor_again(monkeypatch, receipt_is_claimed):
    agent = load_agent()
    wire_calls, receipts = [], []
    monkeypatch.setattr(agent, '_verify', lambda *_: True)
    monkeypatch.setattr(agent, '_claim_seen', lambda *_: not receipt_is_claimed)
    monkeypatch.setattr(agent, '_record_seen', lambda *args: receipts.append(args))
    monkeypatch.setattr(agent, '_validate', lambda *_: (True, []))
    monkeypatch.setattr(agent, '_preview', lambda *_: {})
    monkeypatch.setattr(agent, '_describe', lambda *_: 'fixture')
    monkeypatch.setattr(agent, '_live_eligible', lambda *_: (True, 'fixture'))
    monkeypatch.setattr(agent, 'log', lambda _: None, raising=False)
    async def execute(cmd):
        wire_calls.append(cmd['id'])
        return dict(state='unknown', ok=False, detail='fixture unresolved receipt')
    monkeypatch.setattr(agent, '_execute_live', execute)
    agent._SEEN.clear()
    messages = []
    async def send(message):
        messages.append(json.loads(message))
    cmd = dict(id='pending-fixture', account='pa', type='entry_bracket', expires_at=10**15, payload={})
    async def exercise():
        for _ in range(2):
            await agent._handle_command(N(send=send), json.dumps(cmd), 'fixture')
    asyncio.run(exercise())
    assert len(wire_calls) == (0 if receipt_is_claimed else 1)
    assert messages[-1]['state'] == 'duplicate'


@pytest.mark.parametrize('terminal_state', ['executed', 'unknown'])
def test_patched_session_disconnect_preserves_durable_receipt_and_duplicate_guard(monkeypatch, terminal_state):
    agent = load_agent()
    claimed, wires, receipts, messages = set(), [], [], []
    released = None
    monkeypatch.setattr(agent, '_verify', lambda *_: True)
    def claim(cid, *_):
        if cid in claimed:
            return False
        claimed.add(cid)
        return True
    monkeypatch.setattr(agent, '_claim_seen', claim)
    monkeypatch.setattr(agent, '_record_seen', lambda *args: receipts.append(args))
    monkeypatch.setattr(agent, '_validate', lambda *_: (True, []))
    monkeypatch.setattr(agent, '_preview', lambda *_: {})
    monkeypatch.setattr(agent, '_describe', lambda *_: 'fixture')
    monkeypatch.setattr(agent, '_live_eligible', lambda *_: (True, 'fixture'))
    monkeypatch.setattr(agent, 'log', lambda _: None, raising=False)
    async def execute(cmd):
        wires.append(cmd['id'])
        await released.wait()
        return dict(state=terminal_state, ok=terminal_state == 'executed', detail='fixture result')
    monkeypatch.setattr(agent, '_execute_live', execute)
    agent._SEEN.clear()
    cmd = dict(id='inflight-fixture', type='entry_bracket', account='primary', expires_at=10**15, payload={})
    class Socket:
        def __init__(self):
            self.closed = asyncio.Event()
            self.delivered = False
        async def __aenter__(self):
            return self
        async def __aexit__(self, *_):
            await self.close()
        def __aiter__(self):
            return self
        async def __anext__(self):
            if not self.delivered:
                self.delivered = True
                return json.dumps(dict(type='command', signed=json.dumps(cmd), sig='fixture'))
            await self.closed.wait()
            raise StopAsyncIteration
        async def send(self, raw):
            message = json.loads(raw)
            if message['type'] == 'heartbeat':
                raise OSError('fixture heartbeat loss')
            if self.closed.is_set():
                raise OSError('fixture result transport closed')
            messages.append(message)
        async def close(self):
            self.closed.set()
            released.set()
    async def idle(*_):
        await asyncio.sleep(999)
    async def book():
        return None
    monkeypatch.setitem(__import__('sys').modules, 'execution_connection', connection)
    monkeypatch.setitem(__import__('sys').modules, 'position_action_agent', N(loop=idle))
    env = dict(asyncio=asyncio, json=json, time=time, os=N(getpid=lambda: 1), WS_URL='fixture', TOKEN='fixture',
               HEARTBEAT_S=.001, BOOK_REFRESH_S=20, LIVE_ENABLED=False, _BOOK={},
               in_window=lambda: True, log=lambda _: None, _fetch_book=book,
               _scheduled_option_loop=idle, _handle_command=agent._handle_command)
    exec(compile(patch(FIXTURE.read_text()), '<inert-transport>', 'exec'), env)
    async def exercise():
        nonlocal released
        for number in (1, 2):
            released = asyncio.Event()
            env['_connect'] = lambda *_: Socket()
            with pytest.raises(OSError, match='fixture heartbeat loss'):
                await env['_run_once'](connection.Attempt(number))
            # Every owned transport/maintenance task was reaped.
            assert asyncio.all_tasks() == {asyncio.current_task()}
    asyncio.run(exercise())
    assert wires == ['inflight-fixture']
    assert receipts == [('inflight-fixture', 'entry_bracket', terminal_state)]
    assert any(message.get('state') == 'duplicate' for message in messages)
