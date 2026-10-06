"""Execution bridge connection health only; never submits or replays commands."""
from __future__ import annotations

import asyncio
import json
import random
import re
import time
import uuid
from dataclasses import dataclass

STABLE_SECONDS = 30
STABLE_ACKS = 2
ACK_WARN_SECONDS = 30
DIAGNOSTIC_SECONDS = 60


def secrets_from(ns):
    return tuple(value for key, value in ns.items()
                 if re.search(r'TOKEN|SECRET|KEY', key) and isinstance(value, str) and len(value) >= 8)


def safe_text(value, secrets=()):
    text = str(value).replace('\r', ' ').replace('\n', ' ')
    for secret in sorted((s for s in secrets if isinstance(s, str) and s), key=len, reverse=True):
        text = text.replace(secret, '[redacted]')
    text = re.sub(r'(?i)(bearer\s+|authorization[\s:=]+)\S+', r'\1[redacted]', text)
    text = re.sub(r'(?i)(?:wss?|https?)://\S+', '[endpoint]', text)
    return text[:240]


def exception_fields(exc, secrets=()):
    fields = {'type': type(exc).__name__, 'detail': safe_text(exc, secrets)}
    for name in ('errno', 'winerror'):
        value = getattr(exc, name, None)
        if isinstance(value, int):
            fields[name] = value
    for name in ('rcvd', 'sent'):
        frame = getattr(exc, name, None)
        if frame is not None:
            fields[name] = {'code': getattr(frame, 'code', None), 'reason': safe_text(getattr(frame, 'reason', ''), secrets)}
    cause = exc.__cause__ or exc.__context__
    if cause is not None and cause is not exc:
        fields['cause'] = {'type': type(cause).__name__, 'detail': safe_text(cause, secrets)}
        for name in ('errno', 'winerror'):
            value = getattr(cause, name, None)
            if isinstance(value, int):
                fields['cause'][name] = value
    return fields


@dataclass
class Attempt:
    number: int
    stable: bool = False


class Session:
    def __init__(self, attempt, log, *, clock=time.monotonic, secrets=()):
        self.attempt = attempt
        self.log = log
        self.clock = clock
        self.secrets = secrets
        self.id = uuid.uuid4().hex
        self.started = self.last_report = clock()
        self.last_ack = None
        self.acks = self.sequence = 0
        self.pending = {}
        self.last_rtt = None
        self.max_loop_lag = 0
        self.busy = False
        self.ack_mode = None
        self.emit('connection_open')

    def emit(self, event, **fields):
        self.log('connection ' + json.dumps(dict(event=event, session=self.id, attempt=self.attempt.number, **fields), sort_keys=True))

    def heartbeat(self):
        self.sequence += 1
        now = self.clock()
        self.pending[self.sequence] = now
        self.pending = {seq: sent for seq, sent in self.pending.items() if now-sent <= 120}
        return {'type': 'heartbeat', 't': time.time(), 'session': self.id, 'seq': self.sequence}

    def observe(self, msg):
        if not isinstance(msg, dict) or msg.get('type') != 'ack' or msg.get('of') != 'heartbeat':
            return
        now = self.clock()
        correlated = 'seq' in msg or 'session' in msg
        if correlated:
            seq = msg.get('seq')
            if msg.get('session') != self.id or type(seq) is not int or seq not in self.pending:
                self.emit('ack_unmatched')
                return
            age = now-self.pending.pop(seq)
            if age > ACK_WARN_SECONDS:
                self.emit('ack_delayed', age_s=round(age, 3), consumer_busy=self.busy)
                return
            self.last_rtt = age
        self.ack_mode = 'correlated' if correlated else 'legacy'
        self.last_ack = now
        self.acks += 1
        if not self.attempt.stable and now-self.started >= STABLE_SECONDS and self.acks >= STABLE_ACKS:
            self.attempt.stable = True
            self.emit('connection_stable', uptime_s=round(now-self.started, 3), ack_mode=self.ack_mode)

    def loop_wake(self, due):
        self.max_loop_lag = max(self.max_loop_lag, max(0, self.clock()-due))

    def summary(self):
        now = self.clock()
        ack_age = None if self.last_ack is None else max(0, now-self.last_ack)
        return dict(uptime_s=round(now-self.started, 3), heartbeat_seq=self.sequence, heartbeat_acks=self.acks,
                    ack_mode=self.ack_mode, ack_age_s=None if ack_age is None else round(ack_age, 3),
                    ack_missing=(ack_age if ack_age is not None else now-self.started) >= ACK_WARN_SECONDS,
                    last_rtt_s=None if self.last_rtt is None else round(self.last_rtt, 3),
                    max_loop_lag_s=round(self.max_loop_lag, 3), consumer_busy=self.busy,
                    stable=self.attempt.stable)

    def tick(self):
        if self.clock()-self.last_report >= DIAGNOSTIC_SECONDS:
            self.emit('heartbeat_health', **self.summary())
            self.last_report = self.clock()
            self.max_loop_lag = 0

    def finish(self, ws=None):
        self.emit('connection_end', **self.summary(), close_code=getattr(ws, 'close_code', None),
                  close_reason=safe_text(getattr(ws, 'close_reason', '') or '', self.secrets))


async def supervise(receiver, heartbeat, close):
    """Surface a dead heartbeat task; leave command handling in its serial receiver."""
    done, _ = await asyncio.wait({receiver, heartbeat}, return_when=asyncio.FIRST_COMPLETED)
    if heartbeat in done:
        try:
            await heartbeat
        except asyncio.CancelledError:
            raise
        except Exception as heartbeat_error:
            # Close transport, then let an already-running serial handler finish
            # its bounded subprocess and durable receipt. Never cancel/replay it
            # merely because the heartbeat send failed.
            try:
                await close()
                await receiver
            except asyncio.CancelledError:
                raise
            except Exception as receive_error:
                raise heartbeat_error from receive_error
            raise
        await receiver  # normal heartbeat completion means run-window shutdown
    else:
        await receiver


async def run(ns, run_once, *, clock=time.monotonic, sleep=asyncio.sleep, jitter=random.uniform):
    if not ns.get('WS_URL') or not ns.get('TOKEN'):
        raise SystemExit('Set EXEC_BROKER_WS and EXEC_AGENT_TOKEN env vars first.')
    ns['_load_seen']()
    ns['_load_schedules']()
    backoff = 1
    number = 0
    secrets = secrets_from(ns)
    while ns['in_window']():
        number += 1
        state = Attempt(number)
        started = clock()
        error = {'type': 'CleanClose'}
        try:
            await run_once(state)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            error = exception_fields(exc, secrets)
        if not ns['in_window']():
            return
        if state.stable:
            backoff = 1
        cap = max(1, ns.get('BACKOFF_MAX_S', 30))
        delay = max(.8, min(cap, backoff*jitter(.8, 1.2)))
        ns['log']('connection ' + json.dumps(dict(event='reconnect', attempt=number, stable=state.stable,
                   attempt_duration_s=round(clock()-started, 3), backoff_s=backoff,
                   retry_s=round(delay, 3), error=error), sort_keys=True))
        await sleep(delay)
        backoff = min(cap, backoff*2)
