"""Pure agent proposal contract and durable whole-idea execution claims.

No broker, network, environment or filesystem action occurs at import.
The installed adapter is disabled independently of all legacy execution gates.
"""
from __future__ import annotations

import datetime as dt
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
from zoneinfo import ZoneInfo

ET = ZoneInfo('America/New_York')
UUID = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.I)
EMPTY = (None, '')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def frozen(value):
    text = canonical(value)
    return {'payload': json.loads(text), 'canonical': text,
            'hash': hashlib.sha256(text.encode()).hexdigest()}


def verify(value):
    if not isinstance(value, dict) or not isinstance(value.get('canonical'), str):
        raise ValueError('immutable envelope required')
    if json.loads(value['canonical']) != value.get('payload'):
        raise ValueError('immutable payload changed')
    if hashlib.sha256(value['canonical'].encode()).hexdigest() != value.get('hash'):
        raise ValueError('immutable hash changed')
    return json.loads(value['canonical'])


def instant(value):
    parsed = dt.datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        raise ValueError('timezone-aware timestamp required')
    return parsed


def number(value, label, *, positive=True):
    if isinstance(value, bool):
        raise ValueError(f'{label} must be numeric')
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{label} must be numeric') from exc
    if not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f'{label} must be finite and positive')
    return value


def quantity(value):
    q = number(value, 'quantity')
    if q != int(q):
        raise ValueError('quantity must be a positive whole number')
    return int(q)


def truthy(value):
    return value is True or str(value).strip().lower() in {'1', 'true', 'yes', 'y'}


def validate_request(command, accounts, now):
    if command.get('type') != 'review_execution' or not UUID.fullmatch(str(command.get('id', ''))):
        raise ValueError('review execution type and UUID required')
    if type(command.get('dry_run')) is not bool:
        raise ValueError('explicit dry_run boolean required')
    if not number(command.get('expires_at'), 'command expiry') > now.timestamp() * 1000:
        raise ValueError('command expired')
    p = command.get('payload') or {}
    op = p.get('operation')
    if op not in {'preview', 'execute', 'reconcile'}:
        raise ValueError('explicit preview/execute/reconcile operation required')
    account = command.get('account')
    product = p.get('product')
    bindings = accounts.get(product)
    if product not in {'pitch', 'seasonal'} or not isinstance(bindings, (list, tuple)) or not bindings:
        raise ValueError('product execution accounts are unassigned')
    if account not in {'primary', 'pa'} or account not in bindings:
        raise ValueError('account does not match configured product bindings')
    if not isinstance(p.get('actor'), str) or not p['actor'].strip():
        raise ValueError('verified execution actor required')
    if op == 'reconcile':
        if command['dry_run'] is not True or not p.get('run_key'):
            raise ValueError('reconciliation must be explicitly read-only')
        return p
    proposal = verify(p.get('proposal'))
    if proposal.get('schema') != 1:
        raise ValueError('unsupported proposal schema')
    expected = f"{product}:{proposal.get('source_idea_id')}:{p['proposal']['hash'][:16]}"
    if p['proposal'].get('id') != expected or proposal.get('product') != product:
        raise ValueError('proposal identity/product changed')
    binding = (proposal.get('account_proposals') or {}).get(account)
    if not isinstance(binding, dict) or binding.get('account') != account:
        raise ValueError('explicit account proposal binding unavailable')
    if binding.get('status') != 'requires_fresh_account_preview':
        raise ValueError(binding.get('reason') or 'account proposal sizing blocked')
    review = p.get('review') or {}
    if (review.get('decision') != 'approve_review' or review.get('scope') != 'human_review_only'
            or review.get('execution') != 'not_submitted' or not review.get('actor')
            or review.get('proposal_id') != expected
            or review.get('proposal_hash') != p['proposal']['hash'] or not review.get('id')):
        raise ValueError('matching explicit approved review required')
    if not p.get('delivery_id') or p.get('current_at_authorization') is not True:
        raise ValueError('confirmed current delivery required')
    if proposal.get('source_date') != now.astimezone(ET).date().isoformat():
        raise ValueError('proposal is not for the current ET session')
    if instant(proposal.get('review_deadline')) <= now:
        raise ValueError('proposal expired')
    if instant(review.get('at')) >= instant(proposal['review_deadline']):
        raise ValueError('review was recorded after expiry')
    if not instant(proposal.get('published_at')) <= instant(review.get('at')) <= now:
        raise ValueError('review publication/decision clock is invalid')
    rows = proposal.get('orders')
    if not isinstance(rows, list) or not 1 <= len(rows) <= 4:
        raise ValueError('whole idea must contain one to four legs')
    ids = [quantity(row.get('Leg')) for row in rows]
    if len(set(ids)) != len(ids) or any(row.get('Idea_Id') != proposal['source_idea_id'] for row in rows):
        raise ValueError('leg identity is incomplete or duplicated')
    if op == 'execute':
        if p.get('confirmed') is not True or not p.get('plan_hash'):
            raise ValueError('separate explicit execution confirmation and plan hash required')
        if len(rows) > 1 and p.get('non_atomic_ack') is not True:
            raise ValueError('multi-leg submission is non-atomic; explicit acknowledgement required')
    return p


def native_payload(row, product, source_id, today, session_open=None):
    """Map the original grammar into the established entry_bracket vocabulary."""
    if truthy(row.get('Manual_Only')) or row.get('Proxy_Ticker'):
        raise ValueError('publisher marked leg manual/proxy; publish an exact supported instruction')
    if any(row.get(k) not in EMPTY and number(row[k], k, positive=False) != 0
           for k in ('Trail_Arm_ATR', 'Trail_ATR')):
        raise ValueError('conditional MFE trail is not expressible by entry_bracket; manual execution required')
    if row.get('Sec_Type') != 'STK' or row.get('Contract'):
        raise ValueError('unmapped derivative/contract; no proxy, front-month or symbol substitution')
    if row.get('Currency', 'USD') != 'USD' or row.get('Exchange', 'SMART') != 'SMART':
        raise ValueError('only explicit US stock/ETF SMART USD instructions are supported')
    symbol = str(row.get('Ticker') or '')
    if not re.fullmatch(r'[A-Z][A-Z0-9.\-]{0,14}', symbol):
        raise ValueError('invalid exact stock symbol')
    if row.get('Execute_On') != today:
        raise ValueError('leg is not for this execution session')
    if row.get('Action') not in {'BUY', 'SELL_SHORT'}:
        raise ValueError('unsupported source action')
    exit_date = str(row.get('Time_Exit_Date') or '')
    if dt.date.fromisoformat(exit_date) <= dt.date.fromisoformat(today):
        raise ValueError('explicit future time exit required')
    exit_kind = row.get('Time_Exit_Order')
    if exit_kind not in {'MOO', 'MOC'}:
        raise ValueError('unsupported time exit instruction')
    kind = row.get('Entry_Type')
    stop = None if row.get('Stop_Price') in EMPTY else number(row['Stop_Price'], 'stop')
    target = None if row.get('Target_Price') in EMPTY else number(row['Target_Price'], 'target')
    expiry = None
    if kind == 'LIMIT':
        if row.get('Order_Type') != 'LMT' or row.get('TIF') not in {'DAY', 'GTD'}:
            raise ValueError('limit order type/TIF mismatch')
        anchor = row.get('Entry_Anchor')
        if anchor == 'CLOSE':
            entry = number(row.get('Limit_Price'), 'published limit')
            for atr_key, price in [('Stop_ATR', stop), ('Target_ATR', target)]:
                if row.get(atr_key) not in EMPTY and number(row[atr_key], atr_key, positive=False) != 0 and price is None:
                    raise ValueError(f'published {atr_key} has no exact price')
        elif anchor == 'OPEN':
            opening = number(session_open, 'broker-verified current session open')
            atr = number(row.get('ATR'), 'published ATR')
            offset = number(row.get('Entry_Offset_ATR'), 'published open offset', positive=False)
            direction = 1 if row['Action'] == 'BUY' else -1
            entry = round(opening + offset * atr, 2)  # exact pitch_moo.price_open_row rule
            stop = (round(entry - direction * number(row['Stop_ATR'], 'Stop_ATR') * atr, 2)
                    if row.get('Stop_ATR') not in EMPTY and float(row['Stop_ATR']) != 0 else None)
            target = (round(entry + direction * number(row['Target_ATR'], 'Target_ATR') * atr, 2)
                      if row.get('Target_ATR') not in EMPTY and float(row['Target_ATR']) != 0 else None)
        else:
            raise ValueError('explicit CLOSE/OPEN anchor required')
        if row['TIF'] == 'GTD':
            expiry = str(row.get('Entry_Expire_Date') or '')
            if not today <= expiry < exit_date:
                raise ValueError('explicit GTD expiry must precede time exit')
            dt.date.fromisoformat(expiry)
        native_kind = 'LMT'
    elif kind in {'MOO', 'MOC'}:
        if row.get('Order_Type') != 'MKT' or row.get('TIF') != ('OPG' if kind == 'MOO' else 'MOC'):
            raise ValueError('auction order type/TIF mismatch')
        if stop is not None or target is not None or any(row.get(k) not in EMPTY and float(row[k]) != 0 for k in ['Stop_ATR', 'Target_ATR']):
            raise ValueError('unpriced auction with price exits cannot be substituted')
        entry = number(row.get('Ref_Close'), 'auction risk reference')
        native_kind = kind
    else:
        raise ValueError('unsupported source entry order type')
    action = 'BUY' if row['Action'] == 'BUY' else 'SELL'
    if entry <= 0 or (stop is not None and (stop >= entry if action == 'BUY' else stop <= entry)):
        raise ValueError('stop/entry ordering is invalid')
    if target is not None and (target <= entry if action == 'BUY' else target >= entry):
        raise ValueError('target/entry ordering is invalid')
    return {'symbol': symbol, 'sec_type': 'STK', 'exchange': 'SMART', 'currency': 'USD',
            'action': action, 'source_action': row['Action'], 'quantity': quantity(row.get('Quantity')), 'entry_type': native_kind,
            'entry': entry, 'stop': stop, 'target': target, 'expiry': expiry,
            'time_stop': exit_date, 'time_stop_at': 'open' if exit_kind == 'MOO' else 'close',
            'stop_arm': 'next_session' if stop is not None else 'fill',
            'strategy': ('Pitch-' if product == 'pitch' else 'Seasonal_Agent-') + source_id,
            'ref_date': today, 'risk_ack': False}


def run_key(product, source_id, account, proposal_hash=None):
    # The old three-field key remains usable only to reconcile older receipts.
    values = [product, source_id, proposal_hash, account] if proposal_hash else [product, source_id, account]
    return hashlib.sha256(canonical(values).encode()).hexdigest()


class Journal:
    """Permanent SQLite claims; a missing/corrupt journal never becomes empty."""
    def __init__(self, path):
        self.path = Path(path)

    @staticmethod
    def initialize(path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb'):
            pass
        with sqlite3.connect(path) as db:
            db.execute('CREATE TABLE meta (schema_version INTEGER NOT NULL)')
            db.execute('INSERT INTO meta VALUES (1)')
            db.execute('CREATE TABLE previews (hash TEXT PRIMARY KEY, payload TEXT NOT NULL)')
            db.execute('CREATE TABLE runs (key TEXT PRIMARY KEY, command_id TEXT UNIQUE NOT NULL, payload TEXT NOT NULL)')
        return Journal(path)

    @contextlib.contextmanager
    def connect(self):
        if not self.path.is_file():
            raise ValueError('review execution journal missing; explicit reviewed initialization required')
        db = sqlite3.connect(self.path.resolve().as_uri() + '?mode=rw', uri=True, timeout=5)
        try:
            db.execute('PRAGMA synchronous=FULL')
            if db.execute('SELECT schema_version FROM meta').fetchall() != [(1,)]:
                raise ValueError('review execution journal schema unavailable')
            with db:
                yield db
        finally:
            db.close()

    @contextlib.contextmanager
    def operation(self):
        """Serialize full previews/execution/reconciliation across processes."""
        if not self.path.is_file():
            raise ValueError('review execution journal missing')
        handle = self.path.with_suffix(self.path.suffix + '.operation.lock').open('a+b')
        try:
            if handle.seek(0, 2) == 0:
                handle.write(b'0'); handle.flush()
            handle.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield
        finally:
            handle.close()

    def save_preview(self, plan):
        verify(plan)
        with self.connect() as db:
            db.execute('INSERT OR IGNORE INTO previews VALUES (?,?)', (plan['hash'], canonical(plan)))

    def preview(self, digest):
        with self.connect() as db:
            row = db.execute('SELECT payload FROM previews WHERE hash=?', (digest,)).fetchone()
        if not row:
            raise ValueError('verified local preview unavailable')
        plan = json.loads(row[0]); verify(plan)
        return plan

    def get(self, key):
        with self.connect() as db:
            row = db.execute('SELECT payload FROM runs WHERE key=?', (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def allocated_risk(self, product, account, source_date):
        with self.connect() as db:
            records = [json.loads(row[0]) for row in db.execute('SELECT payload FROM runs')]
        total = 0
        for record in records:
            p = verify(record['plan'])
            if (p['product'], p['account'], p.get('source_date', p['source_idea_id'][:10])) == (product, account, source_date):
                # Count every submitted/uncertain/closed idea against staged-day
                # risk. Only a proved zero-fill whole-chain cancellation releases it.
                if record['state'] != 'cancelled':
                    total += max(p['risk_usd'], p.get('sizing', {}).get('sizing_risk_usd', 0))
        return total

    def claim(self, key, command_id, plan):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT payload FROM runs WHERE key=? OR command_id=?', (key, command_id)).fetchone()
            if row:
                old = json.loads(row[0])
                if old['key'] != key or old['plan']['hash'] != plan['hash'] or old['command_id'] != command_id:
                    raise ValueError('permanent idea/account execution claim exists; reconcile, never re-enter')
                return old, False
            proposed = verify(plan)
            if key != run_key(proposed['product'], proposed['source_idea_id'], proposed['account'], proposed['proposal_hash']):
                raise ValueError('version/account execution key mismatch')
            # Keep the permanent source/account guard as well as version-scoped
            # records. Prior schema-1 receipts are preserved and cannot re-enter.
            for stored, in db.execute('SELECT payload FROM runs'):
                existing = verify(json.loads(stored)['plan'])
                if (existing['product'], existing['source_idea_id'], existing['account']) == (proposed['product'], proposed['source_idea_id'], proposed['account']):
                    raise ValueError('permanent source/account execution claim exists; reconcile, never re-enter')
            value = {'key': key, 'command_id': command_id, 'plan': plan, 'state': 'claimed',
                     'legs': [{'state': 'not_sent'} for _ in plan['payload']['legs']]}
            db.execute('INSERT INTO runs VALUES (?,?,?)', (key, command_id, canonical(value)))
            return value, True

    def update(self, record):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            result = db.execute('UPDATE runs SET payload=? WHERE key=? AND command_id=?',
                                (canonical(record), record['key'], record['command_id']))
            if result.rowcount != 1:
                raise ValueError('durable execution claim changed')


def summarize(record):
    states = [leg['state'] for leg in record['legs']]
    if any(s in {'submitting', 'unknown', 'rejected', 'unprotected', 'cancelled_partial'} for s in states):
        return 'needs_reconciliation'
    if any(s == 'not_sent' for s in states):
        return 'needs_reconciliation'
    if all(s == 'filled' for s in states):
        return 'filled'
    if all(s == 'closed' for s in states):
        return 'closed'
    if any(s in {'closed','partially_closed'} for s in states):
        return 'partially_closed'
    if any(s in {'filled', 'partially_filled'} for s in states):
        return 'partially_filled'
    if all(s == 'cancelled' for s in states):
        return 'cancelled'
    return 'working'
