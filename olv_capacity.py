"""Sizing-only OLV capacity from one broker observation; never exit inventory."""
from dataclasses import dataclass, field
import json
import math
import os
import pandas as pd
from equity_sessions import calendar, last_settled_session, session_close
from olv_sizing import STRATEGY, pending_entry_notionals


@dataclass
class Capacity:
    known: bool = False
    nav: float | None = None
    held: dict = field(default_factory=dict)
    pending: dict = field(default_factory=dict)
    source: str = 'unavailable'
    observed_at: str | None = None
    reason: str = ''


def stamp(value):
    result = pd.Timestamp(value)
    if pd.isna(result) or result.tzinfo is None:
        raise ValueError('capacity timestamp is missing or timezone-naive')
    return result


def symbol_key(value):
    """Broker share classes use spaces/dots; scanner identifiers use dashes."""
    return str(value or '').strip().upper().replace('.', '-').replace(' ', '-')


def utc_now():
    return pd.Timestamp.now(tz='UTC')


def from_book(book, *, now, source='broker_conservative'):
    """Validate NAV, whole held stock exposure and pending parents together.

    Unattributed stock holdings also reserve capacity. Absence of an order ref
    is not evidence that a position is outside OLV; this may tighten the cap.
    The broker account is resolved by the existing Primary-only collector.
    """
    now = stamp(now)
    accounts = [a for a in (book or {}).get('accounts', []) if a.get('key') == 'primary']
    if len(accounts) != 1:
        raise ValueError('capacity requires exactly one Primary account')
    primary = accounts[0]
    account = primary.get('broker_account')
    if not account or primary.get('error'):
        raise ValueError('Primary broker observation failed')
    observed = pd.Timestamp(round(float(primary['orders_source_at'])*1000), unit='ms', tz='UTC')
    if pd.isna(observed) or not pd.Timedelta(0) <= now-observed <= pd.Timedelta(seconds=90):
        raise ValueError('Primary capacity observation is stale or future-dated')
    nav = float(primary['nlv'])
    if not math.isfinite(nav) or nav <= 0:
        raise ValueError('Primary NAV must be finite and positive')
    if not all(isinstance(primary.get(k), list) for k in ('positions', 'orders')):
        raise ValueError('Primary positions or orders are incomplete')
    started = stamp(book.get('capacity_query_started_at'))
    completed = stamp(book.get('capacity_query_completed_at'))
    execution_at = pd.Timestamp(float(primary['fills_source_at']), unit='ms', tz='UTC')
    query_from = stamp(primary.get('fills_query_from'))
    if (primary.get('fills_complete') is not True or primary.get('fills_error')
            or not isinstance(primary.get('fills'), list) or pd.isna(execution_at)
            or not query_from <= started <= observed <= execution_at <= completed <= now
            or started.tz_convert('America/New_York').date() != completed.tz_convert('America/New_York').date()):
        raise ValueError('current-query execution proof is incomplete')
    contracts, symbols = set(), {}
    stock_values = {}
    for row in primary['positions']:
        if row.get('account') != account:
            raise ValueError('position belongs to a different account')
        contract, qty = float(row['con_id']), float(row['position'])
        if (not math.isfinite(contract) or contract <= 0 or contract != int(contract)
                or contract in contracts or not math.isfinite(qty)):
            raise ValueError('position contract or quantity is invalid')
        contracts.add(contract)
        if not row.get('sec_type'):
            raise ValueError('position security type is missing')
        if row.get('sec_type') != 'STK' or not qty:
            continue
        symbol = symbol_key(row.get('symbol'))
        if not symbol or row.get('currency') != 'USD' or qty != int(qty):
            raise ValueError('stock position identity is unsupported')
        if symbol in symbols and symbols[symbol] != contract:
            raise ValueError('stock symbol has multiple contracts')
        symbols[symbol] = contract
        value = float(row['market_value'])
        if not math.isfinite(value) or value == 0 or value*qty <= 0:
            raise ValueError('held stock market value is unavailable or invalid')
        stock_values[(symbol, STRATEGY)] = abs(value)
    # On an OLV signal, reserve all held stock exposure in that same symbol.
    # A different sleeve's tag does not prove that no OLV shares coexist.
    held = stock_values
    raw_pending = pending_entry_notionals(book, account, asof=now.isoformat())
    pending = {}
    for (symbol, strategy), value in raw_pending.items():
        key = (symbol_key(symbol), strategy)
        pending[key] = pending.get(key, 0) + value
    # Orders are copied after positions. A BUY during collection can disappear
    # from remaining orders before appearing in copied holdings. Reserve these
    # buys in addition; an already-reflected fill can only tighten capacity.
    seen = set()
    for row in primary['fills']:
        at = stamp(row.get('time'))
        if not started <= at <= execution_at:
            continue
        identity = row.get('exec_id')
        if not identity or identity in seen or row.get('account') != account:
            raise ValueError('in-collection fill identity is ambiguous')
        seen.add(identity)
        if not row.get('sec_type'):
            raise ValueError('in-collection fill security type is missing')
        if row['sec_type'] != 'STK':
            continue
        symbol = symbol_key(row.get('symbol'))
        contract, qty, price = float(row['con_id']), float(row['qty']), float(row['price'])
        if (not symbol or row.get('currency') != 'USD' or row.get('side') not in {'BOT','SLD'}
                or not all(math.isfinite(v) and v > 0 for v in (contract,qty,price))
                or contract != int(contract) or qty != int(qty)
                or (symbol in symbols and symbols[symbol] != contract)):
            raise ValueError('in-collection stock fill is invalid')
        symbols[symbol] = contract
        if row['side'] == 'BOT':
            key = (symbol, STRATEGY)
            held[key] = held.get(key,0) + qty*price
    for row in primary['orders']:
        if (row.get('symbol'),STRATEGY) in raw_pending:
            symbol = symbol_key(row['symbol'])
            if symbol in symbols and symbols[symbol] != float(row['con_id']):
                raise ValueError('held and pending stock contracts disagree')
            symbols[symbol] = float(row['con_id'])
    return Capacity(True, nav, held, pending, source, observed.isoformat())


def snapshot_key(session):
    return f'ops/olv_capacity/{pd.Timestamp(session).date()}.json'


def make_snapshot(book, *, now):
    now = stamp(now)
    capacity = from_book(book, now=now)
    session = now.tz_convert('America/New_York').date()
    if stamp(capacity.observed_at) < session_close(session)+pd.Timedelta(minutes=5):
        raise ValueError('capacity capture must follow cash close by five minutes')
    return dict(schema_version=1, purpose='olv_sizing_only', session=str(session),
                observed_at=capacity.observed_at, book=book)


def read_snapshot(snapshot, *, now):
    now = stamp(now)
    session = last_settled_session(now)
    if now >= calendar().session_open(calendar().next_session(session)):
        raise ValueError('closing capacity expires at the next cash open')
    if (snapshot.get('schema_version') != 1 or snapshot.get('purpose') != 'olv_sizing_only'
            or snapshot.get('session') != str(session.date())):
        raise ValueError('required closing capacity session is missing')
    observed = stamp(snapshot['observed_at'])
    if (observed > now or observed < session_close(session)+pd.Timedelta(minutes=5)
            or observed.tz_convert('America/New_York').date() != session.date()):
        raise ValueError('closing capacity timestamp is invalid')
    collected = stamp(snapshot['book'].get('capacity_query_completed_at'))
    if collected > now or collected-observed > pd.Timedelta(seconds=90):
        raise ValueError('closing capacity collection is invalid')
    result = from_book(snapshot['book'], now=collected, source='prior_close_conservative')
    if stamp(result.observed_at) != observed:
        raise ValueError('closing capacity and book timestamps disagree')
    return result


def load_capacity(inventory, *, now=None, bookend=False, reader=None, book_loader=None):
    """Prefer reconciled inputs, then saved closing capacity, then a fresh book.

    Never promote the fallback to TaggedInventory.status=known. A failed
    fallback preserves the owner's explicit policy to scan without this cap.
    """
    fixed_now = now
    now = stamp(now if now is not None else utc_now())
    failures = []
    if inventory.status == 'known':
        try:
            from actual_inventory_io import load_primary_nav, load_pending_entry_notionals
            at = inventory.asof_utc if inventory.source_kind == 'prior_close' else now.isoformat()
            nav = load_primary_nav(inventory, asof=at)
            pending = load_pending_entry_notionals(inventory, asof=at)
            return Capacity(True, nav, dict(inventory.notionals), pending,
                            inventory.source_kind, inventory.asof_utc)
        except Exception as exc:
            failures.append('reconciled capacity: '+type(exc).__name__)
    if bookend:
        try:
            if reader is None:
                from cache_io import _client, _r2_creds
                client, creds = _client(), _r2_creds()
                if client is None or creds is None:
                    raise ValueError('capacity storage unavailable')
                reader = lambda key: json.loads(client.get_object(Bucket=creds['R2_BUCKET'], Key=key)['Body'].read())
            return read_snapshot(reader(snapshot_key(last_settled_session(now))), now=now)
        except Exception as exc:
            failures.append('closing capacity: '+type(exc).__name__)
    try:
        book = book_loader() if book_loader else query_local_book()
        checked_at = stamp(fixed_now if fixed_now is not None else utc_now())
        return from_book(book, now=checked_at)
    except Exception as exc:
        failures.append('live capacity: '+type(exc).__name__)
    return Capacity(reason='; '.join(failures))


def query_local_book():
    """Existing Primary-only read-only collector; no relay or order mutation."""
    import subprocess
    import sys
    from pathlib import Path
    snapshot = Path(os.environ.get('INVENTORY_SNAPSHOT_PATH', ''))
    if not snapshot.is_file():
        raise ValueError('local broker observation is not configured')
    query = Path(__file__).parent/'scripts/query_capacity_snapshot.py'
    try:
        result = subprocess.run([os.environ.get('INVENTORY_SNAPSHOT_PYTHON', sys._base_executable),
                                 str(query), str(snapshot)], cwd=str(snapshot.parent),
                                capture_output=True, text=True, encoding='utf-8', timeout=60, check=True)
        return json.loads(result.stdout)
    except (subprocess.SubprocessError, ValueError):
        raise ValueError('Primary read-only observation failed') from None
