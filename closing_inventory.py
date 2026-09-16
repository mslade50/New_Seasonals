"""Verified 16:05 ET OLV inventory for the next cash-session morning.

This is a dated closing observation, not a claim of live broker connectivity.
Original reviewed attribution and execution coverage travel with the snapshot.
"""
import hashlib
import json
import math
import pandas as pd
from actual_inventory_io import load_reviewed_seed, load_primary_nav, load_pending_entry_notionals
from equity_sessions import calendar, last_settled_session, session_close
from tagged_inventory import TaggedInventory, build_tagged_inventory

STRATEGY = 'Oversold Low Volume'


def seed_digest(seed):
    return hashlib.sha256(json.dumps(seed, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def snapshot_key(session):
    return f'ops/olv_closing_inventory/{pd.Timestamp(session).date()}.json'


def make_snapshot(inventory, *, now):
    now = pd.Timestamp(now)
    if now.tzinfo is None:
        raise ValueError('closing capture requires timezone-aware time')
    day = now.tz_convert('America/New_York').date()
    close = session_close(day)
    if now < close + pd.Timedelta(minutes=5):
        raise ValueError('closing capture must follow the completed cash session by five minutes')
    if inventory.status != 'known' or not inventory.source_evidence:
        raise ValueError('closing inventory has not been verified')
    observed = pd.Timestamp(inventory.asof_utc)
    if observed < close + pd.Timedelta(minutes=5) or not pd.Timedelta(0) <= now-observed <= pd.Timedelta(seconds=90):
        raise ValueError('closing inventory observation is not current and post-close')
    load_primary_nav(inventory, asof=now)
    load_pending_entry_notionals(inventory, asof=now)
    return dict(schema_version=1, session=str(day), observed_at=inventory.asof_utc,
                seed_sha256=seed_digest(inventory.source_evidence['seed']),
                evidence=inventory.source_evidence)


def read_snapshot(snapshot, *, now, reviewed_seed):
    now = pd.Timestamp(now)
    if now.tzinfo is None:
        raise ValueError('closing inventory read requires timezone-aware time')
    session = last_settled_session(now)
    next_session = calendar().next_session(session)
    if now >= calendar().session_open(next_session):
        raise ValueError('closing inventory is only valid before the next cash-session open')
    if snapshot.get('schema_version') != 1 or snapshot.get('session') != str(session.date()):
        raise ValueError('required prior-session closing inventory is missing or stale')
    evidence = snapshot['evidence']
    if snapshot['seed_sha256'] != seed_digest(evidence['seed']) or snapshot['seed_sha256'] != seed_digest(reviewed_seed):
        raise ValueError('closing inventory predates the current reviewed attribution')
    observed = pd.Timestamp(snapshot['observed_at'])
    if (observed.tzinfo is None or observed > now
            or observed < session_close(session) + pd.Timedelta(minutes=5)
            or observed.tz_convert('America/New_York').date() != session.date()):
        raise ValueError('closing observation timestamp is invalid')
    result = build_tagged_inventory(evidence['seed'], evidence['fills'], evidence['coverage'],
                                   asof=observed.isoformat(), algo_strategies={STRATEGY},
                                   entry_metadata=evidence['entry_metadata'])
    if result.status != 'known':
        raise ValueError('saved closing inventory no longer verifies')
    result.observed_book = evidence['book']
    load_primary_nav(result, asof=observed)
    load_pending_entry_notionals(result, asof=observed)
    primary = next(a for a in evidence['book']['accounts'] if a['key']=='primary')
    actual = {}
    for row in primary['positions']:
        contract = int(row['con_id'])
        quantity = float(row['position'])
        if row['account'] != result.broker_account or contract in actual or not math.isfinite(quantity):
            raise ValueError('closing broker position identity or quantity is invalid')
        actual[contract] = quantity
    owned = {}
    for tranche in result.tranches:
        contract = tranche['con_id']
        owned[contract] = owned.get(contract, 0) + tranche['signed_qty']
    for contract, quantity in owned.items():
        net = actual.get(contract, 0)
        if quantity*net <= 0 or abs(quantity) > abs(net):
            raise ValueError('closing algorithm holdings do not reconcile to broker positions')
    result.source_kind = 'prior_close'
    return result


def load_closing_inventory(*, asof=None, algo_strategies=None, reader=None):
    try:
        if set(algo_strategies or ()) != {STRATEGY}:
            raise ValueError('closing inventory covers OLV only')
        now = pd.Timestamp(asof or pd.Timestamp.now(tz='UTC'))
        key = snapshot_key(last_settled_session(now))
        if reader is None:
            from cache_io import _client, _r2_creds
            client, creds = _client(), _r2_creds()
            if client is None or creds is None:
                raise ValueError('closing inventory storage is unavailable')
            reader = lambda key: json.loads(client.get_object(Bucket=creds['R2_BUCKET'], Key=key)['Body'].read())
        return read_snapshot(reader(key), now=now, reviewed_seed=load_reviewed_seed())
    except Exception as exc:
        return TaggedInventory(reasons=[f'prior-close OLV inventory could not be verified ({type(exc).__name__})'])
