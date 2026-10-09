"""Adapters into the EXISTING agent mutex, executor, guarded placement and capture.

Installed only by a separate user-controlled runtime handoff. No import-time I/O.
Preview/reconciliation use a read-only connection; execution needs every old and
new arming gate. Approved open-derived limits wait in the persistent journal.
No automatic retry of uncertain submissions, cancellation or position edit.
"""
from __future__ import annotations

import copy
import asyncio
import contextlib
import datetime as dt
import io
import json
import os
from pathlib import Path
import time

try:
    from . import review_execution as contract
    from . import review_sizing as sizing
except ImportError:
    import review_execution as contract
    import review_sizing as sizing


def configuration():
    return {'preview_enabled': os.environ.get('REVIEW_EXECUTION_PREVIEW_ENABLED', '0') == '1',
            'live_enabled': os.environ.get('REVIEW_EXECUTION_LIVE_ENABLED', '0') == '1',
            'accounts': {'pitch': ['primary', 'pa'], 'seasonal': ['primary', 'pa']},
            'risk_multipliers': {'primary': 1.0, 'pa': sizing.agent_risk_multiplier(
                                 os.environ.get('REVIEW_EXECUTION_PA_RISK_MULTIPLIER'))},
            'db': os.environ.get('REVIEW_EXECUTION_DB'),
            'max_risk_bps': min(100.0, contract.number(os.environ.get('REVIEW_EXECUTION_MAX_RISK_BPS', '100'), 'idea risk cap'))}


def gate(g, command, cfg):
    if not cfg['preview_enabled']:
        raise ValueError('review execution preview adapter disabled')
    op = (command.get('payload') or {}).get('operation')
    if op != 'reconcile':
        sizing.policy(cfg, command['payload']['product'], command['account'])
    if op != 'reconcile' and (command.get('payload') or {}).get('product') == 'pitch':
        directory = g.get('_THIS_DIR') or g.get('_DIR')
        if not directory or (Path(directory)/'pitch_moo_enabled.flag').exists():
            raise ValueError('legacy Pitch runner state is unknown or armed; one execution path is required')
    if op in {'execute', 'stage'} and command.get('dry_run') is False:
        if not cfg['live_enabled']:
            raise ValueError('review execution live adapter disabled')
        if not g.get('LIVE_ENABLED') or command.get('account') not in g.get('LIVE_ACCOUNTS', set()):
            raise ValueError('existing agent/account live gate is not armed')
        if not {'review_execution', 'entry_bracket'} <= g.get('LIVE_TYPES', set()):
            raise ValueError('existing review_execution AND entry_bracket live-type gates required')


async def handle_agent(g, command):
    """Called only AFTER the existing signature/command-expiry checks."""
    try:
        cfg = configuration()
        contract.validate_request(command, cfg['accounts'], _clock(g))
        gate(g, command, cfg)
        due = opening_stage_time(command)
        if due is not None and due > _clock(g):
            if not cfg['db']:
                raise ValueError('explicit REVIEW_EXECUTION_DB required')
            return contract.Journal(cfg['db']).schedule(command, due)
        # Existing shared mutex and bounded subprocess path, including unknown on
        # timeout/malformed output. Preview cannot reach a placement in executor.
        result = await g['_execute_live'](command)
        inner = (result.get('fill') or {}).get('review_execution')
        if not isinstance(inner, dict):
            return {'ok': False, 'state': 'unknown', 'detail': result.get('detail', 'No trustworthy review result; reconcile'),
                    'fill': result.get('fill')}
        return {'ok': result['ok'], 'state': inner['state'], 'detail': result['detail'],
                'preview': inner.get('plan'), 'fill': inner.get('record')}
    except (ValueError, TypeError, KeyError) as exc:
        return {'ok': False, 'state': 'rejected', 'detail': str(exc)}


def opening_stage_time(command):
    """Only open-anchored limits need a later price; auctions stage immediately."""
    p = command.get('payload') or {}
    if p.get('operation') != 'stage':
        return None
    source = contract.verify(p['proposal'])
    if not any(row.get('Entry_Type') == 'LIMIT' and row.get('Entry_Anchor') == 'OPEN'
               for row in source['orders']):
        return None
    return dt.datetime.combine(dt.date.fromisoformat(source['source_date']),
                               dt.time(9, 32), contract.ET)


async def process_due_staging(g, ws):
    cfg = configuration()
    if not cfg['db']:
        return
    journal = contract.Journal(cfg['db'])
    for command in journal.due_staging(_clock(g)):
        if not journal.start_scheduled(command['id']):
            continue
        # Reuse the original UUID and authorization. Expiry/live/account checks
        # run again; this never rolls a missed opening instruction into tomorrow.
        result = await handle_agent(g, command)
        journal.finish_scheduled(command['id'], result)
    for command_id, result in journal.unreported_staging():
        await ws.send(json.dumps({'type': 'result', 'id': command_id, **result}))
        journal.reported_staging(command_id)


async def staging_loop(g, ws):
    """Runs beside the existing heartbeat; pending open limits survive restart."""
    cfg = configuration()
    if cfg['db']:
        # A prior executor may have reached the broker. Never retry a processing
        # job after restart; publish uncertainty on its original UUID.
        contract.Journal(cfg['db']).interrupt_staging()
    while True:
        try:
            await process_due_staging(g, ws)
        except Exception as exc:
            g['log'](f'review staging queue: {type(exc).__name__}; check stored intent')
        await asyncio.sleep(10)


def preflight_context(*, payload, broker_account, contract_id, quantity, entry,
                      stop, target, risk, nlv, stop_gat, time_gat, parent_gtd):
    """Injected immediately before the existing executor's placement loop."""
    if not contract_id or not broker_account:
        raise ValueError('exact broker account/contract qualification required')
    risk = contract.number(risk, 'broker-verified risk estimate')
    nlv = contract.number(nlv, 'exact-account NetLiquidation')
    clean = {k: copy.deepcopy(v) for k, v in payload.items() if not k.startswith('_')}
    clean.update(quantity=quantity, entry=entry, stop=stop, target=target)
    return {'payload': clean, 'broker_account': broker_account, 'con_id': int(contract_id),
            'risk_usd': risk, 'nlv': nlv, 'risk_ack_required': stop is None and risk / nlv * 1e4 > 50,
            'timing': {'stop_good_after': stop_gat, 'time_good_after': time_gat, 'parent_good_till': parent_gtd}}


def _capture(g, ib, account, con_id):
    if callable(g.get('_review_capture')):  # dependency-injected fake brokers in tests
        return g['_review_capture'](ib, account, con_id)
    import broker_reconciliation
    return broker_reconciliation.capture(ib, account, con_id, open_reader=g['_fresh_open_trades'])


def _open_price(g, ib, row, now):
    if row.get('Entry_Type') != 'LIMIT' or row.get('Entry_Anchor') != 'OPEN':
        return None
    c = g['Stock'](row['Ticker'], 'SMART', 'USD')
    matches = ib.qualifyContracts(c)
    if len(matches) != 1 or not int(matches[0].conId or 0):
        raise ValueError('open reference contract is ambiguous')
    bars = ib.reqHistoricalData(matches[0], endDateTime='', durationStr='1 D',
                               barSizeSetting='1 min', whatToShow='TRADES', useRTH=True,
                               formatDate=2, keepUpToDate=False)
    if not bars or not isinstance(bars[0].date, dt.datetime) or bars[0].date.tzinfo is None:
        raise ValueError('current session opening bar unavailable or timezone ambiguous')
    bar_at = bars[0].date.astimezone(contract.ET)
    if bar_at.date() != now.astimezone(contract.ET).date() or (bar_at.hour, bar_at.minute) != (9, 30):
        raise ValueError('opening bar is stale or is not the true session open')
    return contract.number(bars[0].open, 'broker session open')


def _reference(native):
    return f"{native['symbol']}|{native['source_action']}|{native['strategy']}|{native['ref_date']}"


def _invoke_entry(g, ib, payload, account):
    # The existing _out PRINTS JSON and returns process exit code 0. Capture it
    # rather than emitting multiple JSON documents into the subprocess protocol.
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        result = g['_do_entry_bracket'](ib, payload, account)
    if isinstance(result, dict):  # fake broker fixture, same semantic contract
        return result
    if result != 0:
        raise ValueError('existing entry executor returned an invalid process result')
    result = json.loads(stream.getvalue())
    if not isinstance(result, dict) or type(result.get('ok')) is not bool:
        raise ValueError('existing entry executor returned invalid JSON')
    return result



def _clock(g):
    return g['_review_clock']() if callable(g.get('_review_clock')) else dt.datetime.now(dt.timezone.utc)


def _account_snapshot(g, ib, logical, account):
    if callable(g.get('_review_account_snapshot')):
        snapshot = g['_review_account_snapshot'](ib, logical, account)
    else:
        # ib_insync.reqAccountSummary() returns None, not result rows. Capture
        # only this new request ID and await its end; never read cached values.
        request_id = ib.client.getReqId()
        future = ib.wrapper.startReq(request_id)
        previous = ib.wrapper.accountSummary
        rows = []
        def receive(req_id, acct, tag, value, currency):
            previous(req_id, acct, tag, value, currency)
            if req_id == request_id:
                rows.append(type('FreshAccountValue', (), {
                    'account': acct, 'tag': tag, 'value': value, 'currency': currency})())
        ib.wrapper.accountSummary = receive
        try:
            ib.client.reqAccountSummary(request_id, 'All',
                'NetLiquidation,AvailableFunds,BuyingPower,ExcessLiquidity')
            ib._run(future)
        finally:
            ib.wrapper.accountSummary = previous
            ib.client.cancelAccountSummary(request_id)  # release query subscription
        values = {}
        for tag in ['NetLiquidation', 'AvailableFunds', 'BuyingPower', 'ExcessLiquidity']:
            exact = [r for r in rows if str(r.account) == account and str(r.tag) == tag and str(r.currency) == 'USD']
            if len(exact) != 1:
                raise ValueError(f'{logical}: missing/ambiguous fresh exact-account USD {tag}')
            values[tag] = contract.number(exact[0].value, tag, positive=False)
        snapshot = {'account': logical, 'broker_account': account, 'currency': 'USD',
                    'source': 'fresh_broker_account_summary', 'request_completed': True,
                    'observed_at': _clock(g).isoformat(), 'nlv': values['NetLiquidation'],
                    'available_funds': values['AvailableFunds'], 'buying_power': values['BuyingPower'],
                    'excess_liquidity': values['ExcessLiquidity']}
    snapshot = sizing.equity(snapshot, logical, account, _clock(g))
    if contract.number(snapshot.get('excess_liquidity'), 'ExcessLiquidity', positive=False) < 0:
        raise ValueError(f'{logical}: account excess liquidity is negative')
    return snapshot


def _contract_and_capacity(g, ib, native, account):
    c = g['Stock'](native['symbol'], 'SMART', 'USD')
    exact = ib.qualifyContracts(c)
    if (len(exact) != 1 or not int(exact[0].conId or 0) or exact[0].secType != 'STK'
            or exact[0].currency != 'USD' or str(getattr(exact[0], 'multiplier', '') or '1') not in {'1', '1.0'}):
        raise ValueError('account exact share contract/multiplier qualification failed')
    c = exact[0]
    # Broker what-if is a margin/permission inquiry; no guarded_place_order is
    # called. Tests supply an object with no connection or mutation methods.
    if callable(g.get('_review_capacity')):
        value = g['_review_capacity'](ib, c, native, account)
    else:
        order = (g['LimitOrder'](native['action'], native['quantity'], native['entry'])
                 if native['entry_type'] == 'LMT' else g['MarketOrder'](native['action'], native['quantity']))
        order.account = account
        order.whatIf = True
        state = ib.whatIfOrder(c, order)
        if state is None or str(getattr(state, 'warningText', '') or '').strip():
            raise ValueError('account instrument/permission/margin preview unavailable or rejected')
        value = {'broker_account': account, 'con_id': int(c.conId), 'quantity': native['quantity'],
                 'initial_margin_change': float(state.initMarginChange),
                 'maintenance_margin_change': float(state.maintMarginChange),
                 'observed_at': _clock(g).isoformat(), 'permission_verified': True}
    if (value.get('broker_account') != account or value.get('con_id') != int(c.conId)
            or value.get('quantity') != native['quantity'] or value.get('permission_verified') is not True):
        raise ValueError('account capacity inquiry identity/quantity/permission mismatch')
    changes = [contract.number(value.get(k), k, positive=False)
               for k in ['initial_margin_change', 'maintenance_margin_change']]
    if any(abs(v) >= 1e100 for v in changes):
        raise ValueError('broker margin evidence unavailable/sentinel')
    age = (_clock(g) - contract.instant(value.get('observed_at'))).total_seconds()
    if not 0 <= age <= sizing.MAX_EQUITY_AGE_SECONDS:
        raise ValueError('account instrument/capacity evidence stale or future dated')
    return int(c.conId), {**value, 'initial_margin_change': max(0, changes[0]),
                        'maintenance_margin_change': max(0, changes[1]), 'lot': 1, 'multiplier': 1}


def preflight(g, ib, command, cfg, now):
    p = contract.validate_request(command, cfg['accounts'], now)
    proposal = contract.verify(p['proposal'])
    account = g['_resolve_broker_account'](ib, command['account'])
    account_policy = sizing.policy(cfg, p['product'], command['account'])
    snapshot = _account_snapshot(g, ib, command['account'], account)
    sized, sizing_summary = sizing.size(proposal, snapshot, account_policy)
    legs = []
    con_ids = set()
    for original, sized_leg in zip(proposal['orders'], sized):
        row = sized_leg['row']
        native = contract.native_payload(row, p['product'], proposal['source_idea_id'],
                                         proposal['source_date'], _open_price(g, ib, row, now))
        exact_con_id, capacity = _contract_and_capacity(g, ib, native, account)
        check = dict(native, _broker_account=account, _command_id=command['id'], _review_preflight_only=True)
        result = _invoke_entry(g, ib, check, command['account'])
        context = (result.get('fill') or {}).get('review_preflight')
        if result.get('ok') is not True or not isinstance(context, dict):
            raise ValueError(result.get('detail') or 'existing executor rejected preflight')
        if context['broker_account'] != account or context['con_id'] != exact_con_id or context['con_id'] in con_ids:
            raise ValueError('ambiguous account or repeated exact contract in multi-leg idea')
        con_ids.add(context['con_id'])
        if not abs(context['nlv'] - snapshot['nlv']) < 1e-6:
            raise ValueError('native risk guard NLV disagrees with fresh exact-account evidence')
        evidence = _capture(g, ib, account, context['con_id'])
        if evidence['account'] != account or evidence['con_id'] != context['con_id']:
            raise ValueError('account inventory evidence identity mismatch')
        ref = _reference(native)
        aliases = {ref, ref.replace('|SELL_SHORT|', '|SELL|')}
        if any(r.get('ref') in aliases for r in evidence['orders'] + evidence['completed'] + evidence['executions']):
            raise ValueError('same source idea already exists in broker orders/executions; reconcile other execution path')
        if evidence['position'] != 0 or any(r.get('status') not in {'Filled', 'Cancelled', 'ApiCancelled'} for r in evidence['orders']):
            raise ValueError(f"{command['account']}: existing position/working order for this contract; no automatic add, netting or reversal")
        legs.append({'leg': row['Leg'], 'original': copy.deepcopy(original), 'ref': ref,
                     'sizing': sized_leg['sizing'], 'capacity': capacity, **context})
    nlv = min(contract.number(leg['nlv'], 'live NLV') for leg in legs)
    total_risk = sum(contract.number(leg['risk_usd'], 'leg risk') for leg in legs)
    total_notional = sum(leg['payload']['quantity'] * leg['payload']['entry'] for leg in legs)
    if total_risk / nlv * 1e4 > account_policy['max_idea_bps']:
        raise ValueError('whole-idea risk exceeds account/publisher cap')
    if total_notional > g['_max_notional'](command['account']):
        raise ValueError('whole-idea notional exceeds existing account cap')
    if total_notional > snapshot['buying_power']:
        raise ValueError(f"{command['account']}: whole-idea notional exceeds fresh account buying power")
    initial = sum(leg['capacity']['initial_margin_change'] for leg in legs)
    maintenance = sum(leg['capacity']['maintenance_margin_change'] for leg in legs)
    if initial > snapshot['available_funds'] or maintenance > snapshot['excess_liquidity']:
        raise ValueError(f"{command['account']}: whole-idea margin exceeds available account capacity")
    journal = contract.Journal(cfg['db'])
    staged = journal.allocated_risk(p['product'], command['account'], proposal['source_date'])
    risk = max(total_risk, sizing_summary['sizing_risk_usd'])
    if (staged + risk)/nlv*1e4 > account_policy['max_daily_bps']:
        raise ValueError(f"{command['account']}: account/product staged-day ATR risk cap exceeded")
    expires = min(contract.instant(proposal['review_deadline']), now + dt.timedelta(minutes=5))
    return contract.frozen({'schema': 'review-execution-plan.v1', 'product': p['product'],
        'source_idea_id': proposal['source_idea_id'], 'source_date': proposal['source_date'], 'proposal_id': p['proposal']['id'],
        'proposal_hash': p['proposal']['hash'], 'review_event_id': p['review']['id'],
        'actor': p['actor'], 'account': command['account'], 'broker_account': account,
        'delivery_id': p['delivery_id'], 'created_at': now.isoformat(), 'expires_at': expires.isoformat(),
        'policy': {'accounts': cfg['accounts'], **account_policy,
                   'max_notional': g['_max_notional'](command['account']), 'max_qty': g['LIVE_MAX_QTY']},
        'risk_usd': total_risk, 'notional_usd': total_notional, 'nlv': nlv,
        'account_equity': snapshot, 'sizing': sizing_summary,
        'capacity': {'initial_margin_change': initial, 'maintenance_margin_change': maintenance,
                     'previous_staged_risk_usd': staged},
        'risk_ack_required': any(leg['risk_ack_required'] for leg in legs),
        'non_atomic': len(legs) > 1,
        'exit_convention': 'Existing executor: market time child at session open or 15:59 close clock; day-2 protective stop. Not a native future auction exit.',
        'legs': legs})


def _same_preflight(plan, fresh):
    old = copy.deepcopy(contract.verify(plan)); new = copy.deepcopy(contract.verify(fresh))
    for value in (old, new):
        for name in ('created_at', 'expires_at', 'nlv'):
            value.pop(name, None)
        value['account_equity'].pop('observed_at', None)
        for leg in value['legs']:
            leg.pop('nlv', None)
            leg['capacity'].pop('observed_at', None)
    # New account/contract/open/price/stop/timing/caps cannot be auto-accepted.
    return old == new


def evidence_leg(g, ib, plan, leg, submitted):
    evidence = _capture(g, ib, plan['broker_account'], leg['con_id'])
    if evidence['account'] != plan['broker_account'] or evidence['con_id'] != leg['con_id']:
        raise ValueError('broker evidence identity mismatch')
    rows = [r for r in evidence['orders'] + evidence['completed'] if r.get('ref') == leg['ref']]
    # Raw stable orders must uniquely prove the parent AND every protective child.
    by_id = {}
    for row in rows:
        ident = tuple(row['identity'])
        if ident in by_id and by_id[ident] != row:
            raise ValueError('broker order evidence is conflicting')
        by_id[ident] = row
    rows = list(by_id.values())
    native = leg['payload']; expected_count = 2 + int(native['stop'] is not None) + int(native['target'] is not None)
    if len(rows) != expected_count:
        return {'state': 'unknown', 'detail': 'parent/complete exit-chain evidence unavailable'}
    parents = [r for r in rows if not r['parent']]
    if len(parents) != 1:
        return {'state': 'unknown', 'detail': 'parent identity ambiguous'}
    parent = parents[0]
    if (parent['action'] != native['action'] or parent['qty'] != native['quantity']
            or any(r['identity'][:2] != [plan['broker_account'], leg['con_id']] or r['identity'][4] <= 0 for r in rows)):
        return {'state': 'unknown', 'detail': 'broker account/contract/quantity/permId mismatch'}
    if submitted.get('order_ids') and set(submitted['order_ids']) != {r['identity'][3] for r in rows}:
        return {'state': 'unknown', 'detail': 'acknowledged broker order IDs changed'}
    exit_action = 'SELL' if native['action'] == 'BUY' else 'BUY'
    children = [r for r in rows if r is not parent]
    if any(r['parent'] != parent['identity'][3] or r['qty'] != native['quantity'] or r['action'] != exit_action for r in children):
        return {'state': 'unprotected', 'detail': 'exit topology/coverage mismatch'}
    expected_types = ['MKT'] + (['STP'] if native['stop'] is not None else []) + (['LMT'] if native['target'] is not None else [])
    if sorted(r['order_type'] for r in children) != sorted(expected_types) or len({r['oca_group'] for r in children}) != 1 or not children[0]['oca_group']:
        return {'state': 'unprotected', 'detail': 'exit types/OCA mismatch'}
    native_type = 'LMT' if native['entry_type'] == 'LMT' else 'MKT' if native['entry_type'] == 'MOO' else 'MOC'
    native_tif = 'GTD' if native['expiry'] else 'OPG' if native['entry_type'] == 'MOO' else 'DAY'
    if parent['order_type'] != native_type or parent['tif'] != native_tif or parent['good_till'] != (leg['timing']['parent_good_till'] or ''):
        return {'state':'unknown','detail':'parent order type/TIF/expiry changed'}
    if native_type == 'LMT' and parent['limit'] != native['entry']:
        return {'state':'unknown','detail':'parent limit changed'}
    for child in children:
        if child['oca_type'] != 1 or child['tif'] != 'GTC':
            return {'state':'unprotected','detail':'exit OCA type/TIF changed'}
        if child['order_type'] == 'STP' and (child['stop'] != native['stop'] or child['good_after'] != (leg['timing']['stop_good_after'] or '')):
            return {'state':'unprotected','detail':'protective stop level/arming changed'}
        if child['order_type'] == 'LMT' and child['limit'] != native['target']:
            return {'state':'unprotected','detail':'target changed'}
        if child['order_type'] == 'MKT' and (child['good_after'] != leg['timing']['time_good_after'] or not child['outside_rth']):
            return {'state':'unprotected','detail':'time exit changed'}
    actual_fills = [r for r in evidence['executions'] if r.get('ref') == leg['ref'] and r['identity'][4] == parent['identity'][4]]
    if len({r['exec_id'] for r in actual_fills}) != len(actual_fills):
        return {'state': 'unknown', 'detail': 'duplicate/conflicting execution evidence'}
    raw_filled = parent.get('filled')
    if raw_filled is None and not actual_fills:
        return {'state':'unknown','detail':'historical entry fill quantity unavailable'}
    entry_fills = [contract.number(r['cumulative'], 'cumulative fill', positive=False) for r in actual_fills] + [contract.number(raw_filled or 0, 'entry fill', positive=False)]
    if any(v < 0 for v in entry_fills):
        return {'state':'unknown','detail':'negative entry fill evidence'}
    filled = max(entry_fills)
    if filled < 0 or filled > native['quantity']:
        return {'state': 'unknown', 'detail': 'entry overfill'}
    accepted = {'Submitted', 'PreSubmitted', 'Filled', 'Cancelled', 'ApiCancelled'}
    if any(r['status'] not in accepted for r in rows):
        return {'state': 'rejected', 'detail': 'broker rejected or has not acknowledged all legs'}
    if parent['status']=='Filled' and filled != native['quantity']:
        return {'state':'unknown','detail':'Filled status disagrees with actual entry quantity'}
    if parent['status'] in {'Cancelled', 'ApiCancelled'}:
        state = 'cancelled_partial' if filled else 'cancelled'
    elif filled >= native['quantity']:
        state = 'filled'
    elif filled > 0:
        state = 'partially_filled'
    else:
        state = 'working'
    exit_filled = 0
    for child in children:
        actual = [e for e in evidence['executions'] if e.get('ref') == leg['ref'] and e['identity'][4] == child['identity'][4]]
        if len({e['exec_id'] for e in actual}) != len(actual):
            return {'state':'unknown','detail':'duplicate/conflicting exit execution evidence'}
        reported = child.get('filled')
        if reported is None and not actual:
            return {'state':'unknown','detail':'historical exit fill quantity unavailable'}
        fills = [contract.number(reported or 0, 'exit fill', positive=False)] + [contract.number(e['cumulative'], 'exit cumulative fill', positive=False) for e in actual]
        if any(v < 0 or v > child['qty'] for v in fills):
            return {'state':'unknown','detail':'invalid exit fill quantity'}
        quantity = max(fills)
        if child['status'] == 'Filled' and quantity != child['qty']:
            return {'state':'unknown','detail':'Filled exit status disagrees with actual quantity'}
        exit_filled += quantity
    if exit_filled > filled:
        return {'state':'unprotected','detail':'exit overfill/reversed exposure'}
    terminal = {'Filled', 'Cancelled', 'ApiCancelled'}
    if state == 'cancelled' and any(child['status'] not in terminal for child in children):
        return {'state':'unprotected','detail':'zero-fill parent cancelled but owned children remain working'}
    if filled > 0 and exit_filled == filled:
        # Matched fills explain exposure, not the remaining executable orders.
        # A live parent can re-enter; a lingering OCA sibling can reverse it.
        if parent['status'] not in terminal:
            return {'state':'unprotected','detail':'matched fills but entry parent remains working'}
        if any(child['status'] not in terminal for child in children):
            return {'state':'unprotected','detail':'matched fills but owned exit siblings remain working'}
        state='closed'
    elif exit_filled>0:
        state='partially_closed'
    if state in {'working', 'partially_filled', 'filled','partially_closed'} and any(r['status'] in {'Cancelled', 'ApiCancelled'} for r in children):
        # An exit fill can legitimately cancel its OCA siblings. Until separately
        # proved, never report an open entry with cancelled protection as healthy.
        state = 'unprotected'
    return {'state': state, 'entry_filled': filled, 'exit_filled':exit_filled, 'entry_quantity': native['quantity'],
            'identities': [r['identity'] for r in rows], 'evidence_at': evidence['at']}


def execute_batch(g, ib, command, cfg, journal, now):
    p = contract.validate_request(command, cfg['accounts'], now)
    stage = p['operation'] == 'stage'
    if stage:
        source = contract.verify(p['proposal'])
        key = contract.run_key(p['product'], source['source_idea_id'], command['account'], p['proposal']['hash'])
        old = journal.get(key)
        if old:
            if old['command_id'] != command['id']:
                raise ValueError('permanent idea/account claim exists; check existing staging')
            return old
        # The Yes decision authorizes the source risk instruction and all legs.
        # Sizing/contract/capacity checks run here, without a second user action.
        plan = preflight(g, ib, command, cfg, now)
        journal.save_preview(plan)
    else:
        plan = journal.preview(p['plan_hash'])
    payload = contract.verify(plan)
    if (payload['actor'] != p['actor'] or payload['account'] != command['account']
            or payload['product'] != p['product'] or payload['proposal_hash'] != p['proposal']['hash']
            or payload['review_event_id'] != p['review']['id'] or payload['delivery_id'] != p['delivery_id']):
        raise ValueError('confirmed preview identity/account/review changed')
    key = contract.run_key(payload['product'], payload['source_idea_id'], payload['account'], payload['proposal_hash'])
    old = journal.get(key)
    if old:
        if old['command_id'] != command['id'] or old['plan']['hash'] != plan['hash']:
            raise ValueError('permanent idea/account claim exists; reconcile instead of new execution')
        return old  # Never restart not_sent/submitting legs, even after a crash.
    if contract.instant(payload['expires_at']) <= now:
        raise ValueError('preview expired; explicit fresh preview required')
    if payload['risk_ack_required'] and not stage and p.get('risk_ack') is not True:
        raise ValueError('explicit acknowledgement of broker-estimated unprotected risk required')
    fresh = plan if stage else preflight(g, ib, command, cfg, now)
    if not stage and not _same_preflight(plan, fresh):
        raise ValueError('broker plan changed; preview and confirm again')
    record, claimed = journal.claim(key, command['id'], plan)
    record['account_recheck'] = contract.verify(fresh)['account_equity']
    if not claimed:
        return record
    for index, leg in enumerate(payload['legs']):
        now = dt.datetime.now(dt.timezone.utc) if not callable(g.get('_review_clock')) else g['_review_clock']()
        try:
            contract.validate_request(command, cfg['accounts'], now)
            if contract.instant(payload['expires_at']) <= now:
                raise ValueError('preview expired before remaining leg')
            gate(g, command, cfg)
            sizing.equity(record['account_recheck'], payload['account'], payload['broker_account'], _clock(g))
            native = dict(leg['payload'], risk_ack=stage or p.get('risk_ack') is True,
                          _broker_account=payload['broker_account'], _command_id=command['id'] + ':' + str(index),
                          _review_expected_con_id=leg['con_id'])
            record['legs'][index] = {'state': 'submitting'}
            record['state'] = 'needs_reconciliation'; journal.update(record)  # before ANY wire call
            g['_review_wire_started'] = True
            result = _invoke_entry(g, ib, native, command['account'])
            if result.get('ok') is not True:
                record['legs'][index] = {'state': 'rejected' if result.get('state') == 'rejected' else 'unknown',
                                        'detail': result.get('detail'), 'ack': result.get('fill')}
                break
            proof = evidence_leg(g, ib, payload, leg, result.get('fill') or {})
            record['legs'][index] = dict(proof, ack=result.get('fill'))
            if proof['state'] not in {'working', 'partially_filled', 'filled'}:
                break
        except BaseException:
            # A crash leaves the durable submitting claim. Do not guess whether
            # the broker received it and do not finish remaining independent legs.
            record['state'] = 'needs_reconciliation'; journal.update(record)
            raise
        finally:
            record['state'] = contract.summarize(record); journal.update(record)
    record['state'] = contract.summarize(record); journal.update(record)
    return record


def reconcile_batch(g, ib, command, journal):
    p = command['payload']; record = journal.get(p['run_key'])
    if not record:
        raise ValueError('durable execution claim missing; no inventory inferred')
    plan = contract.verify(record['plan'])
    if plan['account'] != command['account'] or plan['product'] != p['product'] or plan['actor'] != p['actor'] or p.get('proposal_hash', plan['proposal_hash']) != plan['proposal_hash']:
        raise ValueError('reconciliation account/product/actor mismatch')
    if g['_resolve_broker_account'](ib, command['account']) != plan['broker_account']:
        raise ValueError('broker account changed; reconciliation stopped')
    for index, leg in enumerate(plan['legs']):
        if record['legs'][index]['state'] == 'not_sent':
            continue  # A read-only reconciliation can NEVER submit the remainder.
        record['legs'][index] = dict(evidence_leg(g, ib, plan, leg, record['legs'][index].get('ack') or {}),
                                    ack=record['legs'][index].get('ack'))
    record['state'] = contract.summarize(record); journal.update(record)
    return record


def run_executor(g, command):
    """Called before legacy main routing, with independent old/new gate checks."""
    ib = None
    operation_lock = None
    g['_review_wire_started'] = False
    try:
        cfg = configuration(); now = g['_review_clock']() if callable(g.get('_review_clock')) else dt.datetime.now(dt.timezone.utc)
        p = contract.validate_request(command, cfg['accounts'], now)
        gate(g, command, cfg)
        if not cfg['db']:
            raise ValueError('explicit REVIEW_EXECUTION_DB required')
        journal = contract.Journal(cfg['db'])
        with journal.connect():
            pass
        pending_lock = journal.operation()
        pending_lock.__enter__()
        operation_lock = pending_lock
        operation = p['operation']
        readonly = operation not in {'execute', 'stage'} or command['dry_run']
        host, port, cid = g['PORTS'][command['account']]
        ib = g['IB'](); ib.connect(host, port, clientId=cid, timeout=8, readonly=readonly)
        ib.errorEvent += g['_on_err']
        if operation == 'reconcile':
            record = reconcile_batch(g, ib, command, journal)
            inner = {'state': record['state'], 'record': record}
        elif readonly:
            plan = preflight(g, ib, command, cfg, now); journal.save_preview(plan)
            inner = {'state': 'preview', 'plan': plan}
        else:
            record = execute_batch(g, ib, command, cfg, journal, now)
            inner = {'state': record['state'], 'record': record}
        ok = inner['state'] in {'preview', 'working', 'partially_filled', 'filled', 'cancelled','closed','partially_closed'}
        return g['_out'](ok, 'executed' if ok else 'unknown',
                         'Read-only preview' if inner['state'] == 'preview' else f"Whole-idea state: {inner['state']}; acknowledgment is not a fill",
                         fill={'review_execution': inner})
    except Exception as exc:
        touched = g['_review_wire_started']
        return g['_out'](False, 'unknown' if touched else 'rejected',
                         f'Review execution stopped: {type(exc).__name__}: {exc}; reconcile before any retry' if touched else str(exc))
    finally:
        try:
            if ib is not None:
                ib.disconnect()
        finally:
            if operation_lock is not None:
                operation_lock.__exit__(None, None, None)
