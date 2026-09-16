"""Read-only evidence report. Working exits are claims, never approved holdings.

OCA siblings count once. Broker quantity and original opening executions are
shown alongside each claim so an operator can review a starting inventory.
This deliberately emits no approved seed and never connects to a broker.
"""
from collections import defaultdict
import math


def _whole(value):
    number = float(value)
    if not math.isfinite(number) or number != int(number) or number < 0:
        raise ValueError('invalid whole quantity')
    return int(number)


def inventory_reconciliation(book, fills, *, strategies):
    from scripts.harvest_fills import normalize, merge_fills
    accounts = [a for a in book.get('accounts', []) if a.get('key') == 'primary']
    if len(accounts) != 1 or accounts[0].get('error') or not accounts[0].get('broker_account'):
        raise ValueError('one identified, readable Primary account is required')
    account = accounts[0]
    broker_account = account['broker_account']
    positions = {}
    for row in account.get('positions', []):
        if row.get('account') != broker_account:
            raise ValueError('position belongs to a different or unspecified account')
        con_id = int(row.get('con_id') or 0)
        if con_id <= 0 or con_id in positions:
            raise ValueError('position contract identity is missing or duplicated')
        positions[con_id] = row
    # Normalize and collapse execution corrections before computing evidence.
    frame, _ = merge_fills(None, normalize(fills))
    effective = frame.to_dict('records')
    claims = defaultdict(list)
    issues = []
    active = [o for o in account.get('orders', [])
              if o.get('status') not in {'Filled', 'Cancelled', 'ApiCancelled', 'Inactive'}]
    parent_ids = {(o.get('client_id'), o.get('order_id')) for o in active}
    for order in active:
        ref = str(order.get('order_ref') or '')
        parts = ref.split('|')
        if len(parts) < 4 or parts[2] not in strategies:
            continue
        con_id = int(order.get('con_id') or 0)
        position = positions.get(con_id)
        if order.get('account') != broker_account:
            raise ValueError('tagged order account mismatch')
        # Children of still-working entry parents do not establish holdings.
        if (order.get('client_id'), order.get('parent_id')) in parent_ids:
            issues.append(f'{parts[0]}: working entry bracket needs fill reconciliation')
            continue
        if position is None or not position.get('position'):
            issues.append(f'{parts[0]}: tagged order without a held contract')
            continue
        close_side = 'SELL' if position['position'] > 0 else 'BUY'
        if order.get('action') != close_side:
            continue
        group = order.get('oca_group')
        if not group:
            issues.append(f'{parts[0]}: ungrouped closing order needs allocation review')
            continue
        claims[(con_id, str(group))].append(order)
    rows = []
    totals = defaultdict(int)
    for (con_id, group), orders in sorted(claims.items()):
        refs = {str(o.get('order_ref')) for o in orders}
        if len(refs) != 1:
            issues.append(f'contract {con_id}: conflicting tags in one OCA group')
            continue
        ref = refs.pop()
        symbol, _, strategy, ref_date, *_ = ref.split('|')
        precise = all(o.get('remaining') is not None and o.get('filled') is not None for o in orders)
        quantities = {_whole(o['remaining'] if precise else o['qty']) for o in orders}
        if len(quantities) != 1:
            issues.append(f'{symbol} {ref_date}: exit siblings disagree on quantity')
            continue
        quantity = quantities.pop()
        sign = 1 if positions[con_id]['position'] > 0 else -1
        opening = [f for f in effective if f['account'] == broker_account and f['con_id'] == con_id
                   and f['order_ref'] == ref and f['side'] in ({'BOT', 'BUY'} if sign > 0 else {'SLD', 'SELL'})]
        opened = sum(float(f['qty']) for f in opening)
        price = sum(float(f['qty']) * float(f['price']) for f in opening) / opened if opened else None
        rows.append(dict(symbol=symbol, strategy=strategy, ref_date=ref_date, con_id=con_id,
                         entry_order_ref=ref, claimed_qty=sign*quantity,
                         quantity_basis='remaining' if precise else 'total_quantity_unverified',
                         observed_opening_qty=opened, observed_entry_price=price,
                         time_orders=[o.get('good_after') for o in orders if o.get('good_after')],
                         target_orders=[o.get('lmt') for o in orders if o.get('order_type') == 'LMT']))
        totals[con_id] += sign*quantity
    balances = [dict(symbol=p['symbol'], con_id=k, broker_qty=p['position'],
                     tagged_exit_claim=totals[k], residual=p['position']-totals[k])
                for k, p in positions.items() if p.get('position')]
    return dict(status='review_required', account_key='primary', broker_account=broker_account,
                book_at=book.get('at'), tranches=rows, balances=balances,
                issues=issues, activation='No inventory approved or installed; matching exits alone do not prove ownership.')
