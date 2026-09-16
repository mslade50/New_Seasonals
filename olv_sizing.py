"""OLV notional accounting at order submission, independent of future fills."""
from __future__ import annotations
import math
import datetime as dt

STRATEGY = 'Oversold Low Volume'
DEAD_STATUSES = {'Filled', 'Cancelled', 'ApiCancelled', 'Inactive'}


def pending_entry_notionals(book, account, *, asof, max_age_seconds=90):
    """Remaining exact Primary OLV BUY parents at their submitted limit.

    Missing remaining quantities are unknown, never assumed to be zero or
    inferred from total quantities. Expired rows still reserve until the
    broker confirms cancellation; the broker is authoritative for live state.
    """
    now = dt.datetime.fromisoformat(str(asof).replace('Z', '+00:00'))
    if not account:
        raise ValueError('OLV broker account must be explicit')
    if now.tzinfo is None:
        raise ValueError('OLV observation requires a timezone')
    accounts = [a for a in book.get('accounts',[]) if a.get('key') == 'primary']
    if len(accounts) != 1 or accounts[0].get('error') or accounts[0].get('broker_account') != account:
        raise ValueError('OLV pending-order account is unverified')
    source = dt.datetime.fromtimestamp(float(accounts[0]['orders_source_at']), dt.timezone.utc)
    if not 0 <= (now-source).total_seconds() <= max_age_seconds:
        raise ValueError('OLV pending-order snapshot is stale or future-dated')
    if not isinstance(accounts[0].get('orders'), list):
        raise ValueError('OLV pending-order response is incomplete')
    used, contracts, seen = {}, {}, set()
    for row in accounts[0]['orders']:
        parts = str(row.get('order_ref') or '').split('|')
        if len(parts) < 3 or parts[2] != STRATEGY or row.get('status') in DEAD_STATUSES:
            continue
        if row.get('action') != 'BUY':
            continue
        if (len(parts) < 4 or parts[1] != 'BUY' or row.get('account') != account
                or row.get('sec_type') != 'STK' or row.get('currency') != 'USD'
                or row.get('order_type') != 'LMT' or int(row.get('parent_id') or 0)):
            raise ValueError('OLV pending entry identity or type is ambiguous')
        symbol = str(row.get('symbol') or '').upper()
        if not symbol or parts[0] != symbol:
            raise ValueError('OLV order symbol disagrees with its reference')
        dt.date.fromisoformat(parts[3])
        con_id = float(row['con_id'])
        perm_id = float(row['perm_id'])
        remaining = float(row['remaining'])
        filled, quantity = float(row['filled']), float(row['qty'])
        limit = float(row['lmt'])
        if (not all(math.isfinite(v) for v in (con_id, perm_id, filled, quantity))
                or con_id <= 0 or con_id != int(con_id) or perm_id <= 0 or perm_id != int(perm_id) or perm_id in seen
                or filled < 0 or filled != int(filled) or filled + remaining != quantity
                or row.get('status') not in {'ApiPending','PendingSubmit','PreSubmitted','Submitted','PendingCancel'}
                or not math.isfinite(remaining) or remaining < 0 or remaining != int(remaining)
                or not math.isfinite(limit) or limit <= 0):
            raise ValueError('OLV pending entry quantity or identity is invalid')
        seen.add(perm_id)
        if symbol in contracts and contracts[symbol] != con_id:
            raise ValueError('OLV symbol maps to multiple contracts')
        contracts[symbol] = con_id
        used[(symbol, STRATEGY)] = used.get((symbol, STRATEGY),0) + remaining*limit
    return used


def held_notionals_from_order_refs(book, account, *, strategy=STRATEGY, lookback_days=30, asof=None):
    """Open notionals by (symbol, strategy), attributed from broker order refs.

    Every staged entry carries `SYMBOL|ACTION|Strategy_Ref|Staged_Date` in its
    orderRef, so the broker already knows which sleeve placed each trade. A
    symbol this strategy traded inside the lookback is attributed whole, at the
    broker's own market value.

    This needs no reviewed seed and no continuous fill history, so it survives
    the gaps that make the reconciled path refuse. It trades precision for
    availability in one direction only: a symbol two sleeves both hold is
    counted entirely against this strategy, which overstates usage and tightens
    the cap. Exits still require reconciled tranche metadata, which this cannot
    supply.
    """
    accounts = [a for a in book.get('accounts', []) if a.get('key') == 'primary']
    if len(accounts) != 1 or accounts[0].get('error'):
        raise ValueError('OLV attribution needs exactly one healthy Primary account')
    primary = accounts[0]
    if account and primary.get('broker_account') != account:
        raise ValueError('OLV attribution account mismatch')

    now = dt.datetime.now(dt.timezone.utc) if asof is None else dt.datetime.fromisoformat(
        str(asof).replace('Z', '+00:00'))
    if now.tzinfo is None:
        raise ValueError('OLV attribution requires a timezone')
    floor = (now - dt.timedelta(days=lookback_days)).date()

    symbols = set()
    for row in list(primary.get('fills') or []) + list(primary.get('orders') or []):
        parts = str(row.get('order_ref') or row.get('orderRef') or '').split('|')
        if len(parts) < 4 or parts[2].strip() != strategy:
            continue
        staged = parts[3].strip()[:10]
        try:
            if dt.date.fromisoformat(staged) < floor:
                continue
        except ValueError:
            # An unparseable staged date is not evidence of age. Keep the symbol
            # rather than silently dropping a position from the cap.
            pass
        symbol = str(row.get('symbol') or parts[0]).strip()
        if symbol:
            symbols.add(symbol)

    held = {}
    for row in primary.get('positions') or []:
        symbol = str(row.get('symbol') or '').strip()
        if symbol not in symbols or not float(row.get('position') or 0):
            continue
        value = row.get('market_value')
        if value is None:
            raise ValueError(f'{symbol} has no broker market value; cap input would be understated')
        held[(symbol, strategy)] = held.get((symbol, strategy), 0.0) + abs(float(value))
    return held


def clip_quantity(quantity, limit, cap, used):
    values = [float(v) for v in (quantity, limit, cap, used)]
    if not all(math.isfinite(v) for v in values) or min(values) < 0 or limit <= 0:
        raise ValueError('invalid OLV capacity inputs')
    return min(int(quantity), max(0, math.floor((cap - used) / limit)))


class ModelReservations:
    """Signal-time orders: reserve through expiry, then carry actual filled cost.

    Later signals consult only events at/before their date. Unfilled orders
    reserve budget just as filled orders do; future outcomes cannot free it.
    Quantity scaling is supplied by the engine's completed prior-day cap.
    """
    def __init__(self):
        self.orders = {}

    def used_fraction(self, key, date, quantity_scale):
        total = 0.0
        for order in self.orders.get(key, ()):
            filled = order.get('fill_date') is not None and order['fill_date'] <= date
            if filled:
                if order.get('exit_date') is not None and order['exit_date'] <= date:
                    continue
                price = order['fill_price']
            else:
                if order['expiry'] <= date:
                    continue
                price = order['limit']
            qty = math.floor(order['quantity'] * quantity_scale(order))
            total += qty * price / order['equity']
        return total

    def reserve(self, **order):
        self.orders.setdefault(order['key'], []).append(order)
        return order
