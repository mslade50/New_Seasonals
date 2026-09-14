"""Pure sizing for scheduled long options: hard premium cap, excluding fees."""
from decimal import Decimal, InvalidOperation, ROUND_CEILING
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

POLICY = "capped_limit_v1"
MAX_QUOTE_AGE_SECONDS = 30


def capped_size(budget, ask, tick, multiplier=100):
    budget, ask, tick, multiplier = [Decimal(str(x)) for x in (budget, ask, tick, multiplier)]
    if any(not x.is_finite() or x <= 0 for x in (budget, ask, tick, multiplier)):
        raise ValueError("positive finite budget, live ask, tick and multiplier required")
    limit = (ask / tick).to_integral_value(rounding=ROUND_CEILING) * tick
    quantity = int(budget // (limit * multiplier))
    if quantity < 1:
        raise ValueError("one contract at the rounded limit exceeds the premium budget")
    return quantity, float(limit)


def market_increments(ib, contract, details):
    """Resolve the rule for the order's exchange, never the global minTick.

    IBKR pairs validExchanges and marketRuleIds by position. minTick can be
    smaller than the applicable premium-band increment on that exchange.
    """
    exchange = str(getattr(contract, "exchange", "") or "").upper()
    exchanges = [x.strip().upper() for x in str(getattr(details, "validExchanges", "") or "").split(",")]
    rule_ids = [x.strip() for x in str(getattr(details, "marketRuleIds", "") or "").split(",")]
    if not exchange or len(exchanges) != len(rule_ids) or exchanges.count(exchange) != 1:
        raise ValueError("exact exchange market rule is unavailable; nothing submitted")
    rule_id = rule_ids[exchanges.index(exchange)]
    if not rule_id.isdigit() or int(rule_id) <= 0:
        raise ValueError("exact exchange market rule is unavailable; nothing submitted")
    rows = ib.reqMarketRule(int(rule_id))
    return _validated_increments(rows)


def _validated_increments(rows):
    try:
        bands = [(Decimal(str(row.lowEdge)), Decimal(str(row.increment))) for row in rows]
    except (TypeError, AttributeError, InvalidOperation) as exc:
        raise ValueError("option market rule is unavailable or malformed") from exc
    if not bands or any(not low.is_finite() or not tick.is_finite() or low < 0 or tick <= 0
                        for low, tick in bands):
        raise ValueError("option market rule requires finite price bands and positive increments")
    if bands[0][0] != 0 or any(a[0] >= b[0] for a, b in zip(bands, bands[1:])):
        raise ValueError("option market rule price bands must start at zero and increase")
    return bands


def capped_market_size(budget, ask, bands, multiplier=100):
    """Round into the applicable validated rule band before sizing the premium."""
    budget, ask, multiplier = [Decimal(str(x)) for x in (budget, ask, multiplier)]
    if any(not x.is_finite() or x <= 0 for x in (budget, ask, multiplier)):
        raise ValueError("positive finite budget, live ask and multiplier required")
    for index, (low, tick) in enumerate(bands):
        upper = bands[index + 1][0] if index + 1 < len(bands) else None
        if upper is not None and ask >= upper:
            continue
        start = max(ask, low)
        limit = low + ((start - low) / tick).to_integral_value(rounding=ROUND_CEILING) * tick
        # Rounding across a boundary must use the next band's increment.
        if upper is not None and limit >= upper:
            continue
        quantity = int(budget // (limit * multiplier))
        if quantity < 1:
            raise ValueError("one contract at the rounded limit exceeds the premium budget")
        return quantity, float(limit)
    raise ValueError("option market rule does not cover the quoted premium")


def fresh_live_ask(quote, con_id, requested_at, now=None):
    """Require an exact, live snapshot received during this pricing request."""
    if int(getattr(quote, "marketDataType", 0) or 0) != 1:
        raise ValueError("scheduled option requires a fresh live quote")
    if int(getattr(getattr(quote, "contract", None), "conId", 0) or 0) != con_id:
        raise ValueError("option quote identity changed")
    received = getattr(quote, "time", None)
    now = now or datetime.now(timezone.utc)
    if (not isinstance(received, datetime) or received.tzinfo is None
            or received < requested_at or not 0 <= (now - received).total_seconds() <= MAX_QUOTE_AGE_SECONDS):
        raise ValueError("scheduled option snapshot is stale or missing its timestamp")
    ask = Decimal(str(quote.ask))
    if not ask.is_finite() or ask <= 0:
        raise ValueError("scheduled option requires a positive finite live ask")
    return float(ask)


def validate_intent(payload, now=None):
    if payload.get("pricing_policy") != POLICY or payload.get("order_type") != "LMT":
        raise ValueError("scheduled option requires the capped-limit contract; legacy MKT intents are blocked")
    if payload.get("tif") != "DAY":
        raise ValueError("scheduled option requires DAY")
    zone = ZoneInfo("America/New_York")
    now = now or datetime.now(zone)
    start = datetime.fromisoformat(payload["execute_date"] + "T" + payload["execute_time"]).replace(tzinfo=zone)
    grace = int(payload.get("grace_minutes", 5))
    if not 1 <= grace <= 30 or not start <= now <= start + timedelta(minutes=grace):
        raise ValueError("scheduled option is outside its execution window; nothing submitted")
