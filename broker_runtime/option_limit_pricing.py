"""Pure sizing for scheduled long options: hard premium cap, excluding fees."""
from decimal import Decimal, ROUND_CEILING
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

POLICY = "capped_limit_v1"


def capped_size(budget, ask, tick, multiplier=100):
    budget, ask, tick, multiplier = [Decimal(str(x)) for x in (budget, ask, tick, multiplier)]
    if any(not x.is_finite() or x <= 0 for x in (budget, ask, tick, multiplier)):
        raise ValueError("positive finite budget, live ask, tick and multiplier required")
    limit = (ask / tick).to_integral_value(rounding=ROUND_CEILING) * tick
    quantity = int(budget // (limit * multiplier))
    if quantity < 1:
        raise ValueError("one contract at the rounded limit exceeds the premium budget")
    return quantity, float(limit)


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
