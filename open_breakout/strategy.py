"""Tick-driven strategy decisions. No broker imports, I/O or break-even logic."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import datetime, time
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
import math
from zoneinfo import ZoneInfo
from .config import LIVE_PILOT_MAX_CONTRACTS

NY = ZoneInfo('America/New_York')

def aware(value):
    ts = datetime.fromisoformat(value) if isinstance(value, str) else value
    if ts.tzinfo is None or ts.utcoffset() is None:
        raise ValueError('Timezone-aware timestamp required')
    return ts

def snap(price, tick, up):
    if not math.isfinite(price) or price <= 0 or tick <= 0:
        raise ValueError('Invalid price/tick')
    q = Decimal(str(price)) / Decimal(str(tick))
    return float(q.to_integral_value(rounding=ROUND_CEILING if up else ROUND_FLOOR) * Decimal(str(tick)))

@dataclass
class State:
    day: str
    market: str
    prior_tr: float
    score: float
    opening: float = 0.
    previous: float = 0.
    last_timestamp: str = ''
    long_armed: bool = True
    short_armed: bool = True
    attempts: int = 0
    phase: str = 'FLAT'
    side: int = 0
    qty: int = 0
    entry_value: float = 0.
    stop: float = 0.
    planned_risk: float = 0.
    entry_order: int = 0
    stop_order: int = 0
    time_order: int = 0
    oca: str = ''
    note: str = ''

    @property
    def distance(self):
        return .25 * self.prior_tr

    @property
    def entry(self):
        return self.entry_value / self.qty if self.qty else 0.

    def record(self):
        return asdict(self)


def on_price(state, timestamp, price, tick, now, *, stale_seconds=3., open_delay=2.):
    """Return a new side only on a fresh crossing. Caller journals intent before send."""
    ts, now = aware(timestamp), aware(now)
    local = ts.astimezone(NY)
    if not math.isfinite(price) or price <= 0:
        raise ValueError('Nonpositive or nonfinite trade price')
    if state.last_timestamp and ts < aware(state.last_timestamp):
        return None
    if (now-ts).total_seconds() < -1 or (now-ts).total_seconds() > stale_seconds:
        state.note = 'STALE_TRADE'
        return None
    if local.date().isoformat() != state.day or local.time() < time(9,30):
        return None
    if not state.opening:
        boundary = local.replace(hour=9, minute=30, second=0, microsecond=0)
        if (local-boundary).total_seconds() > open_delay:
            state.phase, state.note = 'BLOCKED', 'MISSED_0930_OPEN'
            return None
        state.opening = state.previous = price
        state.last_timestamp = ts.isoformat()
        return None
    state.last_timestamp = ts.isoformat()
    upper = snap(state.opening + state.distance, tick, True)
    lower = snap(state.opening - state.distance, tick, False)
    old = state.previous
    state.previous = price
    if state.phase != 'FLAT':
        return None
    if old < upper:
        state.long_armed = True
    if old > lower:
        state.short_armed = True
    if local.time() >= time(11,30) or state.attempts >= 3:
        return None
    if state.long_armed and old < upper <= price:
        return 1
    if state.score >= 20 and state.short_armed and old > lower >= price:
        return -1
    return None


def size_order(state, market, side, bid, ask, equity, config):
    """Risk budget includes outward stop rounding, fees and exit-slippage reserve."""
    if not all(math.isfinite(v) and v > 0 for v in [bid,ask,equity]) or bid > ask:
        raise ValueError('Invalid executable quote/equity')
    if (ask-bid)/((ask+bid)/2) > .002:
        raise ValueError('Spread exceeds 20bp entry guard')
    tick, mult = market.execution.tick, market.execution.multiplier
    price = ask if side == 1 else bid
    limit = snap(price + side*config.max_entry_slippage_ticks*tick, tick, side==1)
    stop = snap(limit-side*state.distance, tick, side==-1)
    per_contract = abs(limit-stop)*mult + 2*config.fee_per_contract_side + config.exit_slippage_reserve_ticks*tick*mult
    budget = equity*market.risk_bps/10000
    qty = min(market.max_contracts, math.floor(budget/per_contract))
    if config.mode == 'live':
        # Pilot clamp after all sizing; a computed 0 remains no trade.
        qty = min(qty, LIVE_PILOT_MAX_CONTRACTS)
    return dict(side=side, qty=qty, limit=limit, risk=qty*per_contract, stop_distance=state.distance)
