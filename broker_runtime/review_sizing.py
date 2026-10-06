"""Pure account sizing for the delivered Pitch grammar, never a broker action.

Agent bps are already effective. Do not apply systematic GRM/tilt, copy the
publisher's fixed-basis quantities, or use the other account's residual budget.
"""
from __future__ import annotations

import copy
import datetime as dt
import math

try:
    from . import review_execution as C
except ImportError:
    import review_execution as C

ACCOUNTS = ('primary', 'pa')
MAX_IDEA_BPS = 60.0
MAX_DAILY_BPS = 150.0
MAX_EQUITY_AGE_SECONDS = 60


def instruction(idea, orders, product):
    """Seal the delivered sizing semantics, not an inferred rounded-share ratio.

    Constants/formula agree with pitch_grammar.build_orders/check_risk_budget;
    parity tests exercise risk_bps, NAV percent, unequal weights and multipliers.
    """
    source = idea.get('sizing') or {}
    legs = idea.get('legs')
    if not isinstance(source, dict) or not isinstance(legs, list) or len(legs) != len(orders) or not legs:
        raise ValueError('delivered sizing/complete source legs unavailable; reference quantities cannot be resized')
    mode = str(source.get('mode', 'risk_bps')).lower()
    if mode not in {'risk_bps', 'nav_pct'} or product == 'seasonal' and mode != 'risk_bps':
        raise ValueError('unsupported product sizing mode')
    value = C.number(source.get('risk_bps', 30) if mode == 'risk_bps' else source.get('nav_pct'), 'source budget')
    if mode == 'risk_bps' and not value <= 100 or mode == 'nav_pct' and not value <= .5:
        raise ValueError('source budget outside Pitch grammar')
    if product == 'seasonal' and not 15 <= value <= 50:
        raise ValueError('Seasonal source risk must be 15..50 bps')
    horizon = C.quantity(idea.get('horizon_td'))
    if horizon > 63:
        raise ValueError('source horizon outside grammar')
    ex = idea.get('exit') or {}
    default_stop = 1.0 if horizon <= 5 else 1.3 if horizon <= 10 else 1.6
    stop = C.number(source.get('stop_atr_for_sizing') or ex.get('stop_atr') or default_stop, 'source sizing ATR')
    if product == 'seasonal' and (source.get('stop_atr_for_sizing') is None or stop < (1 if ex.get('stop_atr') else 3)):
        raise ValueError('Seasonal explicit sizing/catastrophe ATR unavailable')
    weights = [C.number(leg.get('weight', 1), 'leg weight') for leg in legs]
    total = sum(weights)
    derived = []
    for index, (leg, row, weight) in enumerate(zip(legs, orders, weights), 1):
        if (C.quantity(row.get('Leg')) != index or str(leg.get('ticker', '')).upper() != row.get('Ticker')
                or str(leg.get('side', '')).upper() != ('LONG' if row.get('Action') == 'BUY' else 'SHORT')):
            raise ValueError('source sizing leg identity/side changed')
        mult = C.number(leg.get('multiplier', 1), 'source multiplier')
        if row.get('Multiplier', mult) != mult:
            raise ValueError('published multiplier differs from delivered source')
        derived.append({'leg': index, 'weight': weight/total, 'atr': C.number(row.get('ATR'), 'published ATR'),
                        'reference_close': C.number(row.get('Ref_Close'), 'published close'),
                        'reference_date': str(row.get('Ref_Date') or ''), 'multiplier': mult,
                        'risk_unit': stop*C.number(row.get('ATR'), 'published ATR')*mult})
    return {'schema': 'agent-sizing.v1', 'product': product, 'mode': mode, 'value': value,
            'stop_atr_for_sizing': stop, 'max_idea_bps': MAX_IDEA_BPS, 'max_daily_bps': MAX_DAILY_BPS,
            'systematic_grm_applied': False, 'legs': derived}


def policy(cfg, product, account):
    value = (cfg.get('risk_multipliers') or {}).get(account)
    if value is None:
        raise ValueError(f'{account}: agent account risk multiplier is unconfigured; no sizing inferred')
    mult = C.number(value, 'account agent risk multiplier')
    # The only source-supported pending alternatives are parity or the existing
    # systematic PA footprint. A different policy needs a reviewed source change.
    if account == 'primary' and mult != 1 or account == 'pa' and mult not in {1, 1.3}:
        raise ValueError('account multiplier has no reviewed sizing contract')
    return {'account': account, 'product': product, 'multiplier': mult,
            'max_idea_bps': min(MAX_IDEA_BPS*mult, cfg['max_risk_bps']),
            'max_daily_bps': MAX_DAILY_BPS*mult, 'systematic_grm_applied': False}


def equity(snapshot, account, broker_account, now):
    if (snapshot.get('account') != account or snapshot.get('broker_account') != broker_account
            or snapshot.get('currency') != 'USD' or snapshot.get('source') != 'fresh_broker_account_summary'
            or snapshot.get('request_completed') is not True):
        raise ValueError('fresh exact-account USD equity/capacity evidence required')
    observed = C.instant(snapshot.get('observed_at'))
    age = (now - observed).total_seconds()
    if not 0 <= age <= MAX_EQUITY_AGE_SECONDS or observed.astimezone(C.ET).date() != now.astimezone(C.ET).date():
        raise ValueError('account equity evidence stale or future dated')
    nlv = C.number(snapshot.get('nlv'), 'exact-account NetLiquidation')
    bp = C.number(snapshot.get('buying_power'), 'exact-account BuyingPower')
    af = C.number(snapshot.get('available_funds'), 'exact-account AvailableFunds', positive=False)
    if af < 0:
        raise ValueError('account has negative available funds')
    return {**copy.deepcopy(snapshot), 'nlv': nlv, 'buying_power': bp, 'available_funds': af}


def size(proposal, snapshot, account_policy):
    spec = proposal.get('source_sizing')
    binding = (proposal.get('account_proposals') or {}).get(account_policy['account'])
    if (not isinstance(spec, dict) or spec.get('schema') != 'agent-sizing.v1'
            or not binding or binding.get('account') != account_policy['account']
            or binding.get('status') != 'requires_fresh_account_preview'
            or binding.get('sizing_hash') != C.frozen(spec)['hash']):
        raise ValueError('explicit account proposal/source sizing binding unavailable')
    if spec.get('product') != proposal['product'] or spec.get('systematic_grm_applied') is not False:
        raise ValueError('agent source sizing product/GRM mismatch')
    if spec.get('mode') not in {'risk_bps', 'nav_pct'} or proposal['product'] == 'seasonal' and spec['mode'] != 'risk_bps':
        raise ValueError('unsupported source sizing mode')
    value = C.number(spec.get('value'), 'source budget')
    if value > (100 if spec['mode'] == 'risk_bps' else .5):
        raise ValueError('source budget outside grammar')
    rows = proposal['orders']
    if len(rows) != len(spec.get('legs', [])) or not math.isclose(sum(x['weight'] for x in spec['legs']), 1, abs_tol=1e-12):
        raise ValueError('whole-idea sizing weights incomplete')
    nlv = snapshot['nlv']; multiplier = account_policy['multiplier']
    budget = nlv*value*multiplier/(1e4 if spec['mode']=='risk_bps' else 1)
    result=[];risk=0
    for row, leg in zip(rows, spec['legs']):
        if leg['leg'] != row['Leg'] or leg['multiplier'] != C.number(row.get('Multiplier', 1), 'multiplier'):
            raise ValueError('sizing leg/multiplier binding changed')
        if row.get('Sec_Type') != 'STK' or leg['multiplier'] != 1 or row.get('Contract'):
            raise ValueError('account instrument unavailable: only exact shares/ETFs with multiplier 1, whole-share lot 1 supported')
        unit = C.number(leg['risk_unit'], 'source sizing risk unit')
        denominator = unit if spec['mode']=='risk_bps' else C.number(leg['reference_close'], 'source NAV reference')*leg['multiplier']
        units = budget*C.number(leg['weight'], 'source leg weight')/denominator
        qty = math.floor(units)  # supported STK min lot/increment are exactly one share
        if qty < 1:
            raise ValueError(f"{account_policy['account']}: leg {row['Leg']} rounds to zero; whole account idea blocked, no leg/budget reassigned")
        risk += qty*unit
        result.append({'row': {**copy.deepcopy(row), 'Quantity': qty}, 'sizing': {
            'reference_quantity': row.get('Quantity'), 'quantity': qty, 'weight': leg['weight'],
            'risk_unit': unit, 'sizing_risk_usd': qty*unit, 'multiplier': 1, 'lot': 1,
            'source_atr': leg['atr'], 'source_reference_close': leg['reference_close'],
            'source_reference_date': leg['reference_date']}})
    if risk/nlv*1e4 > account_policy['max_idea_bps'] + 1e-9:
        raise ValueError('account whole-idea ATR risk exceeds publisher/account bound')
    return result, {'mode': spec['mode'], 'source_value': spec['value'], 'budget_usd': budget,
                    'account_multiplier': multiplier, 'sizing_risk_usd': risk, 'sizing_risk_bps': risk/nlv*1e4,
                    'systematic_grm_applied': False}
