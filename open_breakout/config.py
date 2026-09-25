"""Explicit contracts, effective risk budgets, and opt-in routing boundaries."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import date
from pathlib import Path
import hashlib
import json
import math
import os

SPECS = {'NQ': (20., .25), 'MNQ': (2., .25), 'ES': (50., .25), 'MES': (5., .25)}
FAMILY = {'NQ': 'NQ', 'MNQ': 'NQ', 'ES': 'ES', 'MES': 'ES'}
# Hard code-level cap for live routing; config caps may be lower, never higher.
LIVE_PILOT_MAX_CONTRACTS = 1
ACK_ENV = 'OPEN_BREAKOUT_LIVE_ACK'

def live_ack(session: str, account: str) -> str:
    return f'LIVE {date.fromisoformat(session).isoformat()} {account}'

def positive(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')
    return value

@dataclass(frozen=True)
class Contract:
    symbol: str
    expiry: str
    con_id: int

    @classmethod
    def parse(cls, value):
        c = cls(**value)
        if c.symbol not in SPECS or type(c.con_id) is not int or c.con_id <= 0:
            raise ValueError('An explicit supported futures symbol and positive con_id are required')
        if len(c.expiry) != 8 or not c.expiry.isdigit():
            raise ValueError('expiry must be the exact YYYYMMDD from contract details')
        date(int(c.expiry[:4]), int(c.expiry[4:6]), int(c.expiry[6:]))
        return c

    @property
    def multiplier(self):
        return SPECS[self.symbol][0]

    @property
    def tick(self):
        return SPECS[self.symbol][1]

@dataclass(frozen=True)
class Market:
    name: str
    signal: Contract
    execution: Contract
    risk_bps: float
    max_contracts: int

@dataclass(frozen=True)
class Config:
    mode: str
    host: str
    port: int
    client_id: int
    account: str
    markets: tuple[Market, ...]
    shadow_equity: float
    max_daily_risk_bps: float
    max_open_risk_bps: float
    max_margin_fraction: float
    fee_per_contract_side: float
    max_entry_slippage_ticks: int
    exit_slippage_reserve_ticks: int
    stale_seconds: float
    max_open_delay_seconds: float
    allow_live: bool
    fingerprint: str
    pilot_max_contracts: int = 0
    # Watchdog/stream-gap tolerance; entry-time freshness still uses stale_seconds.
    watchdog_stale_seconds: float = 30.

    @classmethod
    def load(cls, path):
        raw = json.loads(Path(path).read_text(encoding='utf-8-sig'))
        allowed = {'mode','host','port','client_id','account','markets','shadow_equity',
                   'max_daily_risk_bps','max_open_risk_bps','max_margin_fraction',
                   'fee_per_contract_side','max_entry_slippage_ticks',
                   'exit_slippage_reserve_ticks','stale_seconds','max_open_delay_seconds','allow_live'}
        optional = {'pilot','watchdog_stale_seconds'}
        if not allowed <= set(raw) or set(raw) - allowed - optional:
            raise ValueError(f'Configuration fields differ: {sorted((set(raw) - optional) ^ allowed)}')
        if 'watchdog_stale_seconds' in raw:
            raw['watchdog_stale_seconds'] = positive(raw['watchdog_stale_seconds'], 'watchdog_stale_seconds')
            if raw['watchdog_stale_seconds'] > 300:
                raise ValueError('watchdog_stale_seconds exceeds 300')
        pilot = raw.get('pilot')
        pilot_max = 0
        if pilot is not None:
            if not isinstance(pilot, dict) or set(pilot) != {'max_contracts_per_market'}:
                raise ValueError('pilot block must be {"max_contracts_per_market": <int>}')
            pilot_max = pilot['max_contracts_per_market']
            if not isinstance(pilot_max, int) or isinstance(pilot_max, bool) or pilot_max < 1:
                raise ValueError('pilot.max_contracts_per_market must be a positive integer')
        if raw['mode'] not in {'shadow','paper','live'}:
            raise ValueError('mode must be shadow, paper or live')
        if not isinstance(raw['allow_live'], bool):
            raise ValueError('allow_live must be a JSON boolean')
        if not isinstance(raw['account'], str) or not raw['account']:
            raise ValueError('Explicit account required, including shadow data connections')
        if type(raw['port']) is not int or type(raw['client_id']) is not int or not 0 < raw['port'] < 65536 or raw['client_id'] <= 0:
            raise ValueError('Explicit port and nonzero dedicated client_id required')
        markets = []
        for item in raw['markets']:
            signal, execution = Contract.parse(item['signal']), Contract.parse(item['execution'])
            if item['name'] not in {'NQ','ES'} or FAMILY[signal.symbol] != item['name'] or FAMILY[execution.symbol] != item['name']:
                raise ValueError('Signal and execution contracts must share the configured index family')
            if signal.symbol != item['name']:
                raise ValueError('Signal contract must be the NQ or ES mini')
            if signal.expiry != execution.expiry:
                raise ValueError('Signal and execution must use the same expiry')
            risk = positive(item['risk_bps'], 'risk_bps')
            count = item['max_contracts']
            if not isinstance(count, int) or isinstance(count, bool) or count < 1:
                raise ValueError('max_contracts must be a positive integer')
            markets.append(Market(item['name'], signal, execution, risk, count))
        if not markets or len({m.name for m in markets}) != len(markets):
            raise ValueError('One configuration per market required')
        for k in ['shadow_equity','max_daily_risk_bps','max_open_risk_bps','max_margin_fraction','fee_per_contract_side','stale_seconds','max_open_delay_seconds']:
            raw[k] = positive(raw[k], k)
        if raw['max_margin_fraction'] > 1 or raw['max_open_risk_bps'] > raw['max_daily_risk_bps']:
            raise ValueError('Invalid risk or margin caps')
        for k in ['max_entry_slippage_ticks','exit_slippage_reserve_ticks']:
            if not isinstance(raw[k], int) or isinstance(raw[k], bool) or raw[k] < 0:
                raise ValueError(f'{k} must be a nonnegative integer')
        if raw['max_entry_slippage_ticks'] > 20 or raw['stale_seconds'] > 30 or raw['max_open_delay_seconds'] > 10:
            raise ValueError('Quote/open tolerances exceed supported operating envelope')
        if max(m.risk_bps for m in markets) > raw['max_open_risk_bps']:
            raise ValueError('Per-trade risk exceeds the open-risk cap')
        digest = hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()
        raw.pop('pilot', None)
        config = cls(**{**raw, 'markets': tuple(markets), 'fingerprint': digest, 'pilot_max_contracts': pilot_max})
        if config.mode == 'live':
            config.validate_live()
        return config

    def validate_live(self):
        """Structural live-pilot limits; the session acknowledgement is checked in authorize()."""
        if not self.allow_live:
            raise PermissionError('Live mode requires allow_live=true')
        if self.account.startswith('DU'):
            raise PermissionError('Live mode cannot use a DU paper account')
        if not self.pilot_max_contracts:
            raise PermissionError('Live mode requires the pilot block')
        if self.pilot_max_contracts > LIVE_PILOT_MAX_CONTRACTS:
            raise PermissionError(f'pilot.max_contracts_per_market exceeds {LIVE_PILOT_MAX_CONTRACTS}')
        if any(m.max_contracts != 1 or m.max_contracts > LIVE_PILOT_MAX_CONTRACTS for m in self.markets):
            raise PermissionError(f'Live mode requires max_contracts == {LIVE_PILOT_MAX_CONTRACTS} for every market')

    def authorize(self, session=None):
        """Checked before connecting and again before every broker mutation. Returns the live acknowledgement."""
        if self.mode == 'shadow':
            return None
        if self.mode == 'paper':
            if not self.account.startswith('DU') or self.port not in {7497,4002} or self.allow_live:
                raise PermissionError('Paper mode requires a DU account, paper port, and allow_live=false')
            return None
        self.validate_live()
        if session is None:
            raise PermissionError('Live routing requires an explicit session date')
        expected = live_ack(session, self.account)
        if os.environ.get(ACK_ENV) != expected:
            raise PermissionError(f'Live routing requires {ACK_ENV} to equal "LIVE <session YYYY-MM-DD> <full account>"')
        return expected
