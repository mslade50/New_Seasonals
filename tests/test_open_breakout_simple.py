"""Offline tests for open_breakout/simple.py with a fake gateway. No IBKR connection."""
import asyncio
from dataclasses import replace
from datetime import datetime, timedelta
import json
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
from open_breakout.config import Config, RangeFilter, live_ack
from open_breakout.inputs import range_decision
from open_breakout import simple
from open_breakout.simple import Gateway, Runner, globex_open, prior_r, run_proof, start_problems
from open_breakout.strategy import NY, State, size_order

ROOT = Path(__file__).resolve().parents[1]
DAY = '2026-10-12'
SYM = {'NQ': 'MNQ', 'ES': 'MES'}


def at(text, day=DAY):
    return datetime.fromisoformat(f'{day}T{text}').replace(tzinfo=NY)


@pytest.fixture
def config(tmp_path):
    raw = json.loads((ROOT/'artifacts'/'open_breakout_runs'/'config-20260925-live.json').read_text(encoding='utf-8-sig'))
    raw['account'] = 'U9999999'
    p = tmp_path/'c.json'
    p.write_text(json.dumps(raw))
    return Config.load(p)


class FakeGW:
    def __init__(self):
        self.sent, self.cancelled, self.q, self.st, self.up = [], [], {}, {}, True
        self.ids, self.connects, self.syncs = iter(range(100, 100000)), 0, 0
        self.pos, self.working = {'MNQ': 0, 'MES': 0}, []
        self.on_fill = self.on_status = self.on_error = lambda *a: None

    def connected(self): return self.up
    def next_id(self): return next(self.ids)
    def quote(self, n): return self.q[n]
    def status(self, oid): return self.st.get(oid, 'UNKNOWN')

    async def connect(self, on_tick):
        self.up = True
        self.connects += 1

    def send(self, name, oid, body):
        self.sent.append((name, oid, body))
        self.st[oid] = 'Submitted'

    def cancel(self, oid):
        self.cancelled.append(oid)
        if oid in getattr(self, 'stuck', ()):
            return
        self.st[oid] = 'Cancelled'
        for n, o, b in self.sent:   # the broker cancels a parent's children with it
            if b.get('parent') == oid and not getattr(self, 'orphan_children', False):
                self.st[o] = 'Cancelled'

    async def sync(self):
        self.syncs += 1
        return dict(own_positions=dict(self.pos), own_positions_error=None, working_orders=list(self.working))

    def filled(self, oid): return getattr(self, 'part', {}).get(oid, 0)

    async def open_orders(self):
        """The broker's working orders; `alive` ids are working there whatever the local status says."""
        return [dict(id=o, status='Submitted') for n, o, b in self.sent if self.st[o] not in ('Cancelled', 'Filled') or o in getattr(self, 'alive', ())]


async def nosleep(x): return None


def make(config, tmp_path, score=50., prior_tr=40., skip=()):
    man = dict(score=score, markets={m.name: dict(prior_tr=prior_tr, prior_range_status='OK', skip_prior_range=m.name in skip,
                                                  prior_range_reason='test') for m in config.markets})
    now, gw, alerts = [at('09:29:00')], FakeGW(), []
    r = Runner(config, DAY, gw, man, tmp_path/'run', alerts.append, 750000., clock=lambda: now[0].astimezone(), sleep=nosleep)
    return r, gw, now, alerts


def tick(r, gw, now, price, hms, names=('NQ', 'ES')):
    now[0] = at(hms)
    for n in names:
        px = price if n == 'NQ' else price/4
        gw.q[n] = (px-.25, px, now[0])
        r.on_tick(n, now[0], px)


def entry_orders(gw, name, attempt=None):
    return [(oid, b) for n, oid, b in gw.sent if n == name and b['ref'].endswith('ENTRY') and (attempt is None or f'-{attempt}-' in b['ref'])]


def fill_entry(r, gw, name, side, px, n=[0]):
    oid, b = next((o, b) for o, b in entry_orders(gw, name, r.s[name].attempts) if b['side'] == side)
    n[0] += 1
    gw.st[oid] = 'Filled'
    gw.pos[SYM[name]] += side*b['qty']
    r.on_fill(oid, f'0001.{n[0]:04d}.01.01', b['qty'], px)
    return oid, b


def fill_role(r, gw, name, role, px, side=-1, n=[100], oca_cancel=True):
    oid, b = next((o, b) for nm, o, b in reversed(gw.sent) if nm == name and b['ref'].endswith(role) and b['side'] == side)
    n[0] += 1
    gw.st[oid] = 'Filled'
    gw.pos[SYM[name]] += b['side']*b['qty']
    for _, o, x in gw.sent:
        if oca_cancel and o != oid and x.get('oca') == b['oca']:
            gw.st[o] = 'Cancelled'
    r.on_fill(oid, f'0002.{n[0]:04d}.01.01', b['qty'], px)
    return oid


def test_signal_triggers_sizes_and_bracket(config, tmp_path):
    async def go():
        r, gw, now, alerts = make(config, tmp_path)
        tick(r, gw, now, 20000., '09:30:00')
        now[0] = at('09:30:00') + timedelta(milliseconds=300)
        await r.step()
        sent = [x for x in gw.sent if x[0] == 'NQ']
        assert len(sent) == 6 and len([x for x in gw.sent if x[0] == 'ES']) == 6
        (lid, le), (sid, se), (tid, te) = sent[0][1:], sent[1][1:], sent[2][1:]
        m = r.mk['NQ']
        assert le['kind'] == 'STP LMT' and le['side'] == 1 and le['stop'] == 20010. and le['limit'] == 20010.5
        assert le['tif'] == 'GTD' and le['good_till'] == '20261012 11:30:00 America/New_York' and le['trigger_method'] == 2
        assert le['oca_type'] == 1 and le['transmit'] is False and 'parent' not in le
        assert se['kind'] == 'STP' and se['side'] == -1 and se['stop'] == 20000. and se['tif'] == 'GTC' and se['parent'] == lid and se['transmit'] is False
        assert te['kind'] == 'MKT' and te['tif'] == 'GTC' and te['good_after'] == '20261012 15:55:00 America/New_York'
        assert te['parent'] == lid and te['transmit'] is True and se['oca'] == te['oca'] and se['oca_type'] == te['oca_type'] == 1
        assert se['oca'] != le['oca']
        short, sstop = sent[3][2], sent[4][2]
        assert short['side'] == -1 and short['stop'] == 19990. and short['limit'] == 19989.5 and short['oca'] == le['oca']
        assert sstop['stop'] == 20000. and sstop['oca'] != se['oca'] and sstop['side'] == 1
        st = State(DAY, 'NQ', 40., 50.)
        assert le['qty'] == size_order(st, m, 1, 20010., 20010., 750000., config)['qty'] > 0
        assert le['ref'] == 'MNQ|BUY|OpenBreakout|2026-10-12|NQ-1-ENTRY' and se['ref'].endswith('NQ-1-STOP') and te['ref'].endswith('NQ-1-TIME')
        assert json.loads((tmp_path/'run'/'day.json').read_text())['markets']['NQ']['attempts'] == 1
    asyncio.run(go())


def test_shorts_gated_below_score_20(config, tmp_path):
    async def go():
        r, gw, now, _ = make(config, tmp_path, score=19.9)
        tick(r, gw, now, 20000., '09:30:00')
        await r.step()
        assert {b['side'] for n, o, b in gw.sent if b['ref'].endswith('ENTRY')} == {1}
        assert len(gw.sent) == 6
    asyncio.run(go())


def test_missed_open_means_no_orders(config, tmp_path):
    async def go():
        r, gw, now, _ = make(config, tmp_path)
        tick(r, gw, now, 20000., '09:30:05')
        await r.step()
        assert not gw.sent
    asyncio.run(go())


def test_prior_range_skip_marks_market_and_places_nothing(config, tmp_path):
    async def go():
        r, gw, now, _ = make(config, tmp_path, skip=('NQ',))
        tick(r, gw, now, 20000., '09:30:00')
        await r.step()
        assert not [x for x in gw.sent if x[0] == 'NQ'] and [x for x in gw.sent if x[0] == 'ES']
    asyncio.run(go())


def test_range_filter_rule_and_missing_values():
    f = RangeFilter.parse(dict(enabled=True, threshold=1.25, mode='skip', require_prior_big_win=True, big_win_r=2.0), 'live')
    ok = dict(atr20=100., ratio=1.3, atr20_window=['a', 'b'], roll_excluded=[])
    assert range_decision(f, ok, dict(r=2.5, source='own_journal', note=''))['skip_prior_range'] is True
    assert range_decision(f, ok, dict(r=1.0, source='own_journal', note=''))['skip_prior_range'] is False
    assert range_decision(f, ok, dict(r=None, source=None, note=''))['skip_prior_range'] is False
    assert range_decision(f, dict(atr20=100., ratio=1.1, atr20_window=[], roll_excluded=[]), dict(r=3., source='x', note=''))['skip_prior_range'] is False
    assert range_decision(f, 'no atr20', dict(r=None, source=None, note=''))['skip_prior_range'] is True


def test_prior_session_r_from_own_json_then_old_artifacts(tmp_path):
    names = ['NQ', 'ES']
    first = prior_r(tmp_path, DAY, names, 'live')   # Friday 2026-10-09 has no artifacts at all
    assert all(v['r'] is None for v in first.values())
    d = tmp_path/'2026-10-09-simple'
    d.mkdir()
    (d/'day.json').write_text(json.dumps(dict(day='2026-10-09', mode='live', markets={
        'NQ': dict(r=3.0, first_r=2.4, trades=[1]), 'ES': dict(r=None, first_r=None, trades=[])})))
    got = prior_r(tmp_path, DAY, names, 'live')
    assert got['NQ']['r'] == 2.4 and got['NQ']['source'] == 'own_journal' and got['ES']['r'] is None
    # The real repo: the last session that ran (2026-10-08) is read by the old-journal fallback.
    real = prior_r(ROOT/'artifacts'/'open_breakout_runs', '2026-10-09', names, 'live')
    assert real['NQ']['r'] is None or isinstance(real['NQ']['r'], float)


def test_start_check_blocks_when_not_flat_or_orders_working():
    flat = dict(own_positions={'MNQ': 0, 'MES': 0}, own_positions_error=None, working_orders=[])
    assert start_problems(flat) == []
    assert start_problems({**flat, 'own_positions': {'MNQ': 3, 'MES': 0}})
    assert start_problems({**flat, 'working_orders': [dict(symbol='MES', type='STP', client_id=1)]})
    assert start_problems({**flat, 'own_positions': None, 'own_positions_error': 'timeout'})


def test_second_try_after_stop_out_up_to_three_and_not_after_1130(config, tmp_path):
    async def go():
        r, gw, now, alerts = make(config, tmp_path)
        tick(r, gw, now, 20000., '09:30:00', names=('NQ',))
        await r.step()
        for attempt in (1, 2, 3):
            assert r.s['NQ'].attempts == attempt and len(entry_orders(gw, 'NQ', attempt)) == 2
            fill_entry(r, gw, 'NQ', 1, 20010.)
            assert r.x['NQ'].status == 'OPEN'
            await asyncio.sleep(0)
            assert gw.cancelled   # leftover short parent cancelled by settle
            tick(r, gw, now, 19990., f'09:4{attempt}:00', names=('NQ',))
            fill_role(r, gw, 'NQ', 'STOP', 20000.)
            assert r.x['NQ'].status == 'IDLE' and r.x['NQ'].pos == 0 and r.x['NQ'].trades[-1]['r'] == pytest.approx(-1., abs=.01)
            tick(r, gw, now, 20000., f'09:5{attempt}:00', names=('NQ',))
            await r.step()
        assert r.s['NQ'].attempts == 3 and r.x['NQ'].status == 'IDLE'   # 4th never parked
        tick(r, gw, now, 20000., '10:10:00', names=('NQ',))
        await r.step()
        assert len(entry_orders(gw, 'NQ')) == 6
        assert entry_orders(gw, 'NQ', 2)[0][1]['ref'].endswith('NQ-2-ENTRY')
    asyncio.run(go())


def test_no_second_try_after_1130_or_outside_triggers(config, tmp_path):
    async def go():
        r, gw, now, alerts = make(config, tmp_path)
        tick(r, gw, now, 20000., '09:30:00', names=('NQ',))
        await r.step()
        fill_entry(r, gw, 'NQ', 1, 20010.)
        fill_role(r, gw, 'NQ', 'STOP', 19990.)
        n = len(gw.sent)
        tick(r, gw, now, 20015., '10:00:00', names=('NQ',))          # above the long trigger: not inside
        await r.step()
        assert len(gw.sent) == n
        tick(r, gw, now, 20000., '11:30:01', names=('NQ',))
        await r.step()
        assert len(gw.sent) == n and any('entry window closed' in a for a in alerts)
    asyncio.run(go())


def test_child_rejection_after_fill_flattens_market(config, tmp_path):
    async def go():
        r, gw, now, alerts = make(config, tmp_path)
        tick(r, gw, now, 20000., '09:30:00', names=('NQ',))
        await r.step()
        _, b = fill_entry(r, gw, 'NQ', 1, 20010.)
        stop_oid = next(o for n, o, x in gw.sent if x['ref'].endswith('NQ-1-STOP') and x['side'] == -1 and x['parent'] == entry_orders(gw, 'NQ')[0][0])
        r.on_error(stop_oid, 201, 'Order rejected')
        for _ in range(5):
            await asyncio.sleep(0)
        flat = [x for x in gw.sent if x[2]['ref'].endswith('FLAT')]
        assert len(flat) == 1 and flat[0][2]['kind'] == 'MKT' and flat[0][2]['side'] == -1 and flat[0][2]['qty'] == b['qty']
        assert any('FLATTEN' in a for a in alerts) and any('REJECTED' in a for a in alerts)
        r.on_fill(flat[0][1], '0003.0001.01.01', b['qty'], 20009.)
        assert r.x['NQ'].status == 'OFF' and r.x['NQ'].pos == 0
        tick(r, gw, now, 20000., '09:50:00', names=('NQ',))
        await r.step()
        assert r.s['NQ'].attempts == 1
    asyncio.run(go())


def test_reconnect_resyncs_and_keeps_trading(config, tmp_path):
    async def go():
        r, gw, now, alerts = make(config, tmp_path)
        await r.step()
        gw.up, gw.syncs = False, 0
        await r.step()
        assert gw.connects == 1 and gw.syncs == 1 and any('reconnect' in a for a in alerts)
        tick(r, gw, now, 20000., '09:30:00')
        await r.step()
        assert gw.sent   # no halt state exists: orders still placed after the reconnect
        # position filled while disconnected shows up in the broker's own-execution view
        gw.up, gw.pos['MNQ'] = False, 7
        await r.step()
        assert r.x['NQ'].pos == 7 and r.x['NQ'].status == 'OPEN'
    asyncio.run(go())


def test_unknown_position_after_reconnect_blocks_new_orders(config, tmp_path):
    async def go():
        r, gw, now, _ = make(config, tmp_path)

        async def bad():
            return dict(own_positions=None, own_positions_error='timeout', working_orders=[])
        gw.sync = bad
        r.synced = False
        tick(r, gw, now, 20000., '09:30:00')
        await r.step()
        assert not gw.sent
    asyncio.run(go())


def test_fat_finger_cap_60(config, tmp_path):
    async def go():
        r, gw, now, _ = make(config, tmp_path, prior_tr=2.)
        tick(r, gw, now, 20000., '09:30:00', names=('NQ',))
        await r.step()
        assert [b['qty'] for n, o, b in gw.sent if b['ref'].endswith('ENTRY')] == [60, 60]
    asyncio.run(go())


def test_open_and_daily_risk_caps_size_down(config, tmp_path):
    async def go():
        r, gw, now, _ = make(config, tmp_path)
        r.reserved = 750000*70/1e4        # 5 bps of the 75 bps daily budget left
        tick(r, gw, now, 20000., '09:30:00', names=('NQ',))
        await r.step()
        q = entry_orders(gw, 'NQ')[0][1]['qty']
        assert 0 < q <= 15
    asyncio.run(go())


def test_dry_run_never_calls_place_order_and_live_send_sets_parent_and_transmit(config, monkeypatch, capsys):
    asyncio.set_event_loop(asyncio.new_event_loop())
    from open_breakout.ibkr import IBKR
    monkeypatch.setenv('OPEN_BREAKOUT_LIVE_ACK', live_ack(DAY, config.account))
    calls = []
    for dry in (True, False):
        gw = Gateway(config, DAY, dry)
        gw.t = IBKR(config, session=DAY)
        gw.t.contracts = {config.markets[0].execution.con_id: 'CONTRACT'}
        gw.t.ib.placeOrder = lambda c, o: calls.append((c, o)) or NS(order=o)
        body = dict(kind='STP', side=-1, qty=3, stop=19990., tif='GTC', oca='G', oca_type=1, parent=55, transmit=False,
                    account=config.account, ref='MNQ|BUY|OpenBreakout|2026-10-12|NQ-1-STOP')
        gw.send('NQ', 56, body)
        if dry:
            assert not calls and 'DRY-RUN would place' in capsys.readouterr().out
    (c, o), = calls
    assert o.parentId == 55 and o.transmit is False and o.orderType == 'STP' and o.ocaType == 1 and o.auxPrice == 19990.
    with pytest.raises(PermissionError):
        gw.send('NQ', 57, {**body, 'qty': 61})


def spin():
    async def _():
        for _ in range(8):
            await asyncio.sleep(0)
    return _()


def test_day_json_is_mode_tagged_and_dry_never_resumes_or_feeds_prior_r(config, tmp_path):
    async def go():
        r, gw, now, _ = make(config, tmp_path)
        tick(r, gw, now, 20000., '09:30:00', names=('NQ',))
        await r.step()
        d = json.loads((tmp_path/'run'/'day.json').read_text())
        assert d['mode'] == 'live' and d['markets']['NQ']['attempts'] == 1 and d['reserved'] > 0
        r2 = Runner(config, DAY, FakeGW(), r.man, tmp_path/'run', lambda *a: None, 750000., dry=True)
        assert r2.s['NQ'].attempts == 0 and r2.reserved == 0          # a dry runner never resumes a live file
        r2.save()
        assert json.loads((tmp_path/'run'/'day.json').read_text())['mode'] == 'dry'
        r3 = Runner(config, DAY, FakeGW(), r.man, tmp_path/'run', lambda *a: None, 750000.)
        assert r3.s['NQ'].attempts == 0 and r3.reserved == 0          # and a live runner ignores a dry file
    asyncio.run(go())
    d = tmp_path/'2026-10-09-simple'
    d.mkdir()
    (d/'day.json').write_text(json.dumps(dict(day='2026-10-09', mode='dry', markets={'NQ': dict(r=5., first_r=5., trades=[1]), 'ES': dict(r=5., first_r=5., trades=[1])})))
    assert all(v['r'] is None for v in prior_r(tmp_path, DAY, ['NQ', 'ES'], 'live').values())


def test_1130_cancels_every_entry_and_alerts_if_one_stays_alive(config, tmp_path):
    async def go(stuck):
        r, gw, now, alerts = make(config, tmp_path / ('s' if stuck else 'o'))
        tick(r, gw, now, 20000., '09:30:00')
        await r.step()
        entries = [oid for oid, b in entry_orders(gw, 'NQ') + entry_orders(gw, 'ES')]
        assert len(entries) == 4 and not gw.cancelled
        gw.stuck = {entries[0]} if stuck else set()
        tick(r, gw, now, 20000., '11:30:01')
        await r.step()
        assert set(entries) <= set(gw.cancelled)
        assert any('STILL ALIVE' in a and str(entries[0]) in a for a in alerts) is stuck
    asyncio.run(go(False))
    asyncio.run(go(True))


def two_leg_fill(r, gw, now):
    tick(r, gw, now, 20000., '09:30:00', names=('NQ',))
    return r.step()


def test_cancelled_stop_with_open_position_flattens_unless_time_filled(config, tmp_path):
    async def go():
        r, gw, now, alerts = make(config, tmp_path / 'a')
        await two_leg_fill(r, gw, now)
        lid, b = fill_entry(r, gw, 'NQ', 1, 20010.)
        await spin()
        stops = {o: x for n, o, x in gw.sent if x['ref'].endswith('NQ-1-STOP')}
        short_stop = next(o for o, x in stops.items() if x['side'] == 1)
        long_stop = next(o for o, x in stops.items() if x['side'] == -1)
        r.on_status(short_stop, 'Cancelled')                 # the cancelled short sibling's stop is not our protection
        await spin()
        assert not [x for x in gw.sent if x[2]['ref'].endswith('FLAT')]
        gw.st[long_stop] = 'Cancelled'
        r.on_status(long_stop, 'Cancelled')
        await spin()
        flat = [x for x in gw.sent if x[2]['ref'].endswith('FLAT')]
        assert len(flat) == 1 and flat[0][2]['side'] == -1 and flat[0][2]['qty'] == b['qty'] and any('STOP LOST' in a for a in alerts)
        # TIME sibling filled within the second: closed by OCA, nothing to flatten
        r, gw, now, alerts = make(config, tmp_path / 'b')
        await two_leg_fill(r, gw, now)
        fill_entry(r, gw, 'NQ', 1, 20010.)
        long_stop = next(o for n, o, x in gw.sent if x['ref'].endswith('NQ-1-STOP') and x['side'] == -1)
        gw.st[long_stop] = 'Cancelled'
        r.on_status(long_stop, 'Cancelled')
        fill_role(r, gw, 'NQ', 'TIME', 20020.)
        await spin()
        assert not [x for x in gw.sent if x[2]['ref'].endswith('FLAT')] and r.x['NQ'].pos == 0
    asyncio.run(go())


def test_only_hard_reject_codes_count(config, tmp_path):
    async def go(code, flat):
        r, gw, now, _ = make(config, tmp_path / str(code))
        await two_leg_fill(r, gw, now)
        fill_entry(r, gw, 'NQ', 1, 20010.)
        stop = next(o for n, o, x in gw.sent if x['ref'].endswith('NQ-1-STOP') and x['side'] == -1)
        r.on_error(stop, code, 'x')
        await spin()
        assert bool([x for x in gw.sent if x[2]['ref'].endswith('FLAT')]) is flat
    for code in (10147, 10148, 161, 202, 2104):
        asyncio.run(go(code, False))
    for code in (110, 200, 201, 203, 321):
        asyncio.run(go(code, True))


def test_first_r_is_attempt_one_only_and_prior_r_uses_it(config, tmp_path):
    async def go():
        r, gw, now, _ = make(config, tmp_path, skip=('ES',))
        tick(r, gw, now, 20000., '09:30:00')
        await r.step()
        fill_entry(r, gw, 'NQ', 1, 20010.)
        await spin()
        fill_role(r, gw, 'NQ', 'STOP', 19990.)                          # attempt 1: about -2R
        tick(r, gw, now, 20000., '09:50:00')
        await r.step()
        assert r.s['NQ'].attempts == 2
        fill_entry(r, gw, 'NQ', 1, 20010.)
        await spin()
        fill_role(r, gw, 'NQ', 'STOP', 20000.)                          # attempt 2: about -1R
        d = json.loads((tmp_path/'run'/'day.json').read_text())['markets']
        t1, t2 = d['NQ']['trades']
        assert d['NQ']['first_r'] == t1['r'] != t2['r'] and d['NQ']['r'] == pytest.approx(t1['r']+t2['r'])
        assert d['ES']['first_r'] is None and d['ES']['trades'] == []     # range-skipped: no trade, no unfiltered R computed
    asyncio.run(go())
    p = tmp_path/'2026-10-09-simple'
    p.mkdir()
    (p/'day.json').write_text(json.dumps(dict(day='2026-10-09', mode='live', markets={
        'NQ': dict(r=-3., first_r=1.5, trades=[1, 2]), 'ES': dict(r=None, first_r=None, trades=[]), 'X': {}})))
    got = prior_r(tmp_path, DAY, ['NQ', 'ES'], 'live')
    assert got['NQ']['r'] == 1.5 and got['NQ']['source'] == 'own_journal' and got['ES']['r'] is None and got['ES']['source'] != 'own_journal'


def test_no_repark_while_position_or_earlier_orders_alive(config, tmp_path):
    async def go():
        r, gw, now, _ = make(config, tmp_path)
        await two_leg_fill(r, gw, now)
        fill_entry(r, gw, 'NQ', 1, 20010.)
        await spin()
        fill_role(r, gw, 'NQ', 'STOP', 19990., oca_cancel=False)        # TIME sibling still alive at the broker
        tick(r, gw, now, 20000., '09:50:00', names=('NQ',))
        await r.step()
        assert r.s['NQ'].attempts == 1 and r.x['NQ'].status == 'IDLE'
        time_oid = next(o for n, o, x in gw.sent if x['ref'].endswith('NQ-1-TIME') and x['side'] == -1)
        gw.st[time_oid] = 'Cancelled'
        r.x['NQ'].pos = 1                                              # still a position per the books
        await r.step()
        assert r.s['NQ'].attempts == 1
        r.x['NQ'].pos = 0
        await r.step()
        assert r.s['NQ'].attempts == 2
    asyncio.run(go())


def test_settle_cancels_the_attempt_it_was_spawned_for(config, tmp_path):
    async def go():
        r, gw, now, _ = make(config, tmp_path)
        await two_leg_fill(r, gw, now)
        (lid, _), (sid, _) = entry_orders(gw, 'NQ', 1)
        r.x['NQ'].cur = 2                                              # the attempt counter moved on before settle ran
        await r.settle('NQ', lid, 1)
        assert sid in gw.cancelled and lid not in gw.cancelled
    asyncio.run(go())


# ---- simple-proof ----

class ProofGW(FakeGW):
    def __init__(self, now):
        super().__init__()
        self.now, self.script = now, {}

    def quote(self, n): return (19999.75, 20000.25, self.now())

    def send(self, name, oid, body):
        super().send(name, oid, body)
        if oid in self.script:
            self.script[oid](oid)

    async def open_orders(self):
        return [dict(id=o, ref=b['ref'], oca=b.get('oca'), oca_type=b.get('oca_type'), parent=b.get('parent', 0), kind=b['kind'], tif=b['tif'],
                     till=b.get('good_till'), after=b.get('good_after'), status=self.st[o]) for n, o, b in self.sent if self.st[o] not in ('Cancelled', 'Filled')]


def proof_setup(config, tmp_path, when='2026-10-12T10:00:00', **cfg):
    nq = tuple(m for m in config.markets if m.name == 'NQ')
    c = replace(config, client_id=927489, markets=nq, **cfg)
    clock = lambda: datetime.fromisoformat(when).replace(tzinfo=NY)
    return c, ProofGW(clock), clock, []


def run(config, gw, clock, alerts, out):
    return asyncio.run(run_proof(gw, config, DAY, out, alerts.append, clock=clock, sleep=nosleep))


def test_globex_hours():
    f = lambda t: globex_open(datetime.fromisoformat(t).replace(tzinfo=NY))
    assert f('2026-10-12T10:00:00') and f('2026-10-11T18:00:00')
    assert not f('2026-10-10T12:00:00') and not f('2026-10-11T17:59:00') and not f('2026-10-09T17:00:00') and not f('2026-10-13T17:30:00')
    assert f('2026-10-13T18:00:00') and f('2026-10-09T16:59:00')


def test_proof_happy_path_report_and_structure(config, tmp_path):
    c, gw, clock, alerts = proof_setup(config, tmp_path)
    ok, rep = run(c, gw, clock, alerts, tmp_path)
    assert ok, rep['flagged']
    assert len(gw.sent) == 6 and {b['qty'] for n, o, b in gw.sent} == {1} and {n for n, o, b in gw.sent} == {'NQ'}
    ents = [(o, b) for n, o, b in gw.sent if b['ref'].endswith('ENTRY')]
    assert {b['kind'] for o, b in ents} == {'STP LMT'} and {b['tif'] for o, b in ents} == {'GTD'} and len({b['oca'] for o, b in ents}) == 1
    long_e, short_e = next(x for x in ents if x[1]['side'] == 1), next(x for x in ents if x[1]['side'] == -1)
    assert long_e[1]['stop'] == pytest.approx(21000., abs=.5) and short_e[1]['stop'] == pytest.approx(19000., abs=.5)
    assert all(b['ref'].startswith('MNQ|') and '|OpenBreakout|' in b['ref'] and '-9-' in b['ref'] for n, o, b in gw.sent)
    assert gw.cancelled[0] == long_e[0]                                 # the long parent first, then the rest
    assert all(gw.st[o] == 'Cancelled' for n, o, b in gw.sent)
    o = rep['oca']
    assert o['entries_share'] and o['long_children_share'] and o['short_children_share'] and o['distinct'] and o['types_ok'] and o['parents_ok']
    assert rep['position'] == 0 and not rep['flagged'] and rep['client_id'] == 927489
    f, = tmp_path.glob('proof-*.json')
    assert json.loads(f.read_text())['ok'] is True


@pytest.mark.parametrize('case', ['closed', 'live_client', 'near_trigger', 'not_flat', 'working'])
def test_proof_refusals_place_nothing(config, tmp_path, case, monkeypatch):
    c, gw, clock, alerts = proof_setup(config, tmp_path, when='2026-10-10T12:00:00' if case == 'closed' else '2026-10-12T10:00:00')
    if case == 'live_client':
        c = replace(c, client_id=927481)
    if case == 'near_trigger':
        monkeypatch.setattr(simple, 'PROOF_PCT', .02)
    if case == 'not_flat':
        gw.pos['MNQ'] = 1
    if case == 'working':
        gw.working = [dict(symbol='MES', type='STP', client_id=1)]
    ok, rep = run(c, gw, clock, alerts, tmp_path)
    assert not ok and not gw.sent and not gw.cancelled and 'refused' in rep['why']


def test_proof_flags_inactive_and_10147_and_orphaned_children(config, tmp_path):
    c, gw, clock, alerts = proof_setup(config, tmp_path / 'a')
    gw.script[100 + 1] = lambda oid: (gw.st.__setitem__(oid, 'Inactive'), gw.on_status(oid, 'Inactive'))   # first stop child
    ok, rep = run(c, gw, clock, alerts, tmp_path / 'a')
    assert not ok and any('Inactive' in x for x in rep['flagged'])
    c, gw, clock, alerts = proof_setup(config, tmp_path / 'b')
    gw.script[100 + 4] = lambda oid: gw.on_error(oid, 10147, 'OrderId that needs to be cancelled is not found')
    ok, rep = run(c, gw, clock, alerts, tmp_path / 'b')
    assert not ok and any('10147' in x for x in rep['flagged']) and all(gw.st[o] == 'Cancelled' for n, o, b in gw.sent)
    c, gw, clock, alerts = proof_setup(config, tmp_path / 'c')
    gw.orphan_children = True
    ok, rep = run(c, gw, clock, alerts, tmp_path / 'c')
    assert not ok and any('children did not cancel' in x for x in rep['flagged']) and all(gw.st[o] == 'Cancelled' for n, o, b in gw.sent)


def test_proof_cleans_up_when_the_gateway_fails_midway(config, tmp_path):
    c, gw, clock, alerts = proof_setup(config, tmp_path)

    async def boom():
        raise RuntimeError('lost')
    gw.open_orders = boom
    with pytest.raises(RuntimeError):
        run(c, gw, clock, alerts, tmp_path)
    assert gw.sent and all(gw.st[o] == 'Cancelled' for n, o, b in gw.sent)
    f, = tmp_path.glob('proof-*.json')
    assert json.loads(f.read_text())['ok'] is False


# ---- re-review hardening ----

def flats(gw):
    return [x for x in gw.sent if x[2]['ref'].endswith('FLAT')]


def reject_stop(r, gw):
    stop = next(o for n, o, x in gw.sent if x['ref'].endswith('NQ-1-STOP') and x['side'] == -1)
    r.on_error(stop, 201, 'Order rejected')
    return stop


def test_defend_flattens_the_broker_quantity_never_a_stale_local_one(config, tmp_path):
    async def go(case):
        r, gw, now, alerts = make(config, tmp_path / case)
        await two_leg_fill(r, gw, now)
        fill_entry(r, gw, 'NQ', 1, 20010.)
        local = r.x['NQ'].pos
        if case == 'late_time_fill':
            gw.pos['MNQ'] = 0                                   # TIME exit filled at the broker; the callback has not arrived
        elif case == 'partial':
            gw.pos['MNQ'] = 3
        elif case == 'unknown':
            async def bad():
                return dict(own_positions=None, own_positions_error='timeout', working_orders=[])
            gw.sync = bad
        elif case == 'opposite':
            gw.pos['MNQ'] = -2
        reject_stop(r, gw)
        await spin()
        assert r.x['NQ'].off
        if case == 'same':
            f, = flats(gw)
            assert f[2]['side'] == -1 and f[2]['qty'] == local
        elif case == 'partial':
            f, = flats(gw)
            assert f[2]['side'] == -1 and f[2]['qty'] == 3
        else:
            assert not flats(gw) and r.x['NQ'].status == 'OFF'
            assert any(('NO FLATTEN SENT' if case == 'unknown' else 'no flatten') in a for a in alerts)
    for case in ('same', 'partial', 'late_time_fill', 'unknown', 'opposite'):
        asyncio.run(go(case))


def test_defend_cancels_every_non_flat_order_whatever_the_local_status(config, tmp_path):
    async def go():
        r, gw, now, _ = make(config, tmp_path)
        await two_leg_fill(r, gw, now)
        fill_entry(r, gw, 'NQ', 1, 20010.)
        mine = [o for n, o, x in gw.sent if n == 'NQ']
        for o in mine:
            gw.st[o] = 'Cancelled'                              # local view says dead; the broker may disagree
        gw.cancelled.clear()
        reject_stop(r, gw)
        await spin()
        assert set(mine) <= set(gw.cancelled)
    asyncio.run(go())


def test_locally_cancelled_stop_still_working_at_broker_is_not_flattened(config, tmp_path):
    async def go():
        r, gw, now, alerts = make(config, tmp_path)
        await two_leg_fill(r, gw, now)
        fill_entry(r, gw, 'NQ', 1, 20010.)
        long_stop = next(o for n, o, x in gw.sent if x['ref'].endswith('NQ-1-STOP') and x['side'] == -1)
        gw.st[long_stop] = 'Cancelled'
        gw.alive = {long_stop}
        r.on_status(long_stop, 'Cancelled')
        await spin()
        assert not flats(gw) and not any('STOP LOST' in a for a in alerts) and not r.x['NQ'].defended
        gw.alive = set()                                        # now the broker agrees it is gone
        r.on_status(long_stop, 'Cancelled')
        await spin()
        assert len(flats(gw)) == 1 and any('STOP LOST' in a for a in alerts)
    asyncio.run(go())


def test_unknown_status_after_reconnect_uses_cache_and_sync_marks_absent_orders_gone(config, tmp_path):
    async def go():
        r, gw, now, alerts = make(config, tmp_path)
        await two_leg_fill(r, gw, now)
        fill_entry(r, gw, 'NQ', 1, 20010.)
        await spin()
        fill_role(r, gw, 'NQ', 'STOP', 19990., oca_cancel=False)
        time_oid = next(o for n, o, x in gw.sent if x['ref'].endswith('NQ-1-TIME') and x['side'] == -1)
        r.orders[time_oid]['status'] = 'Submitted'
        del gw.st[time_oid]                                     # reconnect: the gateway no longer knows it
        assert gw.status(time_oid) == 'UNKNOWN' and r.st(time_oid) == 'Submitted'
        tick(r, gw, now, 20000., '09:50:00', names=('NQ',))
        await r.step()
        await r.step()
        assert r.s['NQ'].attempts == 1 and len([a for a in alerts if 're-park blocked' in a]) == 1
        gw.working = []                                         # broker's book has no such order
        await r.sync()
        assert r.orders[time_oid]['status'] == 'Gone' and r.st(time_oid) == 'Gone'
        await r.step()
        assert r.s['NQ'].attempts == 2
    asyncio.run(go())


def test_1130_skips_partially_filled_entry_and_alerts(config, tmp_path):
    async def go():
        r, gw, now, alerts = make(config, tmp_path)
        tick(r, gw, now, 20000., '09:30:00')
        await r.step()
        entries = [oid for oid, b in entry_orders(gw, 'NQ') + entry_orders(gw, 'ES')]
        gw.part = {entries[0]: 2}
        tick(r, gw, now, 20000., '11:30:01')
        await r.step()
        assert entries[0] not in gw.cancelled and set(entries[1:]) <= set(gw.cancelled)
        assert any('partially filled' in a and str(entries[0]) in a for a in alerts) and not any('STILL ALIVE' in a for a in alerts)
    asyncio.run(go())


def test_dry_run_without_client_id_never_takes_the_live_client(config):
    live = replace(config, client_id=927481)
    assert simple.effective_config(live, None, True).client_id == 927486
    assert simple.effective_config(live, None, False).client_id == 927481
    assert simple.effective_config(live, 927490, True).client_id == 927490


def test_paper_mode_is_tagged_and_uses_its_own_folder(config, tmp_path):
    assert simple.run_folder('live', False) == 'simple' and simple.run_folder('paper', False) == 'simple-paper'
    assert simple.run_folder('paper', True) == 'simple-dry' and simple.run_folder('live', True) == 'simple-dry'
    paper = replace(config, mode='paper')
    man = dict(score=50., markets={m.name: dict(prior_tr=40., prior_range_status='OK', skip_prior_range=False, prior_range_reason='t') for m in config.markets})
    r = Runner(paper, DAY, FakeGW(), man, tmp_path/'p', lambda *a: None, 750000.)
    r.s['NQ'].attempts = 2
    r.save()
    assert json.loads((tmp_path/'p'/'day.json').read_text())['mode'] == 'paper'
    assert Runner(paper, DAY, FakeGW(), man, tmp_path/'p', lambda *a: None, 750000.).s['NQ'].attempts == 2   # paper resumes paper
    assert Runner(config, DAY, FakeGW(), man, tmp_path/'p', lambda *a: None, 750000.).s['NQ'].attempts == 0   # live ignores it
    d = tmp_path/'2026-10-09-simple-paper'
    d.mkdir()
    (d/'day.json').write_text(json.dumps(dict(day='2026-10-09', mode='paper', markets={'NQ': dict(first_r=1.5), 'ES': dict(first_r=None)})))
    assert prior_r(tmp_path, DAY, ['NQ', 'ES'], 'paper')['NQ']['source'] == 'own_journal'
    assert prior_r(tmp_path, DAY, ['NQ', 'ES'], 'live')['NQ']['source'] != 'own_journal'


def test_startup_connect_retries_with_backoff_until_the_deadline():
    class G:
        def __init__(self, fail): self.fail, self.n = fail, 0

        async def connect(self, on_tick):
            self.n += 1
            if self.n <= self.fail:
                raise TimeoutError('API not ready')
    slept, alerts, t0 = [], [], at('09:20:00')

    async def sl(d): slept.append(d)
    g = G(3)
    asyncio.run(simple.connect_retry(g, None, alerts.append, at('09:27:00'), clock=lambda: t0, sleep=sl))
    assert g.n == 4 and slept == [2, 5, 10] and len(alerts) == 3
    g = G(99)
    with pytest.raises(TimeoutError):
        asyncio.run(simple.connect_retry(g, None, alerts.append, at('09:20:03'), clock=lambda: t0, sleep=sl))
    assert g.n == 2   # first retry (2s) fits, the second (5s) would pass the deadline
