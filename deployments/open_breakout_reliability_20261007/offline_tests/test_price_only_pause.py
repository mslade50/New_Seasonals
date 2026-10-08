"""Offline price/execution separation regressions; never connects or submits to a broker."""
import asyncio
from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace as NS

import pytest
from test_existing_open_breakout import config, setup_service, tick, at, manifest
from open_breakout.replay import SimBroker
from open_breakout.service import Service


def enabled(config):
    return replace(config, entry_order_type='stop_limit', allow_price_only_pause=True)


async def parked(config, tmp_path, broker_class=SimBroker):
    service, broker, store, now = setup_service(enabled(config), tmp_path, broker_class)
    await tick(service, broker, now, 20000, '09:30:00')
    await service.watchdog()
    assert not service.halted
    return service, broker, store, now


async def pause(service, now):
    now[0] = at('09:30:41')
    await service.watchdog()
    assert service.price_paused and not service.halted


def test_price_gap_keeps_entries_and_actual_fill_creates_exits(config, tmp_path):
    async def run():
        service, b, store, now = await parked(config, tmp_path)
        try:
            s = service.states['NQ']; entries = s.entry_orders.copy()
            budget = store.get('daily_reserved'); opening = s.opening
            await pause(service, now)
            assert all(b.status(oid) == 'Submitted' for oid in entries)
            buy = next(oid for oid in entries if b.orders[oid]['body']['side'] == 1)
            b.execute(buy, b.orders[buy]['body']['qty'], 20010.)
            await service.drain()
            assert not service.halted and s.qty and s.stop_order and s.time_order
            assert b.orders[s.stop_order]['body']['qty'] == s.qty
            assert b.status(s.stop_order) == b.status(s.time_order) == 'Submitted'
            assert not service.working_entries(s)
            assert s.opening == opening and store.get('daily_reserved') == budget and s.attempts == 1
            await service.watchdog()
            assert service.price_paused and not service.halted
        finally: store.close()
    asyncio.run(run())


def test_gap_resume_waits_for_fresh_quote_and_verified_watchdog(config, tmp_path):
    async def run():
        service, b, store, now = await parked(config, tmp_path)
        try:
            await pause(service, now); count = len(store.orders())
            service.tick('NQ', now[0], 20000.)
            await service.watchdog()
            assert service.price_paused  # Signal returned; micro quote is still old.
            b.update_quote('NQ', 19999.75, 20000., now[0])
            assert service.price_paused  # A quote callback alone is not recovery proof.
            await service.watchdog()
            assert not service.price_paused and not service.halted
            await service.watchdog()
            assert len(store.orders()) == count
            events = [r[0] for r in store.db.execute('SELECT kind FROM events')]
            assert events.count('PRICE_FEED_PAUSED') == events.count('PRICE_FEED_RECOVERED') == 1
        finally: store.close()
    asyncio.run(run())


@pytest.mark.parametrize('cutoff', [False, True])
def test_reparking_after_pause_obeys_cutoff_and_consumed_attempt_budget(config, tmp_path, cutoff):
    async def run():
        service, b, store, now = await parked(config, tmp_path)
        try:
            s = service.states['NQ']; entries = s.entry_orders.copy(); budget = store.get('daily_reserved')
            await pause(service, now)
            # Only the existing cutoff path cancels; the price pause leaves entries intact.
            if cutoff:
                # Maintain actual feed callbacks through the session so cutoff is the cause.
                while now[0] + timedelta(seconds=500) < at('11:29:59'):
                    now[0] += timedelta(seconds=500)
                    service.tick('NQ', now[0], 20000.)
                    b.update_quote('NQ', 19999.75, 20000., now[0])
                now[0] = at('11:29:59')
                service.tick('NQ', now[0], 20000.)
                b.update_quote('NQ', 19999.75, 20000., now[0])
                now[0] = at('11:30:01'); await service.watchdog(); await service.drain()
                assert all(b.status(oid) == 'Cancelled' for oid in entries)
            else:
                # Model a completed exit/cycle with entry cancellation already confirmed.
                await service.cancel_resting_entries()
                assert s.phase == 'FLAT'
            service.tick('NQ', now[0], 20000.)
            b.update_quote('NQ', 19999.75, 20000., now[0])
            await service.watchdog()
            assert not service.price_paused and not service.halted
            await tick(service, b, now, 20000., '11:30:02' if cutoff else '09:30:42')
            assert s.opening == 20000.
            if cutoff:
                assert s.attempts == 1 and len(store.orders()) == 2 and store.get('daily_reserved') == budget
            else:
                assert s.attempts == 2 and store.get('daily_reserved') > budget
        finally: store.close()
    asyncio.run(run())


@pytest.mark.parametrize('failure', ['missing', 'cached', 'aged', 'slow', 'disconnected', 'unhealthy', 'unknown', 'ownership'])
def test_missing_or_invalid_execution_proof_halts_and_cancels_entries(config, tmp_path, failure):
    class BadProof(SimBroker):
        fail = False
        async def snapshot(self):
            result = await super().snapshot()
            if not self.fail: return result
            proof = result['execution_health']
            if failure == 'missing': result.pop('execution_health')
            elif failure == 'cached': proof['fresh_executions'] = False
            elif failure == 'aged': proof['received_at'] = (self.clock() - timedelta(seconds=6)).isoformat()
            elif failure == 'slow': proof['roundtrip_seconds'] = 6.
            elif failure == 'disconnected': proof['connected'] = False
            elif failure == 'unhealthy': proof['healthy'] = False
            elif failure == 'unknown': result['own'] = None
            else: proof['client_id'] += 1
            return result
    async def run():
        service, b, store, now = await parked(config, tmp_path, BadProof)
        try:
            b.fail = True; now[0] = at('09:30:41')
            await service.watchdog(); await service.drain()
            assert service.halted and 'Execution health unverified' in store.get('halt_reason')
            assert not service.working_entries(service.states['NQ'])
            # Fresh prices must never clear an execution/protection/manual halt.
            b.fail = False; await tick(service, b, now, 20000., '09:30:42'); await service.watchdog()
            assert service.halted
        finally: store.close()
    asyncio.run(run())


def test_execution_timeout_during_price_pause_halts_without_dropping_exits(config, tmp_path):
    class Timeout(SimBroker):
        fail = False
        async def snapshot(self):
            if self.fail: raise TimeoutError('order/execution responses absent')
            return await super().snapshot()
    async def run():
        service, b, store, now = await parked(config, tmp_path, Timeout)
        try:
            s = service.states['NQ']; buy = s.entry_orders[0]
            b.execute(buy, 1, 20010.); await service.drain()
            exits = [s.stop_order, s.time_order]
            b.fail = True; now[0] = at('09:30:41')
            await service.watchdog(); await service.drain()
            assert service.halted and all(b.status(oid) == 'Submitted' for oid in exits)
        finally: store.close()
    asyncio.run(run())


def test_partials_and_duplicate_execution_during_price_pause(config, tmp_path):
    async def run():
        service, b, store, now = await parked(config, tmp_path)
        try:
            await pause(service, now)
            s = service.states['NQ']; buy = s.entry_orders[0]; cid = service.markets['NQ'].execution.con_id
            for i, price in enumerate([20010., 20010.25]):
                b.record(cid, 1, b.orders[buy]['body']['ref'], f'PART-{i}')
                service.fill(buy, f'PART-{i}', 1, price)
            await service.drain()
            stop, timed = s.stop_order, s.time_order
            service.fill(buy, 'PART-1', 1, 20010.25)
            await service.drain()
            assert s.qty == 2 and s.entry == 20010.125 and s.stop_order == stop and s.time_order == timed
            assert b.orders[stop]['body']['qty'] == b.orders[timed]['body']['qty'] == 2
            assert not service.halted and b.status(buy) == 'Cancelled'
        finally: store.close()
    asyncio.run(run())


def test_cancel_fill_race_during_cutoff_still_protects_and_halts(config, tmp_path):
    class Race(SimBroker):
        race = False
        def cancel(self, oid):
            if self.race and self.orders[oid]['body']['side'] == 1 and self.orders[oid]['body']['kind'] == 'STP LMT':
                self.race = False
                self.execute(oid, 1, 20010.)
            else: super().cancel(oid)
    async def run():
        service, b, store, now = await parked(config, tmp_path, Race)
        try:
            await pause(service, now); b.race = True; now[0] = at('11:30:00')
            await service.watchdog(); await service.drain()
            s = service.states['NQ']
            assert service.halted and s.qty == 1 and s.stop_order and s.time_order
            assert b.status(s.stop_order) == b.status(s.time_order) == 'Submitted'
        finally: store.close()
    asyncio.run(run())


def test_stop_creation_failure_during_price_pause_remains_fatal(config, tmp_path):
    class Reject(SimBroker):
        def send(self, oid, market, body):
            if body['kind'] == 'STP': raise RuntimeError('stop refused before transmission')
            super().send(oid, market, body)
    async def run():
        service, b, store, now = await parked(config, tmp_path, Reject)
        try:
            await pause(service, now); s = service.states['NQ']
            b.execute(s.entry_orders[0], 1, 20010.); await service.drain()
            assert service.halted and s.qty == 1 and not s.time_order
            assert 'EXECUTION_OR_STOP' in store.get('halt_reason')
        finally: store.close()
    asyncio.run(run())


def test_restart_is_not_price_recovery(config, tmp_path):
    async def run():
        service, b, store, now = await parked(config, tmp_path)
        try:
            await pause(service, now); count = len(store.orders())
            restarted = Service(service.config, manifest(service.config), store, b, lambda: now[0])
            await restarted.drain()
            assert restarted.halted and len(store.orders()) == count
            assert 'RESTART_REQUIRES_READ_ONLY_RECONCILIATION' in store.get('halt_reason')
        finally: store.close()
    asyncio.run(run())


def test_recovery_never_refunds_risk_or_bypasses_three_attempt_limit(config, tmp_path):
    async def run():
        service, b, store, now = await parked(config, tmp_path)
        try:
            await pause(service, now); await service.cancel_resting_entries()
            s = service.states['NQ']; s.attempts = 3
            budget = store.get('daily_reserved')
            await tick(service, b, now, 20000., '09:30:42'); await service.watchdog()
            await tick(service, b, now, 20000., '09:30:43')
            assert not service.price_paused and not service.halted and s.attempts == 3
            assert len(store.orders()) == 2 and store.get('daily_reserved') == budget
        finally: store.close()
    asyncio.run(run())


def test_adapter_proof_requires_fresh_execution_request_every_snapshot(config):
    async def run():
        from open_breakout.ibkr import IBKR
        adapter = IBKR(enabled(config)); calls = []
        async def empty(): return []
        async def executions(filt): calls.append(filt); return []
        adapter.ib = NS(client=NS(clientId=config.client_id),
                        reqAllOpenOrdersAsync=empty, reqPositionsAsync=empty, reqExecutionsAsync=executions,
                        isConnected=lambda: True)
        first = await adapter.snapshot(); second = await adapter.snapshot()
        assert len(calls) == 2  # Cached attributed positions are not execution-health evidence.
        assert first['execution_health']['fresh_executions'] and second['execution_health']['fresh_orders']
        async def unavailable(filt): raise TimeoutError('execution responses absent')
        adapter.ib.reqExecutionsAsync = unavailable
        result = await adapter.snapshot()
        assert result['own'] is None and not result['execution_health']['fresh_executions']
    asyncio.run(run())


def test_opt_in_default_and_validation(config, tmp_path):
    import json
    from open_breakout.config import Config
    assert config.allow_price_only_pause is False
    raw = json.loads((tmp_path/'config.json').read_text())
    for value in ('true', 1, None):
        raw['allow_price_only_pause'] = value
        (tmp_path/'bad.json').write_text(json.dumps(raw))
        with pytest.raises(ValueError): Config.load(tmp_path/'bad.json')
    raw['allow_price_only_pause'] = True
    (tmp_path/'bad.json').write_text(json.dumps(raw))
    with pytest.raises(ValueError): Config.load(tmp_path/'bad.json')
    raw['entry_order_type'] = 'stop_limit'
    (tmp_path/'enabled.json').write_text(json.dumps(raw))
    approved = Config.load(tmp_path/'enabled.json')
    assert approved.allow_price_only_pause and approved.fingerprint != config.fingerprint


def test_farm_warning_keeps_price_pause_but_connection_loss_is_fatal(config, tmp_path):
    async def run():
        from open_breakout.ibkr import IBKR
        service, b, store, now = await parked(config, tmp_path)
        try:
            await pause(service, now)
            adapter = IBKR(service.config)
            adapter.halt_callback = service.halt
            adapter.diagnostic_callback = service.broker_diagnostic
            for _ in range(4): adapter._error(-1, 2103, 'market data farm unavailable', None)
            assert adapter.healthy and not service.halted
            assert store.db.execute("SELECT COUNT(*) FROM events WHERE kind='BROKER_DIAGNOSTIC'").fetchone()[0] == 1
            adapter._error(-1, 1100, 'connectivity lost', None)
            await service.drain()
            assert service.halted and not service.working_entries(service.states['NQ'])
        finally: store.close()
    asyncio.run(run())


def test_explicit_stop_rejection_during_pause_retains_emergency_guards(config, tmp_path):
    async def run():
        service, b, store, now = await parked(config, tmp_path)
        try:
            await pause(service, now); s = service.states['NQ']
            b.execute(s.entry_orders[0], 1, 20010.); await service.drain()
            b.error_codes[s.stop_order] = 201
            service.order_error(s.stop_order, 201, 'protective stop rejected')
            await service.drain()
            assert service.halted and s.qty == 0 and 'NQ' in service.flattened
            count = len(store.orders())
            service.order_error(s.stop_order, 201, 'duplicate rejection')
            await service.drain()
            assert len(store.orders()) == count  # No speculative second flatten.
        finally: store.close()
    asyncio.run(run())


def test_recovery_does_not_bypass_pooled_risk_or_manual_halt(config, tmp_path):
    async def run():
        service, b, store, now = await parked(config, tmp_path)
        try:
            await pause(service, now); await service.cancel_resting_entries()
            maximum = service.config.shadow_equity * service.config.max_daily_risk_bps / 10000
            store.set('daily_reserved', maximum)
            await tick(service, b, now, 20000., '09:30:42'); await service.watchdog()
            await tick(service, b, now, 20000., '09:30:43')
            assert not service.price_paused and len(store.orders()) == 2 and service.states['NQ'].attempts == 1
            service.halt('MANUAL_HALT'); await service.drain()
            await tick(service, b, now, 20000., '09:30:44'); await service.watchdog()
            assert service.halted and store.get('daily_reserved') == maximum
        finally: store.close()
    asyncio.run(run())


@pytest.mark.parametrize('age,expired', [(599.999, False), (600., True), (600.001, True)])
@pytest.mark.parametrize('missing', ['trade', 'quote', 'both'])
def test_exact_ten_minute_boundary_per_required_feed(config, tmp_path, age, expired, missing):
    async def run():
        service, b, store, now = await parked(config, tmp_path)
        try:
            # Independent virtual monotonic time also permits wall-clock correction tests.
            elapsed = [0.]
            service.monotonic = lambda: elapsed[0]
            for value in service.price_observed.values(): value['observed_monotonic'] = 0.
            service.price_started['NQ'] = 0.
            elapsed[0] = age; now[0] += timedelta(seconds=age)
            if missing == 'trade': b.update_quote('NQ', 19999.75, 20000., now[0])
            if missing == 'quote': service.tick('NQ', now[0], 20000.)
            await service.watchdog(); await service.drain()
            assert service.halted is expired
            entries = service.states['NQ'].entry_orders
            assert all(b.status(oid) == ('Cancelled' if expired else 'Submitted') for oid in entries)
            if expired:
                assert 'PRICE_OUTAGE_GRACE_EXPIRED' in store.get('halt_reason')
                body = store.db.execute("SELECT body FROM events WHERE kind='PRICE_OUTAGE_GRACE_EXPIRED'").fetchone()
                assert body
            else: assert service.price_paused
        finally: store.close()
    asyncio.run(run())


def test_intermittent_valid_prices_reset_only_their_own_timers(config, tmp_path):
    async def run():
        service, b, store, now = await parked(config, tmp_path)
        try:
            start = now[0]
            now[0] = start + timedelta(seconds=599)
            service.tick('NQ', now[0], 20000.)
            b.update_quote('NQ', 19999.75, 20000., now[0])
            await service.watchdog()
            assert not service.halted and not service.price_paused
            now[0] = start + timedelta(seconds=600)
            await service.watchdog()
            assert not service.halted
            now[0] = start + timedelta(seconds=1198.999)
            await service.watchdog()
            assert service.price_paused and not service.halted
            now[0] = start + timedelta(seconds=1199)
            await service.watchdog(); await service.drain()
            assert service.halted
        finally: store.close()
    asyncio.run(run())


def test_other_market_and_farm_recovery_warning_do_not_refresh_stale_market(config, tmp_path):
    async def run():
        from open_breakout.ibkr import IBKR
        service, b, store, now = await parked(config, tmp_path)
        try:
            service.tick('ES', now[0], 5000.)
            b.update_quote('ES', 4999.75, 5000., now[0]); await service.drain()
            start = now[0]
            adapter = IBKR(service.config)
            adapter.halt_callback = service.halt; adapter.diagnostic_callback = service.broker_diagnostic
            now[0] = start + timedelta(seconds=599)
            service.tick('NQ', now[0], 20000.); b.update_quote('NQ', 19999.75, 20000., now[0])
            adapter._error(-1, 2104, 'market data farm connection is OK', None)
            await service.watchdog()
            assert service.price_paused and not service.halted
            now[0] = start + timedelta(seconds=600)
            await service.watchdog(); await service.drain()
            assert service.halted and store.get('halt_reason').endswith('ES:quote,ES:trade')
            assert not service.working_entries(service.states['NQ'])
            assert not service.working_entries(service.states['ES'])
        finally: store.close()
    asyncio.run(run())


def test_duplicate_backdated_invalid_callbacks_and_synthetic_refresh_cannot_reset_grace(config, tmp_path):
    async def run():
        service, b, store, now = await parked(config, tmp_path)
        try:
            start = now[0]; elapsed = [0.]
            service.monotonic = lambda: elapsed[0]
            for value in service.price_observed.values(): value['observed_monotonic'] = 0.
            service.price_started['NQ'] = 0.
            elapsed[0] = 599.; now[0] = start + timedelta(seconds=599)
            service.tick('NQ', start, 20000.)  # Delayed or duplicate source timestamp.
            b.update_quote('NQ', 19999.75, 20000., start)
            service.price_quote('NQ', float('nan'), 20000., now[0])
            service.price_quote('NQ', 20001., 20000., now[0])
            # Even a synthetic fresh cached timestamp and backwards wall clock are insufficient.
            now[0] = start + timedelta(seconds=1)
            b.quotes['NQ'] = (19999.75, 20000., now[0])
            service.states['NQ'].last_timestamp = now[0].isoformat()
            await service.watchdog()
            assert service.price_paused and not service.halted
            elapsed[0] = 600.; await service.watchdog(); await service.drain()
            assert service.halted
        finally: store.close()
    asyncio.run(run())


def test_never_seen_quote_age_is_not_reset_by_trade_updates(config, tmp_path):
    async def run():
        service, b, store, now = await parked(config, tmp_path)
        try:
            start = now[0]
            del service.price_observed['NQ:quote']
            now[0] = start + timedelta(seconds=599)
            service.tick('NQ', now[0], 20000.); await service.watchdog()
            assert service.price_paused and not service.halted
            now[0] = start + timedelta(seconds=600)
            await service.watchdog(); await service.drain()
            assert service.halted and store.get('halt_reason').endswith('NQ:quote')
        finally: store.close()
    asyncio.run(run())


def test_price_grace_cancel_fill_race_retains_accepted_protection(config, tmp_path):
    class Race(SimBroker):
        race = False
        def cancel(self, oid):
            if self.race and self.orders[oid]['body']['kind'] == 'STP LMT':
                self.race = False; self.execute(oid, 1, 20010.)
            else: super().cancel(oid)
    async def run():
        service, b, store, now = await parked(config, tmp_path, Race)
        try:
            b.race = True; now[0] += timedelta(seconds=600)
            await service.watchdog(); await service.drain()
            s = service.states['NQ']
            assert service.halted and s.qty == 1
            assert b.status(s.stop_order) == b.status(s.time_order) == 'Submitted'
            # Late fresh prices cannot clear the grace-expiry halt or duplicate protection.
            count = len(store.orders()); now[0] += timedelta(seconds=1)
            service.tick('NQ', now[0], 20000.); b.update_quote('NQ', 19999.75, 20000., now[0])
            await service.watchdog(); await service.drain()
            assert service.halted and len(store.orders()) == count
        finally: store.close()
    asyncio.run(run())


def test_owner_selected_grace_cannot_be_configured_below_ten_minutes(config, tmp_path):
    import json
    from open_breakout.config import Config
    raw = json.loads((tmp_path/'config.json').read_text())
    raw.update(entry_order_type='stop_limit', allow_price_only_pause=True)
    for value in (0, 30, 599.999, True, float('nan')):
        raw['price_outage_cancel_seconds'] = value
        (tmp_path/'grace.json').write_text(json.dumps(raw))
        with pytest.raises(ValueError): Config.load(tmp_path/'grace.json')
    for value in (600, 600.001):
        raw['price_outage_cancel_seconds'] = value
        (tmp_path/'grace.json').write_text(json.dumps(raw))
        assert Config.load(tmp_path/'grace.json').price_outage_cancel_seconds == value


def test_short_signal_freshness_is_rechecked_after_margin_await(config, tmp_path):
    class SlowMargin(SimBroker):
        async def check_margin(self, market, plan, equity):
            now[0] += timedelta(seconds=4)
            self.update_quote('NQ', 19999.75, 20000., now[0])
    async def run():
        nonlocal now
        service, b, store, now = setup_service(enabled(config), tmp_path, SlowMargin)
        try:
            await tick(service, b, now, 20000., '09:30:00')
            assert not store.orders() and service.price_paused and not service.halted
            assert service.states['NQ'].attempts == 0 and not store.get('daily_reserved', 0.)
        finally: store.close()
    now = None
    asyncio.run(run())


def test_paused_price_updates_do_not_arm_entry_signals(config, tmp_path):
    async def run():
        service, b, store, now = await parked(config, tmp_path)
        try:
            await pause(service, now); await service.cancel_resting_entries()
            s = service.states['NQ']
            assert not s.long_armed and not s.short_armed
            service.tick('NQ', now[0], 20000.)
            assert not s.long_armed and not s.short_armed and s.previous == 20000.
            b.update_quote('NQ', 19999.75, 20000., now[0])
            await service.watchdog()
            assert not service.price_paused and len(store.orders()) == 2
            await tick(service, b, now, 20000., '09:30:42')
            assert s.attempts == 2
        finally: store.close()
    asyncio.run(run())
