"""Minimal Open Breakout runner: one pass, no halts. Checks flat at start, builds the day's inputs, parks an OCA pair
of STP LMT entry brackets (stop + 15:55 MKT children held at the broker) after the 09:30 open, re-parks after a
close before 11:30 (max 3 attempts per market), flattens on a rejected child, reconnects without halting."""
from __future__ import annotations
import asyncio
import json
import math
from dataclasses import replace
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace as NS
import pandas as pd
from .alerts import Alerts, webhook_url
from .config import Config, LIVE_HARD_MAX_CONTRACTS
from .inputs import build_manifest, calendar_dates
from .service import STRATEGY_REF, exec_key, prior_range_skip
from .standby import book_view, fetch_range_bars
from .store import prior_day_r
from .strategy import NY, State, aware, on_price, size_order, snap

ROOT = Path(__file__).resolve().parents[1]
RUNS, BUILD = ROOT/'artifacts'/'open_breakout_runs', ROOT/'artifacts'/'open_breakout_build'
LIVE_CONFIG = RUNS/'config-20260925-live.json'
ENTRY_END, LATEST_START, WATCH_END = time(11,30), time(9,29,30), time(16,10)
DEAD = {'Filled','Cancelled','ApiCancelled','Inactive','Gone'}   # Gone: absent from the broker's working orders after a re-sync
ACKED = {'PreSubmitted','Submitted','Filled'}
DRY_CLIENT = 927486
REJECT = {110,200,201,203,321}   # 10147/10148/161 answer our own cancels of OCA-killed orders: not rejects


def find_risk(day, build=BUILD, any_session=False):
    """Newest risk_*/refresh.json for the session (no risk_error, has legacy_score, parquet present), as launch-live does."""
    for d in sorted(Path(build).glob('risk_*'), reverse=True):
        try:
            info = json.loads((d/'refresh.json').read_text(encoding='utf-8-sig'))
        except Exception:
            continue
        if (not any_session and info.get('session') != day) or 'risk_error' in info or 'legacy_score' not in info:
            continue
        if (ROOT/info['path']).exists():
            return ROOT/info['path'], info
    return None, None


def run_folder(mode, dry):
    return 'simple-dry' if dry else 'simple' if mode == 'live' else f'simple-{mode}'


def effective_config(config, client_id, dry):
    """--client-id wins; a dry/plan run without one uses DRY_CLIENT so it never takes the live client id."""
    return replace(config, client_id=client_id or (DRY_CLIENT if dry else config.client_id))


async def connect_retry(gw, on_tick, alert, deadline, clock=None, sleep=asyncio.sleep, delays=(2, 5, 10, 15)):
    """Startup connect: the port can be up before the API is ready, so retry with backoff until `deadline`, then raise."""
    clock = clock or (lambda: datetime.now(timezone.utc))
    for i in range(1000):
        try:
            return await gw.connect(on_tick)
        except Exception as exc:
            d = delays[min(i, len(delays)-1)]
            if clock()+timedelta(seconds=d) > deadline:
                raise
            alert(f'connect failed ({type(exc).__name__}: {exc}); retry in {d}s')
            await sleep(d)


def prior_r(root, day, names, mode):
    """Prior session's R per market: this runner's day.json first, then the old shadow/live journals (store.prior_day_r).
    A missing value is None and never skips a market. Own value is the FIRST attempt's single-trade R (what big_win_r was
    calibrated on) from a live-mode day.json; a market without one (no trade, or range-skipped) uses the old journals."""
    previous, _ = calendar_dates(day)
    f = Path(root)/f'{previous}-{run_folder(mode, False)}'/'day.json'
    old = prior_day_r(Path(root), previous, 'live' if mode == 'live' else 'shadow', names)
    try:
        d = json.loads(f.read_text(encoding='utf-8'))
        if d['day'] != previous or d.get('mode') != mode or not set(names) <= set(d['markets']):
            return old
        got = {}
        for n in names:
            r = d['markets'][n].get('first_r')
            got[n] = dict(r=r, source='own_journal', note=f'{f.parent.name}: first attempt R') if r is not None else old[n]
        return got
    except Exception:
        return old


def start_problems(v):
    """Reasons not to trade today from a book_view: unknown/non-flat own position, or any working OpenBreakout order."""
    bad = ([f'own position unknown: {v["own_positions_error"]}'] if v['own_positions'] is None
           else [f'own position {k}={q}' for k, q in v['own_positions'].items() if q])
    return bad + [f'working OpenBreakout order {w["symbol"]} {w["type"]} client {w["client_id"]}' for w in v['working_orders']]


class Gateway:
    """Thin IBKR adapter: connect, quotes, order send/cancel, own book. Dry run never reaches placeOrder."""
    def __init__(self, config, session, dry=False):
        self.config, self.session, self.dry, self.t = config, session, dry, None
        self.markets = {m.name: m for m in config.markets}
        self.on_fill = self.on_status = self.on_error = lambda *a: None

    async def connect(self, on_tick):
        from .ibkr import IBKR
        if self.t:
            self.t.close()
        t = self.t = IBKR(self.config, session=self.session)
        t.fill_callback = lambda *a: self.on_fill(*a)
        t.status_callback = lambda *a: self.on_status(*a)
        t.order_error_callback = lambda *a: self.on_error(*a)
        if self.dry:
            def deny(*a, **k):
                raise PermissionError('dry-run: broker orders denied')
            t.ib.client.placeOrder = deny
        await asyncio.wait_for(t.connect(), 30)
        t.subscribe(on_tick, lambda e: None)

    def connected(self): return bool(self.t and self.t.ib.isConnected())
    def next_id(self): return self.t.next_id()
    def quote(self, name): return self.t.quotes[name]
    async def equity(self): return await self.t.equity()
    async def sync(self): return await book_view(self.t)

    def trade(self, oid):
        t = self.t
        return t.trades.get(oid) or next((x for x in t.ib.trades() if x.order.orderId == oid and x.order.clientId == self.config.client_id), None)

    def status(self, oid):
        tr = self.trade(oid)
        return tr.orderStatus.status if tr else 'UNKNOWN'

    def filled(self, oid):
        tr = self.trade(oid)
        return tr.orderStatus.filled if tr else 0

    def cancel(self, oid):
        tr = self.trade(oid)
        if tr:
            self.t.ib.cancelOrder(tr.order)

    async def open_orders(self):
        """This client's orders as the broker reports them (openOrders), with their OCA groups."""
        trs = await asyncio.wait_for(self.t.ib.reqAllOpenOrdersAsync(), 10)
        return [dict(id=x.order.orderId, ref=x.order.orderRef, oca=x.order.ocaGroup, oca_type=x.order.ocaType, parent=x.order.parentId,
                     kind=x.order.orderType, tif=x.order.tif, till=x.order.goodTillDate, after=x.order.goodAfterTime,
                     status=x.orderStatus.status) for x in trs if x.order.clientId == self.config.client_id]

    def send(self, name, oid, body):
        self.config.authorize(self.session)
        if not 0 < body['qty'] <= LIVE_HARD_MAX_CONTRACTS or body['qty'] != int(body['qty']):
            raise PermissionError(f'quantity {body["qty"]} outside [1, {LIVE_HARD_MAX_CONTRACTS}]')
        if body.get('account') != self.config.account:
            raise PermissionError('Order account mismatch')
        if self.dry:
            print(f'  DRY-RUN would place #{oid} {body["kind"]} {"BUY" if body["side"]==1 else "SELL"} {body["qty"]} {self.markets[name].execution.symbol}'
                  f' stop={body.get("stop")} limit={body.get("limit")} tif={body["tif"]} till={body.get("good_till")} after={body.get("good_after")}'
                  f' oca={body.get("oca")}/{body.get("oca_type")} parent={body.get("parent", 0)} transmit={body.get("transmit", True)}'
                  f' ref={body["ref"]}', flush=True)
            return
        o = self.t._order(oid, body)
        o.parentId, o.transmit = body.get('parent', 0), body.get('transmit', True)
        self.t.trades[oid] = self.t.ib.placeOrder(self.t.contracts[self.markets[name].execution.con_id], o)


def place_bracket(gw, cfg, m, day, name, a, p, orders, good_till=None):
    """One parent STP LMT entry (GTD, OCA type 1 with the other side) + STOP and goodAfterTime MKT children in their own OCA.
    Shared by live parking and simple-proof; good_till is overridden only by the proof (a 11:30 that may already be past)."""
    sd, d = p['side'], day.replace('-', '')
    sym, q, cid = m.execution.symbol, p['qty'], cfg.client_id
    ref = lambda role: f'{sym}|{"BUY" if sd == 1 else "SELL"}|{STRATEGY_REF}|{day}|{name}-{a}-{role}'
    eg, xg = f'OB-ENTRY-{cid}-{day}-{name}-{a}', f'OB-EXIT-{cid}-{day}-{name}-{a}-{"L" if sd == 1 else "S"}'
    pid, sid, tid = (gw.next_id() for _ in range(3))
    rows = [(pid, 'ENTRY', dict(kind='STP LMT', side=sd, qty=q, stop=p['trigger'], limit=p['limit'], tif='GTD', trigger_method=2,
                                good_till=good_till or f'{d} 11:30:00 America/New_York', oca=eg, oca_type=1, transmit=False)),
            (sid, 'STOP', dict(kind='STP', side=-sd, qty=q, stop=p['stop_px'], tif='GTC', oca=xg, oca_type=1, parent=pid, transmit=False)),
            (tid, 'TIME', dict(kind='MKT', side=-sd, qty=q, tif='GTC', good_after=f'{d} 15:55:00 America/New_York', oca=xg, oca_type=1,
                               parent=pid, transmit=True))]
    for oid, role, body in rows:
        body.update(account=cfg.account, ref=ref(role))
        orders[oid] = dict(market=name, role=role, side=body['side'], qty=q, attempt=a, stop=p['stop_px'], status='')
        gw.send(name, oid, body)
    return pid


class Runner:
    def __init__(self, config, day, gw, manifest, root, alert, equity, clock=None, sleep=asyncio.sleep, dry=False):
        self.cfg, self.day, self.gw, self.man, self.root, self.alert = config, day, gw, manifest, Path(root), alert
        self.equity, self.sleep, self.dry = equity, sleep, dry
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.mk = {m.name: m for m in config.markets}
        self.s = {n: State(day, n, manifest['markets'][n]['prior_tr'], manifest['score']) for n in self.mk}
        self.x = {n: NS(status='IDLE', pos=0, risk=0., off=False, defended=False, cur=0, recs={}, trades=[], fills=[]) for n in self.mk}
        self.orders, self.seen, self.tasks, self.reserved = {}, set(), set(), 0.
        self.synced, self.noted_end, self.last_note, self.alerted = False, False, {}, set()
        for n, x in self.x.items():
            if prior_range_skip(config, manifest['markets'][n]):
                x.status = 'SKIP'
                alert(f'{n} not armed: {manifest["markets"][n].get("prior_range_reason") or "prior-range filter without an OK decision"}')
        self.restore()
        gw.on_fill, gw.on_status, gw.on_error = self.on_fill, self.on_status, self.on_error

    def restore(self):
        try:
            d = json.loads((self.root/'day.json').read_text(encoding='utf-8'))
            if d['day'] != self.day or d.get('mode') != self.cfg.mode or self.dry:
                return
            self.reserved = d['reserved']
            for n, v in d['markets'].items():
                self.s[n].attempts, self.x[n].trades, self.x[n].cur = v['attempts'], v['trades'], v['attempts']
            self.alert(f'resumed attempts from day.json: { {n: s.attempts for n, s in self.s.items()} }')
        except Exception:
            pass

    def save(self):
        # first_r: attempt 1's single-trade R (None if it never closed). A range-skipped market has no trade, so first_r is None
        # (the unfiltered would-be R needs a minute-bar replay of the day; the runner has only ticks) and prior_r falls back to the journals.
        first = lambda x: next((t['r'] for t in x.trades if t['attempt'] == 1), None)
        mk = {n: dict(attempts=self.s[n].attempts, r=round(sum(t['r'] for t in x.trades if t['r'] is not None), 6) if x.trades else None,
                      first_r=first(x), trades=x.trades, fills=x.fills, status=x.status) for n, x in self.x.items()}
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.root/'day.json.tmp'
        tmp.write_text(json.dumps(dict(day=self.day, mode='dry' if self.dry else self.cfg.mode, reserved=self.reserved, markets=mk), indent=1), encoding='utf-8')
        tmp.replace(self.root/'day.json')

    def spawn(self, coro):
        t = asyncio.ensure_future(coro)
        self.tasks.add(t)
        t.add_done_callback(self.tasks.discard)

    def st(self, oid):
        """Order status; after a reconnect the gateway may not know the order (UNKNOWN), so fall back to the last cached status."""
        s = self.gw.status(oid)
        return (self.orders[oid]['status'] or s) if s == 'UNKNOWN' and oid in self.orders else s

    def alert_once(self, key, text):
        if key not in self.alerted:
            self.alerted.add(key)
            self.alert(text)

    def note(self, name, text):
        if self.last_note.get(name) != text:
            self.last_note[name] = text
            print(f'{name}: {text}', flush=True)

    def on_tick(self, name, ts, price):
        s = self.s[name]
        try:
            on_price(s, ts, price, self.mk[name].signal.tick, self.clock(), stale_seconds=self.cfg.stale_seconds, open_delay=self.cfg.max_open_delay_seconds)
        except ValueError as exc:
            self.note(name, f'bad tick ignored: {exc}')

    def fresh(self, name):
        s = self.s[name]
        bid, ask, ts = self.gw.quote(name)
        now = self.clock()
        if not -1 <= (now-aware(ts)).total_seconds() <= self.cfg.stale_seconds:
            raise ValueError('Stale execution quote')
        if not s.last_timestamp or (now-aware(s.last_timestamp)).total_seconds() > self.cfg.stale_seconds:
            raise ValueError('Stale signal trade')
        if not 0 < bid <= ask or (ask-bid)/((ask+bid)/2) > .002:
            raise ValueError('Spread exceeds 20bp entry guard')
        if abs((bid+ask)/2-s.previous) > max(s.distance, .002*s.previous):
            raise ValueError('Signal/execution price dislocation')
        return bid, ask

    def bracket(self, name, a, p):
        return place_bracket(self.gw, self.cfg, self.mk[name], self.day, name, a, p, self.orders)

    async def park(self, name):
        s, m, x, cfg, eq = self.s[name], self.mk[name], self.x[name], self.cfg, self.equity
        try:
            bid, ask = self.fresh(name)
        except (KeyError, ValueError) as exc:
            return self.note(name, f'not parking yet: {exc}')
        room = min(eq*cfg.max_daily_risk_bps/1e4-self.reserved,
                   eq*cfg.max_open_risk_bps/1e4-sum(y.risk for y in self.x.values() if y.status in ('WORKING', 'OPEN')))
        tick, sides = m.execution.tick, [1] + ([-1] if s.score >= 20 else [])
        trig = {sd: snap(s.opening+sd*s.distance, tick, sd == 1) for sd in sides}
        inside = {sd: (max(s.previous, ask) < trig[sd] if sd == 1 else min(s.previous, bid) > trig[sd]) for sd in sides}
        if s.attempts and (x.pos or any(o['market'] == name and self.st(i) not in DEAD for i, o in self.orders.items())):
            self.alert_once(f'repark-{name}-{s.attempts}', f'{name} re-park blocked: position or earlier-attempt orders still alive at the broker (CHECK TWS if unexpected)')
            return self.note(name, 'not re-parking: position or earlier-attempt orders still alive at the broker')
        if s.attempts and not all(inside.values()):
            return self.note(name, 'flat, waiting for price back inside both triggers')
        plans = []
        for sd in sides:
            if not inside[sd]:
                continue
            p = size_order(s, m, sd, trig[sd], trig[sd], eq, cfg)
            q = min(p['qty'], max(0, math.floor(room/p['per_contract']+1e-9)))
            if q:
                plans.append({**p, 'qty': q, 'trigger': trig[sd], 'stop_px': snap(trig[sd]-sd*s.distance, tick, sd == -1)})
        if not plans:
            if any(inside.values()):
                x.status = 'DONE'
                self.alert(f'{name} no size fits (budget or risk caps, room {room:.0f}); done for the day')
            return
        s.attempts += 1
        a = x.cur = s.attempts
        x.risk = max(p['qty']*p['per_contract'] for p in plans)
        self.reserved += x.risk
        x.status, x.recs[a] = 'WORKING', dict(side=0, stop=None, eq=0, ev=0., xq=0, xv=0.)
        self.save()
        try:
            pids = [self.bracket(name, a, p) for p in plans]
        except Exception as exc:
            self.alert(f'{name} PLACEMENT FAILED ({type(exc).__name__}: {exc}); cancelling the market')
            return await self.defend(name, 'placement failed')
        for _ in range(0 if self.dry else 30):
            if all(self.gw.status(i) in ACKED for i in pids):
                break
            await self.sleep(.1)
        self.alert(f'{name} {"WOULD PARK" if self.dry else "PARKED"} attempt {a}: ' + '; '.join(
            f'{"BUY" if p["side"] == 1 else "SELL"} {p["qty"]} {m.execution.symbol} stop {p["trigger"]} limit {p["limit"]} -> exit stop {p["stop_px"]}, MKT 15:55'
            for p in plans) + '; entries expire 11:30 ET')

    def on_fill(self, oid, eid, qty, price):
        o, key = self.orders.get(oid), exec_key(eid)
        if o is None or key in self.seen:
            return
        self.seen.add(key)
        n = o['market']
        x, r = self.x[n], self.x[n].recs[o['attempt']]
        x.pos += o['side']*qty
        x.fills.append(dict(id=key, order=oid, role=o['role'], qty=qty, price=price, at=self.clock().isoformat()))
        if o['role'] == 'ENTRY':
            r.update(side=o['side'], stop=o['stop'], eq=r['eq']+qty, ev=r['ev']+qty*price)
            if x.status == 'WORKING':
                x.status = 'OPEN'
                self.spawn(self.settle(n, oid, o['attempt']))
            self.alert(f'{n} ENTRY FILL: {"BUY" if o["side"] == 1 else "SELL"} {qty} @ {price} (order {oid}); broker stop {o["stop"]} and 15:55 exit are children of it'
                       + ('' if r['eq'] >= o['qty'] else f'; PARTIAL {r["eq"]}/{o["qty"]}, verify child quantities in TWS'))
        else:
            r['xq'] += qty
            r['xv'] += qty*price
            self.alert(f'{n} {o["role"]} FILL: {"BUY" if o["side"] == 1 else "SELL"} {qty} @ {price}')
            if x.pos == 0:
                self.close(n, r)
        self.save()

    def close(self, n, r):
        x = self.x[n]
        rr = None
        if r['eq'] and r['xq'] and r['stop'] is not None and r['ev']/r['eq'] != r['stop']:
            e = r['ev']/r['eq']
            rr = round(r['side']*(r['xv']/r['xq']-e)/abs(e-r['stop']), 6)
        x.trades.append(dict(attempt=x.cur, side=r['side'], qty=r['eq'], entry=r['ev']/r['eq'] if r['eq'] else None,
                             exit=r['xv']/r['xq'] if r['xq'] else None, stop=r['stop'], r=rr))
        x.risk = 0.
        x.status = 'OFF' if x.off else 'IDLE'
        self.alert(f'{n} closed attempt {x.cur}: R {rr if rr is not None else "n/a"}' + (f'; {3-self.s[n].attempts} attempt(s) left' if not x.off else ''))

    async def settle(self, name, filled, attempt):
        await self.sleep(2)
        for oid, o in list(self.orders.items()):
            if o['market'] == name and o['role'] == 'ENTRY' and oid != filled and o['attempt'] == attempt and self.st(oid) not in DEAD:
                self.gw.cancel(oid)
                self.alert(f'{name} cancelled leftover entry {oid} after the sibling filled')

    def on_status(self, oid, status):
        o = self.orders.get(oid)
        if o:
            o['status'] = status
            if status == 'Inactive':
                self.reject(oid, 'Inactive')
            x = self.x[o['market']]
            if (o['role'] == 'STOP' and status in ('Cancelled', 'ApiCancelled') and x.pos and not x.defended and not x.off
                    and o['attempt'] == x.cur and o['side'] == (-1 if x.pos > 0 else 1)):
                self.spawn(self.stop_lost(oid))

    async def stop_lost(self, oid):
        """The protective STOP of an open position was cancelled (not rejected). A TIME fill or the position closing in the next
        second is benign (OCA); otherwise the position has no stop, so flatten it (service.py STOP_DEAD intent)."""
        o = self.orders[oid]
        n, x = o['market'], self.x[o['market']]
        await self.sleep(1)
        timed = any(y['market'] == n and y['role'] == 'TIME' and y['attempt'] == o['attempt'] and self.st(i) == 'Filled' for i, y in self.orders.items())
        if x.pos and not timed and not x.defended:
            try:   # ib_insync marks an order Cancelled locally on some errors: trust only the broker's working orders
                still = any(w['id'] == oid and w['status'] not in DEAD for w in await self.gw.open_orders())
            except Exception as exc:
                still = False
                self.alert(f'{n} could not confirm STOP {oid} at the broker ({type(exc).__name__}: {exc}); treating it as gone')
            if still:
                return self.note(n, f'STOP {oid} shows Cancelled locally but is still working at the broker; not flattening')
            x.defended = True
            self.alert(f'STOP LOST {n} order {oid} cancelled with position {x.pos} open; flattening')
            await self.defend(n, 'protective stop cancelled')

    async def cancel_entries(self):
        """11:30 ET: explicitly cancel every own ENTRY not dead (GTD is only the backstop), confirm, alert loudly if any remain."""
        live = []
        for i, o in self.orders.items():
            if o['role'] != 'ENTRY' or self.st(i) in DEAD:
                continue
            f = self.gw.filled(i) or sum(y['qty'] for y in self.x[o['market']].fills if y['order'] == i)
            if f > 0:
                self.alert(f'{o["market"]} ENTRY {i} is partially filled ({f}/{o["qty"]}); NOT cancelling at 11:30, CHECK TWS')
            else:
                live.append(i)
        for i in live:
            self.gw.cancel(i)
        for _ in range(30):
            live = [i for i in live if self.st(i) not in DEAD]
            if not live:
                break
            await self.sleep(1)
        if live:
            self.alert(f'ENTRY ORDERS STILL ALIVE after the 11:30 cancel: {live}. CHECK TWS NOW')

    def on_error(self, oid, code, msg):
        if oid in self.orders and code in REJECT:
            self.reject(oid, f'{code} {msg}')

    def reject(self, oid, why):
        o = self.orders[oid]
        x = self.x[o['market']]
        self.alert(f'REJECTED {o["market"]} {o["role"]} order {oid}: {why}')
        if not x.defended:
            x.defended = True
            self.spawn(self.defend(o['market'], f'{o["role"]} rejected'))

    async def defend(self, name, why):
        """A bracket order failed: cancel the market's working orders, flatten any position with a MKT, take the market off."""
        x = self.x[name]
        x.off = True
        for oid, o in list(self.orders.items()):   # unconditional: the local status can say Cancelled while the order still works
            if o['market'] == name and o['attempt'] == x.cur and o['role'] != 'FLAT':
                self.gw.cancel(oid)
        if x.pos:
            await self.sleep(.5)
            # A TIME/STOP fill may have landed at the broker already: flatten the broker's quantity, never the local one.
            try:
                own = (await self.gw.sync())['own_positions']
            except Exception as exc:
                own = None
                self.alert(f'{name} broker position read failed during defend ({type(exc).__name__}: {exc})')
            sym = self.mk[name].execution.symbol
            bq = None if own is None else own[sym]
            if bq is None:
                x.status = 'OFF'
                self.alert(f'{name} POSITION UNKNOWN after {why}: NO FLATTEN SENT, market off. CHECK TWS NOW (local pos {x.pos})')
                return self.save()
            if bq == 0 or (bq > 0) != (x.pos > 0):
                x.status = 'OFF'
                self.alert(f'{name} no flatten after {why}: broker position {bq} vs local {x.pos}' + ('' if bq == 0 else '. OPPOSITE SIGN, CHECK TWS NOW'))
                return self.save()
        if x.pos:
            r, m, oid = x.recs[x.cur], self.mk[name], self.gw.next_id()
            sd, q = (-1 if bq > 0 else 1), abs(bq)
            body = dict(kind='MKT', side=sd, qty=q, tif='DAY', account=self.cfg.account,
                        ref=f'{m.execution.symbol}|{"BUY" if r["side"] == 1 else "SELL"}|{STRATEGY_REF}|{self.day}|{name}-{x.cur}-FLAT')
            self.orders[oid] = dict(market=name, role='FLAT', side=sd, qty=q, attempt=x.cur, stop=r['stop'], status='')
            self.gw.send(name, oid, body)
            self.alert(f'{name} FLATTEN: MKT {"BUY" if sd == 1 else "SELL"} {q} {m.execution.symbol} after {why}')
        else:
            x.status = 'OFF'
            self.alert(f'{name} taken off for the day after {why}')
        self.save()

    async def sync(self):
        """Own positions and working orders from the broker (orderRef based); never places anything."""
        v = await self.gw.sync()
        if v['own_positions'] is None:
            raise RuntimeError(f'own position unknown: {v["own_positions_error"]}')
        here = {w.get('order_id') for w in v['working_orders'] if w.get('client_id', self.cfg.client_id) == self.cfg.client_id}
        for i, o in self.orders.items():   # after a reconnect the gateway may not know our orders: absent from the broker's book = dead
            if i not in here and o['status'] not in DEAD:
                o['status'] = 'Gone'
        for n, m in self.mk.items():
            x, sym = self.x[n], m.execution.symbol
            pos, working = v['own_positions'][sym], any(w['symbol'] == sym for w in v['working_orders'])
            if pos != x.pos:
                self.alert(f'{n} position {x.pos} -> {pos} per broker executions')
                x.pos = pos
            if x.status == 'OPEN' and not pos:
                self.close(n, x.recs.get(x.cur) or dict(side=0, stop=None, eq=0, ev=0., xq=0, xv=0.))
            elif pos and x.status in ('WORKING', 'IDLE'):
                x.status = 'OPEN'
            elif x.status == 'WORKING' and not working and not pos:
                x.status = 'IDLE'
        self.save()

    async def reconnect(self):
        self.alert('CONNECTION LOST; reconnecting (working orders stay at the broker)')
        self.synced = False
        for delay in (2, 5, 10):
            try:
                await self.gw.connect(self.on_tick)
                return self.alert('reconnected; re-reading positions and orders before placing anything')
            except Exception as exc:
                self.note('conn', f'reconnect failed: {type(exc).__name__}: {exc}')
                await self.sleep(delay)
        raise RuntimeError('reconnect failed')

    async def step(self):
        """One pass. Returns True when the session is finished."""
        t = self.clock().astimezone(NY).time()
        if not self.gw.connected():
            try:
                await self.reconnect()
            except RuntimeError:
                return t >= WATCH_END
        if not self.synced:
            try:
                await self.sync()
                self.synced = True
            except Exception as exc:
                self.note('sync', f'sync failed, not trading: {exc}')
                return t >= WATCH_END
        for n, x in self.x.items():
            s = self.s[n]
            if s.phase == 'BLOCKED' and x.status == 'IDLE':
                x.status = 'OFF'
                self.alert(f'{n} skipped for the day: {s.note}')
            if x.status == 'IDLE' and s.opening and t < ENTRY_END and s.attempts < 3 and not x.off:
                await self.park(n)
        if t >= ENTRY_END and not self.noted_end:
            self.noted_end = True
            if not self.dry:
                await self.cancel_entries()
            self.alert('11:30 ET: entry window closed. ' + ', '.join(f'{n} {x.status} pos {x.pos}' for n, x in self.x.items())
                       + '. Broker holds any stop and 15:55 exit; no new orders from here.')
        open_pos = any(x.status == 'OPEN' or x.pos for x in self.x.values())
        return t >= WATCH_END or (t >= time(11,30,10) and not open_pos and not self.tasks)

    async def run(self):
        while not await self.step():
            await self.sleep(.2)
        for n, x in self.x.items():
            if x.pos:
                self.alert(f'{n} still holds {x.pos}; the broker stop and 15:55 exit remain, CHECK TWS')
        self.save()

    async def plan(self, opens):
        """Offline plan: pretend each market opened at `opens`, park once (dry gateway prints the orders)."""
        now = self.clock().isoformat()
        for n, px in opens.items():
            s = self.s[n]
            s.opening = s.previous = px
            s.last_timestamp = now
        self.fresh = lambda n: (opens[n], opens[n])
        for n in opens:
            if self.x[n].status == 'IDLE':
                await self.park(n)


async def run_cli(args):
    cfg_path = Path(args.config) if args.config else (LIVE_CONFIG if args.mode == 'live' else None)
    if cfg_path is None:
        raise SystemExit('paper mode needs --config (the repo has no paper config)')
    config = Config.load(cfg_path)
    if config.mode != args.mode:
        raise SystemExit(f'config mode {config.mode} != --mode {args.mode}')
    now = datetime.now(timezone.utc)
    today = now.astimezone(NY).date().isoformat()
    day = args.session or today
    plan = args.plan
    dry = args.dry_run or plan
    config = effective_config(config, args.client_id, dry)
    alert = Alerts(f'OpenBreakout SIMPLE {args.mode}{" DRY" if dry else ""} {day}', webhook_url())
    if not plan and (day != today or now.astimezone(NY).time() >= LATEST_START):
        alert(f'not starting: session {day}, now {now.astimezone(NY):%H:%M:%S} ET (need today before 09:29:30)')
        return 4
    risk, info = find_risk(day)
    if risk is None and plan:
        risk, info = find_risk(day, any_session=True)
        alert(f'PLAN: no risk refresh for {day}; PROVISIONAL score from refresh for {info and info["session"]} (risk_latest {info and info["risk_latest"]})')
    if args.risk_parquet:
        risk = Path(args.risk_parquet)
    if risk is None:
        alert(f'NO RISK REFRESH for {day}; not trading')
        return 4
    gw = Gateway(config, day, dry)
    root, holder = RUNS/f'{day}-{run_folder(config.mode, dry)}', []
    deadline = now+timedelta(seconds=60) if dry else datetime.fromisoformat(day+'T09:27:00').replace(tzinfo=NY)
    try:
        await connect_retry(gw, lambda *a: holder and holder[0].on_tick(*a), alert, deadline)
        print(f'connected: account ...{config.account[-4:]} client {config.client_id} port {config.port} mode {config.mode}', flush=True)
        v = await gw.sync()
        bad = start_problems(v)
        if bad:
            alert(f'START CHECK FAILED, no trading: {bad}')
            return 3
        equity = await gw.equity()
        bars, (rb, rerr) = await asyncio.gather(gw.t.history(), fetch_range_bars(gw.t))
        frame = pd.read_parquet(risk)
        previous, _ = calendar_dates(day)
        if plan and pd.to_datetime(frame.index).tz_localize(None).max() < pd.Timestamp(previous):
            extra = frame.iloc[[-1]].copy()
            extra.index = [pd.Timestamp(previous)]
            frame = pd.concat([frame, extra])
        prior = prior_r(RUNS, day, [m.name for m in config.markets], config.mode)
        man = build_manifest(config, day, frame, {k: b for k, b in bars.items()}, roll_verified=True, range_bars=rb, range_error=rerr,
                             prior_day=prior, now=datetime.fromisoformat(day+'T09:00:00').replace(tzinfo=NY) if plan else datetime.now(timezone.utc))
        alert(f'inputs ready: score {man["score"]:.1f} (shorts {"ON" if man["score"] >= 20 else "OFF"}); ' + '; '.join(
            f'{n} prior_tr {v["prior_tr"]:.2f} ratio {v["ratio"] if v["ratio"] is None else round(v["ratio"], 3)} prior R {v["prior_day_r"]} skip {v["skip_prior_range"]}'
            for n, v in man['markets'].items()))
        runner = Runner(config, day, gw, man, root, alert, equity, dry=dry)
        holder.append(runner)
        if plan:
            for n, v in man['markets'].items():
                print(f'  {n}: prior-R source {v["prior_day_source"]} note: {v["prior_day_note"]}; reason: {v["prior_range_reason"]}')
            await runner.plan({n: float(b.close.iloc[-1]) for n, b in bars.items()})
            return 0
        runner.synced = True
        await runner.run()
        return 0
    except BaseException as exc:
        alert(f'EXITING on {type(exc).__name__}: {exc}; broker-held stops and exits remain, CHECK TWS')
        raise
    finally:
        alert('exit')
        alert.flush()
        if gw.t:
            gw.t.close()


PROOF_CLIENT, PROOF_ATTEMPT, LIVE_CLIENT, PROOF_PCT = 927489, 9, 927481, .05


def globex_open(ny):
    """CME equity-index Globex: Sun 18:00 ET to Fri 17:00 ET, daily break 17:00-18:00 ET Mon-Thu (holidays not modelled)."""
    wd, t = ny.weekday(), ny.time()
    return not (wd == 5 or (wd == 4 and t >= time(17)) or (wd == 6 and t < time(18)) or (wd < 4 and time(17) <= t < time(18)))


async def run_proof(gw, config, day, out_dir, alert, clock=None, sleep=asyncio.sleep):
    """One-shot broker proof: a 1-lot MNQ long+short bracket pair built by place_bracket, triggers +/-5% from the quote, so nothing can
    fill. Places it, reports statuses/errors/OCA groups, cancels the long parent (children must follow), cancels the rest, checks flat.
    Returns (ok, report); the report is written to out_dir/proof-<timestamp>.json either way."""
    clock = clock or (lambda: datetime.now(timezone.utc))
    ny = clock().astimezone(NY)
    m = config.markets[0]
    rep = dict(day=day, client_id=config.client_id, started=ny.isoformat(), market=m.name, symbol=m.execution.symbol, steps=[], errors=[],
               flagged=[], orders={}, oca={}, ok=False)
    orders, phase = {}, ['place']
    step = lambda text, **kw: (rep['steps'].append(dict(at=clock().astimezone(NY).isoformat(), text=text, **kw)), alert(f'PROOF: {text}'))
    gw.on_fill = lambda *a: rep['flagged'].append(f'UNEXPECTED FILL {a}')
    gw.on_status = lambda oid, st: st == 'Inactive' and oid in orders and rep['flagged'].append(f'order {oid} {orders[oid]["role"]} Inactive')

    def on_error(oid, code, msg):
        if oid in orders:
            rep['errors'].append(dict(id=oid, role=orders[oid]['role'], code=code, msg=str(msg), phase=phase[0]))
            if phase[0] == 'place' or code in REJECT:
                rep['flagged'].append(f'order {oid} {orders[oid]["role"]} error {code} {msg}')
    gw.on_error = on_error
    dead = lambda i: gw.status(i) in DEAD

    async def wait(pred, n, dt=.5):
        for _ in range(n):
            if pred():
                return True
            await sleep(dt)
        return pred()

    def finish(ok, why=None):
        rep.update(ok=ok, why=why, ended=clock().astimezone(NY).isoformat(), final={str(i): gw.status(i) for i in orders})
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        f = Path(out_dir)/f'proof-{ny:%Y%m%d-%H%M%S}.json'
        f.write_text(json.dumps(rep, indent=1, default=str), encoding='utf-8')
        alert(f'PROOF {"PASSED" if ok else "FAILED"}{": "+why if why else ""}; report {f}')
        return ok, rep

    async def cleanup():
        phase[0] = 'cancel'
        left = [i for i in orders if not dead(i)]
        for i in left:
            gw.cancel(i)
        if left:
            await wait(lambda: all(dead(i) for i in orders), 60)

    if config.client_id == LIVE_CLIENT or m.execution.symbol != 'MNQ' or len(config.markets) != 1:
        return finish(False, 'refused: proof is MNQ only and must not use the live client id')
    if not globex_open(ny):
        return finish(False, f'refused: Globex is closed at {ny:%a %H:%M} ET')
    try:
        bad = start_problems(await gw.sync())
        if bad:
            return finish(False, f'refused: start check failed {bad}')
        for _ in range(20):
            try:
                bid, ask, ts = gw.quote('NQ')
                if 0 < bid <= ask and -1 <= (clock()-aware(ts)).total_seconds() <= config.stale_seconds:
                    break
            except (KeyError, ValueError):
                pass
            await sleep(.5)
        else:
            return finish(False, 'refused: no fresh MNQ quote')
        last, tick = (bid+ask)/2, m.execution.tick
        trig = {sd: snap(last*(1+PROOF_PCT*sd), tick, sd == 1) for sd in (1, -1)}
        rep.update(last=last, triggers=trig)
        if any(abs(t/last-1) < .03 for t in trig.values()):
            return finish(False, f'refused: a trigger is within 3% of last {last}')
        dist = snap(last*.005, tick, True)
        plans = {sd: dict(side=sd, qty=1, trigger=trig[sd], limit=trig[sd]+2*tick*sd, stop_px=snap(trig[sd]-sd*dist, tick, sd == -1)) for sd in (1, -1)}
        till = f'{(ny.date() if ny.time() < time(11, 30) else ny.date()+pd.Timedelta(days=1)):%Y%m%d} 11:30:00 America/New_York'
        pids = {}
        for sd, p in plans.items():
            pids[sd] = place_bracket(gw, config, m, day, 'NQ', PROOF_ATTEMPT, p, orders, good_till=till)
        by = lambda sd, role: next(i for i, o in orders.items() if o['role'] == role and o['side'] == (sd if role == 'ENTRY' else -sd))
        ids = {(sd, r): by(sd, r) for sd in (1, -1) for r in ('ENTRY', 'STOP', 'TIME')}
        step('placed 1-lot MNQ pair', plans=plans, till=till, ids={f'{sd}/{r}': i for (sd, r), i in ids.items()})
        await wait(lambda: all(gw.status(i) in ACKED for i in orders), 40)
        rep['orders'] = {str(i): dict(role=o['role'], side=o['side'], status=gw.status(i)) for i, o in orders.items()}
        step('statuses after placement', statuses=rep['orders'])
        rep['flagged'] += [f'order {i} not acknowledged: {gw.status(i)}' for i in orders if gw.status(i) not in ACKED]
        oo = {x['id']: x for x in await gw.open_orders()}
        eg, lx, sx = (oo.get(ids[(1, 'ENTRY')], {}).get('oca'), oo.get(ids[(1, 'STOP')], {}).get('oca'), oo.get(ids[(-1, 'STOP')], {}).get('oca'))
        rep['oca'] = dict(listed=sorted(oo), entry=eg, long_exit=lx, short_exit=sx,
                          entries_share=eg and eg == oo.get(ids[(-1, 'ENTRY')], {}).get('oca'),
                          long_children_share=lx and lx == oo.get(ids[(1, 'TIME')], {}).get('oca'),
                          short_children_share=sx and sx == oo.get(ids[(-1, 'TIME')], {}).get('oca'),
                          distinct=len({eg, lx, sx}) == 3,
                          types_ok=all(oo.get(i, {}).get('oca_type') == 1 for i in ids.values()),
                          parents_ok=all(oo.get(ids[(sd, r)], {}).get('parent') == ids[(sd, 'ENTRY')] for sd in (1, -1) for r in ('STOP', 'TIME')))
        step('openOrders OCA groups', oca=rep['oca'])
        if not (len(oo) >= 6 and all(rep['oca'][k] for k in ('entries_share', 'long_children_share', 'short_children_share', 'distinct', 'types_ok', 'parents_ok'))):
            rep['flagged'].append(f'OCA structure not as expected: {rep["oca"]}')
        phase[0] = 'cancel'
        gw.cancel(ids[(1, 'ENTRY')])
        chain = [ids[(1, r)] for r in ('ENTRY', 'STOP', 'TIME')]
        ok = await wait(lambda: all(dead(i) for i in chain), 40)
        step('cancelled long parent', long_chain={str(i): gw.status(i) for i in chain}, short_parent=gw.status(ids[(-1, 'ENTRY')]))
        if not ok:
            rep['flagged'].append(f'long children did not cancel with their parent: {[gw.status(i) for i in chain]}')
    except BaseException as exc:
        rep['flagged'].append(f'EXCEPTION {type(exc).__name__}: {exc}')
        await cleanup()
        finish(False, rep['flagged'][-1])
        raise
    await cleanup()
    allc = all(gw.status(i) in ('Cancelled', 'ApiCancelled') for i in orders)
    step('cancelled everything left', statuses={str(i): gw.status(i) for i in orders})
    if not allc:
        rep['flagged'].append(f'not all Cancelled: {[gw.status(i) for i in orders]}')
    v = await gw.sync()
    pos = None if v['own_positions'] is None else v['own_positions'].get('MNQ')
    rep.update(position=pos, residual=start_problems(v))
    step('final book', position=pos, residual=rep['residual'])
    if pos != 0 or rep['residual']:
        rep['flagged'].append(f'book not clean: position {pos} residual {rep["residual"]}')
    return finish(not rep['flagged'], '; '.join(rep['flagged']) or None)


async def proof_cli(args):
    config = Config.load(Path(args.config) if args.config else LIVE_CONFIG)
    if config.mode != 'live':
        raise SystemExit('simple-proof is live only')
    now = datetime.now(timezone.utc)
    day = now.astimezone(NY).date().isoformat()
    if args.session and args.session != day:
        raise SystemExit(f'--session must be today ({day})')
    nq = tuple(m for m in config.markets if m.name == 'NQ' and m.execution.symbol == 'MNQ')
    if not nq or (args.client_id or PROOF_CLIENT) == LIVE_CLIENT:
        raise SystemExit('proof needs the NQ/MNQ market and a client id other than the live one')
    if not globex_open(now.astimezone(NY)):
        raise SystemExit(f'refused: Globex is closed at {now.astimezone(NY):%a %H:%M} ET')
    config = replace(config, client_id=args.client_id or PROOF_CLIENT, markets=nq)
    alert = Alerts(f'OpenBreakout SIMPLE PROOF {day}', webhook_url())
    gw = Gateway(config, day)
    try:
        await gw.connect(lambda *a: None)
        print(f'connected: account ...{config.account[-4:]} client {config.client_id} port {config.port}', flush=True)
        ok, _ = await run_proof(gw, config, day, RUNS, alert)
        return 0 if ok else 1
    finally:
        alert.flush()
        if gw.t:
            gw.t.close()
