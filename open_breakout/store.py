"""Durable intent journal. No cleanup/pruning; one process owns a session DB."""
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import sqlite3
import time

class Store:
    def __init__(self, path, fingerprint, day, *, lock=True):
        self.path = Path(path).resolve()
        if any(part.lower().startswith('onedrive') for part in self.path.parts):
            raise ValueError('Runtime state must be outside OneDrive')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_handle = None
        if lock:
            self.lock_handle = open(str(self.path)+'.lock','a+b')
            self.lock_handle.seek(0)
            if os.name == 'nt':
                import msvcrt
                self.lock_handle.write(b'0');self.lock_handle.flush();self.lock_handle.seek(0)
                try: msvcrt.locking(self.lock_handle.fileno(), msvcrt.LK_NBLCK, 1)
                except OSError:
                    self.lock_handle.close();raise RuntimeError('Another breakout process owns this state')
            else:
                import fcntl
                try: fcntl.flock(self.lock_handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except OSError:
                    self.lock_handle.close();raise RuntimeError('Another breakout process owns this state')
        self.db = sqlite3.connect(self.path, isolation_level=None)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS states (market TEXT PRIMARY KEY, body TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS orders (id INTEGER PRIMARY KEY, market TEXT NOT NULL, role TEXT NOT NULL, status TEXT NOT NULL, body TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS fills (exec_id TEXT PRIMARY KEY, body TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY AUTOINCREMENT, time REAL NOT NULL, kind TEXT NOT NULL, body TEXT NOT NULL);
        ''')
        for key, val in [('fingerprint',fingerprint),('day',day)]:
            old = self.get(key)
            if old is not None and old != val:
                self.close();raise ValueError(f'State {key} differs; preserve this DB and use a distinct session DB')
            self.set(key,val)

    @contextmanager
    def transaction(self):
        self.db.execute('BEGIN IMMEDIATE')
        try:
            yield
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def get(self, key, default=None):
        row = self.db.execute('SELECT value FROM meta WHERE key=?',(key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set(self, key, value):
        self.db.execute('INSERT INTO meta VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value',(key,json.dumps(value,allow_nan=False)))

    def save(self, state):
        self.db.execute('INSERT INTO states VALUES (?,?) ON CONFLICT(market) DO UPDATE SET body=excluded.body',(state.market,json.dumps(state.record(),allow_nan=False)))

    def states(self):
        return [json.loads(r[0]) for r in self.db.execute('SELECT body FROM states')]

    def event(self, kind, body):
        self.db.execute('INSERT INTO events(time,kind,body) VALUES (?,?,?)',(time.time(),kind,json.dumps(body,allow_nan=False,default=str)))

    def order(self, order_id, market, role, body):
        # Intent is durable before invoking the broker. IDs are never reused.
        self.db.execute('INSERT INTO orders VALUES (?,?,?,?,?)',(order_id,market,role,'PREPARED',json.dumps(body,allow_nan=False)))

    def order_status(self, order_id, status):
        self.db.execute('UPDATE orders SET status=? WHERE id=?',(status,order_id))

    def orders(self):
        return {r[0]:dict(id=r[0],market=r[1],role=r[2],status=r[3],body=json.loads(r[4])) for r in self.db.execute('SELECT * FROM orders')}

    def add_fill(self, exec_id, body):
        cur = self.db.execute('INSERT OR IGNORE INTO fills VALUES (?,?)',(exec_id,json.dumps(body,allow_nan=False)))
        return cur.rowcount == 1

    def fill_ids(self):
        return [r[0] for r in self.db.execute('SELECT exec_id FROM fills')]

    def day_r(self, markets):
        return journal_day_r(self.db, markets)

    def close(self):
        if getattr(self,'db',None):
            self.db.close();self.db=None
        if self.lock_handle:
            self.lock_handle.close();self.lock_handle=None


# ---- Prior-day own R (owner amendment 2026-09-28 to the prior-range filter) ----

def _attempt_key(order):
    """(market, attempt) from the orderRef tail '<market>-<attempt>-<role>'."""
    tail = str(order['body'].get('ref') or '').split('|')[-1]
    parts = tail.rsplit('-', 2)
    if len(parts) != 3 or not parts[1].isdigit():
        raise ValueError(f'Unparseable orderRef {order["body"].get("ref")!r}')
    return order['market'], int(parts[1])

def journal_day_r(db, markets) -> dict[str, dict]:
    """Per market, summed R over the session's closed trades from journal fills (not marks):
    R = side x (avg exit fill - avg entry fill) / |avg entry fill - stop| per contract; the stop is the
    last journaled protective stop for that attempt. r is None with no closed trade."""
    orders = {r[0]:dict(market=r[1],role=r[2],body=json.loads(r[4])) for r in db.execute('SELECT * FROM orders')}
    stops = {}
    for oid,o in orders.items():
        if o['role'] == 'STOP':
            stops[oid] = o['body'].get('stop')
    for (body,) in db.execute("SELECT body FROM events WHERE kind='MODIFY_INTENT' ORDER BY seq"):
        e = json.loads(body)
        if e.get('id') in stops and 'stop' in (e.get('body') or {}):
            stops[e['id']] = e['body']['stop']
    trades = {}
    for (body,) in db.execute('SELECT body FROM fills'):
        f = json.loads(body)
        o = orders.get(f['order_id'])
        if o is None:
            raise ValueError(f'Fill for unjournaled order {f["order_id"]}')
        t = trades.setdefault(_attempt_key(o), dict(side=0,entry_qty=0,entry_value=0.,exit_qty=0,exit_value=0.,stop=None))
        if o['role'] == 'ENTRY':
            t['side'] = o['body']['side'];t['entry_qty'] += f['qty'];t['entry_value'] += f['qty']*f['price']
        else:
            t['exit_qty'] += f['qty'];t['exit_value'] += f['qty']*f['price']
    for oid,o in sorted(orders.items()):
        if o['role'] == 'STOP' and _attempt_key(o) in trades and stops[oid] is not None:
            trades[_attempt_key(o)]['stop'] = float(stops[oid])
    out = {}
    for name in markets:
        rs, open_count = [], 0
        for (market,attempt),t in sorted(trades.items()):
            if market != name or not t['entry_qty']:
                continue
            if t['exit_qty'] != t['entry_qty']:
                open_count += 1;continue
            entry, exit_ = t['entry_value']/t['entry_qty'], t['exit_value']/t['exit_qty']
            if t['stop'] is None or entry == t['stop'] or t['side'] not in (1,-1):
                raise ValueError(f'{name} attempt {attempt}: no usable stop or side for R')
            rs.append(t['side']*(exit_-entry)/abs(entry-t['stop']))
        note = f'{len(rs)} closed trade(s)' if rs else 'no closed trades'
        if open_count:
            note += f'; {open_count} not closed in the journal (excluded)'
        out[name] = dict(r=round(sum(rs),6) if rs else None, trades=len(rs), trades_r=[round(x,6) for x in rs], note=note)
    return out

def _valid_r(entry):
    r = entry.get('r') if isinstance(entry, dict) else 'invalid'
    return r is None or (isinstance(r, (int, float)) and not isinstance(r, bool) and math.isfinite(r))

def _read_only(path):
    # Read-only URI with a short busy timeout: a locked or corrupt journal raises quickly and is caught.
    return sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro', uri=True, timeout=1.)

def prior_session_dir(runs_root, previous, mode):
    """Highest attempt of <previous>-<mode>[-N] that has a trades journal, or None."""
    best = None
    for d in Path(runs_root).glob(f'{previous}-{mode}*'):
        rest = d.name[len(f'{previous}-{mode}'):]
        if rest and not (rest.startswith('-') and rest[1:].isdigit()):
            continue
        n = int(rest[1:]) if rest else 1
        if (d/'trades.sqlite').exists() and (best is None or n > best[0]):
            best = (n, d)
    return None if best is None else best[1]

def _session_r(runs_root, previous, mode, markets):
    """(per-market {r, note}, 'runtime_meta'|'journal', None) from one session's journal, or
    (None, None, reason) when it is missing, for another day, has no fills, or is unreadable."""
    d = prior_session_dir(runs_root, previous, mode)
    if d is None:
        return None, None, f'no {mode} journal for {previous}'
    try:
        c = _read_only(d/'trades.sqlite')
        try:
            day = json.loads((c.execute("SELECT value FROM meta WHERE key='day'").fetchone() or ['null'])[0])
            if day != previous:
                return None, None, f'{d.name} journal is for {day}, not {previous}'
            if not c.execute('SELECT COUNT(*) FROM fills').fetchone()[0]:
                # Halted before trading, failed, or no signal: nothing this session can say about the day.
                return None, None, f'{d.name} journal has no fills'
            runtime = d/'runtime.sqlite'
            if runtime.exists():
                try:
                    rc = _read_only(runtime)
                    try:
                        meta = {k:json.loads(v) for k,v in rc.execute("SELECT key,value FROM meta WHERE key IN ('day','day_r')")}
                    finally:
                        rc.close()
                except Exception:
                    meta = {}
                summary = meta.get('day_r')
                if (meta.get('day') == previous and isinstance(summary, dict) and set(markets) <= set(summary)
                        and all(_valid_r(summary[m]) for m in markets)):
                    return ({m:dict(r=summary[m].get('r'), note=f'{d.name}: {summary[m].get("note")}') for m in markets},
                            'runtime_meta', None)
            result = journal_day_r(c, markets)
        finally:
            c.close()
    except Exception as exc:
        return None, None, f'{d.name} unreadable ({type(exc).__name__}: {exc})'
    return {m:dict(r=v['r'], note=f'{d.name}: {v["note"]}') for m,v in result.items()}, 'journal', None

def prior_day_r(runs_root, previous, mode, markets) -> dict[str, dict]:
    """{market: {r, source, note}} for the previous XNYS session (owner amendment 2026-09-28, research row (v)).
    The UNFILTERED strategy's result is the prior: the shadow journal first, because a live day the filter
    skipped has no trades and could never be a big win. A live session ('live' mode) falls back to its own
    live journal when the shadow journal is missing, has no fills or is unreadable. source is
    shadow_runtime_meta | shadow_journal | own_runtime_meta | own_journal | None. Never raises; any
    missing input gives r None, and a missing prior-day R never causes a skip."""
    def empty(note):
        return {m:dict(r=None, source=None, note=note) for m in markets}
    try:
        order = [('shadow','shadow')] + ([('live','own')] if mode == 'live' else [])
        reasons = []
        for session_mode, label in order:
            result, kind, reason = _session_r(runs_root, previous, session_mode, markets)
            if result is None:
                reasons.append(reason);continue
            fallback = f' (after {"; ".join(reasons)})' if reasons else ''
            return {m:dict(r=v['r'], source=f'{label}_{kind}', note=v['note']+fallback) for m,v in result.items()}
        return empty('; '.join(reasons))
    except Exception as exc:
        return empty(f'prior-day R unreadable ({type(exc).__name__}: {exc})')
