"""Durable intent journal. No cleanup/pruning; one process owns a session DB."""
from contextlib import contextmanager
import json
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

    def close(self):
        if getattr(self,'db',None):
            self.db.close();self.db=None
        if self.lock_handle:
            self.lock_handle.close();self.lock_handle=None
