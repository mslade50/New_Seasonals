"""Read-only intraday execution snapshots for the Primary morning email.

Uses runner journals only. Never imports trading code, connects to a broker,
opens a writable database, or reports a prepared intent as a confirmed fill.
"""
from contextlib import closing
from datetime import datetime
from html import escape
import json
import math
from pathlib import Path
import re
import sqlite3
from zoneinfo import ZoneInfo

NY = ZoneInfo('America/New_York')


def _text(value):
    return re.sub(r'(?<![A-Za-z0-9])(?:DU|DF|U|F)\d{5,}(?!\d)', '[account]', str(value or ''))[:500]


def _number(value):
    try:
        n = float(value)
        return f'{n:,.2f}' if math.isfinite(n) else '—'
    except (TypeError, ValueError):
        return '—'


def _legend(root, day):
    result_path = root / 'legend_ema_fut_last_result.json'
    journal_path = root / 'legend_ema_fut_journal.jsonl'
    if not any(p.exists() for p in (result_path, journal_path, root / 'legend_ema_fut_enabled.flag')):
        return None
    result = {}
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding='utf-8'))
    current = result.get('date') == day
    rows = {r['Symbol']: dict(r) for r in result.get('symbols', []) if current}
    # A qualified run writes its final result after 10:32. Its live 09:31
    # decision/entry journal must take precedence over yesterday's result.
    if journal_path.exists():
        with journal_path.open(encoding='utf-8') as stream:
            for line in stream:
                try:
                    record = json.loads(line)
                except ValueError:
                    continue  # possibly an append still in progress
                if (record.get('date') != day or record.get('strategy') != 'Legend_EMA'
                        or record.get('forced') or record.get('kind') == 'forced_entry'):
                    continue
                symbol = record.get('symbol')
                if symbol not in {'MES', 'MNQ'}:
                    continue
                kind = record.get('kind')
                if kind not in {'decision', 'entry', 'entry_failed', 'abort'}:
                    continue
                row = rows.setdefault(symbol, {'Symbol': symbol})
                row.update(Side=record.get('side') or row.get('Side', '—'),
                           Target=record.get('target', row.get('Target')))
                if kind == 'decision':
                    row.update(Status='Decision; entry not yet reported' if record.get('side') else 'NO_TRADE',
                               Note=record.get('reason', ''))
                else:
                    row.update(Status=record.get('status') or kind.upper(),
                               Qty=record.get('qty', row.get('Qty', 0)),
                               Note=record.get('note') or record.get('reason') or '')
                    if kind == 'entry':
                        row['fill'] = f"{record.get('filled_qty', 0)} ct @ {_number(record.get('avg_fill_price'))} (runner reported)"
                        row['exit'] = record.get('exit_at') or '10:30 ET'
    details = []
    for row in rows.values():
        desc = f"{row.get('Side', '—')} {row.get('Qty', 0)} ct; {row.get('Status', 'unknown')}"
        if row.get('fill'):
            desc += f"; fill {row['fill']}"
        if row.get('Target') is not None:
            desc += f"; target {_number(row['Target'])}; time exit {row.get('exit', '10:30 ET')}"
        if row.get('Note'):
            desc += f"; {row['Note']}"
        details.append((row['Symbol'], desc))
    if current and result.get('error'):
        details.append(('Runner error', result['error']))
    if not details:
        details.append(('Status', 'Today’s decision/entry is not yet available; previous-session results are excluded.'))
    return ('Legend EMA futures', 'Armed' if (root / 'legend_ema_fut_enabled.flag').is_file() else 'Disarmed', details)


def _meta(db):
    return {key: json.loads(value) for key, value in db.execute('SELECT key,value FROM meta')}


def _open_db(path):
    return sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True, timeout=1)


def _momentum(runs, day, now):
    if not runs.is_dir():
        return None
    pattern = re.compile(re.escape(day) + r'-live(?:-(\d+))?$')
    folders = [(int(m.group(1) or 1), p) for p in runs.iterdir()
               if p.is_dir() and (m := pattern.fullmatch(p.name))]
    if not folders:
        return ('Momentum / Open Breakout', 'Unavailable', [('Status', 'No live session journal for today; shadow and previous sessions are excluded.')])
    folder = max(folders, key=lambda item: item[0])[1]
    with closing(_open_db(folder / 'runtime.sqlite')) as db:
        meta = _meta(db)
    if meta.get('session') != day or meta.get('mode') != 'live':
        raise ValueError('live session identity mismatch')
    beat = meta.get('heartbeat') or {}
    timestamp = datetime.fromisoformat(beat['at']) if beat.get('at') else None
    fresh = timestamp is not None and timestamp.utcoffset() is not None and -5 <= (now - timestamp).total_seconds() <= 120
    state = meta.get('phase', 'unknown')
    if not fresh or beat.get('connected') is not True:
        state += '; stale/disconnected snapshot — working status not verified'
    details = []
    ledger = folder / 'trades.sqlite'
    if ledger.is_file():
        with closing(_open_db(ledger)) as db:
            lm = _meta(db)
            if lm.get('day') != day or lm.get('fingerprint') != meta.get('fingerprint'):
                raise ValueError('trade journal identity mismatch')
            if lm.get('halted'):
                state += '; HALTED: ' + _text(lm.get('halt_reason'))
            for market, role, status, body in db.execute('SELECT market,role,status,body FROM orders ORDER BY id'):
                if role != 'ENTRY':
                    continue
                order = json.loads(body)
                ref = str(order.get('ref', '')).split('|')
                if len(ref) < 4 or ref[2:4] != ['OpenBreakout', day]:
                    continue
                desc = f"{ref[1]} {order.get('qty', 0)} ct; {order.get('kind', '')}; runner status {status}"
                if status == 'PREPARED':
                    desc += ' (intent only; submission unconfirmed)'
                for key, label in (('stop', 'trigger'), ('limit', 'limit')):
                    if order.get(key) is not None:
                        desc += f"; {label} {_number(order[key])}"
                desc += '; entries expire 11:30 ET; position time exit 15:55 ET'
                details.append((ref[0], desc))
            if not details:
                for market, body in db.execute('SELECT market,body FROM states ORDER BY market'):
                    s = json.loads(body)
                    details.append((market, f"{s.get('phase', 'unknown')}; no entry intent; {s.get('note', '')}"))
    if not details:
        details.append(('Status', 'No entry intent yet; awaiting opening levels/arming.'))
    for key in ('last_error', 'feed_error'):
        if meta.get(key):
            details.append(('Runner warning', meta[key]))
    return ('Momentum / Open Breakout', state, details)


def render_intraday_section(executor_root, runs_root, now=None):
    """Best effort email-only section, including zero-stock-staging mornings."""
    now = now or datetime.now(NY)
    if now.tzinfo is None:
        now = now.replace(tzinfo=NY)
    now = now.astimezone(NY)
    day = now.date().isoformat()
    cards = []
    for label, reader, args in (
        ('Legend EMA futures', _legend, (Path(executor_root), day)),
        ('Momentum / Open Breakout', _momentum, (Path(runs_root), day, now)),
    ):
        try:
            snapshot = reader(*args)
        except Exception as exc:  # one malformed snapshot must never prevent the email
            snapshot = (label, 'Unavailable', [('Status', f'Could not read runner evidence ({type(exc).__name__}).')])
        if snapshot is None:
            continue
        name, status, rows = snapshot
        body = ''.join(f'<tr><td style="padding:6px;vertical-align:top"><strong>{escape(_text(symbol))}</strong></td>'
                       f'<td style="padding:6px">{escape(_text(note))}</td></tr>' for symbol, note in rows)
        cards.append(f'<h4 style="margin:14px 0 4px">{escape(name)} · {escape(_text(status))}</h4>'
                     f'<table style="width:100%;border-collapse:collapse;font:12px Arial,sans-serif">{body}</table>')
    if not cards:
        return ''
    return ('<div style="font:13px Arial,sans-serif;margin-top:20px;padding:14px;border:1px solid #d0d7de;border-radius:6px">'
            '<h3 style="margin:0 0 6px">Intraday staged execution · Primary</h3>'
            f'<div style="color:#57606a">{day} · Runner snapshots as of {now:%H:%M:%S} ET. '
            'Separate from stock staging totals; intents and submitted orders are not fills. '
            'Momentum BUY/SELL entries are alternatives in one OCA cycle.</div>'
            + ''.join(cards) + '</div>')
