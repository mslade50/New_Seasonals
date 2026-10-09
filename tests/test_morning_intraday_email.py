import ast
from datetime import datetime
from html import escape
import json
from pathlib import Path
import sqlite3

import pytest

from broker_runtime import intraday_email as email
from broker_runtime.prepare_morning_intraday_email import patch
import trading_ibkr_locations as _tloc  # TRADING_IBKR_SOURCE, else the runtime root, else OneDrive

NOW = datetime(2026, 10, 7, 9, 31, 25, tzinfo=email.NY)
DAY = '2026-10-07'


def legend(root, date=DAY, **row):
    root.mkdir(exist_ok=True)
    (root / 'legend_ema_fut_enabled.flag').touch()
    (root / 'legend_ema_fut_last_result.json').write_text(json.dumps({
        'date': date, 'symbols': [{'Symbol': 'MES', 'Status': 'NO_SETUP', 'Note': 'EMA touched', **row}]}))


def journal(root, records):
    (root / 'legend_ema_fut_journal.jsonl').write_text('\n'.join(json.dumps({
        'date': DAY, 'symbol': 'MES', 'strategy': 'Legend_EMA', **r}) for r in records) + '\n')


def momentum(root, suffix='', *, mode='live', day=DAY, age=0, status='Submitted'):
    folder = root / (DAY + '-live' + suffix)
    folder.mkdir(parents=True)
    with sqlite3.connect(folder / 'runtime.sqlite') as db:
        db.execute('CREATE TABLE meta(key TEXT,value TEXT)')
        meta = dict(mode=mode, session=day, fingerprint='one', phase='RUNNING_LIVE',
                    heartbeat=dict(at=f'{DAY}T09:{31-age:02}:25-04:00', connected=True))
        db.executemany('INSERT INTO meta VALUES(?,?)', [(k, json.dumps(v)) for k,v in meta.items()])
    with sqlite3.connect(folder / 'trades.sqlite') as db:
        db.executescript('CREATE TABLE meta(key TEXT,value TEXT);'
                         'CREATE TABLE states(market TEXT,body TEXT);'
                         'CREATE TABLE orders(id INTEGER,market TEXT,role TEXT,status TEXT,body TEXT);')
        db.executemany('INSERT INTO meta VALUES(?,?)', [('day', json.dumps(day)), ('fingerprint', '"one"')])
        order = dict(side=1, qty=5, stop=31500, limit=31500.5, kind='STP LMT',
                     ref=f'MNQ|BUY|OpenBreakout|{DAY}|NQ-1-ENTRY')
        db.execute('INSERT INTO orders VALUES(1,?,?,?,?)', ('NQ', 'ENTRY', status, json.dumps(order)))
    return folder


def test_current_legend_no_setup_and_html_escaping(tmp_path):
    legend(tmp_path, Note='<script> U12345678')
    html = email.render_intraday_section(tmp_path, tmp_path / 'absent', NOW)
    assert 'NO_SETUP' in html and 'Legend EMA futures · Armed' in html
    assert '&lt;script&gt;' in html and 'U12345678' not in html


def test_yesterday_is_not_todays_staging(tmp_path):
    legend(tmp_path, date='2026-10-06', Status='Filled', Qty=20)
    html = email.render_intraday_section(tmp_path, tmp_path / 'absent', NOW)
    assert 'not yet available' in html and 'Filled' not in html and '20 ct' not in html


def test_live_journal_overrides_old_result_before_final_write(tmp_path):
    legend(tmp_path, date='2026-10-06')
    journal(tmp_path, [dict(kind='decision', side='BUY', target=7850, reason='LONG'),
                       dict(kind='entry', side='BUY', qty=4, target=7850, status='SENT',
                            filled_qty=4, avg_fill_price=7800.25, exit_at='10:30 ET')])
    html = email.render_intraday_section(tmp_path, tmp_path / 'absent', NOW)
    assert 'BUY 4 ct; SENT' in html and '4 ct @ 7,800.25 (runner reported)' in html
    assert 'target 7,850.00' in html and '10:30 ET' in html


def test_decision_only_does_not_claim_submission(tmp_path):
    legend(tmp_path, date='2026-10-06')
    journal(tmp_path, [dict(kind='decision', side='BUY', target=7850)])
    html = email.render_intraday_section(tmp_path, tmp_path / 'absent', NOW)
    assert 'entry not yet reported' in html and 'SENT' not in html


def test_test_and_partial_journal_rows_are_excluded(tmp_path):
    legend(tmp_path, date='2026-10-06')
    journal(tmp_path, [dict(kind='entry', qty=99, forced=True),
                       dict(kind='entry', qty=98, strategy='Legend_EMA_TEST')])
    with (tmp_path / 'legend_ema_fut_journal.jsonl').open('a') as f:
        f.write('{"kind":')
    html = email.render_intraday_section(tmp_path, tmp_path / 'absent', NOW)
    assert '99 ct' not in html and '98 ct' not in html and 'not yet available' in html


@pytest.mark.parametrize('status', ['Submitted', 'PREPARED', 'Filled', 'Cancelled'])
def test_momentum_entry_snapshot_and_read_only(tmp_path, status):
    folder = momentum(tmp_path / 'runs', status=status)
    before = {p.name: p.read_bytes() for p in folder.iterdir()}
    html = email.render_intraday_section(tmp_path / 'absent', tmp_path / 'runs', NOW)
    assert f'runner status {status}' in html and 'MNQ' in html and 'BUY 5 ct' in html
    assert 'trigger 31,500.00' in html and 'limit 31,500.50' in html
    assert '11:30 ET' in html and '15:55 ET' in html and 'OCA' in html
    if status == 'PREPARED':
        assert 'submission unconfirmed' in html
    assert before == {p.name: p.read_bytes() for p in folder.iterdir()}


def test_stale_momentum_and_latest_attempt(tmp_path):
    momentum(tmp_path / 'runs', status='Filled')
    momentum(tmp_path / 'runs', '-2', age=3, status='PREPARED')
    html = email.render_intraday_section(tmp_path / 'absent', tmp_path / 'runs', NOW)
    assert 'stale/disconnected' in html and 'PREPARED' in html and 'Filled' not in html


def test_shadow_and_previous_session_never_used(tmp_path):
    runs = tmp_path / 'runs'
    (runs / (DAY + '-shadow')).mkdir(parents=True)
    (runs / '2026-10-06-live').mkdir()
    html = email.render_intraday_section(tmp_path / 'absent', runs, NOW)
    assert 'No live session journal for today' in html


def test_mismatched_identity_and_corrupt_data_cannot_break_email(tmp_path):
    momentum(tmp_path / 'runs', day='2026-10-06')
    (tmp_path / 'legend_ema_fut_last_result.json').write_text('[]')
    html = email.render_intraday_section(tmp_path, tmp_path / 'runs', NOW)
    assert html.count('Could not read runner evidence') == 2


def test_absent_strategies_omit_section(tmp_path):
    assert email.render_intraday_section(tmp_path, tmp_path / 'absent', NOW) == ''


def test_real_email_patch_primary_only_even_without_stock_orders():
    source = _tloc.source_dir() / 'morning_order_summary.py'
    if not source.exists():
        pytest.skip('Installed email source unavailable')
    original = source.read_text(encoding='utf-8')
    # This check remains valid after installation.
    candidate = original if 'from intraday_email import render_intraday_section' in original else patch(original)
    tree = ast.parse(candidate)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'build_account_email')
    ns = dict(account_section=lambda acc: ('STOCK_SECTION', 0, 0, 0), html_escape=escape,
              BAD_RED='red', OK_GREEN='green', GREY='grey', ACCOUNT_VALUE=750000,
              div_adjust_banner=lambda: '', datetime=datetime)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), '<candidate>', 'exec'), ns)
    primary = dict(label='Primary (TWS)', intraday_html='INTRADAY_SECTION')
    if candidate != original:
        before = next(n for n in ast.parse(original).body
                      if isinstance(n, ast.FunctionDef) and n.name == 'build_account_email')
        before_ns = dict(ns)
        exec(compile(ast.Module(body=[before], type_ignores=[]), '<original>', 'exec'), before_ns)
        _, before_html = before_ns['build_account_email'](primary, '09:31 ET', 'Primary')
        assert 'INTRADAY_SECTION' not in before_html  # regression demonstrated before hook
    _, html = ns['build_account_email'](primary, '09:31 ET', 'Primary')
    assert 'STOCK_SECTION' in html and 'INTRADAY_SECTION' in html
    assert 'No stock order-chain orders' in html
    _, pa = ns['build_account_email'](dict(label='PA'), '09:31 ET', 'PA')
    assert 'INTRADAY_SECTION' not in pa
    with pytest.raises(ValueError):
        patch(candidate)
