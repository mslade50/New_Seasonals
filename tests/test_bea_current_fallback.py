import json

import pandas as pd
import pytest

from scripts.build_official_macro_releases import bea_current_release, collect

NOW = pd.Timestamp("2026-10-08T02:00:00Z")
OLD = pd.Timestamp("2026-08-26T12:30:00Z")
URL = "https://www.bea.gov/news/2026/personal-income-and-outlays-august-2026"


def index(*items):
    return '<table>' + ''.join(
        f'<tr class="release-row"><td><a href="{url}">{title}</a></td>'
        f'<td><time datetime="{date}"></time></td></tr>'
        for title, url, date in items) + '</table>'


def item(url=URL, date="2026-09-30T08:30:00-04:00"):
    return "Personal Income and Outlays, August 2026", url, date


def test_explicit_index_discovery_ignores_other_releases_and_future_items():
    raw = index(item(url='/news/2026/personal-income-and-outlays-august-2026'),
                item(url='/news/2026/future', date='2026-10-29T08:30:00-04:00'),
                ('Personal Consumption Expenditures by State, 2025', '/news/state', '2026-10-01T08:30:00-04:00'))
    assert bea_current_release(raw, 'pce', after=OLD, captured=NOW) == (URL, pd.Timestamp('2026-09-30T12:30:00Z'))


@pytest.mark.parametrize('raw', [
    index(item(date='2026-08-26T08:30:00-04:00')),
    index(item(date='2026-10-29T08:30:00-04:00')),
    index(item(date='2026-09-30T08:30:00')),
    index(item(url='https://example.com/news/pce')),
    index(item(), item(url='/news/2026/conflicting')),
    '<a href="/news/2026/pce">Personal Income and Outlays, August 2026</a>',
])
def test_bad_or_noncurrent_index_fails_closed(raw):
    with pytest.raises(ValueError):
        bea_current_release(raw, 'pce', after=OLD, captured=NOW)


@pytest.mark.parametrize('date,next_date,accepted', [
    ('September 30, 2026', 'October 29, 2026', True),
    ('September 29, 2026', 'October 29, 2026', False),
    ('September 30, 2026', 'October 7, 2026', False),
])
def test_collector_replay_fallback_keeps_provenance_and_gates(tmp_path, monkeypatch, date, next_date, accepted):
    import scripts.build_official_macro_releases as collector
    original = pd.Timestamp

    class Clock(original):
        @classmethod
        def now(cls, tz=None):
            return NOW

    monkeypatch.setattr(collector.pd, 'Timestamp', Clock)
    raw = tmp_path / 'raw'
    raw.mkdir()
    (raw / 'bea_rss.txt').write_text('''<rss><channel><item>
    <title>Personal Income and Outlays, July 2026</title>
    <link>https://www.bea.gov/news/2026/personal-income-and-outlays-july-2026</link>
    <pubDate>Wed, 26 Aug 2026 08:30:00 EDT</pubDate></item></channel></rss>''')

    def html(released, period, previous, upcoming):
        return f'''<main>EMBARGOED UNTIL RELEASE AT 8:30 a.m. EDT, Wednesday, {released}
        Personal Income and Outlays, {period} 2026
        <table><tr><th>Measure</th><th>{previous}</th><th>{period}</th></tr>
        <tr><td>PCE price index</td><td>0.1</td><td>0.3</td></tr>
        <tr><td>PCE price index excluding food and energy</td><td>0.1</td><td>0.2</td></tr></table>
        From the same month one year ago, the PCE price index increased 3.4 percent.
        Excluding food and energy, the PCE price index increased 3.0 percent from one year ago.
        Next release: {upcoming}, at 8:30 a.m. EDT</main>'''

    (raw / 'pce.html').write_text(html('August 26, 2026', 'July', 'June', 'September 30, 2026'))
    (raw / 'bea_current_pce.html').write_text(index(item()))
    (raw / 'pce_current.html').write_text(html(date, 'August', 'July', next_date))
    out = tmp_path / 'out'
    out.mkdir()
    report = collect(out, source_dir=raw, calendar_path=collector.ROOT / 'data/macro_events.csv')
    assert not report['published'] and not report['core_data_pass']  # Other sources deliberately absent.
    if accepted:
        assert not any('pce' in gap.lower() for gap in report['gaps'])
        rows = pd.read_parquet(out / 'observations.parquet')
        assert len(rows) == 4 and set(rows.reference_period) == {'2026-08'}
        assert set(rows.source) == {URL}
        manifest = json.loads((out / 'manifest.json').read_text())
        assert all(s['capture_basis'] == 'archived_replay' for s in manifest['sources'])
    else:
        assert any('timestamps disagree' in g or 'missed next announced release: pce' == g for g in report['gaps'])
