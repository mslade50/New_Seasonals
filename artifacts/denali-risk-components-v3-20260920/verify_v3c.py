"""Text-level checks on the rendered v3c PDFs: no retired headings, no book leakage.

Same checks as verify_v3b, plus the wording this revision removed on purpose
(the dual-filter section, the NYSE change-history note and its evidence
paragraph, the SPY explainer paragraph) and a ban on em dashes in prose.
"""
from pathlib import Path
import re

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent
DOCS = {
    'Denali_Risk_Dial_Detailed_v3c': {'pages': 3, 'need': ['NYSE Net Highs']},
    'Denali_Risk_Dial_One_Page_v3c': {'pages': 1, 'need': []},
}
TITLE = 'Denali Risk Dial'
BOOK_TERMS = ['throttle', 'sizing', 'position size', 'exposure leg', 'sleeve', 'multiplier',
              ' bps', 'OVS', 'OLV', 'Monday Dip', 'Weak Close', 'MonFri', 'frag_risk_bands',
              'strategy', 'strategies', 'Overbot', 'Oversold', 'Bear Fade', 'NAV',
              'order staging', '$']
RETIRED = ['5 day dial', '21 day dial', '5-day dial', '21-day dial', 'pre fomc', 'pre-fomc',
           'equity put call', 'equity p/c',
           'dual filter', 'since 18 september', 'why the five-session average',
           'spy is an exchange-traded fund']

fail = []
for name, spec in DOCS.items():
    pdf = ROOT / (name + '_Final.pdf')
    pages = [p.extract_text() or '' for p in PdfReader(str(pdf)).pages]
    text = '\n'.join(pages)
    low = text.lower()
    md = (ROOT / (name + '.md')).read_text(encoding='utf-8')
    headings = re.findall(r'^## (.+)$', md, flags=re.M)
    print(f'== {name}: {len(pages)} pages (expected {spec["pages"]})')
    print('   headings:', headings)
    if len(pages) != spec['pages']:
        fail.append(f'{name}: {len(pages)} pages, expected {spec["pages"]}')
    if md.splitlines()[0] != '# ' + TITLE:
        fail.append(f'{name}: title is "{md.splitlines()[0]}", expected "# {TITLE}"')
    for need in spec['need']:
        if need not in headings:
            fail.append(f'{name}: missing heading "{need}"')
    for term in RETIRED:
        if term in low:
            fail.append(f'{name}: retired term "{term}" present')
    for term in BOOK_TERMS:
        if term.lower() in low:
            i = low.index(term.lower())
            fail.append(f'{name}: book term "{term}": ...'
                        + text[max(0, i - 70):i + 70].replace('\n', ' ') + '...')
    if '—' in text:
        i = text.index('—')
        fail.append(f'{name}: em dash present: ...{text[max(0, i - 70):i + 70]}...')
    for i, line in enumerate(md.splitlines()):
        if line.startswith('|') and '---' not in line:
            if any(c.strip() == '' for c in line.strip('|').split('|')):
                fail.append(f'{name}: empty table cell on md line {i + 1}: {line}')
    if 'â' in text or 'â€' in text:
        fail.append(f'{name}: mojibake in rendered text')
    for n, p in enumerate(pages, 1):
        words = len(p.split())
        print(f'   page {n}: {words} words')
        if words < 60:
            fail.append(f'{name}: page {n} near-empty ({words} words)')

print('\nRESULT:', 'FAIL' if fail else 'PASS')
for f in fail:
    print(' -', f)
