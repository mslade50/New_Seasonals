"""Text-level checks on the rendered PDF: no retired headings, no book leakage."""
from pathlib import Path
import re
import sys

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent

pdf = ROOT / (sys.argv[1] if len(sys.argv) > 1 else 'Denali_Risk_Dial_and_Forward_Returns_v3_Final.pdf')
pages = [p.extract_text() or '' for p in PdfReader(str(pdf)).pages]
text = '\n'.join(pages)

BANNED_HEADINGS = ['5 day dial', '21 day dial', 'Pre FOMC Rally\n', 'Pre-FOMC']
BOOK_TERMS = ['throttle', 'sizing', 'position size', 'exposure leg', 'sleeve', 'multiplier',
              'bps', 'OVS', 'OLV', 'Monday Dip', 'Weak Close', 'MonFri', 'frag_risk_bands',
              'strategy', 'Overbot', 'order', 'NAV', 'account']
md = (ROOT / (pdf.stem.replace('_Final', '') + '.md')).read_text(encoding='utf-8')
headings = re.findall(r'^## (.+)$', md, flags=re.M)

print('pages with content:', len(pages))
print('headings:', headings)

fail = []
for h in headings:
    low = h.lower()
    if 'pre fomc' in low or '5 day dial' in low or '21 day dial' in low or '5-day' in low:
        fail.append(f'retired heading still present: {h}')
if 'NYSE Net New Highs' not in headings:
    fail.append('NYSE section heading missing')

low = text.lower()
for term in BOOK_TERMS:
    if term.lower() in low:
        ctx = [m.start() for m in re.finditer(re.escape(term.lower()), low)][:3]
        fail.append(f'book term "{term}": ' + ' || '.join(
            text[max(0, i - 70):i + 70].replace('\n', ' ') for i in ctx))

# every markdown table cell must be populated
for i, line in enumerate(md.splitlines()):
    if line.startswith('|') and '---' not in line:
        cells = [c.strip() for c in line.strip('|').split('|')]
        if any(c == '' for c in cells):
            fail.append(f'empty table cell on md line {i + 1}: {line}')

# no page may be near-empty
for n, p in enumerate(pages, 1):
    words = len(p.split())
    print(f'  page {n}: {words} words')
    if words < 60:
        fail.append(f'page {n} looks near-empty ({words} words)')

print('\nRESULT:', 'FAIL' if fail else 'PASS')
for f in fail:
    print(' -', f)
