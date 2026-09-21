"""Freeze the hashes of the verified v3b files before any send."""
from pathlib import Path
import hashlib
import json

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent
FILES = [
    'Denali_Risk_Dial_Detailed_v3b.docx',
    'Denali_Risk_Dial_Detailed_v3b_Final.pdf',
    'Denali_Risk_Dial_One_Page_v3b.docx',
    'Denali_Risk_Dial_One_Page_v3b_Final.pdf',
]
M = json.loads((ROOT / 'v3b_manifest.json').read_text(encoding='utf-8'))

receipt = {
    'date': '2026-09-21',
    'version': '3b',
    'base': ('artifacts/denali-risk-introduction-20260917 — the newest EMAILED Denali report '
             '(accepted by SMTP 2026-09-17T20:51:28Z, subject "Denali risk dial: team '
             'introduction with 5/10/21-day returns"). The later -v2 folder carries no email or '
             'delivery receipt and differs only by one deleted sentence.'),
    'supersedes': ('the v3 files in this folder, which were rebuilt from the earlier '
                   'denali-risk-components-v2-20260917 base; see SUPERSEDED.md'),
    'detailed_pages': len(PdfReader(str(ROOT / FILES[1])).pages),
    'one_page_pages': len(PdfReader(str(ROOT / FILES[3])).pages),
    'study': {'cohort': f"{M['start']} to {M['end']}", 'dates': M['rows'],
              'prices_through': M['prices_through'], 'display_windows': M['display_windows'],
              'dial_basis': M['dial_basis'], 'nyse_model_version': M['nyse_model_version']},
    'verification': ('All rendered pages inspected as images: no overflowing tables, no empty '
                     'sections, every table cell populated, band labels free of mojibake. '
                     'Automated scan confirms 4 + 1 pages, the NYSE and dual-filter headings '
                     'present, no retired 5-day/21-day/Pre-FOMC/put-call wording, and no '
                     'strategy, sizing, exposure or dollar vocabulary in either PDF.'),
    'changes_vs_base': [
        'NYSE Net Highs now arms and resets on the five-session EMA shipped 2026-09-18; its '
        'component description, strength note and one-pager row were edited in place, and a '
        'single evidence-and-caveat note was added rather than a second NYSE section.',
        'Every table recomputed on the production point-in-time dial through 2026-09-18 with the '
        'current NYSE floor; cohort moves from 1,792 dates ending 2026-06-16 to 1,790 ending '
        '2026-06-18 (the EMA warm-up drops 4 dates, 2 new dates qualify at the end).',
        'New book-free section "Proposed dual filter: evaluated, not adopted" reporting the '
        'replay outcome; the proposal document is NOT re-sent.',
        'Closing note on the dial vintage and the stored main_score basis.',
        'The 5/10/21-day framing, the component set, the band structure and the 50 threshold are '
        'unchanged from the base.',
    ],
    'files': {},
}
for name in FILES:
    receipt['files'][name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()

(ROOT / 'delivery_receipt_v3b.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
print(json.dumps(receipt, indent=2))
