"""Freeze the hashes of the verified final files before any send."""
from pathlib import Path
import hashlib
import json

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent
FILES = [
    'Denali_Risk_Dial_and_Forward_Returns_v3.docx',
    'Denali_Risk_Dial_and_Forward_Returns_v3_Final.pdf',
    'Denali_Risk_Dial_One_Page_v3.docx',
    'Denali_Risk_Dial_One_Page_v3_Final.pdf',
]

receipt = {
    'date': '2026-09-20',
    'version': 3,
    'supersedes': 'artifacts/denali-risk-components-v2-20260917 (detailed report) and '
                  'artifacts/denali-risk-components-20260917 (one-page brief)',
    'detailed_pages': len(PdfReader(str(ROOT / FILES[1])).pages),
    'one_page_pages': len(PdfReader(str(ROOT / FILES[3])).pages),
    'verification': ('All rendered pages inspected as images: no overflowing tables, no empty '
                     'sections, every table cell populated. Automated text check confirms no 5-day '
                     'or 21-day dial heading, no Pre FOMC heading, the NYSE section present, and no '
                     'strategy, sizing or exposure vocabulary anywhere in the PDF.'),
    'changes_vs_v2': [
        'Single main dial; 5-day and 21-day dial sections and their forward-return tables removed.',
        'Pre FOMC Rally component section removed after its 17 September 2026 retirement.',
        'Equity Put Call Complacency retained on the board with an explicit note that it '
        'contributes to no displayed dial.',
        'New NYSE Net New Highs section covering the 18 September 2026 EMA5 recovery-reset form.',
        'Forward-return tables recomputed on the production main dial series at 5/10/21/42/63 sessions.',
        'Component tables re-priced from data/master_prices.parquet so every table shares one baseline.',
    ],
    'files': {},
}
for name in FILES:
    data = (ROOT / name).read_bytes()
    receipt['files'][name] = hashlib.sha256(data).hexdigest()

(ROOT / 'delivery_receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
print(json.dumps(receipt, indent=2))
