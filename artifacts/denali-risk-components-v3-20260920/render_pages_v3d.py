"""Rasterise the v3d PDFs so the finished pages can be eyeballed, not just parsed."""
from pathlib import Path

import pypdfium2 as pdfium

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'render-v3d'
OUT.mkdir(exist_ok=True)

for name in ['Denali_Risk_Dial_Detailed_v3d', 'Denali_Risk_Dial_One_Page_v3d']:
    doc = pdfium.PdfDocument(str(ROOT / (name + '_Final.pdf')))
    for i in range(len(doc)):
        img = doc[i].render(scale=1.6).to_pil()
        dst = OUT / f'{name}_p{i + 1}.png'
        img.save(dst)
        print(dst.name, img.size)
