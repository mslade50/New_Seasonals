"""DOCX to PDF through the installed Word engine.

LibreOffice is not present on this machine, which is why the September builds
exported their PDFs from Word by hand. This automates that same export so the
rendered pages match the layout the recipient already has. Run with the
interpreter that carries pywin32.
"""
from pathlib import Path
import sys

import win32com.client

WD_FORMAT_PDF = 17
ROOT = Path(__file__).resolve().parent

names = sys.argv[1:] or ['Denali_Risk_Dial_and_Forward_Returns_v3']
word = win32com.client.DispatchEx('Word.Application')
word.Visible = False
word.DisplayAlerts = 0
try:
    for name in names:
        src = ROOT / (name + '.docx')
        dst = ROOT / (name + '_Final.pdf')
        if not src.exists():
            raise SystemExit(f'missing {src}')
        doc = word.Documents.Open(str(src), ReadOnly=True, AddToRecentFiles=False)
        try:
            doc.ExportAsFixedFormat(OutputFileName=str(dst), ExportFormat=WD_FORMAT_PDF,
                                    OpenAfterExport=False, OptimizeFor=0,
                                    CreateBookmarks=0, DocStructureTags=True)
            pages = doc.ComputeStatistics(2)  # wdStatisticPages
        finally:
            doc.Close(False)
        print(f'{name}: {pages} pages -> {dst.name} ({dst.stat().st_size} bytes)')
finally:
    word.Quit()
