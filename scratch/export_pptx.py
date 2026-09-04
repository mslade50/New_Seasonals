"""Export The_Book.pptx slides to PNG using PowerPoint COM (for visual verification)."""
import shutil
from pathlib import Path

import win32com.client

PPTX = r"C:\Users\McKinley Slade\dev\New_Seasonals\presentation\The_Book.pptx"
OUTDIR = Path(r"C:\Users\McKinley Slade\dev\New_Seasonals\scratch\pptx_shots")

if OUTDIR.exists():
    shutil.rmtree(OUTDIR)
OUTDIR.mkdir(parents=True)

pp = win32com.client.Dispatch("PowerPoint.Application")
try:
    pres = pp.Presentations.Open(PPTX, ReadOnly=True, Untitled=False, WithWindow=False)
    pres.Export(str(OUTDIR), "PNG", 1600, 900)
    pres.Close()
    print("EXPORTED", len(list(OUTDIR.glob("*.PNG")) + list(OUTDIR.glob("*.png"))), "slides")
finally:
    pp.Quit()
