"""READ-ONLY pull of the live Google Sheets tabs that hold staged-signal and
fill history (Trade_Signals_Log sheet1, Trade_Journal if present, Portfolio,
execution tabs). Writes CSV snapshots beside this script. Never writes to Sheets."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import gspread
import pandas as pd
from google.oauth2.service_account import Credentials

OUT = Path(r"C:/Users/McKinley Slade/dev/New_Seasonals/scratch/ultracode_sizing_2026-09-02")
CRED = Path(r"C:/Users/McKinley Slade/OneDrive/trading_ibkr/credentials.json")
SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly", "https://www.googleapis.com/auth/drive.readonly"]

creds = Credentials.from_service_account_file(str(CRED), scopes=SCOPES)
gc = gspread.authorize(creds)
sh = gc.open("Trade_Signals_Log")
summary = {}
for ws in sh.worksheets():
    vals = ws.get_all_values()
    n = len(vals)
    hdr = vals[0] if vals else []
    summary[ws.title] = {"rows": n, "cols": hdr[:40]}
    print(f"{ws.title}: {n} rows; cols={hdr[:25]}")
    if n > 1 and ws.title in {"Sheet1", "Trade_Signals_Log", "Trade_Journal", "Portfolio", "execution", "execution_2", "Trade_Log", "Signals", "Manual Journal"} or ws.index == 0:
        df = pd.DataFrame(vals[1:], columns=[c if c else f"col{i}" for i, c in enumerate(hdr)])
        safe = ws.title.replace(" ", "_")
        df.to_csv(OUT / f"sheets_{safe}.csv", index=False)
(OUT / "estimation_haircut_sheets_summary.json").write_text(json.dumps(summary, indent=1))
