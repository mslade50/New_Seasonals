"""Check the trade ledger for an EWZ entry around 6/9-6/10 2026."""
from pathlib import Path
import pandas as pd
pd.set_option("display.width", 240); pd.set_option("display.max_columns", 60)

ROOT = Path(r"C:\Users\McKinley Slade\dev\New_Seasonals")
LED = ROOT / "data" / "backtest_trades_full.parquet"
if not LED.exists():
    print("no ledger"); raise SystemExit
t = pd.read_parquet(LED)
print("ledger shape:", t.shape)
print("cols:", list(t.columns))
tick_col = next((c for c in t.columns if c.lower() in ("ticker","symbol")), None)
date_cols = [c for c in t.columns if "date" in c.lower() or "entry" in c.lower() or "exit" in c.lower()]
print("ticker col:", tick_col, "date-ish cols:", date_cols)
if tick_col:
    ewz = t[t[tick_col].astype(str).str.upper() == "EWZ"]
    print("EWZ rows in ledger:", len(ewz))
    if len(ewz):
        # show most recent
        showc = [tick_col] + date_cols + [c for c in t.columns if c.lower() in
                 ("strategy","direction","entry_price","entryprice","exit_price","r","price")]
        showc = list(dict.fromkeys([c for c in showc if c in t.columns]))
        print(ewz[showc].tail(12).to_string())
