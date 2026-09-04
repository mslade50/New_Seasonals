"""Dry run of the edited order_staging.py on a SANDBOX COPY.

Goal: prove the Entry_Expire_Time change runs end-to-end through the real
pull_and_stage_orders() pipeline without crashing, and produces correct values,
WITHOUT touching prod (no IBKR, no Google Sheets, no prod staged_orders.csv) so
the real order_staging can run in prod concurrently.

How it stays safe:
  * imports the sandbox COPY (its SCRIPT_DIR/OUTPUT_FOLDER point at the sandbox)
  * connect_to_ibkr -> (None, False)         : no TWS / gateway connection
  * fetch_pa_account_value -> None           : skips execution_2 (no 2nd connect)
  * gspread.service_account -> FakeGC         : synthetic Order_Staging/Overflow
  * update_execution_tab[_2] -> capture       : no sheet writes
Outputs land only in the sandbox; we assert the prod CSV's mtime is unchanged.
"""
import importlib.util
import os
import sys
import pandas as pd

SANDBOX = os.path.join(
    os.environ.get("LOCALAPPDATA", ""),
    "Temp", "claude", "C--Users-McKinley-Slade-dev-New-Seasonals",
    "0af20baf-8b32-4420-b892-fd3bfb69af01", "scratchpad", "dryrun_sandbox",
)
PROD_CSV = r"C:\Users\McKinley Slade\OneDrive\trading_ibkr\staged_orders.csv"

# ---- import the sandbox copy as a module ----
spec = importlib.util.spec_from_file_location(
    "order_staging_sandbox", os.path.join(SANDBOX, "order_staging.py"))
osd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(osd)


# ---- fakes for all live I/O ----
_captured = {}

class _FakeWS:
    def __init__(self, records): self._records = records
    def get_all_records(self): return list(self._records)

class _FakeSheet:
    def __init__(self, tabs): self._tabs = tabs
    def worksheet(self, name):
        if name in self._tabs:
            return _FakeWS(self._tabs[name])
        raise osd.gspread.WorksheetNotFound(name)

class _FakeGC:
    def __init__(self, tabs): self._tabs = tabs
    def open(self, name): return _FakeSheet(self._tabs)


def _row(symbol, strat, time_exit, hold, fill, qty=100, signal_close=100.0,
         atr=2.0, action="BUY", direction="Long", scan_source="Main"):
    """A staging row shaped like daily_scan writes to the sheet (str values,
    as get_all_records would return them where relevant)."""
    return {
        "Scan_Date": "2026-06-24",
        "Symbol": symbol, "SecType": "STK", "Exchange": "SMART",
        "Action": action, "Quantity": qty,
        "Order_Type": "REL_CLOSE", "Limit_Price": 0.0,
        "Offset_ATR_Mult": 0.25, "TIF": "GTC",
        "Frozen_ATR": atr, "Signal_Close": signal_close, "Signal_High": 0,
        "Time_Exit_Date": time_exit,
        "Strategy_Ref": strat,
        "Tgt_ATR_Mult": 2.5, "Stop_ATR_Mult": 1.25,
        "Use_Target": True, "Use_Stop": True,
        "Hold_Days": hold, "Fill_Window_Days": fill,
        "Trade_Direction": direction,
        "Rank_252D": "", "Risk_Amt": 500.0, "Risk_Bps": 35,
        "Scan_Source": scan_source,
    }


# OLV rows (fill=3 < hold=10  -> should shorten). One liquid, one overflow.
# Non-OLV persistent row (fill==hold -> must NOT shorten).
main_rows = [
    _row("AAA", "Oversold Low Volume", "2026-07-15", 10, 3),
    _row("BBB", "LT Trend ST OS",      "2026-07-02",  1, 1),
]
overflow_rows = [
    _row("CCC", "Oversold Low Volume", "2026-07-20", 10, 3, scan_source="Overflow"),
]
tabs = {"Order_Staging": main_rows, "Overflow": overflow_rows}

osd.connect_to_ibkr = lambda: (None, False)
osd.fetch_pa_account_value = lambda: None
osd.update_execution_tab = lambda sh, df: _captured.__setitem__("exec", df.copy())
osd.update_execution_tab_2 = lambda sh, df: _captured.__setitem__("exec2", df.copy())
osd.gspread.service_account = lambda filename=None: _FakeGC(tabs)
osd.OUTPUT_FOLDER = SANDBOX  # belt-and-suspenders (already the sandbox via SCRIPT_DIR)

# ---- record prod CSV mtime to prove we never touch it ----
prod_mtime_before = os.path.getmtime(PROD_CSV) if os.path.exists(PROD_CSV) else None

print("=" * 70)
print("RUNNING pull_and_stage_orders() on the SANDBOX copy (no IBKR / no Sheets)")
print("=" * 70)
osd.pull_and_stage_orders()

# ---- validate ----
print("\n" + "=" * 70)
print("VALIDATION")
print("=" * 70)
out_csv = os.path.join(SANDBOX, "staged_orders.csv")
assert os.path.exists(out_csv), "sandbox staged_orders.csv was not written"
out = pd.read_csv(out_csv)
print(f"sandbox CSV rows: {len(out)}  cols: {list(out.columns)}")

assert "Entry_Expire_Time" in out.columns, "Entry_Expire_Time column missing"
# column sits right after Exit_Condition_Time
ci = list(out.columns)
assert ci.index("Entry_Expire_Time") == ci.index("Exit_Condition_Time") + 1, "column order"

TD = osd.TRADING_DAY
def expect_shorten(exit_str, hold, fill):
    ed = pd.to_datetime(str(exit_str).split(" ")[0])
    return (ed - (1 + hold - fill) * TD).date()

ok = True
for _, r in out.iterrows():
    sym, strat = r["Symbol"], r["Strategy_Ref"]
    ec, ee = str(r["Exit_Condition_Time"]), str(r["Entry_Expire_Time"])
    if strat == "Oversold Low Volume":
        exp = f"{expect_shorten(ec, 10, 3)} 15:59:00"
        good = (ee == exp) and (ee != ec)
        print(f"  {sym:4} OLV       exit={ec}  entry_expire={ee}  expect={exp}  {'OK' if good else 'FAIL'}")
        ok &= good
    else:
        good = ee == ec
        print(f"  {sym:4} {strat:14} exit={ec}  entry_expire={ee}  (mirror) {'OK' if good else 'FAIL'}")
        ok &= good

# prod untouched
prod_mtime_after = os.path.getmtime(PROD_CSV) if os.path.exists(PROD_CSV) else None
prod_safe = prod_mtime_before == prod_mtime_after
print(f"\nprod staged_orders.csv mtime unchanged: {prod_safe}")
print(f"execution tab write captured (not sent): {'exec' in _captured}")

print("\n" + ("ALL DRY-RUN CHECKS PASSED" if (ok and prod_safe) else "*** DRY RUN FAILED ***"))
sys.exit(0 if (ok and prod_safe) else 1)
