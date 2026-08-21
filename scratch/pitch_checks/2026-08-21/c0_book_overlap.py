"""Book overlap for all three candidates against the 23-year ledger.

C2  -- does the book trade SPY/index longs on the same 5d-washout days?
C7  -- what does the book do on bank-breadth BROKEN trigger days, which side?
C10 -- does anything trade a name inside +/-1 session of its own print? OVS
       carries a +/-10 td earnings blackout; confirm it in the ledger rather
       than take it on trust.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

pd.set_option("display.width", 250)

L = pd.read_parquet(ROOT / "data" / "backtest_trades_full.parquet")
print("ledger", L.shape)
print(sorted(L.columns.tolist()))
dc = "Signal_Date" if "Signal_Date" in L.columns else L.columns[0]
for c in L.columns:
    if L[c].dtype.kind == "M":
        print(" datetime col:", c, L[c].min(), L[c].max())
print(L.head(3).to_string())
