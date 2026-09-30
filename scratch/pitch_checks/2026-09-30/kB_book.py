"""Book overlap for C1/C4/C7: what the systematic ledger holds (open at the
2026-09-29 close) in the candidate names/sectors, plus recent history of the
book in those names."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import pandas as pd

ROOTP = Path(__file__).resolve().parents[3]
b = pd.read_parquet(ROOTP / "data" / "backtest_trades_full.parquet")
last = pd.Timestamp("2026-09-29")
opn = b[(b["Exit Date"] >= last) & (b["Time Stop"] > last)]
print(f"ledger rows {len(b)}; open at {last.date()} (exit=last bar, time stop later): {len(opn)}")
print(opn[["Strategy", "Ticker", "Direction", "Signal Date", "Entry Date", "Time Stop",
           "Risk bps"]].to_string(index=False))

names = {"C1": ["XLE", "XLV", "XLU", "XLI", "XLY", "XLB", "XLF", "XLK", "XLP"],
         "C4": ["XLF", "JPM", "C", "WFC", "GS", "BAC", "MS", "KRE", "KBE"],
         "C7": ["XLRE", "IYR", "VNQ", "PLD", "AMT", "EQIX", "SPG", "O", "PSA", "WELL"]}
for c, tk in names.items():
    s = b[b.Ticker.isin(tk)]
    rec = s[s["Signal Date"] >= "2026-06-01"]
    print(f"\n{c}: ledger trades ever in {tk}: {len(s)}; since 2026-06-01: {len(rec)}")
    if len(rec):
        print(rec[["Strategy", "Ticker", "Direction", "Entry Date", "Exit Date",
                   "Return_Pct"]].tail(8).to_string(index=False))
    o = opn[opn.Ticker.isin(tk)]
    print(f"  open now in these names: {len(o)}")

# same-season history: what the book did in these names across past Q3->Q4 turns
b["ed"] = pd.to_datetime(b["Entry Date"])
turn = b[(b.ed.dt.month == 10) & (b.ed.dt.day <= 12)]
for c, tk in names.items():
    t = turn[turn.Ticker.isin(tk)]
    print(f"{c}: book entries in these names Oct 1-12 across all years: {len(t)}"
          + (f" ({t.groupby('Strategy').size().to_dict()})" if len(t) else ""))
