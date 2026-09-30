"""c10/c11 book overlap (2026-09-21): which book strategies can trade SPY / LQD /
IEF / HYG, and what the event sleeve holds inside the c10 hold (09-21 -> 11-02)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
import strategy_config as sc
import event_sleeve as es
import pandas as pd

TK = ["SPY", "^GSPC", "LQD", "IEF", "HYG"]
for s in sc.STRATEGY_BOOK:
    name = s.get("name") if isinstance(s, dict) else str(s)
    uni = s.get("universe_tickers") or s.get("tickers") or [] if isinstance(s, dict) else []
    hit = [t for t in TK if t in (uni or [])]
    if hit:
        side = s.get("settings", {}).get("trade_direction", s.get("direction", "?"))
        print(f"  {name}: trades {hit} ({side})")
print("event sleeve config:")
for k, v in es.EVENT_SLEEVE.items():
    print(f"  {k}: {v}")
ev = pd.read_csv(Path(sc.__file__).parent / "data" / "macro_events.csv")
ev["date"] = pd.to_datetime(ev["date"])
w = ev[(ev.date > "2026-09-21") & (ev.date <= "2026-11-02")]
print("macro events inside the c10 hold (09-21 close -> 11-02 close):")
print(w[["date", "event"]].to_string(index=False))
