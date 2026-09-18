"""Sort the whole 218-name tape on every dimension. Stage B1 input, not a check."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
tape = json.load(open(ROOT / "data" / "pitch_tape.json"))
rows = tape["tickers"] if isinstance(tape, dict) and "tickers" in tape else tape
if isinstance(rows, dict):
    rows = [dict(ticker=k, **v) for k, v in rows.items()]

def top(key, n=14, rev=True):
    ok = [r for r in rows if r.get(key) is not None]
    ok.sort(key=lambda r: r[key], reverse=rev)
    return [(r["ticker"], round(r[key], 2)) for r in ok[:n]]

for key in ["rank_5d", "rank_21d", "rank_63d", "z10", "dist_52w_high_pct",
            "dist_52w_low_pct", "dist_sma200_pct", "ret_21d", "ret_63d",
            "ret_252d", "atr_pct", "vol_vs_63d", "rvol21_ann"]:
    print(f"--- {key} HIGH: {top(key)}")
    print(f"--- {key} LOW : {top(key, rev=False)}")
    print()
