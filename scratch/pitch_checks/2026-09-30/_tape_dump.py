import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
tape = json.loads((ROOT / "data" / "pitch_tape.json").read_text())
rows = tape.get("tickers", tape) if isinstance(tape, dict) else tape
if isinstance(rows, dict) and "tickers" not in tape:
    rows = {k: v for k, v in rows.items() if isinstance(v, dict)}
cols = ["ret_1d", "ret_5d", "ret_21d", "ret_63d", "rank_5d", "rank_21d", "rank_63d",
        "z10", "atr_pct", "dist_52w_high_pct", "dist_52w_low_pct", "dist_sma200_pct", "vol_vs_63d"]
print("ticker  " + " ".join(f"{c[:9]:>9}" for c in cols))
items = rows.items() if isinstance(rows, dict) else ((r.get("ticker"), r) for r in rows)
for t, r in sorted(items, key=lambda kv: kv[1].get("rank_5d") or 0):
    vals = []
    for c in cols:
        v = r.get(c)
        vals.append(f"{v:9.2f}" if isinstance(v, (int, float)) else f"{'na':>9}")
    print(f"{t:8}" + " ".join(vals))
