import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
t = json.loads((ROOT / "data" / "pitch_tape.json").read_text())["tickers"]
cols = ["ret_1d", "ret_5d", "ret_21d", "ret_63d", "rank_5d", "rank_21d", "rank_63d", "z10",
        "atr_pct", "rvol21_ann", "dist_52w_high_pct", "dist_52w_low_pct", "dist_sma200_pct", "vol_vs_63d"]
print("ticker   " + " ".join(f"{c[:9]:>9}" for c in cols))
for name in sorted(t):
    r = t[name]
    vals = []
    for c in cols:
        v = r.get(c)
        vals.append(f"{v:>9.2f}" if isinstance(v, (int, float)) else f"{'NA':>9}")
    print(f"{name:<8} " + " ".join(vals) + ("" if r.get("date") == "2026-09-14" else f"  STALE {r.get('date')}"))
