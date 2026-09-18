import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
t = json.load(open(ROOT / "data" / "pitch_tape.json"))["tickers"]
s = json.load(open(ROOT / "data" / "pitch_state.json"))

cols = ["ret_1d", "ret_5d", "ret_21d", "ret_63d", "rank_5d", "rank_21d", "rank_63d", "z10",
        "atr_pct", "rvol21_ann", "dist_52w_high_pct", "dist_52w_low_pct", "dist_sma200_pct", "vol_vs_63d"]


def row(k, v):
    return f"{k:10s} " + " ".join(f"{(v.get(c) if v.get(c) is not None else float('nan')):7.2f}" for c in cols)


print("HEADLINE")
print(" " * 11 + " ".join(f"{c[:7]:>7s}" for c in cols))
for k, v in s["tape"]["headline"].items():
    print(row(k, v))

print("\nBREADTH", json.dumps(s["tape"]["breadth"]))
print("\nEXTREMES", json.dumps(s["tape"]["extremes"])[:4000])

items = [(k, v) for k, v in t.items() if v.get("z10") is not None]
for key, n in [("z10", 15), ("rank_5d", 12), ("rank_21d", 12), ("rank_63d", 12), ("dist_sma200_pct", 12)]:
    srt = sorted(items, key=lambda kv: kv[1].get(key) if kv[1].get(key) is not None else 0)
    print(f"\nLOWEST {key}")
    for k, v in srt[:n]:
        print(row(k, v))
    print(f"HIGHEST {key}")
    for k, v in srt[-n:]:
        print(row(k, v))

print("\nAT 52W HIGH (<=0.5%)")
for k, v in items:
    if v.get("dist_52w_high_pct") is not None and v["dist_52w_high_pct"] >= -0.5:
        print(row(k, v))
print("\nAT 52W LOW (<=0.5%)")
for k, v in items:
    if v.get("dist_52w_low_pct") is not None and v["dist_52w_low_pct"] <= 0.5:
        print(row(k, v))
print("\nALL TICKERS:", " ".join(sorted(t.keys())))
