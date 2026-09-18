import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
t = json.loads((ROOT / "data" / "pitch_tape.json").read_text(encoding="utf-8"))["tickers"]
rows = [(k, v) for k, v in t.items() if v.get("close") is not None]
print(f"N={len(rows)}")


def show(field: str, n: int = 12, reverse: bool = False) -> None:
    good = [(k, v) for k, v in rows if v.get(field) is not None]
    good.sort(key=lambda kv: kv[1][field], reverse=reverse)
    tag = "TOP" if reverse else "BOTTOM"
    line = ", ".join(f"{k} {v[field]:.2f}" for k, v in good[:n])
    print(f"{field} {tag}: {line}")


for f in ["ret_1d", "rank_5d", "rank_21d", "rank_63d", "z10", "dist_52w_high_pct",
          "dist_52w_low_pct", "dist_sma200_pct", "vol_vs_63d", "ret_252d"]:
    show(f)
    show(f, reverse=True)

print()
print("ALL (ticker: r1d r5d r21d r63d rk5 rk21 rk63 z10 d52h d52l d200 atr%):")
for k, v in sorted(rows):
    print(f"{k}: {v['ret_1d']} {v['ret_5d']} {v['ret_21d']} {v['ret_63d']} | "
          f"{v['rank_5d']} {v['rank_21d']} {v['rank_63d']} z{v['z10']} | "
          f"{v['dist_52w_high_pct']} {v['dist_52w_low_pct']} {v['dist_sma200_pct']} a{v['atr_pct']}")
