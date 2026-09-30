import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
d = json.loads((ROOT / "data/context_state.json").read_text())

for g in d["cells_index"]:
    print(f"\n=== {g['trigger_id']} | {g['lane']} | {g['cell']} | {g.get('anchor_rule','')}")
    extra = {k: v for k, v in g.items() if k not in ("trigger_id", "lane", "cell", "anchor_rule", "subjects")}
    if extra:
        print("   ", extra)
    for s in g["subjects"]:
        keys = [k for k in s if k not in ("subject", "fp")]
        vals = " ".join(f"{k}={s[k]}" for k in keys)
        print(f"  {s['subject']:10s} {vals}")
