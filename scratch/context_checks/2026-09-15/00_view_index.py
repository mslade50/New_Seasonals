import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
d = json.loads((ROOT / "data" / "context_state.json").read_text())
skip = set(sys.argv[1:])
for g in d["cells_index"]:
    if g["trigger_id"] in skip:
        continue
    extra = {k: v for k, v in g.items() if k not in ("subjects",)}
    print("\n==", json.dumps(extra))
    for s in g["subjects"]:
        rest = {k: v for k, v in s.items() if k not in ("subject", "fp")}
        print(f"  {s['subject']:<10} " + " ".join(f"{k}={v}" for k, v in rest.items()))
