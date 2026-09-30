import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[3]
st = json.loads((root / "data" / "pitch_state.json").read_text(encoding="utf-8"))

def size(o) -> int:
    return len(json.dumps(o, default=str))

mode = sys.argv[1] if len(sys.argv) > 1 else "keys"
if mode == "keys":
    for k, v in st.items():
        sub = ""
        if isinstance(v, dict):
            sub = ", ".join(f"{kk}:{size(vv)}" for kk, vv in v.items())
        print(f"{k}: {size(v)}  [{sub}]")
else:
    path = mode.split(".")
    o = st
    for p in path:
        o = o[p] if not p.isdigit() else o[int(p)]
    print(json.dumps(o, indent=1, default=str)[: int(sys.argv[2]) if len(sys.argv) > 2 else 20000])
