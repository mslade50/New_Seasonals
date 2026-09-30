import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
w = json.loads((ROOT / "data/pitch_watchlist.json").read_text(encoding="utf-8"))
out = []
for i, e in enumerate(w["entries"]):
    out.append(f"\n[{i}] added {e.get('added')} exp {e.get('expires')} | cell {e.get('cell')} | {e.get('title')}")
    out.append(f"   TRIGGER: {e.get('trigger', '')}")
    out.append(f"   SCRIPT: {e.get('script')}")
    out.append(f"   NOTE: {e.get('note', '')}")
(Path(__file__).parent / "w_dump_full.txt").write_text("\n".join(out), encoding="utf-8")
print(len(w["entries"]))
