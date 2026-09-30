import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
w = json.loads((ROOT / "data/pitch_watchlist.json").read_text(encoding="utf-8"))
print("expired:", json.dumps(w["expired"])[:600])
for i, e in enumerate(w["entries"]):
    trig = e.get("trigger", "")
    print(f"\n[{i}] {e.get('added')} exp {e.get('expires')} | {e.get('cell')} | {e.get('title')}")
    print(f"   TRIGGER: {trig[:420]}")
    print(f"   script: {e.get('script')}")
