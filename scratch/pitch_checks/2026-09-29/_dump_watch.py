import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[3]
w = json.loads((root / "data" / "pitch_watchlist.json").read_text(encoding="utf-8"))
tlen = int(sys.argv[1]) if len(sys.argv) > 1 else 260
for i, e in enumerate(w["entries"]):
    print(f"[{i}] {e.get('added')} exp {e.get('expires')} | {e.get('title')}")
    print(f"     cell: {e.get('cell')}")
    print(f"     trig: {str(e.get('trigger'))[:tlen]}")
    note = str(e.get("note") or "")
    if note:
        print(f"     note: {note[-200:]}")
print("expired:", w.get("expired"))
