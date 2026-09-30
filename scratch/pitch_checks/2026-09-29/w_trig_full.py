import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[3]
w = json.loads((root / "data" / "pitch_watchlist.json").read_text(encoding="utf-8"))
idx = [int(x) for x in sys.argv[1:]]
for i in idx:
    e = w["entries"][i]
    print(f"[{i}] {e.get('title')} | exp {e.get('expires')} | script {e.get('script')}")
    print(f"  TRIG: {e.get('trigger')}")
    note = str(e.get("note") or "")
    if note:
        print(f"  NOTE(last 400): {note[-400:]}")
    print()
