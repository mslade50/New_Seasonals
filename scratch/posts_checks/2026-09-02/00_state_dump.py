"""Housekeeping for tonight's queue: lint universe membership for candidate
tickers, and the rest of this morning's pitch stand-down (near-misses + kills)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from posts_grammar import _universe_sets  # noqa: E402

liquid, overflow = _universe_sets()
for t in ["EWZ", "UNG", "JPY=X", "XLI", "ITA", "EEM", "SVXY", "TLT", "IEF", "USO", "^BVSP", "UUP", "FXY"]:
    print(f"{t:<8} liquid={t in liquid}  overflow={t in overflow}")

print("\n=== pitch stand-down 2026-09-02: remaining near-misses ===")
for line in open("data/pitch_journal.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r.get("kind") == "stand_down" and r.get("date") == "2026-09-02":
        for c in r["closest"][1:]:
            print("-", c["title"])
            print("   why_died:", c["why_died"][:900])
print("\n=== kills ===")
for line in open("data/pitch_journal.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r.get("kind") == "killed" and r.get("date") == "2026-09-02":
        print("-", r.get("title"), "::", (r.get("reason") or "")[:350])
