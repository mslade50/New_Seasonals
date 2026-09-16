"""Package observed agent review notes with their queue and text hashes. No research or sending."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from episodic_pivot.reviewed_news import PACKET_TYPE


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument(
        "--notes",
        type=Path,
        required=True,
        help="JSON list of actual search-and-read dispositions",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    queue = json.loads(args.queue.read_text(encoding="utf-8"))
    notes = json.loads(args.notes.read_text(encoding="utf-8"))
    if not isinstance(notes, list):
        raise TypeError("review notes must be a list")
    for note in notes:
        for source in note.get("sources", []):
            source["content"] = source["content"].strip()
            source["content_sha256"] = hashlib.sha256(
                source["content"].encode()
            ).hexdigest()
    packet = {
        "record_type": PACKET_TYPE,
        "reviewer": "CODEX_SEARCH_AND_READ",
        "queue": queue,
        "reviews": notes,
    }
    output = args.output.resolve()
    if (ROOT / "artifacts").resolve() not in output.parents:
        raise ValueError("review packet must stay under this worktree's artifacts")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(packet, handle, indent=2)
    print("Review notes packaged; not yet validated. No network or email action.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
