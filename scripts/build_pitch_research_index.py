"""Research-memory index for the Daily Pitch pipeline (Stage A input).

The ideation stage has to know what has already been studied — both so it can
cite an existing finding instead of re-deriving it, and so it stops inventing
things the repo already killed. Reading every research doc each morning is
neither affordable nor necessary; a title + opening line + size + mtime is
enough for the agent to decide which two or three to actually open.

Indexes scratch/ultracode_research/*.md (the research corpus) and the
pre-registration docs alongside them, plus data/pitch_negative_registry.md.

    python scripts/build_pitch_research_index.py [--out data/pitch_research_index.json]

Consumed by scripts/build_pitch_state.py (embeds the index in pitch_state.json).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESEARCH_DIR = ROOT / "scratch" / "ultracode_research"
REGISTRY_PATH = ROOT / "data" / "pitch_negative_registry.md"
DEFAULT_OUT = ROOT / "data" / "pitch_research_index.json"

# Registry entries are "- **slug** — text (cite)" bullets under ## sections.
_BULLET = re.compile(r"^\s*[-*]\s+\*\*(?P<key>[^*]+)\*\*\s*[—:-]\s*(?P<body>.+)$")


def _title_and_lede(path: Path) -> tuple[str, str]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return path.stem, ""
    title, lede = "", ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if not title:
            title = line.lstrip("#").strip()
            continue
        if line.startswith("#"):
            continue
        lede = line
        break
    return title or path.stem, lede[:400]


def index_research(research_dir: Path = RESEARCH_DIR) -> list[dict]:
    docs = []
    for path in sorted(research_dir.glob("*.md")):
        title, lede = _title_and_lede(path)
        stat = path.stat()
        docs.append({
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "title": title,
            "lede": lede,
            "kb": round(stat.st_size / 1024, 1),
            "modified": dt.date.fromtimestamp(stat.st_mtime).isoformat(),
            "is_prereg": "prereg" in path.name.lower(),
        })
    return docs


def parse_registry(path: Path = REGISTRY_PATH) -> list[dict]:
    """Flat list of dead ends from the negative-results registry."""
    if not path.exists():
        return []
    entries, section = [], ""
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.rstrip()
        if line.startswith("##"):
            section = line.lstrip("#").strip()
            continue
        match = _BULLET.match(line)
        if match:
            entries.append({"section": section,
                            "key": match.group("key").strip(),
                            "why_dead": match.group("body").strip()})
    return entries


def build(research_dir: Path = RESEARCH_DIR,
          registry: Path = REGISTRY_PATH) -> dict:
    docs = index_research(research_dir)
    dead = parse_registry(registry)
    return {
        "generated": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "research_dir": str(research_dir.relative_to(ROOT)).replace("\\", "/"),
        "doc_count": len(docs),
        "docs": docs,
        "negative_registry_path": str(registry.relative_to(ROOT)).replace("\\", "/"),
        "negative_registry_count": len(dead),
        "negative_registry": dead,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    payload = build()
    Path(args.out).write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"Indexed {payload['doc_count']} research docs and "
          f"{payload['negative_registry_count']} registry entries -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
