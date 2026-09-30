"""Prepare the read-only workbench extension; no installation or broker I/O."""
from pathlib import Path

ANCHOR = '    mode = str(q.get("mode") or "full").lower()\n'
HOOK = '''    if mode == "last_price":
        from ticket_last_price import resolve
        print(json.dumps(resolve(q)))
        return
'''


def prepare(source):
    if HOOK in source:
        raise ValueError("last-price extension is already installed")
    if source.count(ANCHOR) != 1:
        raise ValueError("workbench entry point differs; review before promotion")
    return source.replace(ANCHOR, ANCHOR + HOOK)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(prepare(args.source.read_text(encoding="utf-8")), encoding="utf-8")
