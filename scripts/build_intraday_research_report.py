"""Build the standalone HTML report for a completed intraday research bundle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.intraday.report import write_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = write_report(args.bundle_dir, overwrite=args.overwrite)
    print(f"Research report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

