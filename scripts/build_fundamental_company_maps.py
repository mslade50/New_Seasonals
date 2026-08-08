"""Build the founder-led and personalized company research maps."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fundamental.company_maps import DEFAULT_OUTPUT, build_company_maps_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", default=str(date.today()))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    report, support = build_company_maps_report(
        as_of=str(args.as_of), output_path=Path(args.output)
    )
    print(f"Company maps: {report}")
    print(f"Support data: {support}")


if __name__ == "__main__":
    main()

