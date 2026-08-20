"""Attach a browser-QA attestation to the latest fundamental run manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fundamental.config import REPORT_ROOT  # noqa: E402
from fundamental.run_manifest import LATEST_MANIFEST_PATH, record_visual_qa  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record a real local-browser visual inspection for an exact report digest."
    )
    parser.add_argument("--manifest", type=Path, default=LATEST_MANIFEST_PATH)
    parser.add_argument("--report", type=Path, default=REPORT_ROOT / "fundamental_daily.html")
    parser.add_argument("--status", choices=["PASS", "FAIL"], required=True)
    parser.add_argument("--notes", required=True)
    args = parser.parse_args()
    qa_path, manifest_path = record_visual_qa(
        manifest_path=args.manifest,
        report_path=args.report,
        passed=args.status == "PASS",
        notes=args.notes,
    )
    print(f"Visual QA: {args.status}")
    print(f"Attestation: {qa_path}")
    print(f"Latest manifest: {manifest_path}")


if __name__ == "__main__":
    main()
