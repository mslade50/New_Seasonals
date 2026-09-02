"""Normalize a full TradingView CSV export into EP research snapshots."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from episodic_pivot.tradingview import (  # noqa: E402
    TradingViewImportError,
    import_tradingview_csv,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a full TradingView export for EP research (dry-run by default)"
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--session", required=True, choices=("premarket", "after_hours"))
    parser.add_argument("--captured-at", required=True, help="timezone-aware capture timestamp")
    parser.add_argument("--screen-id", required=True, help="TradingView saved-screen identifier")
    parser.add_argument(
        "--reported-count",
        type=int,
        help="result count shown by TradingView; mismatch fails closed",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--write-artifact",
        action="store_true",
        help="write the normalized local JSON; never performs research, broker, or production actions",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = import_tradingview_csv(
            args.input,
            session=args.session,
            captured_at=args.captured_at,
            saved_screen_id=args.screen_id,
            reported_result_count=args.reported_count,
        )
    except (OSError, TradingViewImportError) as exc:
        raise SystemExit(f"TradingView import rejected: {exc}") from exc

    verification = "verified" if result.result_count_verified else "not independently verified"
    print(
        f"Validated {result.extracted_row_count} row(s) for "
        f"{result.target_session_date}; displayed count {verification}."
    )
    if not args.write_artifact:
        print("Dry run only: no file was written. Add --write-artifact to create a local snapshot JSON.")
        return 0

    if args.output is None:
        short_hash = result.source_file_sha256[:12]
        args.output = (
            ROOT
            / "artifacts"
            / "episodic_pivot"
            / "imports"
            / f"{result.target_session_date}-{result.session}-{short_hash}.json"
        )
    output = args.output.resolve()
    allowed_root = (ROOT / "artifacts").resolve()
    if output != allowed_root and allowed_root not in output.parents:
        raise SystemExit("--output must stay under this worktree's artifacts directory")
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing import: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote discovery-only snapshot: {output}")
    print("Safety: market status UNKNOWN, tradeable false, broker route absent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
