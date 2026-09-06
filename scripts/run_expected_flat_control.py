"""Render an expected-flat control report from one local JSON manifest.

This command has no broker, network, R2, Sheets, SMTP, or scheduler adapter.
It only reads the named local file and atomically writes JSON, Markdown, and
HTML artifacts under the named local output directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from expected_flat_control import (
    ControlState,
    ManifestError,
    RunMode,
    evaluate,
    load_manifest,
    write_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a local expected-flat fixture/manifest; performs no external I/O."
    )
    parser.add_argument(
        "--input", type=Path, required=True, help="Local frozen-schema JSON manifest"
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Local artifact directory"
    )
    parser.add_argument(
        "--run-mode",
        required=True,
        choices=[mode.value for mode in RunMode],
        help="Must exactly match run.run_mode in the manifest",
    )
    parser.add_argument(
        "--acknowledge-authoritative-live",
        action="store_true",
        help="Second explicit gate required when the manifest claims authoritative LIVE mode",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = load_manifest(args.input)
        requested_mode = RunMode(args.run_mode)
        if requested_mode is not manifest.run.run_mode:
            raise ManifestError(
                f"CLI run mode {requested_mode.value} does not match manifest mode {manifest.run.run_mode.value}"
            )
        if (
            manifest.run.operationally_authoritative
            and not args.acknowledge_authoritative_live
        ):
            raise ManifestError(
                "authoritative LIVE input requires --acknowledge-authoritative-live; no action was taken"
            )
        if (
            args.acknowledge_authoritative_live
            and not manifest.run.operationally_authoritative
        ):
            raise ManifestError(
                "--acknowledge-authoritative-live is invalid for a non-authoritative manifest"
            )
        report = evaluate(manifest)
        paths = write_artifacts(report, args.output_dir)
    except ManifestError as exc:
        print(f"INPUT ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"state={report.state.value}")
    for kind, path in paths.items():
        print(f"{kind}={path.resolve()}")
    if report.state is ControlState.ALERT:
        return 10
    if report.state is ControlState.UNKNOWN:
        return 20
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
