"""Render an expected-flat control report from one local JSON manifest.

This command has no broker, network, R2, Sheets, SMTP, or scheduler adapter.
It only reads the named local file and publishes an immutable JSON, Markdown,
and HTML report bundle under the named local output directory. LIVE and
operationally authoritative input are categorically rejected.
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
    load_manifest_with_digest,
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
        choices=[RunMode.DISABLED.value, RunMode.FIXTURE.value, RunMode.SHADOW.value],
        help="Must exactly match run.run_mode in the manifest",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest, input_sha256 = load_manifest_with_digest(args.input)
        if manifest.run.run_mode is RunMode.LIVE:
            raise ManifestError(
                "the file-only CLI rejects LIVE manifests; use DISABLED, FIXTURE, or SHADOW"
            )
        if manifest.run.operationally_authoritative:
            raise ManifestError(
                "the file-only CLI rejects operationally authoritative manifests"
            )
        requested_mode = RunMode(args.run_mode)
        if requested_mode is not manifest.run.run_mode:
            raise ManifestError(
                f"CLI run mode {requested_mode.value} does not match manifest mode {manifest.run.run_mode.value}"
            )
        report = evaluate(manifest, input_sha256=input_sha256)
        paths = write_artifacts(report, args.output_dir, input_path=args.input)
    except ManifestError as exc:
        print(f"INPUT ERROR: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"OUTPUT ERROR: {exc}", file=sys.stderr)
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
