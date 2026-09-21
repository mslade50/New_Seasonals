"""Quote, submit, download, and convert Databento futures data safely.

The default request targets the volume-based front contracts for ES, NQ, and
RTY.  ``quote`` is metadata-only and cannot incur market-data usage charges.
``submit`` is deliberately guarded by both a cost ceiling and an explicit
confirmation token so an accidental invocation cannot create a billable job.

Databento reads ``DATABENTO_API_KEY`` from the environment.  For local use,
this script also loads the repository's gitignored ``.env`` file when present.
"""

from __future__ import annotations

import argparse
import getpass
import math
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "databento"
DEFAULT_DATASET = "GLBX.MDP3"
DEFAULT_ROOTS = ("ES", "NQ", "RTY")
DEFAULT_SCHEMA = "ohlcv-1m"
DEFAULT_START = "2010-06-06"
SUBMIT_CONFIRMATION = "SUBMIT_BILLABLE_JOB"
KEYRING_SERVICE = "New_Seasonals.Databento"
KEYRING_USERNAME = "prod-001"


@dataclass(frozen=True)
class RequestSpec:
    dataset: str
    symbols: tuple[str, ...]
    schema: str
    start: str
    end: str
    stype_in: str = "continuous"

    def metadata_kwargs(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "symbols": list(self.symbols),
            "schema": self.schema,
            "start": self.start,
            "end": self.end,
            "stype_in": self.stype_in,
        }


def build_spec(args: argparse.Namespace) -> RequestSpec:
    symbols = tuple(f"{root.upper()}.{args.roll_rule}.0" for root in args.roots)
    return RequestSpec(
        dataset=args.dataset,
        symbols=symbols,
        schema=args.schema,
        start=args.start,
        end=args.end,
    )


def _load_databento() -> Any:
    try:
        import databento as db
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise SystemExit(
            "The Databento client is not installed. Run: "
            "python -m pip install -r requirements.txt"
        ) from exc
    return db


def _load_api_key() -> str:
    try:
        from dotenv import load_dotenv
    except ImportError:
        load_dotenv = None

    if load_dotenv is not None:
        load_dotenv(ROOT / ".env")

    key = os.environ.get("DATABENTO_API_KEY", "").strip()
    if key:
        return key

    try:
        import keyring
    except ImportError:
        keyring = None
    if keyring is not None:
        key = (keyring.get_password(KEYRING_SERVICE, KEYRING_USERNAME) or "").strip()
        if key:
            return key

    raise SystemExit(
        "DATABENTO_API_KEY is not configured. Run `python "
        "scripts/databento_futures.py key set` and paste the key into the "
        "hidden prompt, or set the environment variable."
    )


def make_client() -> Any:
    key = _load_api_key()
    if not key:
        raise SystemExit(
            "DATABENTO_API_KEY is empty; never commit or paste it into logs."
        )
    os.environ["DATABENTO_API_KEY"] = key
    return _load_databento().Historical()


def validate_api_key(key: str) -> str:
    key = key.strip()
    if len(key) != 32 or not key.startswith("db-"):
        raise ValueError("Databento API keys must be 32 characters and start with 'db-'.")
    return key


def store_api_key(key: str, keyring_module: Any) -> None:
    keyring_module.set_password(
        KEYRING_SERVICE,
        KEYRING_USERNAME,
        validate_api_key(key),
    )


def store_api_key_in_env_file(key: str, env_path: Path = ROOT / ".env") -> None:
    """Add or replace DATABENTO_API_KEY without exposing other .env entries."""
    key = validate_api_key(key)
    existing = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
    lines = existing.splitlines()
    replacement = f"DATABENTO_API_KEY={key}"
    output: list[str] = []
    replaced = False
    for line in lines:
        if line.lstrip().startswith("DATABENTO_API_KEY="):
            if not replaced:
                output.append(replacement)
                replaced = True
            continue
        output.append(line)
    if not replaced:
        if output and output[-1] != "":
            output.append("")
        output.append(replacement)
    env_path.write_text("\n".join(output) + "\n", encoding="utf-8")


def _finite_nonnegative(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite nonnegative number.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite nonnegative number.") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a finite nonnegative number.")
    return result


def get_quote(client: Any, spec: RequestSpec) -> tuple[float, int]:
    kwargs = spec.metadata_kwargs()
    cost = _finite_nonnegative(client.metadata.get_cost(**kwargs), "Fresh quote")
    size = _finite_nonnegative(client.metadata.get_billable_size(**kwargs), "Billable size")
    if not size.is_integer():
        raise ValueError("Billable size must be an integer number of bytes.")
    billable_size = int(size)
    return cost, billable_size


def format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1000.0 or unit == "TB":
            return f"{value:,.1f} {unit}"
        value /= 1000.0
    raise AssertionError("unreachable")


def print_quote(spec: RequestSpec, cost: float, billable_size: int) -> None:
    print(f"Dataset: {spec.dataset}")
    print(f"Symbols: {', '.join(spec.symbols)} ({spec.stype_in})")
    print(f"Schema: {spec.schema}")
    print(f"Range: {spec.start} through {spec.end} (end exclusive, UTC)")
    print(f"Estimated billable size: {format_bytes(billable_size)}")
    print(f"Estimated cost: ${cost:,.2f}")


def validate_submit(cost: float, max_cost_usd: float | None, confirmation: str) -> None:
    if confirmation != SUBMIT_CONFIRMATION:
        raise ValueError(
            f"Refusing to submit. Pass --confirm {SUBMIT_CONFIRMATION} only after "
            "reviewing the fresh quote."
        )
    if max_cost_usd is None:
        raise ValueError("Refusing to submit without --max-cost-usd.")
    cost = _finite_nonnegative(cost, "Fresh quote")
    max_cost_usd = _finite_nonnegative(max_cost_usd, "Maximum cost")
    if cost > max_cost_usd:
        raise ValueError(
            f"Fresh quote ${cost:,.2f} exceeds --max-cost-usd ${max_cost_usd:,.2f}."
        )


def submit_job(client: Any, spec: RequestSpec, max_cost_usd: float, confirmation: str) -> str:
    cost, billable_size = get_quote(client, spec)
    print_quote(spec, cost, billable_size)
    validate_submit(cost, max_cost_usd, confirmation)
    details = client.batch.submit_job(
        **spec.metadata_kwargs(),
        encoding="dbn",
        compression="zstd",
        split_duration="year",
    )
    return str(details["id"])


def safe_job_summary(details: dict[str, Any]) -> dict[str, Any]:
    allowed = (
        "id",
        "state",
        "progress",
        "cost_usd",
        "dataset",
        "symbols",
        "schema",
        "start",
        "end",
        "record_count",
        "billed_size",
        "actual_size",
        "ts_expiration",
    )
    return {key: details.get(key) for key in allowed if key in details}


def _expand_dbn_inputs(files: Iterable[Path]) -> list[Path]:
    expanded: list[Path] = []
    for source in files:
        if source.is_dir():
            expanded.extend(
                path
                for path in sorted(source.iterdir())
                if path.name.endswith((".dbn", ".dbn.zst"))
            )
        else:
            expanded.append(source)
    return expanded


def convert_dbn_files(db: Any, files: Iterable[Path], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    converted: list[Path] = []
    for source in _expand_dbn_inputs(files):
        if not source.name.endswith((".dbn", ".dbn.zst")):
            continue
        stem = source.name.removesuffix(".zst").removesuffix(".dbn")
        target = output_dir / f"{stem}.parquet"
        db.DBNStore.from_file(source).to_parquet(
            target,
            price_type="float",
            pretty_ts=True,
            map_symbols=True,
        )
        converted.append(target)
    return converted


def add_request_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--roots", nargs="+", default=list(DEFAULT_ROOTS))
    parser.add_argument("--schema", default=DEFAULT_SCHEMA)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument(
        "--end",
        default=date.today().isoformat(),
        help="Exclusive UTC end date (default: today, which includes yesterday).",
    )
    parser.add_argument(
        "--roll-rule",
        choices=("v", "n", "c"),
        default="v",
        help="Continuous roll rule: volume, open interest, or calendar.",
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    quote_parser = subparsers.add_parser(
        "quote", help="Fetch free metadata cost/size estimates; submits no data request."
    )
    add_request_arguments(quote_parser)

    submit_parser = subparsers.add_parser(
        "submit", help="Submit a billable batch job after explicit cost guards."
    )
    add_request_arguments(submit_parser)
    submit_parser.add_argument("--max-cost-usd", type=float, required=True)
    submit_parser.add_argument("--confirm", required=True)

    status_parser = subparsers.add_parser("status", help="Show a sanitized batch-job status.")
    status_parser.add_argument("job_id")

    download_parser = subparsers.add_parser("download", help="Download an existing batch job.")
    download_parser.add_argument("job_id")
    download_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    convert_parser = subparsers.add_parser(
        "convert", help="Convert local DBN files or directories to Parquet."
    )
    convert_parser.add_argument("files", type=Path, nargs="+")
    convert_parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "parquet"
    )

    key_parser = subparsers.add_parser(
        "key", help="Store or check the API key in the operating-system credential vault."
    )
    key_parser.add_argument("action", choices=("set", "set-file", "status"))
    return parser


def main() -> int:
    args = make_parser().parse_args()

    if args.command == "key":
        try:
            import keyring
        except ImportError as exc:
            raise SystemExit("Install the `keyring` package before storing the key.") from exc
        if args.action in {"set", "set-file"}:
            key = getpass.getpass("Paste Databento API key (input hidden): ")
            try:
                if args.action == "set-file":
                    store_api_key_in_env_file(key)
                else:
                    store_api_key(key, keyring)
            except ValueError as exc:
                raise SystemExit(str(exc)) from exc
            if args.action == "set-file":
                print("Databento API key stored in the repository's gitignored .env file.")
            else:
                print("Databento API key stored in the operating-system credential vault.")
        else:
            try:
                from dotenv import load_dotenv

                load_dotenv(ROOT / ".env")
            except ImportError:
                pass
            configured = bool(os.environ.get("DATABENTO_API_KEY", "").strip()) or bool(
                keyring.get_password(KEYRING_SERVICE, KEYRING_USERNAME)
            )
            print("configured" if configured else "not configured")
        return 0

    if args.command == "convert":
        converted = convert_dbn_files(_load_databento(), args.files, args.output_dir)
        for path in converted:
            print(path)
        return 0

    client = make_client()
    if args.command == "quote":
        spec = build_spec(args)
        cost, billable_size = get_quote(client, spec)
        print_quote(spec, cost, billable_size)
        print("Quote only: no data request was submitted.")
    elif args.command == "submit":
        try:
            job_id = submit_job(
                client,
                build_spec(args),
                max_cost_usd=args.max_cost_usd,
                confirmation=args.confirm,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        print(f"Submitted batch job: {job_id}")
    elif args.command == "status":
        print(safe_job_summary(client.batch.get_job_details(args.job_id)))
    elif args.command == "download":
        paths = client.batch.download(job_id=args.job_id, output_dir=args.output_dir)
        for path in paths:
            print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
