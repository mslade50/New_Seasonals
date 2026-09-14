"""Read existing broker/canonical sources and run the Primary exit monitor.

Local observation is the default. --send and --upload are separate operational
actions. This adapter never places orders, stages exits, repairs history or
initializes inventory. A reviewed seed and continuous history are prerequisites.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def publish_status(path, *, client=None, bucket=None):
    """Publish only a newer report, conditionally; delayed runs cannot regress it."""
    from cache_io import _client, _r2_creds
    client = client or _client()
    bucket = bucket or (_r2_creds() or {}).get("R2_BUCKET")
    if client is None or not bucket:
        raise ValueError("status publication is not configured")
    body = Path(path).read_bytes()
    def stamp(data):
        if data.get("schema_version") != 1 or data.get("account_key") != "primary":
            raise ValueError("invalid status schema")
        at = dt.datetime.fromisoformat(data["generated_at"].replace("Z", "+00:00"))
        if at.tzinfo is None:
            raise ValueError("status time is not aware")
        return at
    at = stamp(json.loads(body))
    if at > dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=5):
        raise ValueError("status time is in the future")
    key = "ops/expected_exit_status.json"
    for _ in range(3):
        try:
            existing = client.get_object(Bucket=bucket, Key=key)
            previous = existing["Body"].read()
            prior_at = stamp(json.loads(previous))
            if prior_at >= at:
                return "already_current" if previous == body else "newer_report_retained"
            condition = {"IfMatch": existing["ETag"]}
        except Exception as exc:
            code = str(getattr(exc, "response", {}).get("Error", {}).get("Code", ""))
            if code not in {"NoSuchKey", "404", "NotFound"}:
                raise
            condition = {"IfNoneMatch": "*"}
        try:
            client.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json", **condition)
            return "published"
        except Exception as exc:
            code = str(getattr(exc, "response", {}).get("Error", {}).get("Code", ""))
            if code not in {"PreconditionFailed", "412", "ConditionalRequestConflict", "409"}:
                raise
    raise RuntimeError("status publication conflicted repeatedly")


def observe(seed, catalog, *, inventory_loader=None, book_loader=None, fills_loader=None):
    from actual_inventory_io import load_actual_inventory
    from daily_execution_report import fetch_book, DEFAULT_BROKER_URL
    from scripts.harvest_fills import fetch_fills
    from research.strategy_discovery.family_fit import validate_family_catalog

    validate_family_catalog(catalog)
    # Include every named algorithm, including reference sleeves: a verified
    # tagged holding must remain monitored if its strategy is later deactivated.
    names = {row["name"] for row in catalog["records"]}
    token = os.environ.get("STATUS_TOKEN", "").strip()
    url = os.environ.get("EXEC_BROKER_URL", DEFAULT_BROKER_URL)
    fills, book = {}, {}
    try:
        if token:
            fills = (fills_loader or fetch_fills)(url, token) or {}
    except Exception:
        pass
    if not isinstance(fills, dict):
        fills = {}
    # Use the book and executions from one relay observation when available.
    # In particular, inventory must not independently refetch or invoke its
    # local Gateway refresh-and-publish fallback from this read-only monitor.
    book = fills.get("book") or {}
    try:
        if token and not book:
            book = (book_loader or fetch_book)(url, token) or {}
    except Exception:
        pass
    if not isinstance(book, dict):
        book = {}
    try:
        inventory = (inventory_loader or load_actual_inventory)(seed_path=seed,
            algo_strategies=names, max_age_seconds=90, fills_loader=lambda *_: fills)
        snapshot = {"status": inventory.status, "reasons": inventory.reasons,
                    "asof_utc": inventory.asof_utc, "tranches": inventory.tranches}
    except Exception as exc:
        snapshot = {"status": "unknown", "reasons": [
            f"reviewed inventory source failed ({type(exc).__name__})"], "asof_utc": None, "tranches": []}
    return snapshot, book, fills


def load_read_environment(config_root, exec_env=None):
    """Read existing credentials without loading the broker's write tokens."""
    from dotenv import dotenv_values, load_dotenv
    from scripts.automation_supervisor import resolve_external_secret_paths
    load_dotenv(config_root / ".env", override=False)
    _, execution = resolve_external_secret_paths(config_root=config_root,
        gcp_json_path=None, exec_env_path=exec_env)
    if exec_env is not None and not execution.is_file():
        raise ValueError("explicit broker read configuration is unavailable")
    values = dotenv_values(execution) if execution.is_file() else {}
    for key in ("STATUS_TOKEN", "EXEC_BROKER_URL"):
        if not os.environ.get(key) and values.get(key):
            os.environ[key] = values[key]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--seed", type=Path, help="Explicit reviewed local seed; otherwise use the shared R2 review")
    parser.add_argument("--algorithm-catalog", type=Path, help="Reviewed catalog; default builds the complete catalog from pinned source")
    parser.add_argument("--exec-env", type=Path, help="Existing exec_agent.env; only broker read credentials are loaded")
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts/expected-exits")
    parser.add_argument("--send", action="store_true")
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args(argv)
    try:
        load_read_environment(args.config_root, args.exec_env)
        if args.algorithm_catalog:
            catalog = json.loads(args.algorithm_catalog.read_text(encoding="utf-8-sig"))
        else:
            from scripts.build_algorithm_family_catalog import catalog_from_source
            catalog = catalog_from_source(as_of=dt.datetime.now(dt.timezone.utc).isoformat())
        inventory, book, fills = observe(args.seed, catalog)
        run = args.artifacts / uuid.uuid4().hex
        run.mkdir(parents=True)
        for name, payload in (("inventory", inventory), ("book", book), ("fills", fills)):
            (run / f"{name}.json").write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
        from scripts.monitor_expected_exits import main as monitor
        command = ["--inventory", str(run / "inventory.json"), "--book", str(run / "book.json"),
                   "--fills", str(run / "fills.json"), "--state", str(args.state),
                   "--output", str(run / "status.json")]
        if args.send:
            command.append("--send")
        result = monitor(command)
        # A new report can exist even if its SMTP delivery was ambiguous.
        # Publishing observation and delivering email have independent outcomes.
        if args.upload and (run / "status.json").is_file():
            print("Expected-exit status: " + publish_status(run / "status.json"))
        return result
    except Exception as exc:
        print(f"Expected-exit observation failed ({type(exc).__name__}); prior report must age out", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
