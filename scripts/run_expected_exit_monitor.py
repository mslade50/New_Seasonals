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
    inventory = (inventory_loader or load_actual_inventory)(seed_path=seed, algo_strategies=names, max_age_seconds=90)
    snapshot = {"status": inventory.status, "reasons": inventory.reasons,
                "asof_utc": inventory.asof_utc, "tranches": inventory.tranches}
    token = os.environ.get("STATUS_TOKEN", "").strip()
    url = os.environ.get("EXEC_BROKER_URL", DEFAULT_BROKER_URL)
    if not token:
        return snapshot, {}, {}
    # Each source failure remains missing. The monitor owns the explicit
    # unable-to-verify outcome and preserves previously tracked obligations.
    try:
        book = (book_loader or fetch_book)(url, token) or {}
    except Exception:
        book = {}
    try:
        fills = (fills_loader or fetch_fills)(url, token) or {}
    except Exception:
        fills = {}
    return snapshot, book, fills


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--seed", type=Path, required=True)
    parser.add_argument("--algorithm-catalog", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts/expected-exits")
    parser.add_argument("--send", action="store_true")
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args(argv)
    try:
        from dotenv import load_dotenv
        load_dotenv(args.config_root / ".env", override=False)
        catalog = json.loads(args.algorithm_catalog.read_text(encoding="utf-8-sig"))
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
