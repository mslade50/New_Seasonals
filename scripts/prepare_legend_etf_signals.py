"""Prepare an immutable SPY/QQQ signal plan from read-only IBKR history.

The default path has no Databento dependency. The optional futures research
path retains a zero-dollar cost ceiling and quotes before requesting data.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from legend_etf.paths import machine_runtime_dir, runtime_env_path

RUNTIME_DIR = machine_runtime_dir()
RUNTIME_ENV = runtime_env_path()

RUNTIME_VALUES: dict[str, str] = {}
try:
    from dotenv import dotenv_values, load_dotenv

    raw_runtime = dotenv_values(RUNTIME_ENV)
    RUNTIME_VALUES = {
        str(key): str(value)
        for key, value in raw_runtime.items()
        if value is not None
    }
    load_dotenv(ROOT / ".env", override=False)
except ImportError:
    pass

from legend_etf.config import NY_TZ
from legend_etf.etf_source import ETF_SYMBOLS, prepare_etf_signal_plan

PAID_CONFIRMATION = "I_APPROVE_DATABENTO_CHARGE"


def _default_cache_dir() -> Path:
    configured = RUNTIME_VALUES.get("LEGEND_ETF_DATABENTO_CACHE_DIR", "").strip()
    if configured:
        return Path(configured)
    return RUNTIME_DIR / "futures_cache"


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", choices=("ibkr", "databento-research"), default="ibkr",
        help="SPY/QQQ IBKR history by default; futures retained for research only",
    )
    parser.add_argument("--check-data", action="store_true",
                        help="Inspect current ETF setup inputs without writing an executable plan")
    parser.add_argument(
        "--entry-date",
        help="XNYS entry date (default: today's New York date)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RUNTIME_DIR / "signal_plan.json",
    )
    parser.add_argument("--lookback-days", type=int, default=150)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=_default_cache_dir(),
        help="Machine-global cache and charge ledger shared by all clones",
    )
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=ROOT / "artifacts" / "databento" / "parquet",
        help="Already-purchased archive used to seed the rolling cache",
    )
    parser.add_argument(
        "--max-cost-usd",
        type=float,
        default=float(RUNTIME_VALUES.get("LEGEND_ETF_DATABENTO_MAX_COST_USD", "0")),
        help="Hard daily cost ceiling; default zero or runtime.env value",
    )
    parser.add_argument(
        "--paid-confirmation",
        default=RUNTIME_VALUES.get("LEGEND_ETF_DATABENTO_PAID_CONFIRMATION"),
        help=f"Required for any non-zero quote: {PAID_CONFIRMATION}",
    )
    return parser


def main() -> int:
    args = make_parser().parse_args()
    if args.check_data and args.source != "ibkr":
        raise SystemExit("--check-data only supports the read-only IBKR ETF source")
    now = datetime.now(ZoneInfo(NY_TZ))
    entry_date = args.entry_date or now.date().isoformat()
    if args.source == "ibkr":
        from dataclasses import replace

        from legend_etf.ibkr_adapter import IBKRConnection
        from legend_etf.session import _feed_endpoint

        # Separate from the session feed; this client can never submit orders.
        endpoint = replace(_feed_endpoint(RUNTIME_VALUES), label="signals", client_id=156)
        connection = IBKRConnection(endpoint, live=False)
        try:
            connection.connect()
            connection.assert_server_clock()
            histories = {
                symbol: connection.historical_bars(
                    connection.stock(symbol), duration="21 D", bar_size="15 mins",
                    use_rth=True,
                )
                for symbol in ETF_SYMBOLS
            }
            if args.check_data:
                from legend_etf.etf_source import evaluate_etf_setup

                results = {symbol: evaluate_etf_setup(frame, entry_date=entry_date)
                           for symbol, frame in histories.items()}
                archive = args.output.parent / "etf_data_check" / now.strftime("%Y%m%dT%H%M%S")
                archive.mkdir(parents=True, exist_ok=True)
                for symbol, frame in histories.items():
                    frame.to_parquet(archive / f"{symbol}_15min.parquet")
                print(json.dumps({"ok": True, "read_only": True, "entry_date": entry_date,
                                  "markets": results, "archive": str(archive)}, indent=2))
                return 0
            plan = prepare_etf_signal_plan(
                histories=histories, entry_date=entry_date, as_of=now,
            )
            # Freeze the exact broker input alongside the reviewed plan.
            from legend_etf.storage import atomic_write_json

            archive = args.output.parent / "etf_history" / plan["plan_hash"]
            archive.mkdir(parents=True, exist_ok=True)
            for symbol, frame in histories.items():
                destination = archive / f"{symbol}_15min.parquet"
                if not destination.exists():
                    frame.to_parquet(destination)
            atomic_write_json(args.output, plan)
        finally:
            connection.disconnect()
    else:
        from legend_etf.databento_source import make_client, prepare_signal_plan

        plan = prepare_signal_plan(
            client=make_client(), entry_date=entry_date, as_of=now,
            output_path=args.output, lookback_days=args.lookback_days,
            max_cost_usd=args.max_cost_usd, paid_confirmation=args.paid_confirmation,
            cache_dir=args.cache_dir, archive_dir=args.archive_dir,
        )
    print(
        json.dumps(
            {
                "ok": True,
                "entry_date": entry_date,
                "plan_hash": plan["plan_hash"],
                "qualified": [
                    item["root"] for item in plan["markets"] if item["qualifies"]
                ],
                "quoted_cost_usd": plan["quoted_cost_usd"],
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
