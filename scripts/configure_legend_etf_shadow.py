"""Create a Primary-only, read-only Legend configuration after broker identity proof.

Uses the already configured Primary TWS endpoint. Never starts a trading task,
arms live flags, places an order, or permits paid data. Existing config is never
overwritten. Account IDs stay local; no credentials are copied into runtime.env.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from legend_etf.config import STRATEGY_VERSION
from legend_etf.paths import machine_runtime_dir


def shadow_values(
    account: str, port: int, executor: Path, runtime: Path
) -> dict[str, str]:
    if not re.fullmatch(r"U\d+", account):
        raise ValueError(
            "The configured Primary endpoint must expose one live account; shadow access remains read-only"
        )
    return {
        "LEGEND_ETF_PRIMARY_ACCOUNT": account,
        "LEGEND_ETF_PRIMARY_HOST": "127.0.0.1",
        "LEGEND_ETF_PRIMARY_PORT": str(port),
        "LEGEND_ETF_PRIMARY_CLIENT_ID": "155",
        "LEGEND_ETF_FEED_HOST": "127.0.0.1",
        "LEGEND_ETF_FEED_PORT": str(port),
        "LEGEND_ETF_FEED_CLIENT_ID": "154",
        "LEGEND_ETF_EXECUTOR_ROOT": executor.resolve().as_posix(),
        "LEGEND_ETF_RESERVATION_DIR": (runtime / "reservations").as_posix(),
        "LEGEND_ETF_GUARD_MANIFEST": (
            runtime / "executor_guard_manifest.json"
        ).as_posix(),
        "LEGEND_ETF_GUARD_MANIFEST_SHA256": "",
        "LEGEND_ETF_PORTFOLIO_BUDGET": (runtime / "portfolio_budget.json").as_posix(),
        "LEGEND_ETF_PAPER_PROOF": (runtime / "paper_proof.json").as_posix(),
        "LEGEND_ETF_PAPER_PROOF_SHA256": "",
        "LEGEND_ETF_DATABENTO_MAX_COST_USD": "0",
        "LEGEND_ETF_PRIMARY_LONG_BPS": "10",
        "LEGEND_ETF_PRIMARY_SHORT_BPS": "5",
        "LEGEND_ETF_PRIMARY_CLUSTER_BPS": "30",
        "LEGEND_ETF_PRIMARY_MAX_SHARES_PER_ROOT": "5000",
        "LEGEND_ETF_PRIMARY_MAX_NOTIONAL_PCT": "0.25",
        "LEGEND_ETF_LIVE_ENABLED": "0",
        "LEGEND_ETF_LIVE_DATE": "",
        "LEGEND_ETF_LIVE_ACCOUNTS": "",
        "LEGEND_ETF_ALLOW_LONGS": "0",
        "LEGEND_ETF_ALLOW_SHORTS": "0",
        "LEGEND_ETF_STRATEGY_VERSION": STRATEGY_VERSION,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-port", type=int, default=7496)
    parser.add_argument("--executor-root", type=Path, required=True)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write a new shadow-only config after the read-only check",
    )
    args = parser.parse_args()
    runtime = machine_runtime_dir()
    destination = runtime / "runtime.env"
    if destination.exists():
        raise RuntimeError(
            "Runtime config already exists; inspect it instead of replacing it"
        )
    if not args.executor_root.is_dir() or not 1 <= args.primary_port <= 65535:
        raise ValueError("Invalid executor directory or Primary port")
    from ib_insync import IB

    ib = IB()
    try:
        ib.connect(
            "127.0.0.1", args.primary_port, clientId=155, timeout=8, readonly=True
        )
        accounts = ib.managedAccounts()
        if len(accounts) != 1:
            raise RuntimeError("Primary must expose exactly one account")
        values = shadow_values(
            accounts[0], args.primary_port, args.executor_root, runtime
        )
        if args.apply:
            runtime.mkdir(parents=True, exist_ok=True)
            with destination.open("x", encoding="utf-8") as handle:
                handle.write(
                    "# Primary-only Legend shadow setup. Live execution and paid data disabled.\n"
                )
                for key, value in values.items():
                    handle.write(f'{key}="{value}"\n')
        print(
            json.dumps(
                {
                    "primary_identity_verified": True,
                    "mode": "shadow",
                    "written": args.apply,
                    "paid_data_enabled": False,
                    "tasks_registered": False,
                    "orders_sent": False,
                }
            )
        )
        return 0
    finally:
        ib.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
