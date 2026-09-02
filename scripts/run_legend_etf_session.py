"""Run or preflight the dedicated Legend ETF IBKR session."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from legend_etf.paths import machine_runtime_dir, runtime_env_path

RUNTIME_DIR = machine_runtime_dir()
RUNTIME_ENV = runtime_env_path()

try:
    from dotenv import load_dotenv

    # The dedicated runtime file owns every LEGEND_ETF_* execution gate;
    # inherited shell state must not override its dated kill switch.
    load_dotenv(RUNTIME_ENV, override=True)
except ImportError:
    pass

from legend_etf.session import LegendSession
from legend_etf.storage import StateStore


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        default=RUNTIME_DIR / "signal_plan.json",
    )
    parser.add_argument(
        "--accounts", nargs="+", choices=("primary", "pa"), default=["primary"]
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Request live execution; independent dated environment gates must also pass",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate plan, gates, account identity, and connectivity without waiting",
    )
    parser.add_argument(
        "--shadow-through-exit",
        action="store_true",
        help="In dry-run, keep collecting data through 10:30 for operations validation",
    )
    parser.add_argument(
        "--reconcile-only",
        action="store_true",
        help="Manage/revalidate existing same-day live records; never create an entry",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help=(
            "Read-only all-account SPY/QQQ/IWM correction audit; does not "
            "require a full session or signal plan"
        ),
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="Durable scheduler run identity used by the fenced watchdog lease",
    )
    return parser


def main() -> int:
    args = make_parser().parse_args()
    if args.reconcile_only and not args.live:
        raise SystemExit("--reconcile-only requires --live")
    if args.reconcile_only and args.check:
        raise SystemExit("--reconcile-only cannot be combined with --check")
    if args.audit_only and not args.live:
        raise SystemExit("--audit-only requires --live")
    if args.audit_only and (args.check or args.reconcile_only):
        raise SystemExit("--audit-only cannot be combined with --check/--reconcile-only")
    suffix = "live" if args.live else "dry"
    store = StateStore(
        RUNTIME_DIR / f"session_state_{suffix}.json",
        RUNTIME_DIR / f"audit_{suffix}.jsonl",
        alert_path=RUNTIME_DIR / "critical_alert.json",
        alert_popup=args.live,
    )
    session = LegendSession(
        plan_path=args.plan,
        state_store=store,
        account_labels=args.accounts,
        live_requested=args.live,
        shadow_through_exit=args.shadow_through_exit,
        reconcile_only=args.reconcile_only,
        runtime_env_path=RUNTIME_ENV,
        run_id=args.run_id or None,
    )
    try:
        if args.audit_only:
            with store.exclusive_session():
                result = session.audit_terminal_corrections()
        elif args.check:
            with store.exclusive_session():
                result = session.preflight(connect=True)
        else:
            result = session.run()
        print(json.dumps(result, indent=2, default=str))
        return 0
    finally:
        session.close()


if __name__ == "__main__":
    raise SystemExit(main())
