"""Fail when a scheduled strategy-research run did not reach a final decision."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.strategy_discovery.source_collectors import validate_state
from research_io import read_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=ROOT / "data" / "strategy_source_cursors.json")
    parser.add_argument("--decision", type=Path, default=ROOT / "data" / "strategy_research" / "latest_decision.json")
    parser.add_argument("--marker", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        state = validate_state(read_json(args.state))
        if state["pending"] is not None:
            raise ValueError("source capture remains pending; discovery did not acknowledge it")
        if not args.decision.is_file():
            raise ValueError("latest research decision is missing")
        if args.decision.stat().st_mtime_ns < args.marker.stat().st_mtime_ns:
            raise ValueError("latest research decision predates this scheduled run")
        decision = read_json(args.decision)
        if decision.get("schema_version") != "strategy-research-email-decision.v1":
            raise ValueError("latest research decision has an unsupported schema")
        if decision.get("positions_used") is not False:
            raise ValueError("latest research decision does not prove positions were excluded")
        delivery = decision.get("delivery_status")
        if decision.get("email_required"):
            if delivery not in {"SENT", "ALREADY_SENT"}:
                raise ValueError(f"eligible research lacks confirmed email delivery: {delivery}")
        elif delivery != "NO_EMAIL":
            raise ValueError(f"zero-finding run lacks an explicit NO_EMAIL result: {delivery}")
        print(
            f"strategy research complete: {decision.get('eligible_count', 0)} email-eligible "
            f"of {decision.get('candidate_count', 0)} researched candidate(s)"
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"STRATEGY RESEARCH DELIVERY CHECK FAILED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
