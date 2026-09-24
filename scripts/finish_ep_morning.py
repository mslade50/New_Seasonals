"""Independent deadline check using the same session lock and SMTP receipts."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from episodic_pivot.morning_completion import NY, deliver_once, inspect_morning
from episodic_pivot.email_delivery import EmailDeliveryError, failure_payload, resolve_email_settings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--send", action="store_true")
    args = parser.parse_args(argv)
    now = datetime.now(NY)
    root = (ROOT / "artifacts" / "episodic_pivot").resolve()
    target = now.date().isoformat()
    try:
        state = inspect_morning(root, target)
        if state["status"] == "DEADLINE_MISSED" and now.hour == 9 and 30 <= now.minute <= 50 and args.send:
            payload = failure_payload(phase="morning", target_session_date=target, output_root=root,
                summary="The morning EP research workflow did not produce a confirmed report before the 09:30 ET deadline. Candidate delivery was withheld; saved research remains available for inspection.")
            deliver_once(payload, resolve_email_settings(env_file=args.env_file), root)
            state = inspect_morning(root, target)
        print(json.dumps(state))
        return 2 if state["status"] == "DELIVERY_UNCERTAIN" else 0
    except EmailDeliveryError:
        # A different deadline worker may have won the shared send lock.
        state = inspect_morning(root, target)
        if state["status"] in {"DELIVERED", "FAILURE_REPORTED", "PAUSED_BY_USER"}:
            print(json.dumps(state))
            return 0
        print(json.dumps({"status": "DEADLINE_DELIVERY_UNAVAILABLE", "completion_status": state["status"]}))
        return 2
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "DEADLINE_CHECK_UNAVAILABLE", "error_type": type(exc).__name__}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
