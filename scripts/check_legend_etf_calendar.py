"""Report whether today's XNYS session is eligible for Legend automation."""

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

from legend_etf.calendar import is_full_session, is_session
from legend_etf.config import NY_TZ


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date")
    args = parser.parse_args()
    day = args.date or datetime.now(ZoneInfo(NY_TZ)).date().isoformat()
    regular = is_session(day)
    full = is_full_session(day)
    reason = "full_session" if full else ("early_close" if regular else "non_session")
    print(
        json.dumps(
            {
                "entry_date": day,
                "full_session": full,
                "reason": reason,
            },
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
