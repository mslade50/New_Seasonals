"""Read-only pull of the execution-broker DO's /book snapshot (the same call
daily_execution_report.fetch_book makes) so the what-if pack carries the
ACTUAL live positions and NLV rather than the 2026-08-18 memory note.

Writes live_book_<YYYY-MM-DD>.json beside this script.  Reads STATUS_TOKEN /
EXEC_BROKER_URL from the repo .env; never writes anywhere else.
"""
from __future__ import annotations
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import requests

ROOT = Path(r"C:/Users/McKinley Slade/dev/New_Seasonals")
HERE = Path(__file__).resolve().parent
DEFAULT_BROKER_URL = "https://execution-broker.mckinleyslade.workers.dev"


def load_env() -> dict[str, str]:
    env: dict[str, str] = {}
    for p in (ROOT / ".env", Path(r"C:/Users/McKinley Slade/OneDrive/trading_ibkr/exec_agent.env")):
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    return env


def main() -> int:
    env = load_env()
    token = os.environ.get("STATUS_TOKEN") or env.get("STATUS_TOKEN", "")
    base = os.environ.get("EXEC_BROKER_URL") or env.get("EXEC_BROKER_URL") or DEFAULT_BROKER_URL
    if not token:
        print("STATUS_TOKEN not set; nothing fetched")
        return 2
    r = requests.get(f"{base.rstrip('/')}/book", headers={"Authorization": f"Bearer {token}"}, timeout=30)
    r.raise_for_status()
    payload = r.json()
    book = payload.get("book") or {}
    stamp = datetime.now().strftime("%Y-%m-%d")
    out = HERE / f"live_book_{stamp}.json"
    out.write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")
    print("wrote", out)
    print("top-level keys:", sorted(payload.keys()))
    print("book keys:", sorted(book.keys()) if isinstance(book, dict) else type(book))
    for acct, blob in (book.items() if isinstance(book, dict) else []):
        if not isinstance(blob, dict):
            continue
        pos = blob.get("positions") or []
        print(f"account {acct}: nlv={blob.get('nlv')} snapshot={blob.get('ts') or blob.get('generated') or blob.get('as_of')} positions={len(pos)} orders={len(blob.get('orders') or [])}")
        for p in pos[:60]:
            print("   ", {k: p.get(k) for k in ("symbol", "sec_type", "position", "qty", "avg_cost", "market_price", "market_value", "unrealized_pnl") if k in p})
    return 0


if __name__ == "__main__":
    sys.exit(main())
