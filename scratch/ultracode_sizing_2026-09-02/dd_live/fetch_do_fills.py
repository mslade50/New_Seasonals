"""Read-only pull of the execution-broker DO's /fills ring and /book snapshot.

Uses STATUS_TOKEN from OneDrive/trading_ibkr/exec_agent.env (read bearer, the
same token daily_execution_report.py and the site's exec-fills proxy use).
Writes dd_live/do_fills.json and dd_live/do_book.json. Nothing is uploaded.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
ROOT = Path(r"C:/Users/McKinley Slade/dev/New_Seasonals")
ENV = Path(r"C:/Users/McKinley Slade/OneDrive/trading_ibkr/exec_agent.env")

env: dict[str, str] = {}
for line in ENV.read_text(encoding="utf-8").splitlines():
    if "=" in line and not line.strip().startswith("#"):
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")
token = env.get("STATUS_TOKEN", "")
if not token:
    sys.exit("no STATUS_TOKEN in exec_agent.env")

src = (ROOT / "daily_execution_report.py").read_text(encoding="utf-8")
m = re.search(r"DEFAULT_BROKER_URL\s*=\s*['\"]([^'\"]+)['\"]", src)
base = m.group(1).rstrip("/") if m else None
if not base:
    ws = env.get("EXEC_BROKER_WS", "")
    base = re.sub(r"^wss?://", "https://", ws).split("/agent")[0].rstrip("/")
print("broker base:", base)

hdr = {"Authorization": f"Bearer {token}"}
for path, out in (("/fills", "do_fills.json"), ("/book", "do_book.json")):
    r = requests.get(base + path, headers=hdr, timeout=30)
    print(path, r.status_code, len(r.content), "bytes")
    try:
        data = r.json()
    except Exception as e:  # noqa: BLE001
        print("  non-json:", r.text[:300], e)
        continue
    (HERE / out).write_text(json.dumps(data, indent=1), encoding="utf-8")
    if path == "/fills":
        fills = data.get("fills") or []
        print("  fills rows:", len(fills), "retention_days:", data.get("retention_days"))
        if fills:
            days = sorted({(f.get("time") or "")[:10] for f in fills})
            print("  days:", days[0], "->", days[-1], "n_days", len(days))
            print("  keys:", sorted(fills[0].keys()))
            accts = {}
            for f in fills:
                accts[f.get("account")] = accts.get(f.get("account"), 0) + 1
            print("  by account:", accts)
            tagged = sum(1 for f in fills if f.get("order_ref"))
            print("  with order_ref:", tagged)
    else:
        for k in ("asof", "pushed_at", "ts", "generated_at"):
            if k in data:
                print("  ", k, data[k])
        print("  top keys:", list(data.keys())[:20])
