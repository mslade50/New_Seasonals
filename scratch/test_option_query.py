"""End-to-end test of the option-spread query: POST /option -> agent fetches the
chain -> poll /option for the result. Verifies the broker<->agent option path."""
import json
import sys
import time
import urllib.request

env = {}
with open(r"C:\Users\McKinley Slade\OneDrive\trading_ibkr\exec_agent.env", encoding="utf-8") as f:
    for line in f:
        if "=" in line and not line.startswith("#"):
            k, v = line.strip().split("=", 1)
            env[k] = v
TOKEN = env["STATUS_TOKEN"]
BASE = "https://execution-broker.mckinleyslade.workers.dev"
H = {"Authorization": "Bearer " + TOKEN, "User-Agent": "Mozilla/5.0 (test)"}
TICK = sys.argv[1] if len(sys.argv) > 1 else "AAPL"


def req(path, data=None):
    h = dict(H)
    body = None
    if data is not None:
        h["Content-Type"] = "application/json"; body = json.dumps(data).encode()
    r = urllib.request.Request(BASE + path, data=body, headers=h, method="POST" if data else "GET")
    with urllib.request.urlopen(r, timeout=20) as resp:
        return json.loads(resp.read())


d = req("/option", {"ticker": TICK})
print("POST:", d)
qid = d.get("id")
for i in range(30):
    time.sleep(3)
    q = (req("/option") or {}).get("query") or {}
    if q.get("id") == qid and q.get("result"):
        res = q["result"]
        if res.get("error"):
            print("RESULT error:", res["error"]); break
        print(f"\n{res['ticker']}  spot {res['spot']}  expiry {res['expiry']} ({res['dte']}d)")
        for k in ("call_spread", "put_spread"):
            s = res.get(k) or {}
            if s.get("error"):
                print(f"  {k}: {s['error']}"); continue
            print(f"  {k}: BUY {s['long']['strike']} / SELL {s['short']['strike']}  "
                  f"debit {s['debit']}  width {s['width']}  R/R {s['rr']}  BE {s['breakeven']}")
        break
    print(f"  ...waiting ({i})")
