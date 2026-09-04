"""Exercise the agent's gatekeeper validation + order preview via the broker.
Picks a REAL primary position to flatten (expect dry_run + preview), a fake
symbol (expect rejected), and an entry_bracket (expect preview legs + R:R)."""
import hashlib
import hmac
import json
import time
import urllib.error
import urllib.request

env = {}
with open(r"C:\Users\McKinley Slade\OneDrive\trading_ibkr\exec_agent.env", encoding="utf-8") as f:
    for line in f:
        if "=" in line and not line.startswith("#"):
            k, v = line.strip().split("=", 1)
            env[k] = v
TOKEN = env["STATUS_TOKEN"]
BASE = "https://execution-broker.mckinleyslade.workers.dev"
UA = {"User-Agent": "Mozilla/5.0 (test)"}


def _req(path, data=None):
    h = dict(UA, **{"Authorization": "Bearer " + TOKEN})
    body = None
    if data is not None:
        h["Content-Type"] = "application/json"
        body = json.dumps(data).encode()
    req = urllib.request.Request(BASE + path, data=body, headers=h, method="POST" if data else "GET")
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def send(ctype, payload, nid):
    now = int(time.time() * 1000)
    cmd = {"id": f"{nid}-{now}", "type": ctype, "account": "primary", "dry_run": True,
           "payload": payload, "created_at": now, "expires_at": now + 60000}
    signed = json.dumps(cmd)
    sig = hmac.new(TOKEN.encode(), signed.encode(), hashlib.sha256).hexdigest()
    for _ in range(8):
        st, d = _req("/command", {"signed": signed, "sig": sig})
        if st != 503:
            return cmd["id"]
        time.sleep(1.5)
    return cmd["id"]


def find(cid):
    _, d = _req("/commands")
    m = [c for c in d.get("commands", []) if c["id"] == cid]
    return m[0] if m else None


# real symbol from the live book
_, bk = _req("/book")
prim = next((a for a in (bk.get("book") or {}).get("accounts", []) if a["key"] == "primary"), {})
poss = prim.get("positions", [])
real = poss[0]["symbol"] if poss else "USO"
print(f"primary has {len(poss)} positions; using real symbol: {real}")

c1 = send("flatten", {"symbol": real, "fraction": 0.5, "order_type": "MKT"}, "real")
c2 = send("flatten", {"symbol": "ZZZZ", "fraction": 1.0, "order_type": "MKT"}, "fake")
c3 = send("entry_bracket", {"symbol": "USO", "action": "BUY", "quantity": 100,
                            "entry": 104.80, "stop": 103.29, "target": 123.21}, "entry")
time.sleep(3)
for label, cid in [("flatten REAL", c1), ("flatten FAKE", c2), ("entry_bracket", c3)]:
    r = find(cid) or {}
    print(f"\n{label}: state={r.get('state')}")
    print("  ", json.dumps(r.get("result", {})))
