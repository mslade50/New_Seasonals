"""Live agent-path checks for the 2026-07-01 audit fixes, all no-order-risk:
  1. flatten fraction=1.5 -> REJECTED by the new (0,1] clamp
  2. entry_bracket with dry_run=true -> hard dry-run even though armed
     (the audit's #1 finding: agent used to ignore cmd.dry_run)
  3. option_query TSLA -> standard tradingClass (no 'Unknown contract')
Hits the broker directly like scratch/test_command_roundtrip.py."""
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


def _req(path, data=None):
    headers = {"Authorization": "Bearer " + TOKEN,
               "User-Agent": "Mozilla/5.0 (audit-fix-test)"}
    body = None
    if data is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(data).encode()
    req = urllib.request.Request(BASE + path, data=body, headers=headers,
                                 method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def post_cmd(ctype, payload, note_id, dry_run):
    now = int(time.time() * 1000)
    cmd = {"id": f"{note_id}-{now}", "type": ctype, "account": "pa",
           "dry_run": dry_run, "payload": payload,
           "created_at": now, "expires_at": now + 60000}
    signed = json.dumps(cmd)
    sig = hmac.new(TOKEN.encode(), signed.encode(), hashlib.sha256).hexdigest()
    for _ in range(8):
        st, raw = _req("/command", {"signed": signed, "sig": sig})
        if st != 503:
            return cmd["id"], st, raw
        time.sleep(1.5)
    return cmd["id"], st, raw


def find(cid):
    _, raw = _req("/commands")
    try:
        cmds = json.loads(raw).get("commands", [])
    except Exception:
        return None
    m = [c for c in cmds if c["id"] == cid]
    return m[0] if m else None


results = []

# 1. fraction clamp
cid1, st1, _ = post_cmd("flatten",
                        {"symbol": "ZZZZTEST", "order_type": "MKT", "fraction": 1.5},
                        "fracclamp", dry_run=True)
print(f"1. flatten fraction=1.5 POST -> {st1}")

# 2. entry_bracket dry_run respected while armed
cid2, st2, _ = post_cmd("entry_bracket",
                        {"symbol": "AAPL", "action": "BUY", "quantity": 1,
                         "entry": 100.0, "stop": 90.0, "target": 150.0},
                        "dryrun", dry_run=True)
print(f"2. entry_bracket dry_run POST -> {st2}")

# 3. option query (tradingClass fix) — TSLA had the adjusted-class bug
st3, raw3 = _req("/option", {"ticker": "TSLA"})
print(f"3. option_query TSLA POST -> {st3}")

time.sleep(12)  # agent round-trip

r1 = find(cid1)
res1 = (r1 or {}).get("result") or {}
reasons1 = (res1.get("validation") or {}).get("reasons") or []
p1 = res1.get("ok") is False and any("fraction" in r for r in reasons1)
print(f"\n1. {'PASS' if p1 else 'FAIL'} fraction clamp -> ok={res1.get('ok')} "
      f"reasons={reasons1}")

r2 = find(cid2)
res2 = (r2 or {}).get("result") or {}
d2 = str(res2.get("detail", ""))
p2 = res2.get("ok") is True and d2.startswith("[DRY-RUN]")
print(f"2. {'PASS' if p2 else 'FAIL'} dry_run respected while armed -> "
      f"ok={res2.get('ok')} detail={d2!r}")

deadline = time.time() + 50
opt = None
while time.time() < deadline:
    _, raw = _req("/option")
    q = json.loads(raw).get("query") or {}
    if q.get("ticker") == "TSLA" and q.get("result"):
        opt = q["result"]
        break
    time.sleep(4)
if opt is None:
    print("3. FAIL option_query -> no result within 50s")
else:
    err = str(opt.get("error", "") or "")
    ok3 = "unknown contract" not in err.lower()
    keys = list(opt.keys())
    print(f"3. {'PASS' if ok3 else 'FAIL'} option_query TSLA -> error={err!r} keys={keys}")
    if isinstance(opt.get("call"), dict):
        print("   call spread:", json.dumps(opt["call"])[:200])
