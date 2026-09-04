"""End-to-end 2b test: site->broker->agent->result, hitting the broker directly
(the Pages /exec-command path is behind Access). Sends a valid echo (expect
dry_run) and a tampered-signature command (expect rejected by the agent)."""
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
    # UA matters: Cloudflare 1010-blocks the default Python-urllib agent on the
    # workers.dev zone (curl/browser pass). Test-only; prod paths don't use urllib.
    headers = {"Authorization": "Bearer " + TOKEN, "User-Agent": "Mozilla/5.0 (roundtrip-test)"}
    body = None
    if data is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(data).encode()
    req = urllib.request.Request(BASE + path, data=body, headers=headers,
                                 method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def post_cmd(ctype, payload, note_id, tamper=False):
    now = int(time.time() * 1000)
    cmd = {"id": f"{note_id}-{now}", "type": ctype, "account": "pa", "dry_run": True,
           "payload": payload, "created_at": now, "expires_at": now + 60000}
    signed = json.dumps(cmd)
    sig = "deadbeef00" if tamper else hmac.new(TOKEN.encode(), signed.encode(), hashlib.sha256).hexdigest()
    # retry while the agent is mid-reconnect (503 agent offline)
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


cid1, st1, raw1 = post_cmd("echo", {"note": "2b round-trip"}, "ok")
print(f"POST valid echo -> {st1}: {raw1}")
cid2, st2, raw2 = post_cmd("flatten", {"symbol": "USO", "order_type": "MKT"}, "bad", tamper=True)
print(f"POST bad-sig    -> {st2}: {raw2}")

time.sleep(3)
print("\n--- stored results ---")
print("valid  ->", json.dumps(find(cid1)))
print("badsig ->", json.dumps(find(cid2)))
