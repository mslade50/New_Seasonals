# execution-broker

Standalone Cloudflare Worker + Durable Object for the private site's
live-capable execution bridge. One DO instance (`main`) holds the local agent's
outbound hibernatable WebSocket, stores its latest book/heartbeat, relays signed
commands, and records recent results.

Deployed **separately** from the Pages site so a broker change can't break the
live site deploy.

## Endpoints
- `GET /agent` — WebSocket upgrade; the local agent dials this. Auth: `Authorization: Bearer <AGENT_TOKEN>`.
- `GET /status` — `{online, sockets, last_seen, connected_at, heartbeat_age_ms}`. Auth: `Authorization: Bearer <STATUS_TOKEN>`.
- `GET /book`, `/commands`, `/fills` — read-only execution state. Auth: `STATUS_TOKEN`.
- `POST /command` — signed command relay. Auth: `COMMAND_SECRET`, never `STATUS_TOKEN`.
- `/option`, `/workbench`, `/futures_size`, `/futures_front` — read-only agent queries. Auth: `STATUS_TOKEN`.
- `GET /health` — plain liveness.

The broker does not construct IBKR orders, but it is part of the live-order path.
It independently rejects unknown envelopes, stale heartbeat/book state, unknown
mode, duplicate sockets, books from an earlier agent session, and unarmed live
types/accounts before a command reaches the agent. Reconnecting clears the old
book; no command can flow until the new socket publishes its own snapshot.

## One-time deploy
From this directory, with a recent `wrangler` logged into the Cloudflare account:

```sh
wrangler deploy                       # deploys the Worker + DO + v1 migration
wrangler secret put AGENT_TOKEN       # paste a long random token
wrangler secret put STATUS_TOKEN      # paste a second long random token
wrangler secret put COMMAND_SECRET     # dedicated command bearer + HMAC secret
```

The versioned `[vars]` block in `wrangler.toml` deploys the Worker disarmed;
live switches are policy configuration, not secrets.

Note the deployed URL (e.g. `https://execution-broker.<subdomain>.workers.dev`).

## Wire the site (Pages env vars)
On the `seasonals-mslade` Pages project (dashboard → Settings → Environment
variables / Functions), set:
- `EXEC_BROKER_URL` = the Worker URL above
- `STATUS_TOKEN` = the Worker's read-only token
- `COMMAND_SECRET` = the Worker's command secret and the agent's HMAC secret
- `EXEC_LIVE_ENABLED=0` initially

Until all three core values are set, command submission fails closed. Read-only
status can still report "not configured" or offline.

## Run the local agent (trading machine)
```sh
pip install websockets
set EXEC_BROKER_WS=wss://execution-broker.<subdomain>.workers.dev/agent
set EXEC_AGENT_TOKEN=<the AGENT_TOKEN>
set COMMAND_SECRET=<the COMMAND_SECRET>
python exec_agent.py
```
(`exec_agent.py` lives with the other IBKR scripts in `OneDrive/trading_ibkr`.)

Once the Worker is deployed, the env vars are set, and the agent is running, the
Execution tab flips to **online** within a few seconds.

## Live arming

`Deploy Execution Broker` atomically deploys `EXEC_LIVE_ENABLED=0`, empty live
type/account allowlists, and `EXEC_LIVE_INSTRUMENTS=STK` from `wrangler.toml`,
then applies the same fail-closed state to Pages.
Live bracket entry types reset to `EXEC_LIVE_ENTRY_TYPES=LMT,STP_LMT`.
Position adds/trim-readds reset to `EXEC_LIVE_POSITION_INSTRUMENTS=STK`.
Dry-run previews require no live arming. For a watched live test, set matching
`EXEC_LIVE_TYPES` and `EXEC_LIVE_ACCOUNTS` at both server layers, then set
`EXEC_LIVE_ENABLED=1` at both layers and arm the agent last. See
`docs/site_execution_golive.md`. Turning either server switch off blocks new
live commands.

Risk-increasing live commands remain rejected in both Pages and broker policy
even if those allowlists are changed. They require an atomic aggregate-risk
reservation before arming can be implemented safely.
