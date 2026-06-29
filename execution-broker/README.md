# execution-broker

Standalone Cloudflare Worker + Durable Object — the cloud broker for the site's
execution bridge (Phase 2a). One DO instance (`main`) holds the local agent's
outbound hibernatable WebSocket and tracks a heartbeat; the site reads `/status`
to show an "execution online" light.

Deployed **separately** from the Pages site so a broker change can't break the
live site deploy.

## Endpoints
- `GET /agent` — WebSocket upgrade; the local agent dials this. Auth: `Authorization: Bearer <AGENT_TOKEN>`.
- `GET /status` — `{online, sockets, last_seen, connected_at, heartbeat_age_ms}`. Auth: `Authorization: Bearer <STATUS_TOKEN>`.
- `GET /health` — plain liveness.

Phase 2a carries **no order logic** — it only records liveness and acks
heartbeats. Command dispatch (behind a dry-run gate) is Phase 2b.

## One-time deploy
From this directory, with a recent `wrangler` logged into the Cloudflare account:

```sh
wrangler deploy                       # deploys the Worker + DO + v1 migration
wrangler secret put AGENT_TOKEN       # paste a long random token
wrangler secret put STATUS_TOKEN      # paste a second long random token
```

Note the deployed URL (e.g. `https://execution-broker.<subdomain>.workers.dev`).

## Wire the site (Pages env vars)
On the `seasonals-mslade` Pages project (dashboard → Settings → Environment
variables / Functions), set:
- `EXEC_BROKER_URL` = the Worker URL above
- `STATUS_TOKEN` = the same value you set on the Worker

The `/exec-status` Pages Function reads these; until they're set it returns
`{online:false, configured:false}` and the Execution tab shows "Broker not
configured" (no error).

## Run the local agent (trading machine)
```sh
pip install websockets
set EXEC_BROKER_WS=wss://execution-broker.<subdomain>.workers.dev/agent
set EXEC_AGENT_TOKEN=<the AGENT_TOKEN>
python exec_agent.py
```
(`exec_agent.py` lives with the other IBKR scripts in `OneDrive/trading_ibkr`.)

Once the Worker is deployed, the env vars are set, and the agent is running, the
Execution tab flips to **online** within a few seconds.
