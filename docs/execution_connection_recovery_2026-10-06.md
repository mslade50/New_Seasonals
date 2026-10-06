# Execution bridge reconnect-delay repair — October 6, 2026

This source repair removes the installed reconnect loop's persistent 30-second delay. The new connection policy resets to a roughly one-second retry after a session has lasted at least 30 seconds and received at least two heartbeat acknowledgements. Short sessions and failed handshakes continue exponential backoff, with 20% jitter, a minimum 0.8-second sleep and a maximum 30-second sleep. These are retry sleeps; connection detection, draining an existing request and handshake time can add to total recovery time.

The repair does not claim to stop the underlying TCP resets. The offline threshold stays at 30 seconds. Account/risk/signature/receipt/expiry guards, scheduled intent state and executor code are unchanged. The candidate preparer verifies the reviewed full agent SHA-256, changes only `_run_once`, `main` and `_connect`'s explanatory docstring, and rejects any AST change elsewhere.

## Diagnostics and compatibility

The client records a connection session ID, reconnect attempt and duration, heartbeat ACK age/count, correlated round-trip timing when the relay echoes correlation fields, consumer-busy state and monotonic loop lag. Connection exceptions retain redacted close frames and underlying OS exception fields. Heartbeat-task failure is surfaced to the reconnect loop. Existing in-flight serial requests finish their bounded execution and durable receipt before reconnection; transport failure does not resubmit a command.

The prepared relay source echoes optional session/sequence fields, retains timestamped redacted close/error metadata and adds those fields to `/status`. Legacy clients receive their existing ACK shape, and this client also works with the currently deployed legacy ACKs. Missing ACKs are reported and prevent a session qualifying as stable; this release does not add an ACK-triggered forced reconnect, because long application handlers can delay consumption of ACKs. Protocol ping settings remain unchanged.

Neither relay nor client has been deployed or installed by this task. In particular, `deploy_broker.yml` also resets secrets and rewires Pages; it was not dispatched. Relay deployment is a separate operational promotion. The client delay fix works without deploying it.

## Prepare a reviewed candidate

From the existing repository:

```powershell
python -m broker_runtime.prepare_execution_connection --source 'C:\Users\McKinley Slade\OneDrive\trading_ibkr' --output '<new isolated candidate directory>'
```

The reviewed source hash is `7ec1f52ca49ea66de4504996e4889536fef5494e4ea490ca5312f8b5490bd817`. Only `exec_agent.py` and the new `execution_connection.py` are emitted, plus a manifest; no `.env`, keys or broker data are copied. Stop for review if the source changed rather than overriding the hash.

## Operator installation handoff

The smallest non-disruptive application step is to install after the existing agent exits normally at 21:00 ET. The existing 05:00 ET task will load the candidate tomorrow. There is no need to restart it now or change its schedule.

```powershell
powershell -NoProfile -File scripts\install_execution_connection.ps1 -CandidateDirectory '<reviewed candidate directory>'
```

The operator-run installer refuses daytime installation, requires `ExecAgent` to be Ready, verifies reviewed source and candidate hashes, retains timestamped backups, stages both files and moves the module before the agent. It does not stop/start any task, read credentials, change arming or touch the executor. An earlier activation would need a separate controlled service stop/start after checking pending execution work; that is not part of this handoff. Do not overlap this file promotion with another runtime promotion. Tonight's separately planned 22:00 ET Gateway update is unchanged; re-review if it changes the reviewed agent source.

After the next normal launch, look for `connection_open`, `connection_stable` and `heartbeat_health` records. Following a drop after `connection_stable`, `reconnect` should show `backoff_s:1` and a jittered `retry_s` near one second. Short flaps should continue increasing delay. No live fault injection or trading test is required to verify these diagnostic records.

Rollback uses the retained original agent file during the same stopped service window. The unused helper can remain in place after restoring the original agent; no deletion is needed. The installer was prepared and parsed, not run against production.

## Verification

Offline Python tests exercise stable resets, flapping/jitter bounds, missing/stale/mismatched ACKs, monotonic lag, heartbeat failure during an actual patched session, durable executed/unknown receipts, duplicate commands, cancellation, normal window shutdown and exact-source drift refusal. JavaScript tests exercise correlated/legacy ACK compatibility, redaction and the unchanged offline threshold. Existing execution, scheduled-option, position-action and launcher regressions are included in validation. Hosted Linux and Windows CI status belongs to the exact final commit; runtime activation remains separately unverified.
