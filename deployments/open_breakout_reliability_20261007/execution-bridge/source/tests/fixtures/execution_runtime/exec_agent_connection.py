"""Reviewed transport function fixture only; import never starts an agent."""
def _connect(url: str, headers: dict):
    """websockets.connect with the auth header, tolerant of the lib's header-kwarg
    rename (additional_headers in >=12, extra_headers before).

    ping_interval=None disables the client's WS-level pings: the hibernating
    Durable Object doesn't pong them, so the default ping timeout was tearing the
    socket down every ~20s. Our app-level heartbeat (every 10s) is the keepalive
    and liveness check instead — a failed send surfaces a dead link and triggers
    reconnect."""
    try:
        return websockets.connect(url, additional_headers=headers, open_timeout=15,
                                  ping_interval=None)
    except TypeError:
        return websockets.connect(url, extra_headers=headers, open_timeout=15,
                                  ping_interval=None)


async def _run_once() -> None:
    headers = {"Authorization": f"Bearer {TOKEN}"}
    async with _connect(WS_URL, headers) as ws:
        await ws.send(json.dumps({"type": "hello", "agent": "exec_agent",
                                  "pid": os.getpid(), "t": time.time()}))
        log(f"connected -> {WS_URL}")

        async def beat() -> None:
            while True:
                await asyncio.sleep(HEARTBEAT_S)
                if not in_window():
                    log("end of run window; closing")
                    await ws.close()
                    return
                await ws.send(json.dumps({"type": "heartbeat", "t": time.time()}))

        async def book_loop() -> None:
            # push a read-only book on connect, then refresh on a cadence
            while True:
                try:
                    book = await _fetch_book()
                    if book is not None:
                        # "live" only when explicitly armed (AGENT_LIVE_ENABLED=1);
                        # otherwise hard dry-run. Drives the UI banner.
                        book["mode"] = "live" if LIVE_ENABLED else "dry-run"
                        _BOOK["book"] = book
                        await ws.send(json.dumps({"type": "book", "book": book, "at": time.time()}))
                except Exception as e:  # noqa: BLE001
                    log(f"book_loop error: {e}")
                await asyncio.sleep(BOOK_REFRESH_S)

        hb = asyncio.create_task(beat())
        bk = asyncio.create_task(book_loop())
        so = asyncio.create_task(_scheduled_option_loop(ws))
        import position_action_agent
        pa = asyncio.create_task(position_action_agent.loop(globals(), ws))
        try:
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except Exception:
                    msg = {"type": "raw", "data": raw}
                if msg.get("type") == "command":
                    await _handle_command(ws, msg.get("signed", ""), msg.get("sig", ""))
                elif msg.get("type") == "option_query":      # read-only spread R/R
                    data = await _fetch_option(msg.get("ticker", ""), msg.get("expiry"))
                    await ws.send(json.dumps({"type": "option_result", "id": msg.get("id"),
                                              "ticker": msg.get("ticker"), "data": data}))
                    log(f"option_query {msg.get('ticker')} -> {'ok' if not data.get('error') else data.get('error')}")
                elif msg.get("type") == "workbench_query":    # read-only chain + term structure
                    q = {k: msg.get(k) for k in ("ticker", "mode", "expiry", "max_expiries", "context")
                         if msg.get(k) is not None}
                    data = await _fetch_workbench(q)
                    await ws.send(json.dumps({"type": "workbench_result", "id": msg.get("id"),
                                              "ticker": msg.get("ticker"), "data": data}))
                    log(f"workbench_query {msg.get('ticker')} [{q.get('mode', 'full')}] -> "
                        f"{'ok' if not data.get('error') else data.get('error')}")
                elif msg.get("type") == "futures_size":       # read-only risk sizing (pure, no IBKR)
                    data = _futures_size(msg)
                    await ws.send(json.dumps({"type": "futures_result", "id": msg.get("id"),
                                              "symbol": msg.get("symbol"), "data": data}))
                    log(f"futures_size {msg.get('symbol')} -> "
                        f"{data.get('contracts') if not data.get('error') else data.get('error')}")
                elif msg.get("type") == "futures_front":       # read-only front-month resolve
                    data = await _fetch_futures_front(msg.get("symbol", ""), msg.get("exchange"))
                    await ws.send(json.dumps({"type": "futures_front_result", "id": msg.get("id"),
                                              "symbol": msg.get("symbol"), "data": data}))
                    log(f"futures_front {msg.get('symbol')} -> {data.get('expiry') or data.get('error')}")
                elif msg.get("type") != "ack":
                    log(f"<- {msg}")
        finally:
            hb.cancel()
            bk.cancel()
            so.cancel()
            pa.cancel()
            await asyncio.gather(hb, bk, so, pa, return_exceptions=True)


async def main() -> None:
    if not WS_URL or not TOKEN:
        raise SystemExit("Set EXEC_BROKER_WS and EXEC_AGENT_TOKEN env vars first.")
    _load_seen()                         # dedup survives restarts (last 24h of ids)
    _load_schedules()                    # pending dynamic option intents survive restarts
    backoff = 1
    while True:
        if not in_window():
            log(f"outside run window {START_HOUR:02d}:00-{END_HOUR:02d}:00 (local); exiting")
            return
        try:
            await _run_once()
            backoff = 1                      # clean exit -> reset backoff
        except Exception as e:  # noqa: BLE001 — keep the agent alive across drops
            log(f"disconnected: {e}; retry in {backoff}s")
        await asyncio.sleep(backoff)
        backoff = min(BACKOFF_MAX_S, backoff * 2)
