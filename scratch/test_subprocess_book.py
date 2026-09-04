"""Does asyncio.create_subprocess_exec work in this agent's event loop on Windows?
Mimics exec_agent._fetch_book exactly."""
import asyncio
import json
import sys

BOOK = r"C:\Users\McKinley Slade\OneDrive\trading_ibkr\book_snapshot.py"


async def fetch():
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, BOOK,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
        book = json.loads((out or b"").decode() or "{}")
        print("OK  bytes=", len(out), " accounts=", [a["error"] for a in book.get("accounts", [])])
    except Exception as e:  # noqa: BLE001
        print("FETCH ERROR:", type(e).__name__, "-", e)


print("loop policy:", type(asyncio.get_event_loop_policy()).__name__)
asyncio.run(fetch())
