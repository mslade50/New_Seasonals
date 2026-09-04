"""Read-only probe: is an IBKR PAPER TWS/Gateway running? Paper ports are 7497
(TWS) / 4002 (Gateway); paper account ids start with 'DU' (live start with 'U').
Read-only, places nothing."""
from ib_insync import IB

for label, port in [("paper TWS", 7497), ("paper Gateway", 4002)]:
    ib = IB()
    try:
        ib.connect("127.0.0.1", port, clientId=130, timeout=6, readonly=True)
        accts = ib.managedAccounts()
        kind = "PAPER" if any(a.startswith("DU") for a in accts) else "??"
        print(f"{label} ({port}): UP -> accounts {accts} [{kind}]")
        ib.disconnect()
    except Exception as e:
        print(f"{label} ({port}): not running")
