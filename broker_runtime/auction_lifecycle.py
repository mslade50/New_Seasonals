"""Persistent auction intent claims. Importing this module performs no I/O."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def resolve_primary(ib):
    managed = {str(value).strip() for value in ib.managedAccounts() if str(value).strip()}
    configured = os.environ.get("LEGEND_ETF_PRIMARY_ACCOUNT", "").strip()
    if configured and configured in managed:
        return configured
    if not configured and len(managed) == 1:
        return next(iter(managed))
    raise ValueError("Primary account identity is ambiguous or unconfigured")


def claim(directory, account, contract_id, signal, quantity, order_type, tif):
    """Reserve before submission; never infer non-delivery after an exception.

    A claim survives process restarts and date changes. Missed auctions before
    submission have no claim and retain next-auction intent. Existing submitted
    or uncertain claims require broker reconciliation before another attempt.
    """
    if not account or int(contract_id) <= 0 or not signal or int(quantity) <= 0:
        raise ValueError("auction identity and positive quantity are required")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(f"{account}|{signal}".encode()).hexdigest()
    record = {"account": account, "con_id": int(contract_id), "signal": signal,
              "quantity": int(quantity), "order_type": order_type, "tif": tif,
              "state": "submission_pending_reconciliation"}
    path = directory / f"{key}.json"
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        # Malformed/torn records remain an exception, never a free retry.
        prior = json.loads(path.read_text(encoding="utf-8"))
        if any(prior.get(key) != value for key, value in record.items() if key != "state"):
            raise ValueError("auction intent was reused with different order details")
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(record, stream, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    return True
