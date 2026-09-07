"""Locked durable entry intents; importing this module has no side effects."""
from __future__ import annotations

import datetime as dt
import json
import math
import os
from pathlib import Path
import tempfile


class EntryJournal:
    def __init__(self, path):
        self.path = Path(path)
        self.handle = None
        self.active = set()

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.with_suffix(self.path.suffix + ".lock").open("a+b")
        try:
            if self.handle.seek(0, 2) == 0:
                self.handle.write(b"0")
                self.handle.flush()
            self.handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.read()  # Missing/corrupt history is not an empty account.
            return self
        except Exception:
            self.handle.close()
            self.handle = None
            raise

    def __exit__(self, *args):
        if self.handle is not None:
            self.handle.close()  # Closing releases the native lock, including on failure.
            self.handle = None

    def read(self):
        try:
            rows = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError("entry journal unavailable; reconcile or explicitly initialize reviewed empty history") from exc
        if not isinstance(rows, list):
            raise ValueError("entry journal is malformed")
        seen = set()
        for row in rows:
            if not isinstance(row, dict) or not row.get("sig") or not isinstance(row.get("fp"), list):
                raise ValueError("entry journal record is malformed")
            dt.date.fromisoformat(str(row.get("date")))
            key = row["date"], row["sig"]
            if key in seen:
                raise ValueError("entry journal contains duplicate claims; reconcile before execution")
            seen.add(key)
        return rows

    def _write(self, rows):
        if self.handle is None:
            raise RuntimeError("entry journal must be locked")
        with tempfile.NamedTemporaryFile(mode="w", dir=self.path.parent, suffix=".pending", delete=False, encoding="utf-8") as stream:
            json.dump(rows, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
            pending = stream.name
        os.replace(pending, self.path)

    def references(self, today):
        rows = self.read()
        return {r["sig"] for r in rows}, {tuple(r["fp"]) for r in rows if r["date"] == today}

    def unresolved_refs(self):
        return {r["sig"] for r in self.read() if r.get("state") != "bracket_acknowledged"}

    def record(self, today, sig, fp, state, **metadata):
        rows = self.read()
        found = next((r for r in rows if (r["date"], r["sig"]) == (today, sig)), None)
        if found is None:
            found = {"date": today, "sig": sig, "fp": list(fp)}
            rows.append(found)
        found.update(state=state, **metadata)
        self._write(rows)

    def place(self, today, sig, fp, guarded, ib, contract, order, **kwargs):
        try:
            from auction_lifecycle import resolve_primary
        except ImportError:
            from broker_runtime.auction_lifecycle import resolve_primary
        account = resolve_primary(ib)
        if getattr(order, "account", "") not in {"", account} or kwargs.get("account", account) != account:
            raise ValueError("entry order does not match exact Primary account")
        quantity = float(order.totalQuantity)
        if int(contract.conId) <= 0 or not math.isfinite(quantity) or quantity <= 0 or quantity != int(quantity):
            raise ValueError("entry requires exact contract and positive whole quantity")
        key = today, sig
        prior = next((r for r in self.read() if (r["date"], r["sig"]) == key), None)
        if prior is not None and key not in self.active:
            raise ValueError("prior entry intent requires reconciliation; no blind resubmission")
        order.account = account
        kwargs["account"] = account
        self.record(today, sig, fp, "awaiting_reconciliation", account=account, con_id=int(contract.conId),
                    last_attempted_step=kwargs.get("signal_id"), quantity=int(quantity))
        self.active.add(key)
        return guarded(ib, contract, order, **kwargs)

    def complete(self, today, sig, fp, trades):
        if (today, sig) not in self.active or len(trades) < 2:
            raise ValueError("entry has no active durable intent")
        record = next(r for r in self.read() if (r["date"], r["sig"]) == (today, sig))
        evidence = []
        parent_id = int(trades[0].order.orderId)
        seen = set()
        for index, trade in enumerate(trades):
            order, status = trade.order, str(trade.orderStatus.status)
            if (status not in {"Submitted", "PreSubmitted", "Filled"}
                    or str(order.account) != record["account"]
                    or int(trade.contract.conId) != record["con_id"]
                    or int(order.orderId) <= 0 or int(order.permId) <= 0
                    or (index > 0 and int(getattr(order, "parentId", 0)) != parent_id)):
                raise ValueError("parent/child bracket acknowledgement is incomplete; reconcile before retry")
            key = int(order.clientId), int(order.orderId)
            if key in seen:
                raise ValueError("bracket acknowledgement repeats an order identity")
            seen.add(key)
            evidence.append({"client_id": int(order.clientId), "order_id": int(order.orderId),
                             "perm_id": int(order.permId), "status": status})
        self.record(today, sig, fp, "bracket_acknowledged", orders=evidence)


def initialize_new(path):
    """Explicit reviewed-empty bootstrap, never used by the order runner."""
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write("[]\n")
        stream.flush()
        os.fsync(stream.fileno())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--initialize-reviewed-empty", action="store_true", required=True)
    initialize_new(parser.parse_args().path)
