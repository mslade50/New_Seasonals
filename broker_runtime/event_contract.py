"""Event auction identity; no broker or filesystem I/O on import."""
from datetime import date
import math
import json
from pathlib import Path


# Execution vocabulary mirrors the frozen event_sleeve.EVENT_SLEEVE contract.
TRADES = {
    "T1_FOMC_DRIFT": ("SPY", "BUY", "SELL", "MOO"),
    "T2_FOMC_MIDTERM_SHORT": ("SPY", "SELL_SHORT", "BUY_TO_COVER", "MOO"),
    "T3_SEP_POSTQUAD_SHORT": ("IWM", "SELL_SHORT", "BUY_TO_COVER", "MOC"),
    "T4_DEC_POSTOPEX_LONG": ("IWM", "BUY", "SELL", "MOC"),
    "V2_NOVDEC_VOL": ("SVXY", "BUY", "SELL", "MOC"),
    "V4_POSTOPEX_VOL": ("SVXY", "BUY", "SELL", "MOC"),
}


class AuctionClient:
    """Delegate reads, but enforce the auction deadline at the actual wire call.

    The shared guard's callback is skipped on unmapped instruments and when
    disabled. This proxy also covers those paths without altering that guard.
    """
    def __init__(self, client, before_call):
        self._client = client
        self._before_call = before_call

    def __getattr__(self, name):
        return getattr(self._client, name)

    def placeOrder(self, contract, order):
        self._before_call()
        return self._client.placeOrder(contract, order)


def assert_migrated_cycle(directory, account, row):
    """Hold all pre-install cycles, including unknown/old completed attempts.

    The cutoff is an immutable deployment receipt, never today's date. Missing
    or corrupt migration evidence must never become permission to submit.
    """
    receipt = json.loads((Path(directory) / "event_migration.json").read_text(encoding="utf-8"))
    if (receipt.get("schema") != "event-migration.v1"
            or receipt.get("account") != account
            or receipt.get("policy") != "hold_all_legacy_cycles"):
        raise RuntimeError("Event migration receipt is invalid for Primary")
    cutoff = str(receipt.get("through_entry_date", ""))
    if date.fromisoformat(cutoff).isoformat() != cutoff:
        raise RuntimeError("Event migration cutoff is invalid")
    if entry_identity(row) <= cutoff:
        raise RuntimeError("legacy Event cycle requires explicit reconciliation")


def entry_identity(row):
    """Never substitute a retry date for a missing cycle identity."""
    trade = str(row["Trade"])
    symbol, opening, closing, exit_type = TRADES[trade]
    action = str(row["Action"])
    if row["Ticker"] != symbol or action not in {opening, closing}:
        raise ValueError("Event trade/ticker/action mismatch")
    entry = str(row.get("Entry_Date", ""))
    execute = str(row["Execute_On"])
    # isoformat equality also rejects ambiguous compact/basic ISO dates.
    if date.fromisoformat(entry).isoformat() != entry:
        raise ValueError("Event Entry_Date must be YYYY-MM-DD")
    if date.fromisoformat(execute).isoformat() != execute or entry > execute:
        raise ValueError("Event Execute_On precedes entry or is malformed")
    is_entry = action == opening
    if is_entry and entry != execute:
        raise ValueError("Event entry cannot move to another session")
    kind = "entry" if is_entry else "exit"
    if row.get("Execution_ID") != f"{trade}|{entry}|{kind}":
        raise ValueError("Event Execution_ID does not match entry identity")
    expected_type = "MOC" if is_entry else exit_type
    if row["Order_Type"] != expected_type:
        raise ValueError("Event auction differs from the registered strategy")
    for field in ("Quantity", "Ref_Close"):
        value = float(row[field])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"Event {field} must be finite and positive")
        if field == "Quantity" and not value.is_integer():
            raise ValueError("Event Quantity must be whole shares")
    return entry
